"""Orchestration utilities for copan:LPJmL simulations.

This module provides high-level functions for orchestrating complete coupled
model simulations, including profiling, output writing, and lifecycle
management.

Classes
-------
RunContext
    Bookkeeping data for a single simulation run (paths, settings, etc.).

Functions
---------
run_simulation
    Main entry point for orchestrating a complete coupled simulation.
build_run_context
    Build derived paths and metadata for a simulation run.
read_profiling
    Load profiling setting from config files.
driver_profiler_session
    Context manager for PyInstrument profiling of the driver process.
log_header
    Write execution summary to stdout.

The run module handles:
- Complete simulation lifecycle orchestration
- Driver profiling with PyInstrument
- Output writing to configured formats (NetCDF, Parquet, CSV)
- Graceful shutdown and error handling

Example
-------
>>> from pycopanlpjml.run import run_simulation
>>>
>>> def build_model(config_file):
...     return MyModel(config_file=config_file)
...
>>> run_simulation(
...     config_file="config.json",
...     model_factory=build_model,
...     description="Production run",
... )
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, Literal, Optional

from pycoupler.config import read_yaml
from pycoupler.utils import read_json

from .output import write_outputs_netcdf, write_outputs_tables

# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class RunContext:
    """Bookkeeping data for a single simulation run.

    Contains all paths, settings, and metadata needed throughout the
    simulation lifecycle.

    Attributes
    ----------
    config_file : str
        Path to the main configuration file.
    run_name : str
        Human-readable name derived from config filename.
    output_dir : Path
        Directory for simulation outputs.
    timestamp : str
        Timestamp string for unique file naming.
    profiling : bool
        Enable PyInstrument profiling for the driver process.
    """

    config_file: str
    run_name: str
    output_dir: Path
    timestamp: str
    profiling: bool


# Type alias for model factory functions
ModelFactory = Callable[[str], object]


# ============================================================================
# Logging Utilities
# ============================================================================


def _log_message(
    message: str,
    *,
    stream: Literal["stderr", "stdout", "both"] = "stderr",
) -> None:
    """Emit a message to the requested std stream(s) with flush semantics."""
    targets = []
    if stream in ("stdout", "both"):
        targets.append(sys.stdout)
    if stream in ("stderr", "both"):
        targets.append(sys.stderr)
    for target in targets:
        print(message, file=target, flush=True)


def _stderr_logger(message: str) -> None:
    """Emit a message to stderr."""
    _log_message(message, stream="stderr")


def _status_logger(message: str) -> None:
    """Emit prominent status updates to stdout."""
    _log_message(message, stream="stdout")


# ============================================================================
# Configuration Loading
# ============================================================================


def read_profiling(
    config_file: Optional[str] = None,
) -> bool:
    """Read profiling setting from configuration files.

    Checks model-specific config first (if config_file provided), then falls
    back to library defaults. Model-specific settings override library
    defaults.

    Parameters
    ----------
    config_file : str, optional
        Path to the main config file.

    Returns
    -------
    bool
        Whether profiling is enabled.
    """
    config_paths = []

    if config_file:
        config_dir = Path(config_file).parent
        config_paths.append(config_dir / "pycopanlpjml_config.yaml")
        config_paths.append(config_dir / "config.yaml")

    default_config = Path(__file__).resolve().with_name("config.yaml")
    config_paths.append(default_config)

    enabled = False

    for config_path in config_paths:
        if not config_path.exists() or not config_path.is_file():
            continue

        config_data = None
        suffix = config_path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            config_data = read_yaml(str(config_path), dict)
        elif suffix == ".json":
            config_data = read_json(str(config_path))

        if config_data:
            profiling_val = config_data.get("profiling")
            if isinstance(profiling_val, bool):
                enabled = profiling_val
            elif isinstance(profiling_val, dict):
                enabled = bool(profiling_val.get("driver", enabled))

    if config_file:
        primary_data = read_json(str(config_file))
        if primary_data:
            profiling_val = primary_data.get("profiling")
            if isinstance(profiling_val, bool):
                enabled = profiling_val
            elif isinstance(profiling_val, dict):
                enabled = bool(profiling_val.get("driver", enabled))

            coupled_config = primary_data.get("coupled_config")
            if isinstance(coupled_config, dict):
                coupled_profiling = coupled_config.get("profiling")
                if isinstance(coupled_profiling, bool):
                    enabled = coupled_profiling
                elif isinstance(coupled_profiling, dict):
                    enabled = bool(coupled_profiling.get("driver", enabled))

    return enabled


# ============================================================================
# Run Context Construction
# ============================================================================


def build_run_context(config_file: str) -> RunContext:
    """Build derived paths and metadata for a simulation run.

    Parameters
    ----------
    config_file : str
        Path to the main configuration file.

    Returns
    -------
    RunContext
        Populated context with all paths and settings.
    """
    profiling = read_profiling(config_file)
    config_path = Path(config_file).expanduser().resolve()
    config_dir = config_path.parent

    config_name = config_path.name
    if config_name.startswith("config_") and config_name.endswith(".json"):
        run_name = config_name[len("config_") : -len(".json")]
    else:
        run_name = config_path.stem

    output_dir = config_dir / "output" / run_name
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    output_dir.mkdir(parents=True, exist_ok=True)

    return RunContext(
        config_file=str(config_path),
        run_name=run_name,
        output_dir=output_dir,
        timestamp=timestamp,
        profiling=profiling,
    )


# ============================================================================
# Context Managers
# ============================================================================


@contextmanager
def driver_profiler_session(context: RunContext) -> Iterator[None]:
    """PyInstrument profiling context for the driver process.

    Profiles the main process and saves results to HTML on completion
    or signal interruption.

    Parameters
    ----------
    context : RunContext
        Run context containing profiling settings and paths.

    Yields
    ------
    None
    """
    if not context.profiling:
        yield
        return

    try:
        from pyinstrument import Profiler
    except ImportError:
        _status_logger("⚠ pyinstrument not available - skipping profiling.")
        yield
        return

    output_path = context.output_dir / f"profiling_{context.timestamp}.html"
    profiler = Profiler(interval=0.05)
    saved = False

    def _save(reason: str) -> None:
        nonlocal saved
        if profiler is None or saved:
            return
        try:
            if profiler.is_running:
                profiler.stop()
            profiler.write_html(output_path)
            saved = True
            _status_logger(
                f"✓ Main process profiling saved ({reason}) -> {output_path}"
            )
        except Exception as exc:
            _status_logger(f"✗ Error saving profiler output ({reason}): {exc}")

    def _signal_handler(signum, _frame):
        signal_name = signal.Signals(signum).name
        _status_logger(f"Received {signal_name}; saving profiler before exit.")
        _save(f"on {signal_name.lower()}")
        raise SystemExit(128 + signum)

    original_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            original_handlers[sig] = signal.signal(sig, _signal_handler)
        except Exception:
            original_handlers[sig] = None

    atexit.register(lambda: _save("via exit handler"))
    profiler.start()
    _status_logger(f"✓ Driver profiling started -> {output_path}")

    try:
        yield
    finally:
        _save("at end of run")
        for sig, handler in original_handlers.items():
            if handler is not None:
                signal.signal(sig, handler)


# ============================================================================
# Simulation Execution
# ============================================================================


def _write_outputs_if_configured(
    model, context: RunContext, years: list
) -> None:
    """Write outputs to configured formats after simulation completes.

    Checks output configuration and writes to NetCDF, Parquet, and/or CSV
    as specified.

    Parameters
    ----------
    model
        Model instance with component and config.
    context : RunContext
        Run context with output paths.
    years : list
        List of simulation years for output range.
    """
    if not years:
        return

    try:

        def _formats_from_config(cfg):
            if cfg is None:
                return None
            if isinstance(cfg, Mapping):
                formats = cfg.get("format") or cfg.get("output_formats")
                return list(formats) if formats else None
            formats = getattr(cfg, "format", None) or getattr(
                cfg, "output_formats", None
            )
            if formats:
                return list(formats)
            if hasattr(cfg, "to_dict"):
                data = cfg.to_dict()
                if isinstance(data, Mapping):
                    formats = data.get("format") or data.get("output_formats")
                    if formats:
                        return list(formats)
            return None

        output_formats = None

        if hasattr(model, "config") and hasattr(
            model.config, "coupled_config"
        ):
            config_output = getattr(
                model.config.coupled_config, "output", None
            )
            output_formats = _formats_from_config(config_output)

        if not output_formats and hasattr(model, "pycopanlpjml_config"):
            output_config = getattr(model.pycopanlpjml_config, "output", None)
            output_formats = _formats_from_config(output_config)

        if not output_formats:
            return

        import tempfile

        zarr_store_path = os.path.join(
            tempfile.gettempdir(), f"inseeds_outputs_{os.getpid()}.zarr"
        )

        if hasattr(model, "config") and hasattr(model.config, "sim_path"):
            sim_path = model.config.sim_path
            sim_name = getattr(model.config, "sim_name", context.run_name)
        else:
            config_dir = Path(context.config_file).parent
            sim_path = str(config_dir)
            sim_name = context.run_name

        output_dir = Path(sim_path) / "output" / sim_name
        output_dir.mkdir(parents=True, exist_ok=True)

        start_year = years[0]
        end_year = years[-1]

        _status_logger(
            f"✓ Writing outputs ({', '.join(output_formats)}) "
            f"for years {start_year}-{end_year}..."
        )

        table_export_paths: Dict[str, str] = {}
        if hasattr(model, "finalize_output_streams"):
            try:
                table_export_paths = model.finalize_output_streams()
            except Exception as exc:
                _status_logger(
                    f"  ⚠ Failed to finalize streaming outputs: {exc}"
                )
                table_export_paths = {}

        table_formats = [
            fmt for fmt in output_formats if fmt in {"csv", "parquet"}
        ]
        other_formats = [
            fmt for fmt in output_formats if fmt not in {"csv", "parquet"}
        ]

        if table_formats:
            missing = [
                fmt for fmt in table_formats if fmt not in table_export_paths
            ]
            if missing:
                try:
                    paths = write_outputs_tables(
                        str(zarr_store_path),
                        str(output_dir),
                        start_year,
                        end_year,
                        formats=missing,
                    )
                    table_export_paths.update(paths)
                except Exception as exc:
                    for fmt in missing:
                        _status_logger(
                            f"  Failed to write {fmt} output: {exc}"
                        )
                    missing = []
            for fmt in table_formats:
                path = table_export_paths.get(fmt)
                if path:
                    _status_logger(f"  {fmt.upper()} output written: {path}")

        for fmt in other_formats:
            try:
                if fmt == "netcdf":
                    model_prefix = None
                    if hasattr(model, "config"):
                        model_prefix = getattr(
                            model.config, "coupled_model", None
                        )
                    prefix = model_prefix or context.run_name

                    lpjml_grid_file = None
                    for candidate in [
                        "grid.nc4",
                        "country.nc4",
                        "soilno3.nc4",
                    ]:
                        candidate_path = output_dir / candidate
                        if candidate_path.exists():
                            lpjml_grid_file = str(candidate_path)
                            break

                    nc_paths = write_outputs_netcdf(
                        str(zarr_store_path),
                        str(output_dir),
                        start_year,
                        end_year,
                        file_prefix=prefix,
                        lpjml_grid_file=lpjml_grid_file,
                    )
                    for var_name, file_path in sorted(nc_paths.items()):
                        _status_logger(
                            f"  ✓ NetCDF [{var_name}] -> {file_path}"
                        )  # noqa: E501
                else:
                    _status_logger(f"  ⚠ Unknown output format: {fmt}")
            except Exception as exc:
                _status_logger(f"  ✗ Failed to write {fmt} output: {exc}")
    except Exception as exc:
        # Don't fail simulation if output writing fails
        _status_logger(f"⚠ Output writing failed: {exc}")


def log_header(context: RunContext) -> None:
    """Write execution summary to stdout.

    Parameters
    ----------
    context : RunContext
        Run context with simulation metadata.
    """
    header = [
        "========================================",
        f"✓ copan:LPJmL run: {context.run_name}",
        f"  Config: {context.config_file}",
        f"  Output: {context.output_dir}",
        f"  Profiling: {context.profiling}",
        "========================================",
    ]
    for line in header:
        _status_logger(line)


def _iterate_simulation(
    model,
    context: RunContext,
) -> None:
    """Iterate over LPJmL coupling years and write outputs.

    Parameters
    ----------
    model
        Model instance with update() method and lpjml coupler.
    context : RunContext
        Run context for output writing.
    """
    years = list(model.lpjml.get_sim_years())
    for year in years:
        try:
            model.update(year)
        except Exception as exc:
            _status_logger(f"✗ ERROR during year {year} update: {exc}")
            raise

    _status_logger("Simulation completed!")

    _write_outputs_if_configured(model, context, years)


# ============================================================================
# Main Entry Point
# ============================================================================


def run_simulation(
    config_file: str,
    model_factory: ModelFactory,
    *,
    description: str = "Coupled run",
) -> None:
    """Orchestrate a complete coupled InSEEDS-LPJmL simulation.

    This is the main entry point for running coupled simulations. It handles:
    - Run context setup (paths, profiling)
    - Driver profiling
    - Simulation iteration
    - Output writing
    - Graceful shutdown

    Parameters
    ----------
    config_file : str
        Path to the LPJmL JSON configuration produced by pycoupler.
    model_factory : callable
        Callable that receives the resolved config path and returns a model
        object exposing ``lpjml.get_sim_years()`` and ``update(year)``.
    description : str
        Human-readable description emitted to the log header.

    Example
    -------
    >>> def build_model(config_file):
    ...     return MyModel(config_file=config_file)
    ...
    >>> run_simulation(
    ...     config_file="config.json",
    ...     model_factory=build_model,
    ... )
    """

    context = build_run_context(config_file)
    log_header(context)

    model = model_factory(config_file=context.config_file)
    with driver_profiler_session(context):
        _iterate_simulation(model, context)
