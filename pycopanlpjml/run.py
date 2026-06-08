"""Orchestration utilities for copan:LPJmL simulations.

This module provides high-level functions for orchestrating complete coupled
model simulations, including parallel runtime management, profiling, output
writing, and lifecycle management.

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
load_profiling_enabled
    Load profiling setting from config files.
detect_worker_target
    Determine optimal number of Dask workers for current host.
ensure_single_instance
    Context manager preventing multiple simultaneous instances.
driver_profiler_session
    Context manager for PyInstrument profiling of the driver process.
configure_parallel_runtime
    Context manager for Dask cluster lifecycle.
log_header
    Write execution summary to stdout.

The run module handles:
- Complete simulation lifecycle orchestration
- Automatic parallel runtime detection and configuration
- Driver profiling with PyInstrument
- Output writing to configured formats (NetCDF, Parquet, CSV)
- Graceful shutdown and error handling
- Single-instance locking to prevent conflicts

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
import fcntl
import importlib
import json
import os
import signal
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, Literal, Optional

import psutil
import yaml

from . import parallel as parallel_utils
from .output import write_outputs_netcdf, write_outputs_tables

# ============================================================================
# Constants
# ============================================================================

LOGIN_NODE_WORKER_CAP = int(
    os.environ.get("PYCOPANLPJML_LOGIN_WORKER_CAP", "4")
)


def _load_pycoupler_func(module_name: str, func_name: str):
    """Dynamically load a function from pycoupler if available."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, func_name, None)
    except ImportError:  # pragma: no cover - optional dependency
        return None


_pycoupler_read_yaml = _load_pycoupler_func("pycoupler.config", "read_yaml")
_pycoupler_read_json = _load_pycoupler_func("pycoupler.utils", "read_json")


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
    lock_file : Path
        Path to advisory lock file for single-instance enforcement.
    n_workers : int
        Target number of Dask workers.
    """

    config_file: str
    run_name: str
    output_dir: Path
    timestamp: str
    profiling: bool
    lock_file: Path
    n_workers: int


# Type alias for model factory functions
ModelFactory = Callable[[str], object]

# Environment variables indicating MPI availability
MPI_ENV_VARS = [
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "MPI_LOCALNRANKS",
]

# Environment variables indicating Slurm job context
SLURM_ENV_VARS = [
    "SLURM_JOB_ID",
    "SLURM_NTASKS",
    "SLURM_JOB_NAME",
]


# ============================================================================
# Logging Utilities
# ============================================================================


def _log_message(
    message: str,
    *,
    stream: Literal["stderr", "stdout", "both"] = "stderr",
) -> None:
    """Emit a message to the requested std stream(s) with flush semantics.

    Parameters
    ----------
    message : str
        Message to emit.
    stream : {'stderr', 'stdout', 'both'}
        Target stream(s) for output.
    """
    targets = []
    if stream in ("stdout", "both"):
        targets.append(sys.stdout)
    if stream in ("stderr", "both"):
        targets.append(sys.stderr)
    for target in targets:
        print(message, file=target, flush=True)


def _stderr_logger(message: str) -> None:
    """Emit a message to stderr (for backward compatibility)."""
    _log_message(message, stream="stderr")


def _status_logger(message: str) -> None:
    """Emit prominent status updates to stdout."""
    _log_message(message, stream="stdout")


# ============================================================================
# Configuration Loading
# ============================================================================


def _profiling_from_config(config: Mapping | None) -> dict:
    """Extract profiling settings from a config mapping."""
    if not isinstance(config, Mapping):
        return {}
    if "profiling" in config and isinstance(config["profiling"], Mapping):
        return dict(config["profiling"])
    coupled = config.get("coupled_config")
    if isinstance(coupled, Mapping):
        profile = coupled.get("profiling")
        if isinstance(profile, Mapping):
            return dict(profile)
    return {}


def _read_config_yaml(path: Path) -> dict:
    """Read a YAML config file, using pycoupler if available."""
    if _pycoupler_read_yaml:
        try:
            return _pycoupler_read_yaml(str(path), dict)
        except Exception:
            pass
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}


def _read_config_json(path: Path | str) -> dict:
    """Read a JSON config file, using pycoupler if available."""
    if _pycoupler_read_json:
        try:
            return _pycoupler_read_json(str(path)) or {}
        except Exception:
            pass
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle) or {}
    except Exception:
        return {}


def load_profiling_enabled(
    config_file: Optional[str] = None,
) -> bool:
    """Read profiling setting from configuration files.

    Checks model-specific config first (if config_file provided), then falls
    back to library defaults. Model-specific settings override library
    defaults.

    Supports both old format (profiling.driver: true) and new format
    (profiling: true) for backward compatibility.

    Parameters
    ----------
    config_file : str, optional
        Path to the main config file. If provided, checks for config.yaml in
        the same directory first.

    Returns
    -------
    bool
        Whether profiling is enabled.
    """
    # Build search paths (same order as _load_pycopanlpjml_config)
    config_paths = []

    if config_file:
        config_dir = Path(config_file).parent
        config_paths.append(config_dir / "pycopanlpjml_config.yaml")
        config_paths.append(config_dir / "config.yaml")

    # Add default location
    default_config = Path(__file__).resolve().with_name("config.yaml")
    config_paths.append(default_config)

    # Default: profiling disabled
    enabled = False

    for config_path in config_paths:
        if not config_path.exists() or not config_path.is_file():
            continue

        config_data = None
        suffix = config_path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            config_data = _read_config_yaml(config_path)
        elif suffix == ".json":
            config_data = _read_config_json(config_path)

        if config_data:
            profiling_val = config_data.get("profiling")
            if isinstance(profiling_val, bool):
                enabled = profiling_val
            elif isinstance(profiling_val, dict):
                # Backward compatibility: profiling.driver
                enabled = bool(profiling_val.get("driver", enabled))

    # Primary config (JSON) overrides everything
    if config_file:
        primary_data = _read_config_json(config_file)
        if primary_data:
            # Check top-level profiling
            profiling_val = primary_data.get("profiling")
            if isinstance(profiling_val, bool):
                enabled = profiling_val
            elif isinstance(profiling_val, dict):
                enabled = bool(profiling_val.get("driver", enabled))
            
            # Also check coupled_config.profiling (pycoupler stores settings there)
            coupled_config = primary_data.get("coupled_config")
            if isinstance(coupled_config, dict):
                coupled_profiling = coupled_config.get("profiling")
                if isinstance(coupled_profiling, bool):
                    enabled = coupled_profiling
                elif isinstance(coupled_profiling, dict):
                    enabled = bool(coupled_profiling.get("driver", enabled))

    return enabled


# ============================================================================
# Worker Detection
# ============================================================================


def detect_worker_target(max_cap: int = 128) -> int:
    """Determine optimal number of Dask workers for the current host.

    Checks in order:
    1. PYCOPANLPJML_MAX_WORKERS environment variable
    2. SLURM_CPUS_ON_NODE environment variable
    3. System CPU count (with login node safeguard)

    Parameters
    ----------
    max_cap : int
        Maximum allowed workers (default: 128).

    Returns
    -------
    int
        Target number of workers (at least 1).
    """
    env_cap = os.environ.get("PYCOPANLPJML_MAX_WORKERS")
    if env_cap:
        try:
            desired = max(1, min(int(env_cap), max_cap))
            _status_logger(
                f"✓ Using PYCOPANLPJML_MAX_WORKERS={desired} "
                "for LocalCluster worker target."
            )
            return desired
        except ValueError:
            _status_logger(
                f"⚠ Invalid PYCOPANLPJML_MAX_WORKERS value '{env_cap}'; ignoring."  # noqa: E501
            )

    slurm_cpus = os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_cpus:
        try:
            return max(1, min(int(slurm_cpus), max_cap))
        except ValueError:
            pass

    cpu_count = psutil.cpu_count(logical=True) or max_cap

    if not _parallel_environment_available():
        capped = max(1, min(LOGIN_NODE_WORKER_CAP, cpu_count, max_cap))
        _status_logger(
            "✓ No Slurm/MPI environment detected; "
            f"capping LocalCluster to {capped} workers "
            f"(cpu_count={cpu_count}, login safeguard)."
        )
        return capped

    return max(1, min(cpu_count, max_cap))


def _parallel_environment_available() -> bool:
    """Detect whether a Slurm/MPI job context is active."""
    env = os.environ
    return any(var in env for var in MPI_ENV_VARS + SLURM_ENV_VARS)


# ============================================================================
# Parallelization Settings Extraction
# ============================================================================


def _extract_parallelization_settings(
    model,
    *,
    fallback_file: Optional[str] = None,
) -> tuple[str | None, int | None]:
    """Return (mode, max_workers) from any available config source."""

    def _read_settings(container) -> tuple[str | None, int | None]:
        if container is None:
            return (None, None)
        settings = None
        if isinstance(container, Mapping):
            settings = container.get("parallelization")
        else:
            settings = getattr(container, "parallelization", None)
        if settings is None:
            return (None, None)

        if isinstance(settings, Mapping):
            mode = settings.get("mode")
            max_workers = settings.get("max_workers")
        else:
            mode = getattr(settings, "mode", None)
            max_workers = getattr(settings, "max_workers", None)
        if isinstance(mode, str):
            mode = mode.lower()
        try:
            max_workers = int(max_workers) if max_workers is not None else None
        except (TypeError, ValueError):
            max_workers = None
        return (mode, max_workers)

    sources = [
        getattr(model.config, "coupled_config", None),
        getattr(model, "pycopanlpjml_config", None),
    ]

    final_mode = None
    final_max = None
    for source in sources:
        mode, max_workers = _read_settings(source)
        if mode is not None:
            final_mode = mode
        if max_workers is not None:
            final_max = max_workers
        if final_mode is not None and final_max is not None:
            break

    if final_mode is None and fallback_file:
        try:
            with open(fallback_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            coupled_cfg = data.get("coupled_config") or {}
            settings = coupled_cfg.get("parallelization") or {}
            mode = settings.get("mode")
            max_workers = settings.get("max_workers")
            if isinstance(mode, str):
                final_mode = mode.lower()
            try:
                if max_workers is not None:
                    final_max = int(max_workers)
            except (TypeError, ValueError):
                final_max = None
        except Exception:
            pass

    return final_mode, final_max


def _needs_embedded_dask_runtime(
    model,
    *,
    desired_mode: Optional[str] = None,
) -> bool:
    """Return True if config requests Dask but no client is attached."""

    mode = desired_mode
    if mode is None:
        mode, _ = _extract_parallelization_settings(model)
    wants_dask = mode == "dask"
    if not wants_dask:
        return False

    executor = getattr(model, "_parallel_executor", None)
    config = getattr(executor, "config", None) if executor else None
    has_client = bool(getattr(config, "_client", None))
    is_already_dask = getattr(config, "mode", None) == "dask"
    return not (has_client and is_already_dask)


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
    profiling = load_profiling_enabled(config_file)
    cfg_path = Path(config_file).expanduser().resolve()
    cfg_dir = cfg_path.parent

    cfg_name = cfg_path.name
    if cfg_name.startswith("config_") and cfg_name.endswith(".json"):
        run_name = cfg_name[len("config_") : -len(".json")]
    else:
        run_name = cfg_path.stem

    output_dir = cfg_dir / "output" / run_name
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    output_dir.mkdir(parents=True, exist_ok=True)

    return RunContext(
        config_file=str(cfg_path),
        run_name=run_name,
        output_dir=output_dir,
        timestamp=timestamp,
        profiling=profiling,
        lock_file=Path("/tmp/inseeds_coupling_lock.pid"),
        n_workers=detect_worker_target(),
    )


# ============================================================================
# Context Managers
# ============================================================================


@contextmanager
def ensure_single_instance(lock_file: Path) -> Iterator[None]:
    """Prevent multiple simultaneous instances by holding an advisory lock.

    Uses fcntl file locking to ensure only one simulation runs at a time.

    Parameters
    ----------
    lock_file : Path
        Path to the lock file.

    Yields
    ------
    None

    Raises
    ------
    RuntimeError
        If another instance is already running.
    """
    lock_fd = open(lock_file, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_fd.close()
        raise RuntimeError(
            "Another copan:LPJmL process is already running."
        ) from exc

    try:
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
        yield
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            lock_fd.close()


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
    saved = False  # Track if profiling has already been saved

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
            _status_logger(f"⚠ Error saving profiler output ({reason}): {exc}")

    def _signal_handler(signum, _frame):
        signal_name = signal.Signals(signum).name
        _status_logger(
            f"⚠ Received {signal_name}; saving profiler before exit."
        )
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


@contextmanager
def configure_parallel_runtime(
    model, context: RunContext
) -> Iterator[parallel_utils.LocalDaskRuntime]:
    """Create, scale, and tear down a local Dask cluster.

    Manages the complete lifecycle of a LocalCluster, including startup,
    model configuration, and graceful shutdown.

    Parameters
    ----------
    model
        Model instance to configure for Dask execution.
    context : RunContext
        Run context with worker count and settings.

    Yields
    ------
    LocalDaskRuntime
        The active Dask runtime for the duration of the context.
    """
    runtime = parallel_utils.start_local_dask_cluster(
        context.n_workers,
        scratch_directory=None,
        logger=_stderr_logger,
    )
    _status_logger(
        "✓ Dask cluster online "
        f"(current workers: {runtime.workers}/{context.n_workers})"
    )
    parallel_utils.configure_model_for_dask(
        model,
        runtime,
        backend="dask.distributed (LocalCluster)",
    )
    try:
        yield runtime
    finally:
        try:
            runtime.close()
            _status_logger("✓ Dask cluster shut down")
        except TimeoutError as exc:
            _status_logger(
                "⚠ Timed out while shutting down the Dask cluster; some "
                "worker processes were terminated forcefully."
            )
            _stderr_logger(f"  Details: {exc}")
        except Exception as exc:  # pragma: no cover - defensive logging
            _status_logger(
                "⚠ Failed to shut down the Dask cluster cleanly; continuing."
            )
            _stderr_logger(f"  Details: {exc}")


@contextmanager
def worker_profiler_session(
    runtime: parallel_utils.LocalDaskRuntime, context: RunContext
) -> Iterator[None]:
    """Worker profiling session (removed - no-op for backward compatibility)."""
    yield


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

        # Always use temporary storage for Zarr (outputs are written to final
        # formats after simulation)
        import tempfile

        zarr_store_path = os.path.join(
            tempfile.gettempdir(), f"inseeds_outputs_{os.getpid()}.zarr"
        )

        # Determine output directory for final files (NetCDF/Parquet/CSV)
        if hasattr(model, "config") and hasattr(model.config, "sim_path"):
            sim_path = model.config.sim_path
            sim_name = getattr(model.config, "sim_name", context.run_name)
        else:
            # Fallback: derive from config_file path
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
                            f"  ✗ Failed to write {fmt} output: {exc}"
                        )  # noqa: E501
                    missing = []
            for fmt in table_formats:
                path = table_export_paths.get(fmt)
                if path:
                    _status_logger(f"  ✓ {fmt.upper()} output written: {path}")

        for fmt in other_formats:
            try:
                if fmt == "netcdf":
                    model_prefix = None
                    if hasattr(model, "config"):
                        model_prefix = getattr(
                            model.config, "coupled_model", None
                        )
                    prefix = model_prefix or context.run_name

                    # Look for LPJmL grid file to use as template for alignment
                    lpjml_grid_file = None
                    for candidate in ["grid.nc4", "country.nc4", "soilno3.nc4"]:
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
        f"  Planned Dask workers: {context.n_workers}",
        "  InSEEDS: setup is single-process on the driver; after that, "
        "country updates use Dask (actors by default). LPJmL uses MPI separately.",
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

    # Note: log_file deprecated - stdout/stderr logging is sufficient
    _status_logger("✓ Simulation completed!")

    # Write outputs if configured
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
    - Run context setup (paths, profiling, worker count)
    - Single-instance locking
    - Parallel runtime configuration (if requested)
    - Driver and worker profiling
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
    model = None

    try:
        with ensure_single_instance(context.lock_file):
            model = model_factory(config_file=context.config_file)
            mode, max_workers = _extract_parallelization_settings(
                model, fallback_file=context.config_file
            )
            if max_workers and max_workers > 0:
                desired = max(1, min(context.n_workers, int(max_workers)))
                if desired != context.n_workers:
                    _status_logger(
                        f"✓ Limiting worker target to {desired} "
                        f"(parallelization.max_workers={max_workers})"
                    )
                context.n_workers = desired
            parallel_requested = context.n_workers > 1
            env_parallel_possible = (
                parallel_requested and _parallel_environment_available()
            )
            force_local_dask = _needs_embedded_dask_runtime(
                model, desired_mode=mode
            )
            # Respect explicit serial mode setting
            if mode == "serial":
                parallel_available = False
                if parallel_requested:
                    _status_logger(
                        "✓ Serial mode explicitly requested; "
                        "bypassing parallel execution."
                    )
            else:
                parallel_available = env_parallel_possible or force_local_dask
            if not parallel_available and parallel_requested and mode != "serial":
                _status_logger(
                    "⚠ Parallel execution requested, but no MPI/Slurm "
                    "environment was detected. Falling back to serial mode."
                )
            if force_local_dask and not env_parallel_possible and mode != "serial":
                _status_logger(
                    "✓ Dask mode requested without Slurm/MPI environment; "
                    "starting embedded LocalCluster."
                )
            with driver_profiler_session(context):
                if parallel_available:
                    with configure_parallel_runtime(model, context) as runtime:
                        with worker_profiler_session(runtime, context):
                            _iterate_simulation(
                                model,
                                context,
                            )
                else:
                    _iterate_simulation(
                        model,
                        context,
                    )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        sys.exit(0)
    finally:
        executor = (
            getattr(model, "_parallel_executor", None) if model else None
        )  # noqa: E501
        if executor is not None:
            try:
                executor.close()
            except Exception as exc:  # pragma: no cover - defensive shutdown
                _status_logger(
                    f"⚠ Failed to close parallel executor cleanly: {exc}"
                )  # noqa: E501
