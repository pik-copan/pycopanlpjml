"""Utility helpers for orchestrating coupled InSEEDS–LPJmL runs."""

from __future__ import annotations

import atexit
import fcntl
import os
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import psutil
import yaml

from pycopanlpjml import parallel as parallel_utils


@dataclass(frozen=True)
class ProfilingOptions:
    """Simple container describing profiling toggles."""

    driver: bool = True
    workers: bool = False


@dataclass
class RunContext:
    """Bookkeeping data required during a single simulation run."""

    config_file: str
    run_name: str
    output_dir: Path
    log_file: Path
    profiling_dir: Path
    worker_profile_dir: Path
    dask_report: Path
    timestamp: str
    profiling: ProfilingOptions
    lock_file: Path
    n_workers: int


ModelFactory = Callable[[str], object]

MPI_ENV_VARS = [
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "MPI_LOCALNRANKS",
]
SLURM_ENV_VARS = [
    "SLURM_JOB_ID",
    "SLURM_NTASKS",
    "SLURM_JOB_NAME",
]


def _stderr_logger(message: str) -> None:
    """Write log messages to stderr with flush semantics."""

    print(message, file=sys.stderr, flush=True)


def load_profiling_options() -> ProfilingOptions:
    """Read profiling toggles from the library's default configuration."""

    config_path = Path(__file__).resolve().with_name("config.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    profiling_cfg = config.get("profiling", {})
    return ProfilingOptions(
        driver=bool(profiling_cfg.get("driver", True)),
        workers=bool(profiling_cfg.get("workers", False)),
    )


def detect_worker_target(max_cap: int = 128) -> int:
    """Determine how many Dask workers to request for the current host."""

    slurm_cpus = os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_cpus:
        try:
            return max(1, min(int(slurm_cpus), max_cap))
        except ValueError:
            pass

    cpu_count = psutil.cpu_count(logical=True) or max_cap
    return max(1, min(cpu_count, max_cap))


def _parallel_environment_available() -> bool:
    """Detect whether a Slurm/MPI job context is active."""

    env = os.environ
    return any(var in env for var in MPI_ENV_VARS + SLURM_ENV_VARS)


def build_run_context(config_file: str) -> RunContext:
    """Build derived paths and metadata for a simulation run."""

    profiling = load_profiling_options()
    cfg_path = Path(config_file).expanduser().resolve()
    cfg_dir = cfg_path.parent

    cfg_name = cfg_path.name
    if cfg_name.startswith("config_") and cfg_name.endswith(".json"):
        run_name = cfg_name[len("config_"):-len(".json")]
    else:
        run_name = cfg_path.stem

    output_dir = cfg_dir / "output" / run_name
    profiling_dir = output_dir / "profiling"
    worker_profile_dir = profiling_dir / "workers"
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    output_dir.mkdir(parents=True, exist_ok=True)
    profiling_dir.mkdir(parents=True, exist_ok=True)
    worker_profile_dir.mkdir(parents=True, exist_ok=True)

    dask_report = profiling_dir / f"dask_workers_{timestamp}.html"
    log_file = output_dir / "inseeds_coupling.log"

    return RunContext(
        config_file=str(cfg_path),
        run_name=run_name,
        output_dir=output_dir,
        log_file=log_file,
        profiling_dir=profiling_dir,
        worker_profile_dir=worker_profile_dir,
        dask_report=dask_report,
        timestamp=timestamp,
        profiling=profiling,
        lock_file=Path("/tmp/inseeds_coupling_lock.pid"),
        n_workers=detect_worker_target(),
    )


@contextmanager
def ensure_single_instance(lock_file: Path) -> Iterator[None]:
    """Prevent multiple simultaneous instances by holding an advisory lock."""

    lock_fd = open(lock_file, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_fd.close()
        raise RuntimeError(
            "Another InSEEDS coupling process is already running."
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
    """Optional PyInstrument profiling context for the driver process."""

    if not context.profiling.driver:
        yield
        return

    try:
        from pyinstrument import Profiler
    except ImportError:
        _stderr_logger("⚠ pyinstrument not available on driver - skipping.")
        yield
        return

    output_path = (
        context.profiling_dir / f"profiling_main_{context.timestamp}.html"
    )
    profiler = Profiler(interval=0.05)

    def _save(reason: str) -> None:
        if profiler is None:
            return
        try:
            if profiler.is_running:
                profiler.stop()
            profiler.write_html(output_path)
            _stderr_logger(
                f"✓ Main process profiling saved ({reason}) -> {output_path}"
            )
        except Exception as exc:
            _stderr_logger(f"⚠ Error saving profiler output ({reason}): {exc}")

    def _signal_handler(signum, _frame):
        signal_name = signal.Signals(signum).name
        _stderr_logger(
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
    _stderr_logger(f"✓ Driver profiling started -> {output_path}")

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
    """Create, scale, and tear down the LocalCluster + Dask client."""

    runtime = parallel_utils.start_local_dask_cluster(
        context.n_workers,
        scratch_directory=None,
        logger=_stderr_logger,
    )
    _stderr_logger(
        "✓ Dask cluster started with "
        f"{runtime.workers}/{context.n_workers} workers"
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
            _stderr_logger("✓ Dask cluster shut down")
        except TimeoutError as exc:
            _stderr_logger(
                "⚠ Timed out while shutting down the Dask cluster; some "
                "worker processes were terminated forcefully."
            )
            _stderr_logger(f"  Details: {exc}")
        except Exception as exc:  # pragma: no cover - defensive logging
            _stderr_logger(
                "⚠ Failed to shut down the Dask cluster cleanly; continuing."
            )
            _stderr_logger(f"  Details: {exc}")


@contextmanager
def worker_profiler_session(
    runtime: parallel_utils.LocalDaskRuntime, context: RunContext
) -> Iterator[None]:
    """Emit Dask performance reports and per-worker PyInstrument traces."""

    if not context.profiling.workers:
        yield
        return

    context.worker_profile_dir.mkdir(parents=True, exist_ok=True)
    parallel_utils.register_worker_profiler(
        runtime.client,
        context.worker_profile_dir,
        context.timestamp,
    )
    _stderr_logger(
        "✓ Worker profiling enabled -> "
        f"{context.worker_profile_dir}/worker_<name>_{context.timestamp}.html"
    )

    with parallel_utils.worker_performance_report(
        True, str(context.dask_report)
    ):
        _stderr_logger(f"✓ Dask performance report -> {context.dask_report}")
        yield


def log_header(context: RunContext) -> None:
    """Write a short execution summary to stderr and the run log."""

    header = [
        "========================================",
        f"✓ InSEEDS Coupling run: {context.run_name}",
        f"  Config: {context.config_file}",
        f"  Output: {context.output_dir}",
        f"  Planned Dask workers: {context.n_workers}",
        f"  Profiling (driver/workers): "
        f"{context.profiling.driver}/{context.profiling.workers}",
        "========================================",
    ]
    for line in header:
        _stderr_logger(line)
    with context.log_file.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n")


def _iterate_simulation(
    model,
    context: RunContext,
    progress_every: int = 10,
) -> None:
    """Iterate over LPJmL coupling years, logging periodic progress."""

    years = list(model.lpjml.get_sim_years())
    last_year = years[-1] if years else None

    for year in years:
        _stderr_logger(f"DEBUG: Starting year {year} update...")
        try:
            model.update(year)
        except Exception as exc:
            _stderr_logger(f"✗ ERROR during year {year} update: {exc}")
            raise
        finally:
            _stderr_logger(f"DEBUG: Year {year} update COMPLETE")

        if year % progress_every == 0 or year == last_year:
            with context.log_file.open("a", encoding="utf-8") as handle:
                handle.write(f"  Year {year} completed\n")
            _stderr_logger(f"  Year {year} completed")

    with context.log_file.open("a", encoding="utf-8") as handle:
        handle.write("\n✓ Simulation completed successfully!\n")
    _stderr_logger("✓ Simulation completed!")


def run_simulation(
    config_file: str,
    model_factory: ModelFactory,
    *,
    description: str = "Coupled run",
    progress_every: int = 10,
) -> None:
    """Orchestrate the full coupled workflow for a given model builder.

    Parameters
    ----------
    config_file:
        Path to the LPJmL JSON configuration produced by pycoupler.
    model_builder:
        Callable that receives the resolved config path and returns a model
        object exposing ``lpjml.get_sim_years()`` and ``update(year)``.
    description:
        Human-readable description emitted to the log header.
    progress_every:
        Emit "year completed" messages every N years (default: 10).
    """

    context = build_run_context(config_file)
    header = [
        f"✓ {description}",
        f"  Config: {context.config_file}",
    ]
    for line in header:
        _stderr_logger(line)
    log_header(context)

    parallel_requested = context.n_workers > 1
    parallel_available = (
        parallel_requested and _parallel_environment_available()
    )

    if parallel_requested and not parallel_available:
        _stderr_logger(
            "⚠ Parallel execution requested, but no MPI/Slurm environment "
            "was detected. Falling back to serial mode."
        )

    try:
        with ensure_single_instance(context.lock_file):
            model = model_factory(config_file=context.config_file)
            with driver_profiler_session(context):
                if parallel_available:
                    with configure_parallel_runtime(model, context) as runtime:
                        with worker_profiler_session(runtime, context):
                            _iterate_simulation(
                                model,
                                context,
                                progress_every=progress_every,
                            )
                else:
                    _iterate_simulation(
                        model,
                        context,
                        progress_every=progress_every,
                    )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        sys.exit(0)
