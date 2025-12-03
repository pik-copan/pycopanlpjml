"""Auto-detecting parallel execution module for pycopanlpjml.

This module automatically detects whether the code is running in a parallel
environment (MPI, Dask, or HPC scheduler) and configures the appropriate
parallelization backend.

Key features:
- Transparent serial/parallel switching
- No code changes required in user scripts
- Automatic environment detection
- Supports both Dask and MPI backends
"""

import os
import sys
import warnings
import logging
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, List, Callable, Any, Iterator


class ParallelConfig:
    """Configuration for parallel execution.

    Attributes
    ----------
    mode : str
        Parallelization mode: 'serial', 'dask', 'mpi'
    is_parallel : bool
        Whether parallel execution is active
    rank : int
        Process rank (0 for serial, MPI rank, or Dask worker ID)
    size : int
        Total number of parallel workers
    backend : str
        Backend name for logging/debugging
    """

    def __init__(self):
        self.mode: Literal["serial", "dask", "mpi"] = "serial"
        self.is_parallel: bool = False
        self.rank: int = 0
        self.size: int = 1
        self.backend: Optional[str] = None
        self._client = None
        self._comm = None

    def __repr__(self):
        return (
            f"ParallelConfig(mode='{self.mode}', "
            f"rank={self.rank}/{self.size}, "
            f"backend='{self.backend}')"
        )

    def __getstate__(self):
        """Drop non-serializable backend handles when pickling."""
        state = self.__dict__.copy()
        # Dask client and MPI communicator aren't pickle-friendly.
        state["_client"] = None
        state["_comm"] = None
        return state

    def __setstate__(self, state):
        """Restore pickled state."""
        self.__dict__.update(state)
        # Ensure handles default to None when coming from pickle.
        self._client = None
        self._comm = None


def detect_parallel_environment(
    config: Optional[dict] = None,
) -> ParallelConfig:
    """Automatically detect the parallel execution environment.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary with parallelization settings.
        If None, uses default settings.

    Config options:
        mode : str, default="auto"
            - "auto": Auto-detect (Dask → MPI → serial)
            - "serial": Force serial mode (no parallelization)
            - "dask": Force Dask mode (fail if not available)
            - "mpi": Force MPI mode (fail if not available)
        max_workers : int, default=0
            Maximum number of workers (0 = no limit)
        debug : bool, default=False
            Print detailed detection information

    Detection order (mode="auto"):
    1. Check if user explicitly requested "serial" → return serial
    2. Check if user explicitly requested "mpi" → check MPI only
    3. Check for Dask client/scheduler → use Dask if available
    4. Check for LPJmL MPI environment → reuse LPJmL's MPI processes
    5. Default to serial execution

    Important: MPI detection now checks if we're ACTUALLY in an MPI process,
    not just if MPI environment variables exist (which can be misleading in
    Slurm allocations where Python script runs outside of mpirun/srun).

    Returns
    -------
    ParallelConfig
        Configuration object with detected settings

    Examples
    --------
    >>> # Auto-detection (default)
    >>> config = detect_parallel_environment()
    >>> if config.is_parallel:
    ...     print(f"Running in {config.mode} mode with {config.size} workers")

    >>> # Force serial mode
    >>> config = detect_parallel_environment(
    ...     {"parallelization": {"mode": "serial"}}
    ... )

    >>> # Force Dask mode
    >>> config = detect_parallel_environment(
    ...     {"parallelization": {"mode": "dask"}}
    ... )
    """
    # Get parallelization config with defaults
    parallel_config = _get_parallelization_config(config)

    # Create result config
    result_config = ParallelConfig()

    # 1. Check preferred mode
    preferred_mode = parallel_config["mode"]
    if preferred_mode == "serial":
        result_config.mode = "serial"
        result_config.is_parallel = False
        result_config.backend = "serial (preferred)"
        return result_config

    # 2. If user explicitly wants MPI, check for it first
    if preferred_mode == "mpi" and _is_lpjml_mpi_environment():
        try:
            from mpi4py import MPI

            result_config._comm = MPI.COMM_WORLD
            result_config.rank = result_config._comm.Get_rank()
            result_config.size = result_config._comm.Get_size()
            result_config.mode = "mpi"
            result_config.is_parallel = result_config.size > 1
            result_config.backend = "mpi4py (LPJmL communicator)"

            # Apply max_workers limit if specified
            if parallel_config["max_workers"] > 0:
                result_config.size = min(
                    result_config.size, parallel_config["max_workers"]
                )

            if result_config.rank == 0:
                print(
                    f"✓ Detected LPJmL MPI environment: {result_config.size} processes"  # noqa
                )
                print(
                    "  Will reuse LPJmL's MPI processes during I/O wait time"
                )
                if parallel_config["debug"]:
                    print(
                        f"  Debug: MPI rank {result_config.rank}/{result_config.size}"  # noqa
                    )
            return result_config
        except ImportError:
            warnings.warn(
                "LPJmL MPI environment detected but mpi4py not available. "
                "Falling back to serial execution."
            )

    # 3. Check for Dask client/scheduler (prioritized in auto mode)
    if preferred_mode in ["auto", "dask"]:
        try:
            from dask.distributed import Client, get_client

            # First, check environment variable (set by main script)
            dask_scheduler = os.environ.get("DASK_SCHEDULER_ADDRESS")
            if dask_scheduler:
                result_config._client = Client(dask_scheduler, timeout="5s")
                result_config.mode = "dask"
                result_config.is_parallel = True
                result_config.size = len(
                    result_config._client.scheduler_info()["workers"]
                )
                result_config.rank = 0
                result_config.backend = "dask.distributed (env var)"

                # Apply max_workers limit if specified
                if parallel_config["max_workers"] > 0:
                    result_config.size = min(
                        result_config.size, parallel_config["max_workers"]
                    )

                print("✓ Connected to Dask via DASK_SCHEDULER_ADDRESS")
                print(
                    f"  Dask cluster: {result_config.size} workers available"
                )
                return result_config

            # Second, try to get an existing client (from LocalCluster in same
            # process)
            try:
                result_config._client = get_client(timeout="1s")
                result_config.mode = "dask"
                result_config.is_parallel = True
                result_config.size = len(
                    result_config._client.scheduler_info()["workers"]
                )
                result_config.rank = 0
                result_config.backend = "dask.distributed (existing client)"

                # Apply max_workers limit if specified
                if parallel_config["max_workers"] > 0:
                    result_config.size = min(
                        result_config.size, parallel_config["max_workers"]
                    )

                print("✓ Connected to existing Dask client")
                print(
                    f"  Dask cluster: {result_config.size} workers available"
                )
                if parallel_config["debug"]:
                    print(
                        f"  Debug: Scheduler at {result_config._client.scheduler.address}"  # noqa
                    )
                return result_config
            except (ValueError, OSError):
                # No existing client, try scheduler file
                pass

            # Third, try to connect via scheduler file (from separate dask-mpi
            # cluster)
            dask_scheduler = _get_dask_scheduler_address()
            if dask_scheduler:
                result_config._client = Client(dask_scheduler)
                result_config.mode = "dask"
                result_config.is_parallel = True
                result_config.size = len(
                    result_config._client.scheduler_info()["workers"]
                )
                result_config.rank = 0
                result_config.backend = "dask.distributed (scheduler file)"

                # Apply max_workers limit if specified
                if parallel_config["max_workers"] > 0:
                    result_config.size = min(
                        result_config.size, parallel_config["max_workers"]
                    )

                print(f"✓ Connected to Dask scheduler: {dask_scheduler}")
                print(
                    f"  Dask cluster: {result_config.size} workers available"
                )
                return result_config

        except Exception as e:
            if preferred_mode == "dask":
                # User explicitly wanted Dask but it failed
                warnings.warn(
                    f"Dask mode requested but detection failed: {e}. "
                    "Falling back to serial execution."
                )
            elif parallel_config["debug"]:
                warnings.warn(
                    f"Dask detection failed: {e}. Checking for MPI fallback."
                )

    # 4. Check for MPI as fallback (only in auto mode, after Dask failed)
    if preferred_mode == "auto" and _is_lpjml_mpi_environment():
        try:
            from mpi4py import MPI

            result_config._comm = MPI.COMM_WORLD
            result_config.rank = result_config._comm.Get_rank()
            result_config.size = result_config._comm.Get_size()
            result_config.mode = "mpi"
            result_config.is_parallel = result_config.size > 1
            result_config.backend = "mpi4py (LPJmL communicator)"

            # Apply max_workers limit if specified
            if parallel_config["max_workers"] > 0:
                result_config.size = min(
                    result_config.size, parallel_config["max_workers"]
                )

            if result_config.rank == 0:
                print(
                    f"✓ Detected LPJmL MPI environment: {result_config.size} processes"  # noqa
                )
                print(
                    "  Will reuse LPJmL's MPI processes during I/O wait time"
                )
                if parallel_config["debug"]:
                    print(
                        f"  Debug: MPI rank {result_config.rank}/{result_config.size}"  # noqa
                    )
            return result_config
        except ImportError:
            warnings.warn(
                "LPJmL MPI environment detected but mpi4py not available. "
                "Falling back to serial execution."
            )

    # 5. Default: serial execution
    result_config.mode = "serial"
    result_config.is_parallel = False
    result_config.backend = "serial"
    print("Running in serial mode")

    return result_config


def _get_parallelization_config(config: Optional[dict] = None) -> dict:
    """Get parallelization configuration with defaults.

    Parameters
    ----------
    config : dict or CoupledConfig, optional
        Configuration dictionary or CoupledConfig object. If None, uses
        defaults.

    Returns
    -------
    dict
        Parallelization configuration with defaults applied
    """
    defaults = {
        "max_workers": 0,  # 0 = auto-detect
        "debug": False,
        "mode": "auto",  # 'auto', 'mpi', 'serial'
    }

    if config is None:
        return defaults

    # Handle CoupledConfig objects
    if hasattr(config, "parallelization"):
        parallel_config = config.parallelization
        if hasattr(parallel_config, "__dict__"):
            # Convert CoupledConfig to dict
            parallel_config = {
                k: v
                for k, v in parallel_config.__dict__.items()
                if not k.startswith("_")
            }
        else:
            parallel_config = dict(parallel_config)
    else:
        # Handle regular dict
        parallel_config = config.get("parallelization", {})

    # Convert string booleans to actual booleans
    if "debug" in parallel_config:
        if isinstance(parallel_config["debug"], str):
            parallel_config["debug"] = parallel_config["debug"].lower() in (
                "true",
                "1",
                "yes",
                "on",
            )

    # Apply defaults for missing keys
    for key, default_value in defaults.items():
        if key not in parallel_config or parallel_config[key] is None:
            parallel_config[key] = default_value

    return parallel_config


def _get_profiling_config(config: Optional[dict] = None) -> dict:
    """Return profiling defaults merged with config overrides."""

    defaults = {"driver": True, "workers": False}
    if config is None:
        return defaults

    if hasattr(config, "profiling"):
        profiling = getattr(config, "profiling")
        if hasattr(profiling, "__dict__"):
            profiling = {
                k: v
                for k, v in profiling.__dict__.items()
                if not k.startswith("_")
            }
        else:
            profiling = dict(profiling)
    else:
        profiling = config.get("profiling", {})

    for key, default_value in defaults.items():
        if key not in profiling or profiling[key] is None:
            profiling[key] = default_value

    return profiling


def _is_lpjml_mpi_environment() -> bool:
    """Check if running in LPJmL's MPI environment.

    This detects when pycopanlpjml is running as a coupled script
    launched by LPJmL's MPI processes (via submit_lpjml).

    IMPORTANT: This checks if we're ACTUALLY in an MPI process, not just
    if MPI environment variables exist. Environment variables can be present
    even in non-MPI processes (e.g., Slurm allocations).
    """
    # Check for MPI environment variables that indicate LPJmL MPI launch
    mpi_vars = [
        "OMPI_COMM_WORLD_SIZE",  # Open MPI
        "PMI_SIZE",  # Intel MPI
        "SLURM_NTASKS",  # SLURM with MPI
        "MPI_LOCALNRANKS",  # Various MPI implementations
    ]

    # First check: do MPI-related environment variables exist?
    has_mpi_vars = any(var in os.environ for var in mpi_vars)

    if not has_mpi_vars:
        return False

    # Second check: are we ACTUALLY in an MPI process?
    # Strategy: Check for PMI/runtime indicators that prove we were launched
    # via mpirun

    # Intel MPI specific: PMI_RANK exists only if launched via mpirun/srun
    if "PMI_RANK" in os.environ:
        # We're in a real Intel MPI process
        return True

    # Open MPI specific: OMPI_COMM_WORLD_RANK exists only in real MPI processes
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        # We're in a real Open MPI process
        return True

    # MPICH specific: PMI_ID or MPI_LOCALRANKID
    if "PMI_ID" in os.environ or "MPI_LOCALRANKID" in os.environ:
        # We're in a real MPICH process
        return True

    # If we have MPI vars but no rank-specific vars, we're likely in a Slurm
    # allocation but NOT launched via mpirun/srun
    # Don't risk calling MPI.COMM_WORLD - it can hang!
    return False


def _is_mpi_environment() -> bool:
    """Check if running in an MPI environment."""
    mpi_vars = [
        "OMPI_COMM_WORLD_SIZE",  # Open MPI
        "PMI_SIZE",  # Intel MPI
        "SLURM_NTASKS",  # SLURM with MPI
        "MPI_LOCALNRANKS",  # Various MPI implementations
    ]
    return any(var in os.environ for var in mpi_vars)


def _get_dask_scheduler_address() -> Optional[str]:
    """Get Dask scheduler address from environment or file."""
    # Check environment variable
    if "DASK_SCHEDULER_ADDRESS" in os.environ:
        return os.environ["DASK_SCHEDULER_ADDRESS"]

    # Check for scheduler file (created by dask-mpi)
    scheduler_file = os.environ.get("DASK_SCHEDULER_FILE")
    if scheduler_file and os.path.exists(scheduler_file):
        import json

        with open(scheduler_file) as f:
            info = json.load(f)
            return info.get("address")

    return None


class ParallelExecutor:
    """Executor that automatically uses the appropriate parallelization
    backend.

    This class provides a unified interface for parallel execution that works
    seamlessly in serial, Dask, or MPI environments.

    Parameters
    ----------
    config : ParallelConfig, optional
        Parallel configuration. If None, auto-detects environment.

    Examples
    --------
    >>> executor = ParallelExecutor()
    >>> countries = [...list of 200 countries...]
    >>>
    >>> # This automatically runs in parallel if available, serial otherwise
    >>> results = executor.map(lambda c: c.update(t), countries)
    """

    def __init__(
        self,
        config: Optional[ParallelConfig] = None,
        parallelization_config: Optional[dict] = None,
    ):
        self.config = config or detect_parallel_environment(
            parallelization_config
        )

    def map(self, func: Callable, items: List[Any], **kwargs) -> List[Any]:
        """Map a function over items, using parallel execution if available.

        Parameters
        ----------
        func : callable
            Function to apply to each item
        items : list
            Items to process
        **kwargs : dict
            Additional arguments passed to func

        Returns
        -------
        list
            Results from applying func to each item
        """
        if self.config.mode == "serial":
            return self._map_serial(func, items, **kwargs)
        elif self.config.mode == "dask":
            return self._map_dask(func, items, **kwargs)
        elif self.config.mode == "mpi":
            return self._map_mpi(func, items, **kwargs)

    def _map_serial(self, func, items, **kwargs):
        """Serial execution."""
        return [func(item, **kwargs) for item in items]

    def _map_dask(self, func, items, **kwargs):
        """Dask parallel execution."""
        if kwargs:
            # Wrap function to handle kwargs
            def wrapped_func(item):
                return func(item, **kwargs)

            futures = self.config._client.map(wrapped_func, items)
        else:
            futures = self.config._client.map(func, items)

        return self.config._client.gather(futures)

    def _map_mpi(self, func, items, **kwargs):
        """MPI parallel execution with load balancing.

        Uses a simple work-stealing approach:
        - Rank 0 distributes items
        - Each rank processes its items
        - Results gathered back to rank 0
        """
        comm = self.config._comm
        rank = self.config.rank
        size = self.config.size

        if rank == 0:
            # Master: distribute items
            items_per_rank = len(items) // size
            remainder = len(items) % size

            # Send items to workers
            offset = 0
            for i in range(size):
                count = items_per_rank + (1 if i < remainder else 0)
                if i == 0:
                    # Master keeps its own items
                    my_items = items[offset : offset + count]  # noqa
                else:
                    # Send to worker
                    comm.send(items[offset : offset + count], dest=i, tag=0)  # noqa
                offset += count
        else:
            # Worker: receive items
            my_items = comm.recv(source=0, tag=0)

        # Each rank processes its items
        my_results = [func(item, **kwargs) for item in my_items]

        # Gather results to rank 0
        all_results = comm.gather(my_results, root=0)

        if rank == 0:
            # Flatten list of lists
            return [item for sublist in all_results for item in sublist]
        else:
            return []

    def barrier(self):
        """Synchronization barrier (only relevant for MPI)."""
        if self.config.mode == "mpi":
            self.config._comm.Barrier()

    def is_master(self) -> bool:
        """Check if this is the master process/rank."""
        return self.config.rank == 0

    def close(self):
        """Close parallel resources."""
        if self.config.mode == "dask" and self.config._client:
            # Only close if we created the client
            if not hasattr(self, "_external_client"):
                self.config._client.close()


# Global singleton for easy access
_global_executor: Optional[ParallelExecutor] = None


def get_executor(
    reset: bool = False, config: Optional[dict] = None
) -> ParallelExecutor:
    """Get or create the global parallel executor.

    Parameters
    ----------
    reset : bool, default=False
        Force re-detection of parallel environment
    config : dict, optional
        Configuration dictionary with parallelization settings

    Returns
    -------
    ParallelExecutor
        Global parallel executor instance
    """
    global _global_executor

    if _global_executor is None or reset:
        _global_executor = ParallelExecutor(parallelization_config=config)

    return _global_executor


@dataclass
class LocalDaskRuntime:
    """Lightweight context manager bundling LocalCluster + Client."""

    cluster: Any
    client: Any
    workers: int

    def close(self) -> None:
        try:
            self.client.close()
        finally:
            self.cluster.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _default_logger(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _select_local_directory(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit

    candidates = [
        os.environ.get("SLURM_TMPDIR"),
        os.environ.get("LOCAL_SCRATCH"),
        os.environ.get("TMPDIR"),
        "/tmp",
    ]
    base = next(
        (path for path in candidates if path and os.path.isdir(path)), "/tmp"
    )
    return os.path.join(base, f"inseeds_dask_{os.getpid()}")


def start_local_dask_cluster(
    n_workers: int,
    *,
    scratch_directory: Optional[str] = None,
    wait_timeout: str = "300s",
    allowed_failures: int = 20,
    logger: Optional[Callable[[str], None]] = None,
) -> LocalDaskRuntime:
    """Start a LocalCluster, scale to n_workers, and return runtime handles."""

    from dask.distributed import Client, LocalCluster
    import psutil

    log = logger or _default_logger
    local_directory = _select_local_directory(scratch_directory)
    os.makedirs(local_directory, exist_ok=True)

    available_memory_gb = psutil.virtual_memory().available / 1e9
    per_worker_gb = max(available_memory_gb / max(n_workers, 1), 0.5)
    memory_limit = f"{per_worker_gb:.1f}GB"

    log(f"  Using Dask local_directory: {local_directory}")
    log(f"  Available memory: {available_memory_gb:.1f} GB")
    log(f"  Memory per worker: {memory_limit} ({n_workers} workers)")

    scheduler_kwargs = {"allowed_failures": allowed_failures}
    cluster = LocalCluster(
        n_workers=0,
        threads_per_worker=1,
        processes=True,
        memory_limit=memory_limit,
        silence_logs=False,
        local_directory=local_directory,
        dashboard_address=None,
        scheduler_kwargs=scheduler_kwargs,
    )
    log("  LocalCluster object created (0 workers initially)")

    client = Client(cluster)
    log(
        "  Client connected to cluster "
        f"(scheduler: {client.scheduler.address})"
    )
    log(
        "  Scaling cluster to "
        f"{n_workers} workers and waiting for them to connect..."
    )

    start_time = time.time()
    cluster.scale(n_workers)
    wait_ok = True
    try:
        client.wait_for_workers(n_workers, timeout=wait_timeout)
    except Exception as exc:  # pragma: no cover - depends on cluster behaviour
        wait_ok = False
        log(f"  ⚠️  wait_for_workers timed out: {exc}")

    scheduler_info = client.scheduler_info()
    worker_count = len(scheduler_info.get("workers", {}))
    if worker_count == 0 and hasattr(cluster, "workers"):
        worker_count = len(getattr(cluster, "workers"))

    elapsed = time.time() - start_time
    status = "complete" if wait_ok else "partial"
    log(
        "  Worker startup "
        f"{status}: {worker_count}/{n_workers} connected in {elapsed:.1f}s"
    )

    return LocalDaskRuntime(
        cluster=cluster,
        client=client,
        workers=worker_count,
    )


def configure_model_for_dask(
    model,
    runtime: LocalDaskRuntime,
    backend: str = "dask.distributed (LocalCluster)",
) -> ParallelConfig:
    """Attach a Dask-backed ParallelConfig to the given model."""

    config = ParallelConfig()
    config.mode = "dask"
    config.is_parallel = True
    config._client = runtime.client
    config.size = runtime.workers
    config.rank = 0
    config.backend = backend
    model._parallel_executor.config = config
    return config


def register_worker_profiler(
    client,
    output_dir: Path,
    timestamp: str,
    interval: float = 0.01,
) -> None:
    """Register a PyInstrument worker plugin for per-worker HTML traces."""

    from dask.distributed import WorkerPlugin

    class _PyInstrumentWorkerPlugin(WorkerPlugin):
        def __init__(self, output_dir: str, timestamp: str, interval: float):
            self._output_dir = output_dir
            self._timestamp = timestamp
            self._interval = interval
            self._profiler = None
            self._path = None

        def setup(self, worker):  # pragma: no cover - executed on workers
            try:
                from pyinstrument import Profiler
            except ImportError:
                worker.log_event("pyinstrument not available on worker")
                self._profiler = None
                return

            os.makedirs(self._output_dir, exist_ok=True)
            self._path = os.path.join(
                self._output_dir,
                f"worker_{worker.name}_{self._timestamp}.html",
            )
            self._profiler = Profiler(interval=self._interval)
            self._profiler.start()

        def teardown(self, worker):  # pragma: no cover - executed on workers
            profiler = getattr(self, "_profiler", None)
            if profiler is None or self._path is None:
                return
            try:
                profiler.stop()
                profiler.write_html(self._path)
            except Exception as exc:
                worker.log_event(f"Failed to write worker profile: {exc}")

    plugin = _PyInstrumentWorkerPlugin(str(output_dir), timestamp, interval)
    client.register_worker_plugin(plugin, name=f"pyinstrument-{timestamp}")


@contextmanager
def worker_performance_report(enabled: bool, filename: str) -> Iterator[None]:
    """Context manager that emits a Dask performance report HTML."""

    if not enabled:
        yield
        return

    from dask.distributed import performance_report

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with performance_report(filename=filename):
        yield
