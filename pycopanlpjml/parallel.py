"""Auto-detecting parallel execution module for pycopanlpjml.

This module automatically detects whether the code is running in a parallel
environment (MPI, Dask, or HPC scheduler) and configures the appropriate
parallelization backend.

Classes
-------
ParallelSetup
    Configuration container for parallel execution settings.
ParallelExecutor
    Unified executor interface for Dask, MPI, and serial backends.
LocalDaskRuntime
    Context manager for local Dask cluster lifecycle.

Functions
---------
detect_parallel_environment
    Automatically detect and configure the parallel execution environment.
get_executor
    Get an executor instance based on the parallel configuration.
start_local_dask_cluster
    Start a local Dask cluster with automatic configuration.
configure_model_for_dask
    Configure a model instance for Dask-based parallel execution.

The parallelization system handles:
- Transparent serial/parallel switching
- No code changes required in user scripts
- Automatic environment detection (Dask scheduler, MPI, HPC)
- LPJmL MPI environment reuse

Example
-------
>>> from pycopanlpjml.parallel import detect_parallel_environment
>>>
>>> # Auto-detect parallel environment
>>> options = detect_parallel_environment()
>>> print(options)  # ParallelSetup(mode='dask', rank=0/8, ...)
>>>
>>> # Get an executor for parallel task submission
>>> executor = get_executor(options)
>>> futures = executor.map(process_country, countries)
>>> results = executor.gather(futures)
"""

import logging
import os
import sys
import warnings
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, List, Callable, Any, Iterator

import numpy as np


# Import sync helpers from serial.py
from pycopanlpjml.serial import (
    resolve_dotted_path,
    extract_output_scalar,
    get_sync_attributes,
)


# Increase recursion limit for cloudpickle/Dask serialization.
# pycopancore's mixin metaclasses have deep __getattribute__ chains that
# trigger recursion when cloudpickle inspects class MROs during serialization.
# Default Python limit is ~1000, global runs with ~200 countries need much more.
_ORIGINAL_RECURSION_LIMIT = sys.getrecursionlimit()
if _ORIGINAL_RECURSION_LIMIT < 100000:
    sys.setrecursionlimit(100000)


class ParallelSetup:
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
            f"ParallelSetup(mode='{self.mode}', "
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
) -> ParallelSetup:
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
    ParallelSetup
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
    options = _get_parallelization_config(config)

    # Create result config
    setup = ParallelSetup()

    # 1. Check preferred mode
    preferred_mode = options["mode"]
    if preferred_mode == "serial":
        setup.mode = "serial"
        setup.is_parallel = False
        setup.backend = "serial (preferred)"
        return setup

    # 2. If user explicitly wants MPI, check for it first
    if preferred_mode == "mpi" and _is_lpjml_mpi_environment():
        try:
            from mpi4py import MPI

            setup._comm = MPI.COMM_WORLD
            setup.rank = setup._comm.Get_rank()
            setup.size = setup._comm.Get_size()
            setup.mode = "mpi"
            setup.is_parallel = setup.size > 1
            setup.backend = "mpi4py (LPJmL communicator)"

            # Apply max_workers limit if specified
            if options["max_workers"] > 0:
                setup.size = min(
                    setup.size, options["max_workers"]
                )

            if setup.rank == 0:
                print(
                    f"✓ Detected LPJmL MPI environment: {setup.size} processes"  # noqa
                )
                print(
                    "  Will reuse LPJmL's MPI processes during I/O wait time"
                )
                if options["debug"]:
                    print(
                        f"  Debug: MPI rank {setup.rank}/{setup.size}"  # noqa
                    )
            return setup
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
                setup._client = Client(dask_scheduler, timeout="5s")
                setup.mode = "dask"
                setup.is_parallel = True
                setup.size = len(
                    setup._client.scheduler_info()["workers"]
                )
                setup.rank = 0
                setup.backend = "dask.distributed (env var)"

                # Apply max_workers limit if specified
                if options["max_workers"] > 0:
                    setup.size = min(
                        setup.size, options["max_workers"]
                    )

                print("✓ Connected to Dask via DASK_SCHEDULER_ADDRESS")
                print(
                    f"  Dask cluster: {setup.size} workers available"
                )
                return setup

            # Second, try to get an existing client (from LocalCluster in same
            # process)
            try:
                setup._client = get_client(timeout="1s")
                setup.mode = "dask"
                setup.is_parallel = True
                setup.size = len(
                    setup._client.scheduler_info()["workers"]
                )
                setup.rank = 0
                setup.backend = "dask.distributed (existing client)"

                # Apply max_workers limit if specified
                if options["max_workers"] > 0:
                    setup.size = min(
                        setup.size, options["max_workers"]
                    )

                print("✓ Connected to existing Dask client")
                print(
                    f"  Dask cluster: {setup.size} workers available"
                )
                if options["debug"]:
                    print(
                        f"  Debug: Scheduler at {setup._client.scheduler.address}"  # noqa
                    )
                return setup
            except (ValueError, OSError):
                # No existing client, try scheduler file
                pass

            # Third, try to connect via scheduler file (from separate dask-mpi
            # cluster)
            dask_scheduler = _get_dask_scheduler_address()
            if dask_scheduler:
                setup._client = Client(dask_scheduler)
                setup.mode = "dask"
                setup.is_parallel = True
                setup.size = len(
                    setup._client.scheduler_info()["workers"]
                )
                setup.rank = 0
                setup.backend = "dask.distributed (scheduler file)"

                # Apply max_workers limit if specified
                if options["max_workers"] > 0:
                    setup.size = min(
                        setup.size, options["max_workers"]
                    )

                print(f"✓ Connected to Dask scheduler: {dask_scheduler}")
                print(
                    f"  Dask cluster: {setup.size} workers available"
                )
                return setup

        except Exception as e:
            if preferred_mode == "dask":
                # User explicitly wanted Dask but it failed
                warnings.warn(
                    f"Dask mode requested but detection failed: {e}. "
                    "Falling back to serial execution."
                )
            elif options["debug"]:
                warnings.warn(
                    f"Dask detection failed: {e}. Checking for MPI fallback."
                )

    # 4. Check for MPI as fallback (only in auto mode, after Dask failed)
    if preferred_mode == "auto" and _is_lpjml_mpi_environment():
        try:
            from mpi4py import MPI

            setup._comm = MPI.COMM_WORLD
            setup.rank = setup._comm.Get_rank()
            setup.size = setup._comm.Get_size()
            setup.mode = "mpi"
            setup.is_parallel = setup.size > 1
            setup.backend = "mpi4py (LPJmL communicator)"

            # Apply max_workers limit if specified
            if options["max_workers"] > 0:
                setup.size = min(
                    setup.size, options["max_workers"]
                )

            if setup.rank == 0:
                print(
                    f"✓ Detected LPJmL MPI environment: {setup.size} processes"  # noqa
                )
                print(
                    "  Will reuse LPJmL's MPI processes during I/O wait time"
                )
                if options["debug"]:
                    print(
                        f"  Debug: MPI rank {setup.rank}/{setup.size}"  # noqa
                    )
            return setup
        except ImportError:
            warnings.warn(
                "LPJmL MPI environment detected but mpi4py not available. "
                "Falling back to serial execution."
            )

    # 5. Default: serial execution
    setup.mode = "serial"
    setup.is_parallel = False
    setup.backend = "serial"
    if options.get("debug"):
        print("Running in serial mode")

    return setup


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
        options = config.parallelization
        if hasattr(options, "__dict__"):
            # Convert CoupledConfig to dict
            options = {
                k: v
                for k, v in options.__dict__.items()
                if not k.startswith("_")
            }
        else:
            options = dict(options)
    else:
        # Handle regular dict
        options = config.get("parallelization", {})

    # Convert string booleans to actual booleans
    if "debug" in options:
        if isinstance(options["debug"], str):
            options["debug"] = options["debug"].lower() in (
                "true",
                "1",
                "yes",
                "on",
            )

    # Apply defaults for missing keys
    for key, default_value in defaults.items():
        if key not in options or options[key] is None:
            options[key] = default_value

    return options


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
    config : ParallelSetup, optional
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
        config: Optional[ParallelSetup] = None,
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
                    comm.send(
                        items[offset : offset + count], dest=i, tag=0
                    )  # noqa
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
    workers: int  # Actual connected workers
    planned_workers: int = 0  # Target/planned workers (for actor deployment)

    def close(self) -> None:
        import logging
        import time
        # Give workers a moment to finish cleanup before closing connections
        time.sleep(0.1)
        # Temporarily suppress Dask's shutdown race condition messages
        dask_loggers = ["distributed.comm", "distributed.batched", "distributed.core"]
        original_levels = {}
        for name in dask_loggers:
            logger = logging.getLogger(name)
            original_levels[name] = logger.level
            logger.setLevel(logging.CRITICAL)
        try:
            self.client.close()
        finally:
            self.cluster.close()
            # Restore logging levels
            for name, level in original_levels.items():
                logging.getLogger(name).setLevel(level)

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
    
    # First try Dask's wait_for_workers (may return early)
    wait_ok = True
    try:
        client.wait_for_workers(n_workers, timeout=wait_timeout)
    except Exception as exc:  # pragma: no cover - depends on cluster behaviour
        wait_ok = False
        log(f"  ⚠️  wait_for_workers returned: {exc}")

    # Check how many workers we actually have
    # Use client.nthreads() which is more reliable than scheduler_info()
    # scheduler_info() can return stale/cached data
    def _get_worker_count():
        """Get worker count using multiple methods for reliability."""
        try:
            # Method 1: nthreads() - most reliable for active workers
            nthreads = client.nthreads()
            if nthreads:
                return len(nthreads)
        except Exception:
            pass
        
        try:
            # Method 2: scheduler_info() - may be stale but better than nothing
            scheduler_info = client.scheduler_info()
            workers = scheduler_info.get("workers", {})
            if workers:
                return len(workers)
        except Exception:
            pass
        
        try:
            # Method 3: cluster.workers attribute
            if hasattr(cluster, "workers"):
                return len(cluster.workers)
        except Exception:
            pass
        
        return 0
    
    worker_count = _get_worker_count()

    # If we got fewer workers than expected, wait a bit for stragglers
    # Most workers should connect within a few seconds
    target_workers = max(1, int(n_workers * 0.95))
    min_workers = max(1, int(n_workers * 0.50))
    max_additional_wait = 60  # Only wait 60 more seconds (workers should be fast)
    poll_interval = 2  # Check every 2 seconds
    
    if worker_count < target_workers:
        log(
            f"  Waiting for workers: {worker_count}/{n_workers} connected "
            f"(target: {target_workers})..."
        )
        additional_wait_start = time.time()
        last_count = worker_count
        stall_count = 0
        
        while worker_count < target_workers:
            elapsed_additional = time.time() - additional_wait_start
            
            # Stop if we've waited too long
            if elapsed_additional >= max_additional_wait:
                # Final check with forced sync
                try:
                    # Force a round-trip to scheduler
                    client.run(lambda: None)
                    worker_count = _get_worker_count()
                except Exception:
                    pass
                break
            
            # Stop if we have minimum and no progress for 20s
            if worker_count >= min_workers and stall_count >= 10:  # 10 * 2s = 20s
                break
            
            time.sleep(poll_interval)
            worker_count = _get_worker_count()
            
            # Track stalls (no new workers)
            if worker_count == last_count:
                stall_count += 1
            else:
                stall_count = 0
                log(f"  ... {worker_count}/{n_workers} workers connected")
            last_count = worker_count

    elapsed = time.time() - start_time
    # Report based on what we actually have
    log(f"  Worker startup: {worker_count}/{n_workers} reported in {elapsed:.1f}s")
    if worker_count < n_workers:
        log(
            "  Note: Dask may report fewer workers than actually running. "
            "Check stderr for 'Register worker' count."
        )

    return LocalDaskRuntime(
        cluster=cluster,
        client=client,
        workers=worker_count,
        planned_workers=n_workers,  # Store planned count for actor deployment
    )


def configure_model_for_dask(
    model,
    runtime: LocalDaskRuntime,
    backend: str = "dask.distributed (LocalCluster)",
) -> ParallelSetup:
    """Attach a Dask-backed ParallelSetup to the given model."""

    config = ParallelSetup()
    config.mode = "dask"
    config.is_parallel = True
    config._client = runtime.client
    # Use planned_workers (not actual) so ActorManager waits for more workers
    # Workers may still be connecting in the background
    config.size = runtime.planned_workers if runtime.planned_workers > 0 else runtime.workers
    config.rank = 0
    config.backend = backend
    model._parallel_executor.config = config
    return config


# =============================================================================
# DASK ACTORS FOR COUNTRY UPDATES
# =============================================================================

from pycopanlpjml.serial import (
    serialize_country_for_worker,
    deserialize_country,
)


class CountryActor:
    """Persistent country wrapper on Dask worker.

    The country is serialized once at deployment (using detach/reattach).
    After that it persists on the worker - only data arrays are transferred.

    Parameters
    ----------
    country_payload : bytes
        Serialized country payload from serialize_country_for_worker.
    """

    def __init__(self, country_payload):
        self.country = deserialize_country(country_payload)
        self.indices = self.country.indices.copy()

        # Cache individual references for output sync
        self._individuals = []
        for cell in getattr(self.country, "_cells", []):
            self._individuals.extend(getattr(cell, "_individuals", []))

    def update(self, t, from_earth_data):
        """Update country with new LPJmL output, return changes.

        Parameters
        ----------
        t : int
            Current simulation year.
        from_earth_data : dict
            LPJmL output data as {var_name: numpy_array}.

        Returns
        -------
        indices : ndarray
            Global cell indices for this country.
        input_updates : dict
            Changed input variables as {var_name: values}.
        individual_updates : dict or None
            Individual attribute updates for output collection.
        """
        # Update sim_year on model view for behaviours that need it
        self._set_sim_year(t)

        # Update country's output data copy with new LPJmL results
        country_output = self.country._output
        if from_earth_data and country_output is not None:
            for var, values in from_earth_data.items():
                if var in country_output.data_vars:
                    country_output[var].values[:] = values

        # Capture pre-update input state for delta detection
        # Only copy ONCE before update (not twice)
        country_input = self.country._input
        pre_input = {}
        if country_input is not None:
            for var in country_input.data_vars:
                pre_input[var] = country_input[var].values.copy()

        # Run country update (cells, individuals)
        self.country.update(t)

        # Compute input deltas (farmer decisions)
        # Compare directly against values (no copy), only copy if changed
        input_updates = {}
        if country_input is not None:
            for var, pre_values in pre_input.items():
                current_values = country_input[var].values  # View, not copy
                if not np.array_equal(current_values, pre_values):
                    input_updates[var] = current_values.copy()  # Only copy changes

        # Extract individual updates for output collection
        individual_updates = self._extract_individual_updates()

        # Extract country statistic cache for cross-country sync
        country_stats = self._extract_country_stats()

        return self.indices, input_updates, individual_updates, country_stats

    def _extract_individual_updates(self):
        """Extract individual attributes for output sync."""
        if not self._individuals:
            return None

        sync_attrs = get_sync_attributes(self._individuals[0])
        if not sync_attrs:
            return None

        values = {attr: [] for attr in sync_attrs}
        indices = []

        for ind in self._individuals:
            indices.append(getattr(ind, "_individual_index", 0))
            for attr in sync_attrs:
                if "." in attr:
                    val = resolve_dotted_path(ind, attr)
                else:
                    val = getattr(ind, attr, None)
                values[attr].append(extract_output_scalar(val))

        return {"indices": indices, "values": values}

    def _extract_country_stats(self):
        """Extract country statistic cache for cross-country sync.

        Returns the country's statistic cache (e.g., CA stats for cluster
        aggregation). Returns None if no statistic or empty cache.
        """
        if not hasattr(self.country, "_statistic"):
            return None

        statistic = self.country._statistic
        if not hasattr(statistic, "_cache"):
            return None

        cache = statistic._cache
        if not cache:
            return None

        # Return country code with the cache for identification
        country_code = getattr(self.country, "country_code", None)
        return {"country_code": country_code, "cache": cache.copy()}

    def _set_sim_year(self, t):
        """Set sim_year on model view for behaviours that access model.lpjml.sim_year."""
        world = getattr(self.country, "_world", None)
        if world is not None:
            model_view = getattr(world, "_model", None)
            if model_view is not None and hasattr(model_view, "lpjml"):
                model_view.lpjml._sim_year = t

    def update_with_broadcast(self, t, from_earth_full, cell_indices=None,
                               world_stats_update=None):
        """Update country using broadcasted from_earth data.

        This method extracts the country's slice from the full from_earth
        data on the worker side, avoiding redundant serialization.

        Parameters
        ----------
        t : int
            Current simulation year.
        from_earth_full : dict
            Full from_earth data as {var_name: numpy_array} for ALL cells.
        cell_indices : ndarray, optional
            Cell indices for this country (uses self.indices if not provided).
        world_stats_update : dict, optional
            Updated world statistics to sync to worker's snapshot.

        Returns
        -------
        tuple
            (indices, input_updates, individual_updates, country_stats)
        """
        # Update worker's world statistic snapshot with latest from driver
        if world_stats_update:
            self._update_world_stats(world_stats_update)

        # Use provided indices or fall back to stored
        indices = cell_indices if cell_indices is not None else self.indices

        # Extract this country's slice from full data
        from_earth_slice = None
        if from_earth_full is not None and len(from_earth_full) > 0:
            from_earth_slice = {}
            for var, full_arr in from_earth_full.items():
                try:
                    if full_arr.ndim == 1:
                        from_earth_slice[var] = full_arr[indices]
                    elif full_arr.ndim == 2:
                        from_earth_slice[var] = full_arr[indices, :]
                    else:
                        from_earth_slice[var] = full_arr[indices]
                except (IndexError, KeyError):
                    pass

        # Delegate to standard update
        return self.update(t, from_earth_slice)

    def _update_world_stats(self, stats_update):
        """Update worker's world statistic snapshot with driver data.

        Parameters
        ----------
        stats_update : dict
            Statistics to merge into the snapshot (e.g., cluster_stats).
        """
        world = getattr(self.country, "_world", None)
        if world is None:
            return

        statistic = getattr(world, "_statistic", None)
        if statistic is None:
            return

        # Update the snapshot's cache with new values
        if hasattr(statistic, "_cache"):
            statistic._cache.update(stats_update)


class ActorManager:
    """Manages country actors across Dask workers.

    Parameters
    ----------
    client : distributed.Client
        Dask client for actor deployment.
    countries : list
        List of Country instances to deploy.
    """

    def __init__(self, client, countries):
        self.client = client
        self.countries = countries
        self.actors = None
        self.deployed = False

    def deploy(self):
        """Deploy countries as persistent actors (one-time cost).

        Countries are pre-serialized using __getstate__ to avoid issues
        with non-serializable objects (asyncio, dask expressions, etc.).

        Before serialization, we clear _parallel_executor and _actor_manager
        from all reachable model instances to prevent indirect references
        to the Dask client (which contains asyncio tasks).
        """
        if self.deployed:
            return

        # Clear model instances that have executor/actor_manager (they contain Dask client)
        models_to_restore = []
        found_model_ids = set()

        def _find_and_clear_model(candidate):
            if candidate is None:
                return
            model_id = id(candidate)
            if model_id in found_model_ids:
                return
            if not hasattr(candidate, "_parallel_executor") and not hasattr(candidate, "_actor_manager"):
                return
            found_model_ids.add(model_id)
            executor = getattr(candidate, "_parallel_executor", None)
            actor_mgr = getattr(candidate, "_actor_manager", None)
            if executor is not None or actor_mgr is not None:
                models_to_restore.append((candidate, executor, actor_mgr))
                if executor is not None:
                    candidate._parallel_executor = None
                if actor_mgr is not None:
                    candidate._actor_manager = None

        # Search paths that could lead to model instances
        for country in self.countries:
            world = getattr(country, "_world", None)
            if world is not None:
                _find_and_clear_model(getattr(world, "_model", None))
                _find_and_clear_model(getattr(world, "model", None))
            _find_and_clear_model(getattr(country, "_model", None))
            _find_and_clear_model(getattr(country, "model", None))

            for cell in getattr(country, "_cells", None) or []:
                _find_and_clear_model(getattr(cell, "_model", None))
                _find_and_clear_model(getattr(cell, "model", None))
                for ind in getattr(cell, "_individuals", None) or set():
                    _find_and_clear_model(getattr(ind, "_model", None))
                    _find_and_clear_model(getattr(ind, "model", None))
                    behaviour = getattr(ind, "behaviour", None)
                    if behaviour is not None:
                        agent = getattr(behaviour, "agent", None)
                        if agent is not None:
                            _find_and_clear_model(getattr(agent, "_model", None))
                            _find_and_clear_model(getattr(agent, "model", None))

        try:
            # Pre-serialize countries to bytes (handles recursion limit internally)
            # This avoids Dask's serializer which triggers pycopancore metaclass recursion
            payloads = [serialize_country_for_worker(c) for c in self.countries]
            
            # Scatter payloads to workers first to avoid inline serialization
            # in the task graph which can cause recursion issues
            scattered = self.client.scatter(payloads, broadcast=False)
            
            # Submit actors using scattered references
            futures = [
                self.client.submit(CountryActor, p, actor=True, pure=False)
                for p in scattered
            ]
            self.actors = self.client.gather(futures)
            self.deployed = True
        finally:
            # Restore the executors and actor managers
            for model, executor, actor_mgr in models_to_restore:
                if executor is not None:
                    model._parallel_executor = executor
                if actor_mgr is not None:
                    model._actor_manager = actor_mgr

    def update_all(self, t, world_output, world_stats_update=None):
        """Update all actors in parallel using broadcast optimization.

        Instead of slicing data on the driver and sending N separate payloads,
        we broadcast the full data once and let workers extract their slices.
        This dramatically reduces serialization overhead for many countries.

        Parameters
        ----------
        t : int
            Current simulation year.
        world_output : xarray.Dataset
            Full world output data from LPJmL.
        world_stats_update : dict, optional
            World statistics to sync to workers (e.g., cluster_stats).

        Returns
        -------
        list of tuples
            Results from each actor: (indices, input_updates, individual_updates, country_stats).
        """
        if not self.deployed:
            self.deploy()

        # Prepare from_earth data as pure numpy (no xarray/dask refs)
        # This is done ONCE and broadcast to all workers
        from_earth_dict = self._prepare_from_earth_broadcast(world_output)

        # Pre-extract cell indices for all countries (avoid references to country objects)
        all_indices = [np.asarray(c.indices, dtype=np.int64) for c in self.countries]

        # Submit updates to all actors using broadcast pattern
        # Each actor extracts its own slice from the full data
        futures = []
        for actor, indices in zip(self.actors, all_indices):
            futures.append(actor.update_with_broadcast(
                t, from_earth_dict, indices, world_stats_update
            ))

        # Collect results
        return [f.result() for f in futures]

    def _prepare_from_earth_broadcast(self, world_output):
        """Convert world_output to pure numpy dict for efficient broadcast.

        Ensures all data is fully materialized with no Dask/xarray references
        that could cause serialization issues.

        Parameters
        ----------
        world_output : xarray.Dataset or None
            LPJmL output data.

        Returns
        -------
        dict
            Variable name -> numpy array mapping.
        """
        import pickle

        if world_output is None:
            return {}

        result = {}
        for var in world_output.data_vars:
            try:
                da = world_output[var]

                # Force computation if Dask-backed
                if hasattr(da, "compute"):
                    da = da.compute()

                data = da.values

                # Force computation if Dask array
                if hasattr(data, "compute"):
                    data = data.compute()

                # Sever ALL references to Dask/xarray via list conversion
                # This is slower but guarantees clean serialization
                dtype = data.dtype
                shape = data.shape
                as_list = data.tolist()
                arr = np.array(as_list, dtype=dtype)
                if arr.shape != shape:
                    arr = arr.reshape(shape)

                result[var] = arr
            except Exception:
                pass

        # Pickle round-trip to ensure no hidden Dask references
        try:
            result = pickle.loads(pickle.dumps(result))
        except Exception:
            pass

        return result

    def shutdown(self):
        """Release actors."""
        self.actors = None
        self.deployed = False
