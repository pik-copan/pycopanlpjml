"""Auto-detecting parallel execution module for pycopanlpjml.

This module automatically detects whether the code is running in a parallel environment
(MPI, Dask, or HPC scheduler) and configures the appropriate parallelization backend.

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
from typing import Optional, Literal, List, Callable, Any


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
        self.mode: Literal['serial', 'dask', 'mpi'] = 'serial'
        self.is_parallel: bool = False
        self.rank: int = 0
        self.size: int = 1
        self.backend: Optional[str] = None
        self._client = None
        self._comm = None
    
    def __repr__(self):
        return (f"ParallelConfig(mode='{self.mode}', "
                f"rank={self.rank}/{self.size}, "
                f"backend='{self.backend}')")


def detect_parallel_environment(config: Optional[dict] = None) -> ParallelConfig:
    """Automatically detect the parallel execution environment.
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary with parallelization settings.
        If None, uses default settings.
    
    Detection order:
    1. Check config.parallelization.mode preference
    2. Check for LPJmL MPI environment → reuse LPJmL's MPI processes
    3. Check for explicit Dask scheduler (dask-mpi)
    4. Default to serial execution
    
    Returns
    -------
    ParallelConfig
        Configuration object with detected settings
    
    Examples
    --------
    >>> config = detect_parallel_environment()
    >>> if config.is_parallel:
    ...     print(f"Running in {config.mode} mode with {config.size} workers")
    ... else:
    ...     print("Running in serial mode")
    """
    # Get parallelization config with defaults
    parallel_config = _get_parallelization_config(config)
    
    # Create result config
    result_config = ParallelConfig()
    
    # 1. Check preferred mode
    preferred_mode = parallel_config['mode']
    if preferred_mode == 'serial':
        result_config.mode = 'serial'
        result_config.is_parallel = False
        result_config.backend = 'serial (preferred)'
        if parallel_config['debug']:
            print("Running in serial mode (preferred by config)")
        return result_config
    
    # 3. Check for LPJmL MPI environment (reuse LPJmL's MPI processes)
    if preferred_mode in ['auto', 'mpi'] and _is_lpjml_mpi_environment():
        try:
            from mpi4py import MPI
            result_config._comm = MPI.COMM_WORLD
            result_config.rank = result_config._comm.Get_rank()
            result_config.size = result_config._comm.Get_size()
            result_config.mode = 'mpi'
            result_config.is_parallel = result_config.size > 1
            result_config.backend = 'mpi4py (LPJmL communicator)'
            
            # Apply max_workers limit if specified
            if parallel_config['max_workers'] > 0:
                result_config.size = min(result_config.size, parallel_config['max_workers'])
            
            if result_config.rank == 0:
                print(f"✓ Detected LPJmL MPI environment: {result_config.size} processes")
                print("  Will reuse LPJmL's MPI processes during I/O wait time")
                if parallel_config['debug']:
                    print(f"  Debug: MPI rank {result_config.rank}/{result_config.size}")
            return result_config
        except ImportError:
            warnings.warn(
                "LPJmL MPI environment detected but mpi4py not available. "
                "Falling back to serial execution."
            )
    
    # 3. Check for explicit Dask scheduler (dask-mpi) - only if MPI not available
    if preferred_mode == 'auto':
        dask_scheduler = _get_dask_scheduler_address()
        if dask_scheduler:
            try:
                from dask.distributed import Client, get_client
                
                # Try to connect to existing scheduler
                try:
                    result_config._client = get_client()
                    print("✓ Connected to existing Dask client")
                except ValueError:
                    result_config._client = Client(dask_scheduler)
                    print(f"✓ Connected to Dask scheduler: {dask_scheduler}")
                
                result_config.mode = 'dask'
                result_config.is_parallel = True
                result_config.size = len(result_config._client.scheduler_info()['workers'])
                result_config.rank = 0
                result_config.backend = 'dask.distributed'
                
                # Apply max_workers limit if specified
                if parallel_config['max_workers'] > 0:
                    result_config.size = min(result_config.size, parallel_config['max_workers'])
                
                print(f"  Dask cluster: {result_config.size} workers available")
                if parallel_config['debug']:
                    print(f"  Debug: Dask scheduler at {dask_scheduler}")
                return result_config
            except Exception as e:
                warnings.warn(
                    f"Dask scheduler detected but connection failed: {e}. "
                    f"Falling back to serial execution."
                )
    
    # 4. Default: serial execution
    result_config.mode = 'serial'
    result_config.is_parallel = False
    result_config.backend = 'serial'
    if parallel_config['debug']:
        print("Running in serial mode (no parallel environment detected)")
    else:
        print("Running in serial mode")
    
    return result_config


def _get_parallelization_config(config: Optional[dict] = None) -> dict:
    """Get parallelization configuration with defaults.
    
    Parameters
    ----------
    config : dict or CoupledConfig, optional
        Configuration dictionary or CoupledConfig object. If None, uses defaults.
    
    Returns
    -------
    dict
        Parallelization configuration with defaults applied
    """
    defaults = {
        'max_workers': 0,  # 0 = auto-detect
        'debug': False,
        'mode': 'auto'  # 'auto', 'mpi', 'serial'
    }
    
    if config is None:
        return defaults
    
    # Handle CoupledConfig objects
    if hasattr(config, 'parallelization'):
        parallel_config = config.parallelization
        if hasattr(parallel_config, '__dict__'):
            # Convert CoupledConfig to dict
            parallel_config = {k: v for k, v in parallel_config.__dict__.items() 
                             if not k.startswith('_')}
        else:
            parallel_config = dict(parallel_config)
    else:
        # Handle regular dict
        parallel_config = config.get('parallelization', {})
    
    # Convert string booleans to actual booleans
    if 'debug' in parallel_config:
        if isinstance(parallel_config['debug'], str):
            parallel_config['debug'] = parallel_config['debug'].lower() in ('true', '1', 'yes', 'on')
    
    # Apply defaults for missing keys
    for key, default_value in defaults.items():
        if key not in parallel_config or parallel_config[key] is None:
            parallel_config[key] = default_value
    
    return parallel_config


def _is_lpjml_mpi_environment() -> bool:
    """Check if running in LPJmL's MPI environment.
    
    This detects when pycopanlpjml is running as a coupled script
    launched by LPJmL's MPI processes (via submit_lpjml).
    """
    # Check for MPI environment variables that indicate LPJmL MPI launch
    mpi_vars = [
        'OMPI_COMM_WORLD_SIZE',  # Open MPI
        'PMI_SIZE',               # Intel MPI
        'SLURM_NTASKS',          # SLURM with MPI
        'MPI_LOCALNRANKS',       # Various MPI implementations
    ]
    
    # Must be in MPI environment AND launched by LPJmL
    has_mpi = any(var in os.environ for var in mpi_vars)
    
    # Additional check: look for LPJmL-specific indicators
    # This could be enhanced with more specific LPJmL environment detection
    lpjml_indicators = [
        'LPJML_NTASKS',          # LPJmL-specific task count
        'LPJML_MPI',             # LPJmL MPI flag
    ]
    
    has_lpjml_mpi = any(var in os.environ for var in lpjml_indicators)
    
    # For now, assume any MPI environment launched via submit_lpjml is LPJmL's
    # This is a reasonable assumption since submit_lpjml launches the coupled script
    return has_mpi


def _is_mpi_environment() -> bool:
    """Check if running in an MPI environment."""
    mpi_vars = [
        'OMPI_COMM_WORLD_SIZE',  # Open MPI
        'PMI_SIZE',               # Intel MPI
        'SLURM_NTASKS',          # SLURM with MPI
        'MPI_LOCALNRANKS',       # Various MPI implementations
    ]
    return any(var in os.environ for var in mpi_vars)


def _get_dask_scheduler_address() -> Optional[str]:
    """Get Dask scheduler address from environment or file."""
    # Check environment variable
    if 'DASK_SCHEDULER_ADDRESS' in os.environ:
        return os.environ['DASK_SCHEDULER_ADDRESS']
    
    # Check for scheduler file (created by dask-mpi)
    scheduler_file = os.environ.get('DASK_SCHEDULER_FILE')
    if scheduler_file and os.path.exists(scheduler_file):
        import json
        with open(scheduler_file) as f:
            info = json.load(f)
            return info.get('address')
    
    return None


# Removed functions that create separate Dask clusters
# We now reuse LPJmL's MPI processes instead of creating new ones


class ParallelExecutor:
    """Executor that automatically uses the appropriate parallelization backend.
    
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
    
    def __init__(self, config: Optional[ParallelConfig] = None, parallelization_config: Optional[dict] = None):
        self.config = config or detect_parallel_environment(parallelization_config)
    
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
        if self.config.mode == 'serial':
            return self._map_serial(func, items, **kwargs)
        elif self.config.mode == 'dask':
            return self._map_dask(func, items, **kwargs)
        elif self.config.mode == 'mpi':
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
                    my_items = items[offset:offset + count]
                else:
                    # Send to worker
                    comm.send(items[offset:offset + count], dest=i, tag=0)
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
        if self.config.mode == 'mpi':
            self.config._comm.Barrier()
    
    def is_master(self) -> bool:
        """Check if this is the master process/rank."""
        return self.config.rank == 0
    
    def close(self):
        """Close parallel resources."""
        if self.config.mode == 'dask' and self.config._client:
            # Only close if we created the client
            if not hasattr(self, '_external_client'):
                self.config._client.close()


# Global singleton for easy access
_global_executor: Optional[ParallelExecutor] = None


def get_executor(reset: bool = False, config: Optional[dict] = None) -> ParallelExecutor:
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







