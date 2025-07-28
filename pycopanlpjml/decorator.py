"""
Decorator for entity data synchronization.
"""

import functools
from pycoupler.data import LPJmLDataSet


# global sync registry
SYNC_REGISTRY = {}


# register an object for synchronization of a given IO type
def register_sync(obj, io_type):
    """
    Register an object for synchronization of a given IO type.

    Parameters
    ----------
    obj : object
        The object to register (World, WorldRegion, Country).
    io_type : str
        'input' or 'output', the dataset type to synchronize.
    """
    SYNC_REGISTRY.setdefault(io_type, []).append(obj)


# check if data is dirty and needs synchronization
def sync_needed(obj, io_type):
    """
    Check whether the object's data for the given IO type is dirty
    and needs synchronization.

    Parameters
    ----------
    obj : object
        The object to check.
    io_type : str
        'input' or 'output'.

    Returns
    -------
    bool
        True if synchronization is needed, False otherwise.
    """
    return obj._dirty.get(io_type, False)


# propagate the most recent data across all registered objects for a specific
#   IO type
def propagate_sync(io_type):
    """
    Propagate the most recent data across all registered objects
    for a specific IO type.

    Parameters
    ----------
    io_type : str
        'input' or 'output'.
    """
    objs = SYNC_REGISTRY.get(io_type, [])
    for obj in objs:
        if sync_needed(obj, io_type):
            obj._sync(io_type)


# decorator for entity data synchronization
def sync_io_test(*args):
    """
    Decorator to automatically synchronize 'input' and/or 'output' datasets 
    before executing the decorated method.

    This decorator supports two usage patterns:

    1. Without arguments:
        Synchronizes both 'input' and 'output' datasets.

        Example:
        @sync_io
        def some_method(self):
            ...

    2. With one or more string arguments specifying which datasets to 
    synchronize:
        Synchronizes only the specified datasets.

        Example:
        @sync_io('input')
        def some_method(self):
            ...

        @sync_io('output')
        def some_method(self):
            ...

        @sync_io('input', 'output')
        def some_method(self):
            ...

    Parameters
    ----------
    *args : str
        Optional variable length argument list. Each argument should be either
        'input' or 'output' indicating which dataset(s) to synchronize.

    Returns
    -------
    function
        The decorated function wrapped with automatic synchronization logic.
    """
    # Case 1: Decorator used without parentheses: @sync_io
    if len(args) == 1 and callable(args[0]):
        func = args[0]

        @functools.wraps(func)
        def wrapper(self, *f_args, **kwargs):
            # Always sync both input and output if needed
            for io_type in ('input', 'output'):
                if sync_needed(self, io_type):
                    propagate_sync(io_type)
            return func(self, *f_args, **kwargs)

        return wrapper

    # Case 2: Decorator used with arguments: @sync_io('input'), etc.
    selected_io_types = args

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *f_args, **kwargs):
            # Sync only the selected io_types if needed
            for io_type in selected_io_types:
                if sync_needed(self, io_type):
                    propagate_sync(io_type)
            return func(self, *f_args, **kwargs)
        return wrapper

    return decorator


def sync_io(func):
    """
    Decorator to trigger Dask computation and synchronization
    for both input and output datasets after the wrapped method is called.

    Parameters
    ----------
    func : callable
        The method to wrap.

    Returns
    -------
    callable
        Wrapped function with synchronization logic.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)

        # Synchronize both input and output if they exist
        for io_attr in ['input', 'output', 'grid', 'country', 'area']:
            ds = getattr(self, io_attr, None)
            if isinstance(ds, LPJmLDataSet.Dataset):
                ds.load()

        return result
    return wrapper
