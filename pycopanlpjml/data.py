"""
DirtyDataset wrapper for xarray data of different entity levels.

This wrapper is used to mark the dataset as dirty when it is modified.
"""

import functools


class DirtyDataset:
    """
    Wrapper for an xarray.Dataset that tracks modifications ("dirty" state)
    and marks parent objects as needing synchronization.

    Parameters
    ----------
    parent : object
        The parent object (World, WorldRegion, Country) owning this dataset.
    ds : xarray.Dataset
        The wrapped dataset.
    io_type : str
        'input' or 'output' indicating which dataset type this is.
    """
    def __init__(self, parent, ds, io_type):
        self._ds = ds
        self._parent = parent
        self._io_type = io_type

    def _mark_dirty(self):
        """
        Mark the dataset and all related objects (upwards and downwards)
        as dirty, so synchronization will be triggered automatically.
        """

        # Mark self dirty
        self._parent._dirty[self._io_type] = True

        # Propagate dirty flag to higher regions
        def mark_higher(obj):
            for higher in obj.next_higher_entities + [obj.world]:
                higher._dirty[self._io_type] = True

        mark_higher(self._parent)

        # Propagate dirty flag to lower regions
        def mark_lower(obj):
            for lower in obj.next_lower_entities:
                lower._dirty[self._io_type] = True

        mark_lower(self._parent)

    def __getitem__(self, key):
        return self._ds[key]

    def __setitem__(self, key, value):
        self._ds[key] = value
        self._mark_dirty()

    def __getattr__(self, attr):
        """
        Delegate attribute access to the underlying xarray.Dataset.
        Wraps certain mutating methods to mark dirty on modification.
        """
        obj = getattr(self._ds, attr)
        if callable(obj):
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                result = obj(*args, **kwargs)
                # Mark dirty if method mutates data
                if attr in ["__setitem__", "loc", "isel", "sel", "update"]:
                    self._mark_dirty()
                return result
            return wrapper
        return obj

    def __repr__(self):
        return repr(self._ds)
