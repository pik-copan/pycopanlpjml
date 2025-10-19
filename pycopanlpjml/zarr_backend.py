"""Zarr-based backend for synchronized data access across World, Country, and Cell levels.

This module provides a thread-safe, memory-efficient backend that allows changes
at any level (World, Country, Cell) to be immediately visible at all other levels.
It uses Zarr arrays as a single source of truth with index-based views.
"""

import numpy as np
import xarray as xr
import zarr
from threading import Lock
from typing import Union, List, Optional, Dict, Any

# Default chunk size for Zarr arrays (number of cells per chunk)
DEFAULT_CHUNK_SIZE = 1000

# Import LPJmL data types for full compatibility
try:
    from pycoupler.data import LPJmLDataSet, LPJmLData

    HAS_PYCOUPLER = True
except ImportError:
    HAS_PYCOUPLER = False
    # Fallback to regular xarray if pycoupler not available
    LPJmLDataSet = xr.Dataset
    LPJmLData = xr.DataArray


def _convert_datetime_coordinate(coord_data, coord_dtype):
    """Helper function to convert datetime coordinates from stored format.

    Parameters
    ----------
    coord_data : list or array
        Coordinate data (int64 timestamps for datetime)
    coord_dtype : str or None
        Data type string, e.g., 'datetime64[ns]'

    Returns
    -------
    array
        Converted coordinate data
    """
    if coord_dtype and coord_dtype.startswith("datetime64"):
        import pandas as pd

        coord_data = pd.to_datetime(coord_data, unit="ns").to_pydatetime()
        if not isinstance(coord_data, list):
            coord_data = coord_data.tolist()
    return coord_data


class ZarrBackend:
    """Manages the shared Zarr store for World, Country, and Cell data.

    This class provides access to a Zarr store that serves as the
    single source of truth for all LPJmL data. It supports:
    - Automatic synchronization across hierarchical levels
    - Thread-safe concurrent access
    - Chunked storage for large datasets
    - xarray-compatible data structures

    Parameters
    ----------
    store_path : str
        Path to the Zarr directory store
    overwrite : bool, default=False
        Whether to overwrite existing store

    Attributes
    ----------
    root : zarr.Group
        Root Zarr group containing all datasets
    lock : threading.Lock
        Thread lock for safe concurrent access
    """

    def __init__(self, store_path: str, overwrite: bool = False):
        self.store_path = store_path

        # Thread-safe lock for concurrent write operations
        self.lock = Lock()

        # Create/open Zarr group (Zarr 3.x API)
        mode = "w" if overwrite else "a"
        self.root = zarr.open_group(store=store_path, mode=mode)

    def initialize_from_xarray(
        self,
        input_ds: xr.Dataset,
        output_ds: xr.Dataset,
        grid: xr.DataArray,
        country: xr.DataArray,
        area: Optional[xr.DataArray] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        """Initialize Zarr store from xarray datasets.

        Parameters
        ----------
        input_ds : xr.Dataset
            LPJmL input dataset
        output_ds : xr.Dataset
            LPJmL output dataset
        grid : xr.DataArray
            Grid coordinates (lon, lat)
        country : xr.DataArray
            Country codes for each cell
        area : xr.DataArray, optional
            Cell areas
        chunk_size : int
            Chunk size for cell dimension
        """
        with self.lock:
            # Store input variables
            input_group = self.root.create_group("input", overwrite=True)

            # Access variables WITHOUT triggering _construct_dataarray to preserve original coord names
            for var_name in input_ds.data_vars:
                # Access the raw variable data to avoid LPJmLDataSet's coordinate renaming
                var_variable = input_ds._variables[var_name]

                # Reconstruct full dimension names from coordinates
                # LPJmLDataSet may normalize dims (e.g., 'band'), but coordinates have full names (e.g., 'band (pft_harvestc)')
                dims = []
                for dim in var_variable.dims:
                    # Check if there's a coordinate with a full name for this dimension
                    full_dim_name = dim
                    for coord_name in input_ds.coords:
                        coord_obj = input_ds.coords[coord_name]
                        if (
                            hasattr(coord_obj, "dims")
                            and len(coord_obj.dims) == 1
                            and coord_obj.dims[0] == dim
                        ):
                            # If the coord name is more specific (e.g., 'band (pft_harvestc)'), use it
                            # Match both 'band (with_tillage)' and 'band'
                            if coord_name.startswith(dim):
                                full_dim_name = coord_name
                                break
                    dims.append(full_dim_name)

                # Determine chunks (chunk along cell dimension)
                chunks = tuple(
                    chunk_size if dim == "cell" else s
                    for dim, s in zip(dims, var_variable.shape)
                )

                # Create array (Zarr 3.x API)
                arr = input_group.create_array(
                    name=var_name,
                    shape=var_variable.shape,
                    chunks=chunks,
                    dtype=var_variable.dtype,
                    overwrite=True,
                )
                # Assign data
                arr[:] = var_variable.values

                # Store metadata including coords - get from DATASET level to preserve names
                var_coords = {}
                # Get coords relevant to this variable's dimensions from dataset
                for coord_name, coord_obj in input_ds.coords.items():
                    # Check if coord is relevant using both normalized and full dims
                    if hasattr(coord_obj, "dims"):
                        # Check if all coord dims match either normalized or full dim names
                        # E.g., coord with dims=('band',) should match variable dim 'band (with_tillage)'
                        coord_relevant = True
                        for coord_dim in coord_obj.dims:
                            # Check if coord_dim matches any variable dim (normalized or full)
                            matched = False
                            for var_dim in dims:
                                # Match if exact or if var_dim starts with coord_dim (e.g., 'band (...)' starts with 'band')
                                if var_dim == coord_dim or var_dim.startswith(
                                    coord_dim + " "
                                ):
                                    matched = True
                                    break
                            if not matched:
                                coord_relevant = False
                                break
                        if coord_relevant:
                            coord_data = coord_obj
                        else:
                            continue
                    elif coord_name in dims:
                        # It's a dimension coordinate
                        coord_data = coord_obj
                    else:
                        # Not relevant to this variable
                        continue
                    # Handle datetime coordinates specially
                    coord_values = coord_data.values
                    if np.issubdtype(coord_values.dtype, np.datetime64):
                        # Keep as timestamp integers (nanoseconds since epoch)
                        data_to_store = coord_values.view("int64").tolist()
                        coord_dtype = "datetime64[ns]"
                    else:
                        data_to_store = (
                            coord_values.tolist()
                            if hasattr(coord_values, "tolist")
                            else list(coord_values)
                        )
                        coord_dtype = str(coord_values.dtype)

                    # Reconstruct coord dims using full dimension names
                    if hasattr(coord_data, "dims"):
                        # Map normalized dims to full dims
                        coord_dims = []
                        for coord_dim in coord_data.dims:
                            # If this coord is 1D and its name starts with the dim, use coord name
                            if len(
                                coord_data.dims
                            ) == 1 and coord_name.startswith(coord_dim):
                                coord_dims.append(coord_name)
                            else:
                                # Find the full dim name from our reconstructed dims list
                                coord_dims.append(
                                    dims[
                                        list(var_variable.dims).index(
                                            coord_dim
                                        )
                                    ]
                                )
                    else:
                        coord_dims = [coord_name]

                    coord_info = {
                        "data": data_to_store,
                        "dims": coord_dims,
                        "attrs": (
                            dict(coord_data.attrs)
                            if hasattr(coord_data, "attrs")
                            else {}
                        ),
                        "dtype": coord_dtype,
                    }
                    var_coords[coord_name] = coord_info

                arr.attrs.update(
                    {
                        "dims": dims,
                        "attrs": dict(var_variable.attrs),
                        "coords": var_coords,
                    }
                )

            # Store input dataset attributes and coords with dtype info
            coords_meta = {}
            for name, coord in input_ds.coords.items():
                # For dimension coordinates (where the coord name differs from its dim),
                # use the coord name as the dimension to preserve full names like 'band (hdate)'
                if hasattr(coord, "dims") and len(coord.dims) == 1:
                    # If coord name contains parentheses (e.g., 'band (hdate)'), it's a full name
                    # Use it as the dimension regardless of what xarray reports
                    if "(" in name and ")" in name:
                        coord_dims = [name]
                    else:
                        coord_dims = list(coord.dims)
                else:
                    coord_dims = (
                        list(coord.dims) if hasattr(coord, "dims") else [name]
                    )

                coords_meta[name] = {
                    "data": (
                        coord.values.view("int64").tolist()
                        if np.issubdtype(coord.values.dtype, np.datetime64)
                        else (
                            coord.values.tolist()
                            if hasattr(coord.values, "tolist")
                            else coord.values
                        )
                    ),
                    "dims": coord_dims,
                    "dtype": (
                        "datetime64[ns]"
                        if np.issubdtype(coord.values.dtype, np.datetime64)
                        else str(coord.values.dtype)
                    ),
                }
            input_group.attrs["coords"] = coords_meta
            input_group.attrs["dataset_attrs"] = dict(input_ds.attrs)

            # Store output variables
            output_group = self.root.create_group("output", overwrite=True)

            # Access variables WITHOUT triggering _construct_dataarray
            for var_name in output_ds.data_vars:
                var_variable = output_ds._variables[var_name]

                # Reconstruct full dimension names from coordinates
                # LPJmLDataSet may normalize dims (e.g., 'band'), but coordinates have full names (e.g., 'band (pft_harvestc)')
                dims = []
                for dim in var_variable.dims:
                    # Check if there's a coordinate with a full name for this dimension
                    full_dim_name = dim
                    for coord_name in output_ds.coords:
                        coord_obj = output_ds.coords[coord_name]
                        if (
                            hasattr(coord_obj, "dims")
                            and len(coord_obj.dims) == 1
                            and coord_obj.dims[0] == dim
                        ):
                            # If the coord name is more specific (e.g., 'band (pft_harvestc)'), use it
                            if (
                                coord_name.startswith(dim + " ")
                                or coord_name == dim
                            ):
                                full_dim_name = coord_name
                                break
                    dims.append(full_dim_name)

                chunks = tuple(
                    chunk_size if dim == "cell" else s
                    for dim, s in zip(dims, var_variable.shape)
                )

                arr = output_group.create_array(
                    name=var_name,
                    shape=var_variable.shape,
                    chunks=chunks,
                    dtype=var_variable.dtype,
                    overwrite=True,
                )
                arr[:] = var_variable.values

                # Store metadata including coords - get from DATASET level to preserve names
                var_coords = {}
                # Get coords relevant to this variable's dimensions from dataset
                for coord_name, coord_obj in output_ds.coords.items():
                    # Check if coord is relevant using both normalized and full dims
                    if hasattr(coord_obj, "dims"):
                        # Check if all coord dims match either normalized or full dim names
                        # E.g., coord with dims=('band',) should match variable dim 'band (pft_harvestc)'
                        coord_relevant = True
                        for coord_dim in coord_obj.dims:
                            # Check if coord_dim matches any variable dim (normalized or full)
                            matched = False
                            for var_dim in dims:
                                # Match if exact or if var_dim starts with coord_dim (e.g., 'band (...)' starts with 'band')
                                if var_dim == coord_dim or var_dim.startswith(
                                    coord_dim + " "
                                ):
                                    matched = True
                                    break
                            if not matched:
                                coord_relevant = False
                                break
                        if coord_relevant:
                            coord_data = coord_obj
                        else:
                            continue
                    elif coord_name in dims:
                        # It's a dimension coordinate
                        coord_data = coord_obj
                    else:
                        # Not relevant to this variable
                        continue
                    # Handle datetime coordinates specially
                    coord_values = coord_data.values
                    if np.issubdtype(coord_values.dtype, np.datetime64):
                        # Keep as timestamp integers (nanoseconds since epoch)
                        data_to_store = coord_values.view("int64").tolist()
                        coord_dtype = "datetime64[ns]"
                    else:
                        data_to_store = (
                            coord_values.tolist()
                            if hasattr(coord_values, "tolist")
                            else list(coord_values)
                        )
                        coord_dtype = str(coord_values.dtype)

                    # Reconstruct coord dims using full dimension names
                    if hasattr(coord_data, "dims"):
                        # Map normalized dims to full dims
                        coord_dims = []
                        for coord_dim in coord_data.dims:
                            # If this coord is 1D and its name starts with the dim, use coord name
                            if len(
                                coord_data.dims
                            ) == 1 and coord_name.startswith(coord_dim):
                                coord_dims.append(coord_name)
                            else:
                                # Find the full dim name from our reconstructed dims list
                                coord_dims.append(
                                    dims[
                                        list(var_variable.dims).index(
                                            coord_dim
                                        )
                                    ]
                                )
                    else:
                        coord_dims = [coord_name]

                    coord_info = {
                        "data": data_to_store,
                        "dims": coord_dims,
                        "attrs": (
                            dict(coord_data.attrs)
                            if hasattr(coord_data, "attrs")
                            else {}
                        ),
                        "dtype": coord_dtype,
                    }
                    var_coords[coord_name] = coord_info

                arr.attrs.update(
                    {
                        "dims": dims,
                        "attrs": dict(var_variable.attrs),
                        "coords": var_coords,
                    }
                )

            # Store output dataset attributes and coords with dtype info
            coords_meta = {}
            for name, coord in output_ds.coords.items():
                # For dimension coordinates (where the coord name differs from its dim),
                # use the coord name as the dimension to preserve full names like 'band (hdate)'
                if hasattr(coord, "dims") and len(coord.dims) == 1:
                    # If coord name contains parentheses (e.g., 'band (hdate)'), it's a full name
                    # Use it as the dimension regardless of what xarray reports
                    if "(" in name and ")" in name:
                        coord_dims = [name]
                    else:
                        coord_dims = list(coord.dims)
                else:
                    coord_dims = (
                        list(coord.dims) if hasattr(coord, "dims") else [name]
                    )

                coords_meta[name] = {
                    "data": (
                        coord.values.view("int64").tolist()
                        if np.issubdtype(coord.values.dtype, np.datetime64)
                        else (
                            coord.values.tolist()
                            if hasattr(coord.values, "tolist")
                            else coord.values
                        )
                    ),
                    "dims": coord_dims,
                    "dtype": (
                        "datetime64[ns]"
                        if np.issubdtype(coord.values.dtype, np.datetime64)
                        else str(coord.values.dtype)
                    ),
                }
            output_group.attrs["coords"] = coords_meta
            output_group.attrs["dataset_attrs"] = dict(output_ds.attrs)

            # Store grid, country, and area as separate arrays with full metadata
            grid_arr = self.root.create_array(
                name="grid",
                shape=grid.shape,
                chunks=(chunk_size, grid.shape[1]),
                dtype=grid.dtype,
                overwrite=True,
            )
            grid_arr[:] = grid.values
            grid_arr.attrs["dims"] = list(grid.dims)
            grid_arr.attrs["attrs"] = dict(grid.attrs)
            # Store coords with full metadata
            grid_coords = {}
            for coord_name, coord_data in grid.coords.items():
                grid_coords[coord_name] = {
                    "data": (
                        coord_data.values.tolist()
                        if hasattr(coord_data.values, "tolist")
                        else list(coord_data.values)
                    ),
                    "dims": (
                        list(coord_data.dims)
                        if hasattr(coord_data, "dims")
                        else [coord_name]
                    ),
                    "attrs": (
                        dict(coord_data.attrs)
                        if hasattr(coord_data, "attrs")
                        else {}
                    ),
                }
            grid_arr.attrs["coords"] = grid_coords
            grid_arr.attrs["name"] = getattr(grid, "name", "grid")

            country_arr = self.root.create_array(
                name="country",
                shape=country.shape,
                chunks=(chunk_size,) + country.shape[1:],
                dtype=country.dtype,
                overwrite=True,
            )
            country_arr[:] = country.values
            country_arr.attrs["dims"] = list(country.dims)
            country_arr.attrs["attrs"] = dict(country.attrs)
            # Store coords with full metadata
            country_coords = {}
            for coord_name, coord_data in country.coords.items():
                country_coords[coord_name] = {
                    "data": (
                        coord_data.values.tolist()
                        if hasattr(coord_data.values, "tolist")
                        else list(coord_data.values)
                    ),
                    "dims": (
                        list(coord_data.dims)
                        if hasattr(coord_data, "dims")
                        else [coord_name]
                    ),
                    "attrs": (
                        dict(coord_data.attrs)
                        if hasattr(coord_data, "attrs")
                        else {}
                    ),
                }
            country_arr.attrs["coords"] = country_coords
            country_arr.attrs["name"] = getattr(country, "name", "country")

            if area is not None:
                area_arr = self.root.create_array(
                    name="area",
                    shape=area.shape,
                    chunks=(chunk_size,) + area.shape[1:],
                    dtype=area.dtype,
                    overwrite=True,
                )
                area_arr[:] = area.values
                area_arr.attrs["dims"] = list(area.dims)
                area_arr.attrs["attrs"] = dict(area.attrs)
                # Store coords with full metadata
                area_coords = {}
                for coord_name, coord_data in area.coords.items():
                    area_coords[coord_name] = {
                        "data": (
                            coord_data.values.tolist()
                            if hasattr(coord_data.values, "tolist")
                            else list(coord_data.values)
                        ),
                        "dims": (
                            list(coord_data.dims)
                            if hasattr(coord_data, "dims")
                            else [coord_name]
                        ),
                        "attrs": (
                            dict(coord_data.attrs)
                            if hasattr(coord_data, "attrs")
                            else {}
                        ),
                    }
                area_arr.attrs["coords"] = area_coords
                area_arr.attrs["name"] = getattr(area, "name", "area")

            # Store global metadata
            self.root.attrs["n_cells"] = len(grid.cell)

    def get_view(
        self,
        group: str,
        indices: Optional[Union[int, slice, np.ndarray, List[int]]] = None,
    ) -> "ZarrDatasetView":
        """Get a view into a dataset group.

        Parameters
        ----------
        group : str
            Group name ('input' or 'output')
        indices : int, slice, array-like, optional
            Cell indices to access. None means all cells.

        Returns
        -------
        ZarrDatasetView
            View object providing xarray-like interface
        """
        return ZarrDatasetView(self, group, indices)

    def get_array_view(
        self,
        path: str,
        indices: Optional[Union[int, slice, np.ndarray, List[int]]] = None,
    ) -> "ZarrDataArrayView":
        """Get a view into a single array (grid, country, area).

        Parameters
        ----------
        path : str
            Path to array ('grid', 'country', or 'area')
        indices : int, slice, array-like, optional
            Cell indices to access

        Returns
        -------
        ZarrDataArrayView
            View object providing xarray-like interface
        """
        return ZarrDataArrayView(self, path, indices)


class WritableCoordinateArray(np.ndarray):
    """Numpy array subclass that writes changes back to Zarr attrs.

    This allows code like: coord.values[0] = new_val
    """

    def __new__(cls, input_array, backend, group, coord_name):
        obj = np.asarray(input_array).view(cls)
        obj._backend = backend
        obj._group = group
        obj._coord_name = coord_name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._backend = getattr(obj, "_backend", None)
        self._group = getattr(obj, "_group", None)
        self._coord_name = getattr(obj, "_coord_name", None)

    def _write_back_to_zarr(self):
        """Write current array values back to Zarr attrs."""
        if self._backend and self._group and self._coord_name:
            with self._backend.lock:
                zarr_group = self._backend.root[self._group]
                coords = zarr_group.attrs.get("coords", {})

                # Get existing dtype info if available
                dtype_info = None
                if isinstance(coords.get(self._coord_name), dict):
                    dtype_info = coords[self._coord_name].get("dtype")

                # Convert for storage
                if np.issubdtype(self.dtype, np.datetime64):
                    # Store as int64 timestamps
                    data_to_store = self.view("int64").tolist()
                    coords[self._coord_name] = {
                        "data": data_to_store,
                        "dtype": "datetime64[ns]",
                    }
                else:
                    if dtype_info:
                        # Preserve existing dtype info
                        coords[self._coord_name] = {
                            "data": self.tolist(),
                            "dtype": dtype_info,
                        }
                    else:
                        coords[self._coord_name] = self.tolist()

                zarr_group.attrs["coords"] = coords

    def __setitem__(self, key, value):
        """Set item and write back to Zarr."""
        # Convert datetime64 values to match array dtype if needed
        if np.issubdtype(self.dtype, np.datetime64) and isinstance(
            value, np.datetime64
        ):
            # Both datetime64, direct assignment works
            super().__setitem__(key, value)
        elif isinstance(value, np.datetime64):
            # Array is int64 storage, value is datetime64 - convert to timestamp
            value = value.view("int64")
            super().__setitem__(key, value)
        else:
            # Normal assignment
            super().__setitem__(key, value)

        # Write back to Zarr
        self._write_back_to_zarr()


class ZarrCoordinateView:
    """Writable view for accessing coordinates with .values attribute.

    This mimics xarray coordinate behavior for compatibility with
    code that accesses coordinates like dataset.time.values.
    Supports setting values which updates the Zarr group attrs.

    Parameters
    ----------
    backend : ZarrBackend
        The Zarr backend managing the data
    group : str
        Group name ('input' or 'output')
    coord_name : str
        Name of the coordinate
    initial_values : array-like
        The initial coordinate values
    """

    def __init__(self, backend, group, coord_name, initial_values):
        self._backend = backend
        self._group = group
        self._coord_name = coord_name
        self._initial_values = np.array(initial_values)

    @property
    def values(self):
        """Get coordinate values as a writable numpy array."""
        # Get fresh values from attrs
        zarr_group = self._backend.root[self._group]
        coords = zarr_group.attrs.get("coords", {})

        if self._coord_name in coords:
            coord_info = coords[self._coord_name]

            # Check if new format with dtype info
            if isinstance(coord_info, dict):
                coord_data = coord_info["data"]
                coord_dtype = coord_info.get("dtype", None)

                # Convert to datetime64 if stored as timestamps
                if coord_dtype and coord_dtype.startswith("datetime64"):
                    values_array = np.array(coord_data, dtype="int64").view(
                        "datetime64[ns]"
                    )
                else:
                    values_array = np.array(coord_data)
            else:
                # Old format (just data)
                values_array = np.array(coord_info)
        else:
            values_array = self._initial_values

        # Return writable array that syncs back to Zarr
        return WritableCoordinateArray(
            values_array, self._backend, self._group, self._coord_name
        )

    @values.setter
    def values(self, new_values):
        """Set coordinate values back to Zarr attrs."""
        with self._backend.lock:
            zarr_group = self._backend.root[self._group]
            coords = zarr_group.attrs.get("coords", {})

            # Handle different value types appropriately
            if isinstance(new_values, np.ndarray):
                # Check if datetime64
                if np.issubdtype(new_values.dtype, np.datetime64):
                    # Store as int64 timestamps (nanoseconds since epoch)
                    coords[self._coord_name] = new_values.view(
                        "int64"
                    ).tolist()
                else:
                    coords[self._coord_name] = new_values.tolist()
            elif hasattr(new_values, "tolist"):
                coords[self._coord_name] = new_values.tolist()
            else:
                coords[self._coord_name] = (
                    list(new_values)
                    if not isinstance(new_values, (int, float, str))
                    else [new_values]
                )

            zarr_group.attrs["coords"] = coords

    def __len__(self):
        """Return length of coordinate (for len(dataset.time))."""
        return len(self.values)

    def __repr__(self):
        return f"ZarrCoordinateView({self._coord_name})"


class ZarrDatasetView:
    """Provides xarray.Dataset-like interface to Zarr-backed data.

    This class provides a view into a Zarr group (input or output) that
    behaves like an xarray Dataset. Changes made through this view are
    immediately visible to all other views.

    Parameters
    ----------
    backend : ZarrBackend
        The Zarr backend managing the data
    group : str
        Group name ('input' or 'output')
    indices : int, slice, array-like, optional
        Cell indices for this view
    """

    def __init__(
        self,
        backend: ZarrBackend,
        group: str,
        indices: Optional[Union[int, slice, np.ndarray, List[int]]] = None,
    ):
        self.backend = backend
        self.group = group

        # Normalize indices
        if indices is None:
            self.indices = slice(None)
        elif isinstance(indices, (int, np.integer)):
            self.indices = indices
        elif isinstance(indices, list):
            self.indices = np.array(indices)
        else:
            self.indices = indices

        self._is_scalar = isinstance(self.indices, (int, np.integer))

        # Performance: Cache frequently accessed coords property and xarray object
        self._coords_cache = None
        self._xarray_cache = None

    def __setattr__(self, name, value):
        """Set attribute (delegates to object.__setattr__ to avoid recursion)."""
        object.__setattr__(self, name, value)

    @property
    def data_vars(self):
        """Return dictionary of variable names."""
        zarr_group = self.backend.root[self.group]
        return {key: self[key] for key in zarr_group.keys()}

    @property
    def coords(self):
        """Return coordinates for this view (cached for performance)."""
        # PERFORMANCE: Cache coords to avoid repeated dict rebuilding and conversions
        if self._coords_cache is not None:
            return self._coords_cache

        zarr_group = self.backend.root[self.group]
        coords_dict = zarr_group.attrs.get("coords", {})

        # Subset coordinates based on indices
        result = {}
        for name, coord_info in coords_dict.items():
            # Handle new format with dtype info
            if isinstance(coord_info, dict) and "data" in coord_info:
                values = coord_info["data"]
                dtype_info = coord_info.get("dtype", None)

                # Convert datetime timestamps to datetime64
                if dtype_info and dtype_info.startswith("datetime64"):
                    values = np.array(values, dtype="int64").view(
                        "datetime64[ns]"
                    )
                else:
                    values = values
            else:
                values = coord_info

            if name == "cell":
                if self._is_scalar:
                    result[name] = self.indices
                elif isinstance(self.indices, slice):
                    result[name] = np.array(values)[self.indices]
                else:
                    result[name] = self.indices
            else:
                result[name] = values

        # Cache the result
        self._coords_cache = result
        return result

    def __getitem__(self, key: str) -> "ZarrDataArrayView":
        """Access a variable by name (returns writable view).

        For read-only access with normalized dimensions, use attribute access (view.variable_name).
        For writable access, use dict-style access (view["variable_name"]).

        Parameters
        ----------
        key : str
            Variable name

        Returns
        -------
        ZarrDataArrayView
            Writable view into the variable data
        """
        return ZarrDataArrayView(
            self.backend, f"{self.group}/{key}", self.indices
        )

    def __setitem__(self, key: str, value):
        """Set a variable's data.

        Parameters
        ----------
        key : str
            Variable name
        value : array-like
            New values
        """
        with self.backend.lock:
            zarr_array = self.backend.root[f"{self.group}/{key}"]
            if self._is_scalar:
                zarr_array[self.indices] = value
            else:
                zarr_array[self.indices] = value

    def _get_xarray(self):
        """Lazy-load the xarray/LPJmLDataSet representation.

        Returns
        -------
        LPJmLDataSet or xr.Dataset
            The xarray representation of this dataset
        """
        if self._xarray_cache is None:
            self._xarray_cache = self.to_xarray()
        return self._xarray_cache

    def __getattr__(self, name: str):
        """Access coordinates, variables, and all xarray methods/attributes.

        This provides full xarray compatibility by delegating unknown attributes
        to the underlying xarray object.

        Parameters
        ----------
        name : str
            Coordinate, variable name, or xarray method/attribute

        Returns
        -------
        ZarrCoordinateView, ZarrDataArrayView, or any xarray attribute
            View into the coordinate/variable, or xarray method/attribute
        """
        # Avoid infinite recursion for private attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        # Check if it's a coordinate
        coords = self.coords
        if name in coords:
            # Return a writable coordinate view with proper dtype
            coord_value = coords[name]
            # coord_value is already converted to proper dtype (datetime64 if needed) by coords property
            return ZarrCoordinateView(
                self.backend, self.group, name, coord_value
            )

        # Delegate to the xarray object for all other attributes
        # This ensures LPJmLDataSet._construct_dataarray is called for variables,
        # which normalizes dimensions ('band (hdate)' -> 'band')
        xarray_obj = self._get_xarray()
        if hasattr(xarray_obj, name):
            return getattr(xarray_obj, name)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def isel(self, indexers: Dict[str, Any], drop: bool = False):
        """Select by integer position (similar to xarray.isel).

        Parameters
        ----------
        indexers : dict
            Dictionary of dimension names to indices
        drop : bool
            Whether to drop scalar dimensions (kept for API compatibility)

        Returns
        -------
        ZarrDatasetView
            New view with updated indices
        """
        if "cell" not in indexers:
            return self

        cell_indexer = indexers["cell"]

        # Combine current indices with new indexer
        if (
            self.indices is None
            or isinstance(self.indices, slice)
            and self.indices == slice(None)
        ):
            new_indices = cell_indexer
        elif self._is_scalar:
            # Already at a single cell, can't index further
            raise IndexError("Cannot index a scalar cell view")
        else:
            # Apply indexer to current indices
            if isinstance(self.indices, np.ndarray):
                new_indices = self.indices[cell_indexer]
            else:
                new_indices = np.arange(self.backend.root.attrs["n_cells"])[
                    self.indices
                ][cell_indexer]

        return ZarrDatasetView(self.backend, self.group, new_indices)

    def to_xarray(self):
        """Convert view to LPJmLDataSet (creates a copy).

        Returns
        -------
        LPJmLDataSet or xr.Dataset
            LPJmLDataSet with current data (falls back to xr.Dataset if pycoupler not available)
        """
        zarr_group = self.backend.root[self.group]

        # Get dataset-level attributes and coords
        dataset_attrs = zarr_group.attrs.get("dataset_attrs", {})
        dataset_coords = zarr_group.attrs.get("coords", {})

        # Build by creating xarray Dataset directly from dict representation
        # This preserves exact coordinate names like 'band (with_tillage)'

        # First, get dataset-level coordinates (these have all coords including 'band (varname)')
        all_coords_data = {}
        for coord_name, coord_info in dataset_coords.items():
            # Skip cell coordinate for scalar views
            if self._is_scalar and coord_name == "cell":
                continue

            if isinstance(coord_info, dict) and "data" in coord_info:
                coord_data = coord_info["data"]
                coord_dtype = coord_info.get("dtype", None)
                coord_dims = coord_info.get("dims", (coord_name,))

                # Convert datetime if needed
                coord_data = _convert_datetime_coordinate(
                    coord_data, coord_dtype
                )

                # Use stored dims (e.g., 'band (pft_harvestc)' has dims ['band (pft_harvestc)'])
                # Ensure dims is a tuple of strings, not a tuple of tuples
                if isinstance(coord_dims, (list, tuple)):
                    dims_tuple = tuple(coord_dims)
                else:
                    dims_tuple = (coord_dims,)

                # For scalar views, drop 'cell' from coord dims
                if self._is_scalar:
                    dims_tuple = tuple(d for d in dims_tuple if d != "cell")

                # Apply subset to coordinates that have 'cell' dimension
                if self.indices is not None and "cell" in dims_tuple:
                    # If this is a 1D coordinate along cell dimension, apply subset
                    if len(dims_tuple) == 1 and dims_tuple[0] == "cell":
                        coord_data = np.array(coord_data)[
                            self.indices
                        ].tolist()

                # Only add if dims is not empty (some coords might only have 'cell')
                if dims_tuple:
                    all_coords_data[coord_name] = {
                        "dims": dims_tuple,
                        "data": coord_data,
                        "attrs": {},
                    }
            else:
                # Legacy format - just has data, use coord_name as dimension
                all_coords_data[coord_name] = {
                    "dims": (coord_name,),
                    "data": coord_info,
                    "attrs": {},
                }

        # Then add any variable-specific coordinates not in dataset coords
        for var_name in zarr_group.keys():
            var_array = zarr_group[var_name]
            var_coords_stored = var_array.attrs.get("coords", {})

            for coord_name, coord_info in var_coords_stored.items():
                # Skip if we already have this coordinate
                if coord_name in all_coords_data:
                    continue

                # Skip cell coordinate for scalar views
                if self._is_scalar and coord_name == "cell":
                    continue

                if isinstance(coord_info, dict) and "data" in coord_info:
                    coord_data = coord_info["data"]
                    coord_dims = tuple(coord_info.get("dims", [coord_name]))
                    coord_attrs = coord_info.get("attrs", {})
                    coord_dtype = coord_info.get("dtype", None)

                    # Convert datetime if needed
                    coord_data = _convert_datetime_coordinate(
                        coord_data, coord_dtype
                    )

                    # For scalar views, drop 'cell' from coord dims
                    if self._is_scalar:
                        coord_dims = tuple(
                            d for d in coord_dims if d != "cell"
                        )

                    # Apply subset to coordinates that have 'cell' dimension
                    if self.indices is not None and "cell" in coord_dims:
                        # If this is a 1D coordinate along cell dimension, apply subset
                        if len(coord_dims) == 1 and coord_dims[0] == "cell":
                            coord_data = np.array(coord_data)[
                                self.indices
                            ].tolist()

                    # Only add if dims is not empty
                    if coord_dims:
                        # Store in from_dict format
                        all_coords_data[coord_name] = {
                            "dims": coord_dims,
                            "data": coord_data,
                            "attrs": coord_attrs,
                        }

        # Build data_vars in from_dict format
        data_vars = {}
        all_used_dims = (
            set()
        )  # Track all dimensions actually used by variables
        for var_name in zarr_group.keys():
            var_array = zarr_group[var_name]
            var_data = var_array[
                self.indices if self.indices is not None else slice(None)
            ]
            var_dims = list(var_array.attrs.get("dims", ()))
            var_attrs = var_array.attrs.get("attrs", {})

            # For scalar views (single cell), drop the 'cell' dimension
            if self._is_scalar:
                var_dims = [d for d in var_dims if d != "cell"]

            # Track which dimensions are actually used
            all_used_dims.update(var_dims)

            data_vars[var_name] = {
                "dims": var_dims,
                "data": var_data.tolist(),
                "attrs": var_attrs,
            }

        # Build data_vars and coords for xarray.Dataset.from_dict()
        # Use from_dict to preserve the exact structure we want
        dict_for_reconstruction = {
            "coords": {},
            "attrs": dataset_attrs,
            "dims": {},
            "data_vars": data_vars,
        }

        # Add ALL coordinates at dataset level (both dimension and non-dimension)
        # LPJmLDataSet will handle dimension normalization properly
        for coord_name, coord_data in all_coords_data.items():
            coord_dims = coord_data["dims"]
            # Include coordinate if its dimensions are used by any variable
            if any(dim in all_used_dims for dim in coord_dims):
                dict_for_reconstruction["coords"][coord_name] = coord_data

        # Create Dataset - use LPJmLDataSet if available for proper _construct_dataarray behavior
        ds = xr.Dataset.from_dict(dict_for_reconstruction)

        # Convert to LPJmLDataSet to get proper dimension normalization behavior
        if HAS_PYCOUPLER:
            ds.__class__ = LPJmLDataSet
            # Set LPJmLDataSet-specific attributes
            if ds.data_vars:
                first_var_name = list(ds.data_vars)[0]
                if first_var_name in ds._variables:
                    first_var_attrs = ds._variables[first_var_name].attrs
                    if "cellsize" in first_var_attrs:
                        ds.attrs["source"] = first_var_attrs.get("source", "")
                        ds.attrs["history"] = first_var_attrs.get(
                            "history", ""
                        )
                        ds.attrs["cellsize"] = first_var_attrs["cellsize"]
                        if "institution" in first_var_attrs:
                            ds.attrs["institution"] = first_var_attrs.get(
                                "institution", ""
                            )
                            ds.attrs["contact"] = first_var_attrs.get(
                                "contact", ""
                            )
                            ds.attrs["comment"] = first_var_attrs.get(
                                "comment", ""
                            )

        return ds

    def to_numpy(self):
        """Convert to dictionary of numpy arrays.

        This method provides compatibility with LPJmL's send_input
        which expects a dict with numpy arrays.

        Returns
        -------
        dict
            Dictionary with variable names as keys and numpy arrays as values
        """
        result = {}
        zarr_group = self.backend.root[self.group]

        for var_name in zarr_group.keys():
            var_view = self[var_name]
            result[var_name] = var_view.values

        return result

    def to_dict(self):
        """Convert to dictionary (for compatibility with xarray interface).

        Returns
        -------
        dict
            Dictionary representation compatible with xarray.Dataset.to_dict()
        """
        # Build dict directly to avoid LPJmLDataSet._construct_dataarray issues
        zarr_group = self.backend.root[self.group]
        dataset_coords = zarr_group.attrs.get("coords", {})
        dataset_attrs = zarr_group.attrs.get("dataset_attrs", {})

        result = {
            "coords": {},
            "attrs": dataset_attrs,
            "dims": {},
            "data_vars": {},
        }

        # Collect all coordinates from all variables (preserves 'band (varname)' names)
        all_coord_names = set()
        for var_name in zarr_group.keys():
            var_coords = zarr_group[var_name].attrs.get("coords", {})
            all_coord_names.update(var_coords.keys())

        # Build coords dict AND populate dims from coordinates
        for coord_name in all_coord_names:
            # Try to get from variable-specific coords first (more complete metadata)
            coord_data = None
            coord_dims = None
            coord_attrs = {}

            for var_name in zarr_group.keys():
                var_coords = zarr_group[var_name].attrs.get("coords", {})
                if coord_name in var_coords:
                    coord_info = var_coords[coord_name]
                    if isinstance(coord_info, dict) and "data" in coord_info:
                        coord_data = coord_info["data"]
                        coord_dims = tuple(
                            coord_info.get("dims", [coord_name])
                        )
                        coord_attrs = coord_info.get("attrs", {})
                        coord_dtype = coord_info.get("dtype", None)

                        # Convert datetime if needed
                        coord_data = _convert_datetime_coordinate(
                            coord_data, coord_dtype
                        )
                        break

            if coord_data is not None:
                result["coords"][coord_name] = {
                    "dims": coord_dims,
                    "data": coord_data,
                    "attrs": coord_attrs,
                }

                # Add to dims dict - use the actual dimension names from coords
                # coords like 'band (with_tillage)' have dims ('band (with_tillage)',)
                if len(coord_dims) == 1:
                    dim_name = coord_dims[0]
                    # Add the actual dimension name (not normalized)
                    result["dims"][dim_name] = (
                        len(coord_data) if isinstance(coord_data, list) else 1
                    )

        # Build data_vars
        for var_name in zarr_group.keys():
            var_array = zarr_group[var_name]
            var_data = var_array[
                self.indices if self.indices is not None else slice(None)
            ]
            var_dims = tuple(var_array.attrs.get("dims", ()))
            var_attrs = var_array.attrs.get("attrs", {})

            result["data_vars"][var_name] = {
                "dims": var_dims,
                "data": var_data.tolist(),
                "attrs": var_attrs,
            }

        return result

    def __repr__(self):
        """Return full xarray/LPJmLDataSet representation.

        Delegates to the underlying xarray object to show complete information
        including dimensions, coordinates, data variables, and attributes.
        """
        return repr(self._get_xarray())


class ZarrDataArrayView:
    """Provides xarray.DataArray-like interface to Zarr-backed arrays.

    This class provides a view into a Zarr array that behaves like an
    xarray DataArray. It uses lazy loading - converting to actual
    xarray/LPJmLData only when needed, while maintaining Zarr synchronization.

    All xarray methods and attributes are available through automatic
    delegation to the underlying xarray object.

    Parameters
    ----------
    backend : ZarrBackend
        The Zarr backend managing the data
    path : str
        Path to the array in the Zarr store
    indices : int, slice, array-like, optional
        Cell indices for this view
    """

    def __init__(
        self,
        backend: ZarrBackend,
        path: str,
        indices: Optional[Union[int, slice, np.ndarray, List[int]]] = None,
    ):
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "path", path)

        # Normalize indices
        if indices is None:
            normalized_indices = slice(None)
        elif isinstance(indices, (int, np.integer)):
            normalized_indices = indices
        elif isinstance(indices, list):
            normalized_indices = np.array(indices)
        else:
            normalized_indices = indices

        object.__setattr__(self, "indices", normalized_indices)
        object.__setattr__(
            self,
            "_is_scalar",
            isinstance(normalized_indices, (int, np.integer)),
        )
        object.__setattr__(self, "_zarr_array", backend.root[path])

        # Performance: Cache frequently accessed metadata and LPJmL representation
        object.__setattr__(self, "_shape_cache", None)
        object.__setattr__(self, "_dims_cache", None)
        object.__setattr__(self, "_dtype_cache", None)
        object.__setattr__(self, "_attrs_cache", None)
        object.__setattr__(
            self, "_lpjml_cache", None
        )  # Cache for LPJmLData representation

    def __setattr__(self, name, value):
        """Set attribute (delegates to object.__setattr__ to avoid recursion)."""
        object.__setattr__(self, name, value)

    def _get_lpjml_representation(self):
        """Get LPJmLData representation with normalized dimensions (lazy-loaded, cached).

        This provides the nice representation and normalized dimension names while
        keeping the underlying Zarr data writable through this view.

        The normalization happens by getting the variable from its parent LPJmLDataSet,
        which calls LPJmLDataSet._construct_dataarray() to normalize dimensions.

        Returns
        -------
        LPJmLData or xr.DataArray
            Normalized representation
        """
        if self._lpjml_cache is None:
            # Get the parent dataset and variable name
            # path is like "output/hdate" or "input/fertilizer"
            parts = self.path.split("/")
            if len(parts) >= 2:
                group = parts[0]
                var_name = parts[1]

                # Get the dataset view and access the variable through it
                # This ensures LPJmLDataSet._construct_dataarray is called
                dataset_view = self.backend.get_view(group, self.indices)
                lpjml_ds = dataset_view._get_xarray()

                # Access the variable - this triggers _construct_dataarray
                if hasattr(lpjml_ds, var_name):
                    object.__setattr__(
                        self, "_lpjml_cache", getattr(lpjml_ds, var_name)
                    )
                else:
                    # Fallback to direct conversion
                    object.__setattr__(self, "_lpjml_cache", self.to_xarray())
            else:
                # Fallback to direct conversion
                object.__setattr__(self, "_lpjml_cache", self.to_xarray())
        return self._lpjml_cache

    def __repr__(self):
        """Return string representation (delegates to LPJmLData for nice formatting)."""
        try:
            return repr(self._get_lpjml_representation())
        except Exception:
            return f"ZarrDataArrayView(path='{self.path}', shape={self.shape}, dtype={self.dtype})"

    def __str__(self):
        """Return string representation (delegates to LPJmLData)."""
        try:
            return str(self._get_lpjml_representation())
        except Exception:
            return self.__repr__()

    @property
    def values(self):
        """Get values as numpy array (creates a view when possible).

        Returns
        -------
        np.ndarray
            Array values at the specified indices
        """
        # Note: Zarr 3.x read operations are thread-safe
        # Lock only needed if concurrent writes are happening
        if self._is_scalar:
            return self._zarr_array[self.indices]
        else:
            # Return fancy indexed values
            return self._zarr_array[self.indices]

    @values.setter
    def values(self, new_values):
        """Set values from numpy array.

        Parameters
        ----------
        new_values : array-like
            New values to set
        """
        with self.backend.lock:
            if self._is_scalar:
                self._zarr_array[self.indices] = new_values
            else:
                self._zarr_array[self.indices] = new_values

            # Invalidate cache after write so repr/dims/coords reflect new data
            object.__setattr__(self, "_lpjml_cache", None)

    @property
    def shape(self):
        """Return shape of the view (cached for performance)."""
        if self._shape_cache is not None:
            return self._shape_cache

        if self._is_scalar:
            shape = self._zarr_array.shape[1:]  # Remove cell dimension
        elif isinstance(self.indices, slice):
            n_cells = len(
                range(*self.indices.indices(self._zarr_array.shape[0]))
            )
            shape = (n_cells,) + self._zarr_array.shape[1:]
        else:
            shape = (len(self.indices),) + self._zarr_array.shape[1:]

        self._shape_cache = shape
        return shape

    @property
    def dtype(self):
        """Return data type (cached for performance)."""
        if self._dtype_cache is None:
            self._dtype_cache = self._zarr_array.dtype
        return self._dtype_cache

    @property
    def dims(self):
        """Return normalized dimension names (e.g., 'band' instead of 'band (hdate)').

        This delegates to the LPJmLData representation for consistent dimension names.
        """
        try:
            return self._get_lpjml_representation().dims
        except Exception:
            # Fallback to raw Zarr dims if LPJmL representation fails
            if self._dims_cache is None:
                self._dims_cache = self._zarr_array.attrs.get("dims", None)
            return self._dims_cache

    @property
    def attrs(self):
        """Return attributes (cached for performance)."""
        if self._attrs_cache is None:
            self._attrs_cache = self._zarr_array.attrs.get("attrs", {})
        return self._attrs_cache

    @property
    def coords(self):
        """Get normalized coordinates dictionary (xarray compatibility).

        This delegates to the LPJmLData representation to provide normalized
        coordinate names (e.g., 'band' instead of 'band (hdate)').

        Returns
        -------
        xr.core.coordinates.DataArrayCoordinates
            Normalized coordinates from LPJmLData
        """
        try:
            return self._get_lpjml_representation().coords
        except Exception:
            # Fallback to raw Zarr coords if LPJmL representation fails
            coords_dict = {}
            array_coords = self._zarr_array.attrs.get("coords", {})

            for coord_name, coord_info in array_coords.items():
                if isinstance(coord_info, dict) and "data" in coord_info:
                    coord_data = coord_info["data"]
                    coord_dtype = coord_info.get("dtype", None)

                    # Convert to datetime64 if needed
                    if coord_dtype and coord_dtype.startswith("datetime64"):
                        coord_data = np.array(coord_data, dtype="int64").view(
                            "datetime64[ns]"
                        )
                    else:
                        coord_data = np.array(coord_data)

                    coords_dict[coord_name] = coord_data
                else:
                    coords_dict[coord_name] = np.array(coord_info)

            return coords_dict

    @property
    def cell(self):
        """Return cell coordinate as a simple object with .values.

        This is for compatibility with code that accesses grid.cell.values.
        """

        class CellCoordinate:
            def __init__(self, values):
                self.values = values

        # Get cell indices for this view
        if self._is_scalar:
            cell_vals = self.indices
        elif isinstance(self.indices, slice):
            # Return the range of cells
            n_cells = self.backend.root.attrs.get(
                "n_cells", self._zarr_array.shape[0]
            )
            cell_vals = np.arange(n_cells)[self.indices]
        else:
            cell_vals = self.indices

        return CellCoordinate(cell_vals)

    def isel(self, indexers: Dict[str, Any], drop: bool = False):
        """Select by integer position.

        Parameters
        ----------
        indexers : dict
            Dictionary of dimension names to indices
        drop : bool
            Whether to drop scalar dimensions

        Returns
        -------
        ZarrDataArrayView
            New view with updated indices
        """
        if "cell" not in indexers:
            # If not indexing cells, return self (for other dimensions)
            # Note: Multi-dimensional indexing for band/time dimensions not yet implemented.
            # Currently only cell indexing is supported. To add band/time indexing:
            # 1. Track band/time slices in addition to cell indices
            # 2. Apply combined indexing to underlying Zarr array
            # 3. Update shape and coords accordingly
            return self

        cell_indexer = indexers["cell"]

        # Combine current indices with new indexer
        if isinstance(self.indices, slice) and self.indices == slice(None):
            new_indices = cell_indexer
        elif self._is_scalar:
            raise IndexError("Cannot index a scalar cell view")
        else:
            if isinstance(self.indices, np.ndarray):
                new_indices = self.indices[cell_indexer]
            else:
                new_indices = np.arange(self.backend.root.attrs["n_cells"])[
                    self.indices
                ][cell_indexer]

        return ZarrDataArrayView(self.backend, self.path, new_indices)

    def to_xarray(self):
        """Convert view to LPJmLData (creates a copy).

        Returns
        -------
        LPJmLData or xr.DataArray
            LPJmLData with current data (falls back to xr.DataArray if pycoupler not available)
        """
        data = self.values
        stored_dims = list(self.dims) if self.dims else []
        attrs = dict(self.attrs)

        # Get the variable/array name from attrs or path
        var_name = self._zarr_array.attrs.get("name", self.path.split("/")[-1])

        # Get coords from array attrs (stored per-variable/array with full metadata)
        coords_stored = self._zarr_array.attrs.get("coords", {})

        # For scalar views (single cell), we need to drop the cell dimension
        if self._is_scalar:
            # Remove 'cell' from dims
            dims = tuple(d for d in stored_dims if d != "cell")
        else:
            dims = tuple(stored_dims)

        # Reconstruct coords as xarray-compatible dict
        # IMPORTANT: Only include coordinates whose dimensions are present in this variable
        coords_dict = {}
        for coord_name, coord_info in coords_stored.items():
            # Skip cell coordinate for scalar views
            if self._is_scalar and coord_name == "cell":
                continue

            if isinstance(coord_info, dict) and "data" in coord_info:
                # New format with dims and attrs
                coord_dims = tuple(coord_info["dims"])
                coord_data = coord_info["data"]
                coord_attrs = coord_info.get("attrs", {})
                coord_dtype = coord_info.get("dtype", None)

                # Reconstruct datetime if needed
                coord_data = _convert_datetime_coordinate(
                    coord_data, coord_dtype
                )

                # For scalar views, drop 'cell' from coord dims too
                if self._is_scalar:
                    coord_dims = tuple(d for d in coord_dims if d != "cell")

                # Only add coordinate if:
                # 1. Its dimensions are not empty
                # 2. All its dimensions are present in this variable's dimensions
                if coord_dims and all(cd in dims for cd in coord_dims):
                    # Create xarray coordinate
                    coords_dict[coord_name] = (
                        coord_dims,
                        coord_data,
                        coord_attrs,
                    )
            else:
                # Old format (just data)
                coords_dict[coord_name] = coord_info

        # Create xr.DataArray first
        da = xr.DataArray(
            data, dims=dims, coords=coords_dict, attrs=attrs, name=var_name
        )

        # Convert to LPJmLData for proper dimension normalization behavior
        # This ensures 'band (hdate)' -> 'band' normalization like pycoupler
        if HAS_PYCOUPLER:
            try:
                # Use LPJmLData constructor to get proper normalization
                return LPJmLData(da)
            except Exception:
                # Fallback to xr.DataArray if conversion fails
                return da
        else:
            return da

    def __getitem__(self, key):
        """Support numpy-style indexing."""
        # This allows things like: cell.input['fertilizer'][0]
        # OPTIMIZED: Direct Zarr array access without loading full data
        # PERFORMANCE: No lock needed for reads (Zarr is thread-safe)
        # Build the full indexer by combining self.indices and key
        if self._is_scalar:
            # Single cell view: self.indices is an int
            if isinstance(key, tuple):
                full_index = (self.indices,) + key
            else:
                full_index = (self.indices, key)
        elif isinstance(self.indices, slice) and self.indices == slice(None):
            # Full view (all cells): key is the full index
            full_index = key
        else:
            # Subset view: need to map key through self.indices
            if isinstance(key, tuple):
                # Multi-dimensional key
                first_dim_key = key[0]
                rest_of_key = key[1:]
                if isinstance(self.indices, np.ndarray):
                    mapped_first = self.indices[first_dim_key]
                else:
                    mapped_first = self.indices
                full_index = (mapped_first,) + rest_of_key
            else:
                # Single dimension key
                if isinstance(self.indices, np.ndarray):
                    full_index = self.indices[key]
                else:
                    full_index = key

        # Direct access to Zarr array - no intermediate copy
        return self._zarr_array[full_index]

    def __setitem__(self, key, value):
        """Support numpy-style setting."""
        with self.backend.lock:
            # Build the full indexer by combining self.indices and key
            if self._is_scalar:
                # Single cell view: self.indices is an int
                if isinstance(key, tuple):
                    full_index = (self.indices,) + key
                else:
                    full_index = (self.indices, key)
            elif isinstance(self.indices, slice) and self.indices == slice(
                None
            ):
                # Full view (all cells): key is the full index
                full_index = key
            else:
                # Subset view: need to map key through self.indices
                if isinstance(key, tuple):
                    # Multi-dimensional key
                    first_dim_key = key[0]
                    rest_of_key = key[1:]
                    if isinstance(self.indices, np.ndarray):
                        mapped_first = self.indices[first_dim_key]
                    else:
                        mapped_first = self.indices
                    full_index = (mapped_first,) + rest_of_key
                else:
                    # Single dimension key
                    if isinstance(self.indices, np.ndarray):
                        full_index = self.indices[key]
                    else:
                        full_index = key

            # Assign directly to Zarr array at the calculated index
            self._zarr_array[full_index] = value

            # Invalidate cache after write so repr/dims/coords reflect new data
            object.__setattr__(self, "_lpjml_cache", None)

    def to_dict(self):
        """Convert to dictionary (for compatibility with xarray interface).

        Returns
        -------
        dict
            Dictionary representation compatible with xarray.DataArray.to_dict()
        """
        return self.to_xarray().to_dict()

    def __getattr__(self, name: str):
        """Access coordinates, methods, and all xarray attributes.

        This provides full xarray/LPJmL compatibility by delegating to the
        cached LPJmLData representation, which handles dimension normalization
        and provides all xarray functionality.

        Parameters
        ----------
        name : str
            Coordinate name or xarray method/attribute

        Returns
        -------
        Any
            Coordinate values, xarray method result, or attribute value

        Examples
        --------
        >>> array.band  # Access normalized 'band' coordinate
        >>> array.time  # Access 'time' coordinate
        >>> array.mean()  # Use xarray methods
        >>> array.plot()  # Use xarray plot methods
        """
        # Avoid infinite recursion - only handle non-private attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        # Delegate all attribute access to the LPJmLData representation
        # This ensures proper dimension normalization and full xarray compatibility
        lpjml_obj = self._get_lpjml_representation()
        if hasattr(lpjml_obj, name):
            return getattr(lpjml_obj, name)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
