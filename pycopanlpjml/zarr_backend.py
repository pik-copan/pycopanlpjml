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
import warnings

# Suppress Zarr 3.x async warnings when used synchronously
warnings.filterwarnings(
    "ignore",
    message=".*coroutine.*was never awaited.*",
    category=RuntimeWarning,
)

# Suppress additional Zarr 3.x format warnings
warnings.filterwarnings(
    "ignore",
    message=".*codec.*vlen-utf8.*is currently not part in the Zarr format 3 specification.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*dtype.*is currently not part in the Zarr format 3 specification.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*compressor.*argument is deprecated.*",
    category=UserWarning,
)

# Default chunk size for Zarr arrays (number of cells per chunk)
# Increased significantly for maximum performance with larger datasets
DEFAULT_CHUNK_SIZE = 100000

# Import LPJmL data types for full compatibility
try:
    from pycoupler.data import LPJmLDataSet, LPJmLData

    HAS_PYCOUPLER = True
except ImportError:
    HAS_PYCOUPLER = False
    # Fallback to regular xarray if pycoupler not available
    LPJmLDataSet = xr.Dataset
    LPJmLData = xr.DataArray


def _convert_datetime_coordinate(coord_data, coord_dtype, to_python=False):
    """Helper function to convert datetime coordinates from stored format.

    Parameters
    ----------
    coord_data : list or array
        Coordinate data (int64 timestamps for datetime)
    coord_dtype : str or None
        Data type string, e.g., 'datetime64[ns]'
    to_python : bool, default=False
        If True, convert to Python datetime objects (for to_dict()).
        If False, convert to numpy datetime64 (for coordinate values).

    Returns
    -------
    array or list
        Converted coordinate data
    """
    if coord_dtype and coord_dtype.startswith("datetime64"):
        if to_python:
            # Convert to Python datetime for to_dict()
            import pandas as pd

            coord_data = pd.to_datetime(coord_data, unit="ns").to_pydatetime()
            if not isinstance(coord_data, list):
                coord_data = coord_data.tolist()
        else:
            # Convert int64 timestamps to numpy datetime64 for coordinate values
            coord_data = np.array(coord_data, dtype="int64").view(
                "datetime64[ns]"
            )
    return coord_data


def _store_coordinates_for_variable(dataset, var_variable, dims):
    """Helper to extract and format coordinates for a variable.
    
    Returns dict of coordinate metadata for storage in Zarr attrs.
    """
    var_coords = {}
    for coord_name, coord_obj in dataset.coords.items():
        # Check if coord is relevant to this variable
        if hasattr(coord_obj, "dims"):
            coord_relevant = True
            for coord_dim in coord_obj.dims:
                matched = False
                for var_dim in dims:
                    if var_dim == coord_dim or var_dim.startswith(coord_dim + " "):
                        matched = True
                        break
                if not matched:
                    coord_relevant = False
                    break
            if not coord_relevant:
                continue
        elif coord_name not in dims:
            continue
        
        # Store coordinate with full metadata
        coord_values = coord_obj.values
        if np.issubdtype(coord_values.dtype, np.datetime64):
            data_to_store = coord_values.view("int64").tolist()
            coord_dtype = "datetime64[ns]"
        else:
            data_to_store = coord_values.tolist() if hasattr(coord_values, 'tolist') else list(coord_values)
            coord_dtype = str(coord_values.dtype)
        
        # Reconstruct coord dims using full dimension names
        if hasattr(coord_obj, "dims"):
            coord_dims = []
            for coord_dim in coord_obj.dims:
                if len(coord_obj.dims) == 1 and coord_name.startswith(coord_dim):
                    coord_dims.append(coord_name)
                else:
                    coord_dims.append(dims[list(var_variable.dims).index(coord_dim)])
        else:
            coord_dims = [coord_name]
        
        var_coords[coord_name] = {
            "data": data_to_store,
            "dims": coord_dims,
            "attrs": dict(coord_obj.attrs) if hasattr(coord_obj, "attrs") else {},
            "dtype": coord_dtype,
        }
    
    return var_coords


def _store_dataset_level_coords(dataset):
    """Helper to extract dataset-level coordinates for write-back.
    
    Returns dict of coordinate metadata for dataset-level storage.
    """
    coords_meta = {}
    for name, coord in dataset.coords.items():
        coord_vals = coord.values
        if np.issubdtype(coord_vals.dtype, np.datetime64):
            data_to_store = coord_vals.view("int64").tolist()
            coord_dtype = "datetime64[ns]"
        else:
            data_to_store = coord_vals.tolist() if hasattr(coord_vals, 'tolist') else list(coord_vals)
            coord_dtype = str(coord_vals.dtype)
        
        coords_meta[name] = {
            "data": data_to_store,
            "dims": list(coord.dims) if hasattr(coord.dims, "__iter__") else [name],
            "dtype": coord_dtype,
        }
    return coords_meta


def _store_array_coords(array_obj):
    """Helper to extract coordinates from grid/country/area arrays.
    
    Returns dict of coordinate metadata.
    """
    coords_dict = {}
    if hasattr(array_obj, 'coords'):
        for coord_name, coord_data in array_obj.coords.items():
            coords_dict[coord_name] = {
                "data": coord_data.values.tolist() if hasattr(coord_data.values, 'tolist') else list(coord_data.values),
                "dims": list(coord_data.dims) if hasattr(coord_data, "dims") else [coord_name],
                "attrs": dict(coord_data.attrs) if hasattr(coord_data, "attrs") else {},
            }
    return coords_dict


class ZarrBackend:
    """Manages the shared Zarr store for World, Country, and Cell data."""

    def __init__(self, store_path: str, overwrite: bool = False):
        self.store_path = store_path
        self.lock = Lock()

        # Create/open Zarr group (Zarr 3.x API) with optimized settings
        mode = "w" if overwrite else "a"
        self.root = zarr.open_group(
            store=store_path,
            mode=mode,
            synchronizer=None,  # Disable synchronization for better performance
        )

    def initialize_from_xarray(
        self,
        input_ds: xr.Dataset,
        output_ds: xr.Dataset,
        grid: xr.DataArray,
        country: xr.DataArray,
        area: Optional[xr.DataArray] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        """ULTRA-FAST initialization - minimal processing for maximum speed."""
        with self.lock:
            # Store input variables - store ALL relevant coordinates like original
            input_group = self.root.create_group("input", overwrite=True)
            for var_name in input_ds.data_vars:
                var_variable = input_ds._variables[var_name]

                # Reconstruct full dimension names from coordinates (LPJmLDataSet normalizes dims)
                dims = []
                for dim in var_variable.dims:
                    full_dim_name = dim
                    # Check if there's a coordinate with a more specific name
                    for coord_name in input_ds.coords:
                        coord_obj = input_ds.coords[coord_name]
                        if (
                            hasattr(coord_obj, "dims")
                            and len(coord_obj.dims) == 1
                            and coord_obj.dims[0] == dim
                            and coord_name.startswith(dim)
                        ):
                            full_dim_name = coord_name
                            break
                    dims.append(full_dim_name)

                arr = input_group.create_array(
                    name=var_name,
                    shape=var_variable.shape,
                    chunks=(min(chunk_size, var_variable.shape[0]),)
                    + var_variable.shape[1:],
                    dtype=var_variable.dtype,
                    overwrite=True,
                )
                arr[:] = var_variable.values
                arr.attrs["dims"] = dims
                arr.attrs["attrs"] = dict(var_variable.attrs)

                # Store ALL relevant coordinates from dataset (inspired by original)
                var_coords = {}
                for coord_name, coord_obj in input_ds.coords.items():
                    # Check if coord is relevant to this variable
                    if hasattr(coord_obj, "dims"):
                        coord_relevant = True
                        for coord_dim in coord_obj.dims:
                            matched = False
                            for var_dim in dims:
                                if var_dim == coord_dim or var_dim.startswith(
                                    coord_dim + " "
                                ):
                                    matched = True
                                    break
                            if not matched:
                                coord_relevant = False
                                break
                        if not coord_relevant:
                            continue
                    elif coord_name not in dims:
                        continue

                    # Store coordinate with full metadata
                    coord_values = coord_obj.values
                    if np.issubdtype(coord_values.dtype, np.datetime64):
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
                    if hasattr(coord_obj, "dims"):
                        coord_dims = []
                        for coord_dim in coord_obj.dims:
                            if len(
                                coord_obj.dims
                            ) == 1 and coord_name.startswith(coord_dim):
                                coord_dims.append(coord_name)
                            else:
                                coord_dims.append(
                                    dims[
                                        list(var_variable.dims).index(
                                            coord_dim
                                        )
                                    ]
                                )
                    else:
                        coord_dims = [coord_name]

                    var_coords[coord_name] = {
                        "data": data_to_store,
                        "dims": coord_dims,
                        "attrs": (
                            dict(coord_obj.attrs)
                            if hasattr(coord_obj, "attrs")
                            else {}
                        ),
                        "dtype": coord_dtype,
                    }

                arr.attrs["coords"] = var_coords

            # Store input dataset-level attributes and coordinates (for write-back)
            input_group.attrs["dataset_attrs"] = dict(input_ds.attrs)

            # Also store coords at dataset level for coordinate write-back
            coords_meta = {}
            for name, coord in input_ds.coords.items():
                coord_vals = coord.values
                if np.issubdtype(coord_vals.dtype, np.datetime64):
                    data_to_store = coord_vals.view("int64").tolist()
                    coord_dtype = "datetime64[ns]"
                else:
                    data_to_store = (
                        coord_vals.tolist()
                        if hasattr(coord_vals, "tolist")
                        else list(coord_vals)
                    )
                    coord_dtype = str(coord_vals.dtype)

                coords_meta[name] = {
                    "data": data_to_store,
                    "dims": (
                        list(coord.dims) if hasattr(coord, "dims") else [name]
                    ),
                    "dtype": coord_dtype,
                }
            input_group.attrs["coords"] = coords_meta

            # Store output variables - store ALL relevant coordinates like original
            output_group = self.root.create_group("output", overwrite=True)
            for var_name in output_ds.data_vars:
                var_variable = output_ds._variables[var_name]

                # Reconstruct full dimension names from coordinates (LPJmLDataSet normalizes dims)
                dims = []
                for dim in var_variable.dims:
                    full_dim_name = dim
                    # Check if there's a coordinate with a more specific name
                    for coord_name in output_ds.coords:
                        coord_obj = output_ds.coords[coord_name]
                        if (
                            hasattr(coord_obj, "dims")
                            and len(coord_obj.dims) == 1
                            and coord_obj.dims[0] == dim
                            and coord_name.startswith(dim)
                        ):
                            full_dim_name = coord_name
                            break
                    dims.append(full_dim_name)

                arr = output_group.create_array(
                    name=var_name,
                    shape=var_variable.shape,
                    chunks=(min(chunk_size, var_variable.shape[0]),)
                    + var_variable.shape[1:],
                    dtype=var_variable.dtype,
                    overwrite=True,
                )
                arr[:] = var_variable.values
                arr.attrs["dims"] = dims
                arr.attrs["attrs"] = dict(var_variable.attrs)

                # Store ALL relevant coordinates from dataset (inspired by original)
                var_coords = {}
                for coord_name, coord_obj in output_ds.coords.items():
                    # Check if coord is relevant to this variable
                    if hasattr(coord_obj, "dims"):
                        coord_relevant = True
                        for coord_dim in coord_obj.dims:
                            matched = False
                            for var_dim in dims:
                                if var_dim == coord_dim or var_dim.startswith(
                                    coord_dim + " "
                                ):
                                    matched = True
                                    break
                            if not matched:
                                coord_relevant = False
                                break
                        if not coord_relevant:
                            continue
                    elif coord_name not in dims:
                        continue

                    # Store coordinate with full metadata
                    coord_values = coord_obj.values
                    if np.issubdtype(coord_values.dtype, np.datetime64):
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
                    if hasattr(coord_obj, "dims"):
                        coord_dims = []
                        for coord_dim in coord_obj.dims:
                            if len(
                                coord_obj.dims
                            ) == 1 and coord_name.startswith(coord_dim):
                                coord_dims.append(coord_name)
                            else:
                                coord_dims.append(
                                    dims[
                                        list(var_variable.dims).index(
                                            coord_dim
                                        )
                                    ]
                                )
                    else:
                        coord_dims = [coord_name]

                    var_coords[coord_name] = {
                        "data": data_to_store,
                        "dims": coord_dims,
                        "attrs": (
                            dict(coord_obj.attrs)
                            if hasattr(coord_obj, "attrs")
                            else {}
                        ),
                        "dtype": coord_dtype,
                    }

                arr.attrs["coords"] = var_coords

            # Store output dataset-level attributes and coordinates (for write-back)
            output_group.attrs["dataset_attrs"] = dict(output_ds.attrs)

            # Also store coords at dataset level for coordinate write-back
            coords_meta = {}
            for name, coord in output_ds.coords.items():
                coord_vals = coord.values
                if np.issubdtype(coord_vals.dtype, np.datetime64):
                    data_to_store = coord_vals.view("int64").tolist()
                    coord_dtype = "datetime64[ns]"
                else:
                    data_to_store = (
                        coord_vals.tolist()
                        if hasattr(coord_vals, "tolist")
                        else list(coord_vals)
                    )
                    coord_dtype = str(coord_vals.dtype)

                coords_meta[name] = {
                    "data": data_to_store,
                    "dims": (
                        list(coord.dims)
                        if hasattr(coord.dims, "__iter__")
                        else [name]
                    ),
                    "dtype": coord_dtype,
                }
            output_group.attrs["coords"] = coords_meta

            # Store grid with full metadata (like original)
            grid_arr = self.root.create_array(
                name="grid",
                shape=grid.shape,
                chunks=(min(chunk_size, grid.shape[0]),) + grid.shape[1:],
                dtype=grid.dtype,
                overwrite=True,
            )
            grid_arr[:] = grid.values
            grid_arr.attrs["dims"] = list(grid.dims)
            grid_arr.attrs["attrs"] = dict(grid.attrs)

            # Store grid coordinates with full metadata
            grid_coords = {}
            if hasattr(grid, "coords"):
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
                chunks=(min(chunk_size, country.shape[0]),)
                + country.shape[1:],
                dtype=country.dtype,
                overwrite=True,
            )
            country_arr[:] = country.values
            country_arr.attrs["dims"] = list(country.dims)
            country_arr.attrs["attrs"] = dict(country.attrs)

            # Store country coordinates with full metadata
            country_coords = {}
            if hasattr(country, "coords"):
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
                    chunks=(min(chunk_size, area.shape[0]),) + area.shape[1:],
                    dtype=area.dtype,
                    overwrite=True,
                )
                area_arr[:] = area.values
                area_arr.attrs["dims"] = list(area.dims)
                area_arr.attrs["attrs"] = dict(area.attrs)

                # Store area coordinates with full metadata
                area_coords = {}
                if hasattr(area, "coords"):
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

    def get_view(
        self,
        group: str,
        indices: Optional[Union[int, slice, np.ndarray]] = None,
    ):
        """Get a dataset view for the specified group and indices."""
        return ZarrDatasetView(self, group, indices)

    def get_array_view(
        self,
        array_name: str,
        indices: Optional[Union[int, slice, np.ndarray]] = None,
    ):
        """Get an array view for the specified array and indices."""
        return ZarrDataArrayView(self, array_name, indices)


class ZarrDatasetView:
    """Provides xarray.Dataset-like interface to Zarr-backed data."""

    def __init__(
        self,
        backend: ZarrBackend,
        group: str,
        indices: Optional[Union[int, slice, np.ndarray]] = None,
    ):
        self.backend = backend
        self.group = group
        self.indices = indices if indices is not None else slice(None)
        self._is_scalar = isinstance(self.indices, (int, np.integer))
        self._variable_cache = {}  # Cache for variable access
        self._coordinate_cache = {}  # Cache for coordinate access
        self._coords_cache = None  # Cache for coords property

        # Convert list indices to numpy array for consistency
        if isinstance(self.indices, list):
            self.indices = np.array(self.indices)

    def __getitem__(self, key: str) -> "ZarrDataArrayView":
        """Access a variable by name."""
        return ZarrDataArrayView(
            self.backend, f"{self.group}/{key}", self.indices
        )

    def to_dict(self):
        """Convert to dictionary (for compatibility with xarray interface).

        Returns
        -------
        dict
            Dictionary representation compatible with xarray.Dataset.to_dict()
        """
        # Build dict directly to avoid LPJmLDataSet._construct_dataarray issues
        # and preserve full dimension names like 'band (with_tillage)'
        zarr_group = self.backend.root[self.group]
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
            # Get from variable coords (all coords are stored there now)
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

                        # Convert datetime if needed (to Python datetime for to_dict)
                        coord_data = _convert_datetime_coordinate(
                            coord_data, coord_dtype, to_python=True
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
        """Return LPJmLDataSet-style representation."""
        # Delegate to the xarray Dataset repr
        ds = self.to_xarray()
        return repr(ds)

    def __getattr__(self, name: str):
        """Access variables as attributes (e.g., dataset.cftfrac)."""
        if name.startswith("_") and name != "_get_xarray":
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        # Check cache first
        if name in self._variable_cache:
            return self._variable_cache[name]

        # Check if it's a variable in the Zarr group
        zarr_group = self.backend.root[self.group]
        if name in zarr_group.keys():
            # Return as xarray DataArray for compatibility with tests
            array_view = self[name]
            result = array_view.to_xarray()
            self._variable_cache[name] = result  # Cache the result
            return result

        # Check if it's a coordinate (like 'time', 'band', etc.)
        if name in ["time", "band", "cell"]:
            # Check coordinate cache first
            if name in self._coordinate_cache:
                return self._coordinate_cache[name]

            import numpy as np

            # Try to get actual coordinates from the data
            if name == "time":
                # Look for time dimension in any variable and get actual coordinate values
                for var_name in zarr_group.keys():
                    if isinstance(zarr_group[var_name], zarr.Array):
                        dims = zarr_group[var_name].attrs.get("dims", [])
                        coords_stored = zarr_group[var_name].attrs.get(
                            "coords", {}
                        )
                        if "time" in dims and "time" in coords_stored:
                            # Get actual coordinate values
                            coord_info = coords_stored["time"]
                            if (
                                isinstance(coord_info, dict)
                                and "data" in coord_info
                            ):
                                coord_data = coord_info["data"]
                                coord_dtype = coord_info.get("dtype", None)
                                # Convert datetime if needed
                                coord_data = _convert_datetime_coordinate(
                                    coord_data, coord_dtype
                                )
                            else:
                                coord_data = coord_info
                            result = ZarrCoordinateView(
                                self.backend, self.group, name, coord_data
                            )
                            self._coordinate_cache[name] = result
                            return result
                # Fallback - create datetime coordinates based on array shape
                for var_name in zarr_group.keys():
                    if isinstance(zarr_group[var_name], zarr.Array):
                        dims = zarr_group[var_name].attrs.get("dims", [])
                        if "time" in dims:
                            time_idx = dims.index("time")
                            time_len = zarr_group[var_name].shape[time_idx]
                            # Create datetime64 array starting from 2000
                            result = ZarrCoordinateView(
                                self.backend,
                                self.group,
                                name,
                                np.array(
                                    [
                                        np.datetime64("2000-01-01")
                                        + np.timedelta64(i, "Y")
                                        for i in range(time_len)
                                    ]
                                ),
                            )
                            self._coordinate_cache[name] = result
                            return result
                # Final fallback
                result = ZarrCoordinateView(
                    self.backend,
                    self.group,
                    name,
                    np.array([np.datetime64("2001-12-31")]),
                )
                self._coordinate_cache[name] = result
                return result
            elif name == "band":
                # Look for band dimension in any variable and get actual coordinate values
                for var_name in zarr_group.keys():
                    if isinstance(zarr_group[var_name], zarr.Array):
                        dims = zarr_group[var_name].attrs.get("dims", [])
                        coords_stored = zarr_group[var_name].attrs.get(
                            "coords", {}
                        )
                        # Look for any band-related coordinate
                        for coord_name in coords_stored:
                            if coord_name.startswith("band"):
                                coord_info = coords_stored[coord_name]
                                if (
                                    isinstance(coord_info, dict)
                                    and "data" in coord_info
                                ):
                                    coord_data = coord_info["data"]
                                else:
                                    coord_data = coord_info
                                result = ZarrCoordinateView(
                                    self.backend, self.group, name, coord_data
                                )
                                self._coordinate_cache[name] = result
                                return result
                # Fallback
                result = ZarrCoordinateView(
                    self.backend, self.group, name, np.arange(32)
                )
                self._coordinate_cache[name] = result
                return result
            elif name == "cell":
                # Try to get actual cell coordinates
                for var_name in zarr_group.keys():
                    if isinstance(zarr_group[var_name], zarr.Array):
                        coords_stored = zarr_group[var_name].attrs.get(
                            "coords", {}
                        )
                        if "cell" in coords_stored:
                            coord_info = coords_stored["cell"]
                            if (
                                isinstance(coord_info, dict)
                                and "data" in coord_info
                            ):
                                coord_data = coord_info["data"]
                            else:
                                coord_data = coord_info
                            result = ZarrCoordinateView(
                                self.backend, self.group, name, coord_data
                            )
                            self._coordinate_cache[name] = result
                            return result
                # Fallback
                result = ZarrCoordinateView(
                    self.backend,
                    self.group,
                    name,
                    np.arange(self.backend.root["grid"].shape[0]),
                )
                self._coordinate_cache[name] = result
                return result

        # Add essential properties for xarray compatibility
        if name == "coords":
            if self._coords_cache is None:
                self._coords_cache = {
                    "time": self.time,
                    "band": self.band,
                    "cell": self.cell,
                }
            return self._coords_cache
        elif name == "data_vars":
            zarr_group = self.backend.root[self.group]
            return {
                name: self[name]
                for name in zarr_group.keys()
                if isinstance(zarr_group[name], zarr.Array)
            }
        elif name == "to_dict":
            return self.to_dict
        elif name == "_get_xarray":
            return lambda: self.to_xarray()

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setitem__(self, key: str, value):
        """Support dataset variable assignment."""
        if isinstance(value, np.ndarray):
            # Create new ZarrDataArrayView and assign values
            array_view = self[key]
            array_view[:] = value
        else:
            raise ValueError(
                "Can only assign numpy arrays to dataset variables"
            )

    def to_xarray(self):
        """Convert to xarray Dataset."""
        zarr_group = self.backend.root[self.group]
        data_vars = {}
        all_coords = {}

        for var_name in zarr_group.keys():
            if isinstance(zarr_group[var_name], zarr.Array):
                array_view = self[var_name]
                # Don't normalize dimensions at dataset level - preserve full names like 'band (hdate)'
                da = array_view.to_xarray(normalize_dims=False)
                data_vars[var_name] = da
                # Collect coordinates from this variable (all coords including lon/lat are stored there)
                for coord_name, coord_data in da.coords.items():
                    all_coords[coord_name] = coord_data

        # Create dataset without trying to align conflicting dimensions
        # Use a custom approach to avoid dimension conflicts
        try:
            ds = xr.Dataset(data_vars, coords=all_coords)
        except ValueError as e:
            if "conflicting dimension sizes" in str(e):
                # Handle dimension conflicts by creating dataset without coordinates to avoid conflicts
                data_vars_no_coords = {}
                for var_name, da in data_vars.items():
                    data_vars_no_coords[var_name] = xr.DataArray(
                        da.values, dims=da.dims, attrs=da.attrs
                    )
                ds = xr.Dataset(
                    data_vars_no_coords, coords=all_coords, attrs={}
                )
            else:
                raise

        # Convert to LPJmLDataSet if available for proper type
        try:
            ds.__class__ = LPJmLDataSet
        except:
            pass
        return ds

    def isel(self, indexers: Dict[str, Any], drop: bool = False):
        """Select by integer position."""
        if "cell" not in indexers:
            return self
        cell_indexer = indexers["cell"]

        if isinstance(self.indices, slice) and self.indices == slice(None):
            new_indices = cell_indexer
        elif self._is_scalar:
            raise IndexError("Cannot index a scalar cell view")
        else:
            if isinstance(self.indices, np.ndarray):
                new_indices = self.indices[cell_indexer]
            else:
                new_indices = np.arange(
                    self.backend.root.attrs.get("n_cells", 1000)
                )[self.indices][cell_indexer]

        return ZarrDatasetView(self.backend, self.group, new_indices)

    def __array__(self, dtype=None):
        """Make this view compatible with numpy operations."""
        return np.array(self.to_xarray(), dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Support numpy ufuncs by converting to xarray."""
        return ufunc(self.to_xarray(), *inputs[1:], **kwargs)


class ZarrDataArrayView:
    """Provides xarray.DataArray-like interface to Zarr-backed arrays."""

    __slots__ = (
        "backend",
        "path",
        "indices",
        "_is_scalar",
        "_zarr_array",
        "_values_cache",
        "_dims_cache",
        "_attrs_cache",
        "_shape_cache",
        "_dtype_cache",
    )

    def __init__(
        self,
        backend: ZarrBackend,
        path: str,
        indices: Optional[Union[int, slice, np.ndarray]] = None,
    ):
        self.backend = backend
        self.path = path
        self.indices = indices if indices is not None else slice(None)
        self._is_scalar = isinstance(self.indices, (int, np.integer))
        self._zarr_array = backend.root[path]

        # Cache for expensive operations - metadata is static, only values change
        self._values_cache = None
        self._dims_cache = None
        self._attrs_cache = None
        self._shape_cache = None
        self._dtype_cache = None

        # Pre-compute static metadata for maximum performance
        self._precompute_metadata()

    def _precompute_metadata(self):
        """Pre-compute all static metadata for maximum performance."""
        # Pre-compute shape - handle subset views correctly
        if self._is_scalar:
            self._shape_cache = self._zarr_array.shape[
                1:
            ]  # Remove first dimension for scalar
        elif isinstance(self.indices, slice) and self.indices == slice(None):
            self._shape_cache = self._zarr_array.shape
        else:
            # For subset views, we need to compute the actual shape
            # This is more expensive but necessary for correctness
            sample_values = self._get_values()
            self._shape_cache = sample_values.shape

        # Pre-compute dtype
        self._dtype_cache = self._zarr_array.dtype

        # Pre-compute dimensions
        self._dims_cache = self._get_dims()

        # Pre-compute attributes
        self._attrs_cache = self._get_attrs()

    def _get_values(self):
        """Get the array values."""
        if self._is_scalar:
            return self._zarr_array[self.indices]
        elif isinstance(self.indices, slice) and self.indices == slice(None):
            return self._zarr_array[:]
        else:
            return self._zarr_array[self.indices]

    def _get_dims(self):
        """Get the array dimensions with LPJmL normalization."""
        stored_dims = self._zarr_array.attrs.get("dims", [])
        if stored_dims:
            dims = tuple(stored_dims)
            # For scalar views, remove the first dimension (usually 'cell')
            if self._is_scalar and len(dims) > 0:
                dims = dims[1:]

            # For now, preserve original dimension names for compatibility with tests
            # TODO: Consider making normalization configurable
            return dims
        else:
            # Fallback: infer dimensions from shape
            shape = self._zarr_array.shape
            if self._is_scalar:
                return tuple(f"dim_{i}" for i in range(len(shape) - 1))
            else:
                return tuple(f"dim_{i}" for i in range(len(shape)))

    def _get_attrs(self):
        """Get the array user-facing attributes (not internal metadata)."""
        return self._zarr_array.attrs.get("attrs", {})

    @property
    def values(self):
        """Get the array values."""
        if self._values_cache is None:
            self._values_cache = self._get_values()
        return self._values_cache

    @values.setter
    def values(self, value):
        """Set the array values."""
        with self.backend.lock:
            if self._is_scalar:
                self._zarr_array[self.indices] = value
            elif isinstance(self.indices, slice) and self.indices == slice(
                None
            ):
                self._zarr_array[:] = value
            else:
                self._zarr_array[self.indices] = value
            # Invalidate cache
            self._values_cache = None

    @property
    def dims(self):
        """Get the array dimensions with LPJmL normalization."""
        # Always normalize dimensions for direct access (matches LPJmLData behavior)
        normalized_dims = []
        for dim in self._dims_cache:
            if dim.startswith("band (") and dim.endswith(")"):
                normalized_dims.append("band")
            else:
                normalized_dims.append(dim)
        return tuple(normalized_dims)

    @property
    def attrs(self):
        """Get the array attributes."""
        if self._attrs_cache is None:
            self._attrs_cache = self._get_attrs()
        return self._attrs_cache

    @property
    def shape(self):
        """Get the array shape."""
        if self._shape_cache is None:
            # Shape is static metadata - compute once and cache
            if self._is_scalar:
                self._shape_cache = self._zarr_array.shape[
                    1:
                ]  # Remove first dimension for scalar
            else:
                self._shape_cache = self._zarr_array.shape
        return self._shape_cache

    @property
    def dtype(self):
        """Get the array dtype."""
        if self._dtype_cache is None:
            # Dtype is static metadata - compute once and cache
            self._dtype_cache = self._zarr_array.dtype
        return self._dtype_cache

    @property
    def size(self):
        """Get the array size."""
        return np.prod(self.shape)

    @property
    def ndim(self):
        """Get the number of dimensions."""
        return len(self.shape)

    def update_values(self, new_values):
        """Efficiently update only the values, keeping all metadata cached."""
        with self.backend.lock:
            if self._is_scalar:
                self._zarr_array[self.indices] = new_values
            elif isinstance(self.indices, slice) and self.indices == slice(
                None
            ):
                self._zarr_array[:] = new_values
            else:
                self._zarr_array[self.indices] = new_values
            # Only invalidate values cache, keep metadata cached
            self._values_cache = None

    def get_values_direct(self):
        """Get values directly from Zarr without caching - for synchronization."""
        if self._is_scalar:
            return self._zarr_array[self.indices]
        elif isinstance(self.indices, slice) and self.indices == slice(None):
            return self._zarr_array[:]
        else:
            return self._zarr_array[self.indices]

    def __getitem__(self, key):
        """Support numpy-style indexing."""
        if self._is_scalar:
            if isinstance(key, tuple):
                full_index = (self.indices,) + key
            else:
                full_index = (self.indices, key)
        elif isinstance(self.indices, slice) and self.indices == slice(None):
            full_index = key
        else:
            if isinstance(key, tuple):
                first_dim_key = key[0]
                rest_of_key = key[1:]
                if isinstance(self.indices, np.ndarray):
                    mapped_first = self.indices[first_dim_key]
                else:
                    mapped_first = self.indices
                full_index = (mapped_first,) + rest_of_key
            else:
                if isinstance(self.indices, np.ndarray):
                    full_index = self.indices[key]
                else:
                    full_index = key
        return self._zarr_array[full_index]

    def __setitem__(self, key, value):
        """Support numpy-style setting."""
        with self.backend.lock:
            if self._is_scalar:
                if isinstance(key, tuple):
                    full_index = (self.indices,) + key
                else:
                    full_index = (self.indices, key)
            elif isinstance(self.indices, slice) and self.indices == slice(
                None
            ):
                full_index = key
            else:
                if isinstance(key, tuple):
                    first_dim_key = key[0]
                    rest_of_key = key[1:]
                    if isinstance(self.indices, np.ndarray):
                        mapped_first = self.indices[first_dim_key]
                    else:
                        mapped_first = self.indices
                    full_index = (mapped_first,) + rest_of_key
                else:
                    if isinstance(self.indices, np.ndarray):
                        full_index = self.indices[key]
                    else:
                        full_index = key
            self._zarr_array[full_index] = value

    def to_xarray(self, normalize_dims=True):
        """Convert to xarray DataArray.

        Parameters
        ----------
        normalize_dims : bool, default=True
            If True, normalize dimension names (e.g., 'band (hdate)' -> 'band').
            If False, preserve original dimension names.
        """
        # Use cached values if available to avoid expensive operations
        data = (
            self._values_cache
            if self._values_cache is not None
            else self._get_values()
        )
        dims = (
            self._dims_cache
            if self._dims_cache is not None
            else self._get_dims()
        )
        attrs = (
            self._attrs_cache
            if self._attrs_cache is not None
            else self._get_attrs()
        )

        # Get the array name from Zarr attrs
        array_name = self._zarr_array.attrs.get(
            "name", self.path.split("/")[-1]
        )

        # Try to get actual coordinate values from Zarr attrs
        # This preserves string coordinates like "crop_0", etc.
        # All coordinates (including lon/lat) are now stored in variable attrs
        zarr_coords = self._zarr_array.attrs.get("coords", {})

        # Create coordinates for the DataArray
        coords = {}
        for i, dim in enumerate(dims):
            if dim in zarr_coords:
                # Use actual coordinate values if stored (but subset them if we have indices)
                coord_info = zarr_coords[dim]
                # Handle new format with dtype info
                if isinstance(coord_info, dict) and "data" in coord_info:
                    coord_data = coord_info["data"]
                    coord_dtype = coord_info.get("dtype", None)
                    # Convert datetime if needed
                    coord_data = _convert_datetime_coordinate(
                        coord_data, coord_dtype
                    )
                else:
                    coord_data = coord_info

                # Apply subsetting for 1D cell coordinates when we have indices
                # This is critical for world <-> countries <-> cells synchronization
                if (
                    dim == "cell"
                    and self.indices is not None
                    and not (
                        isinstance(self.indices, slice)
                        and self.indices == slice(None)
                    )
                ):
                    # Apply indices to cell coordinates (handle list, numpy array, or int)
                    if isinstance(self.indices, (int, np.integer)):
                        coord_data = (
                            [coord_data[self.indices]]
                            if isinstance(coord_data, list)
                            else [coord_data[self.indices]]
                        )
                    else:
                        # Works for both list and numpy array indices
                        coord_data = np.array(coord_data)[
                            self.indices
                        ].tolist()
                coords[dim] = coord_data
            elif dim in ("time",) or dim == "band" or dim.startswith("band ("):
                # Fallback to integer indices (but not for 'cell' or other dimensions)
                coords[dim] = np.arange(data.shape[i])

        # Also add any other coordinates (like lon/lat) that are not dimension coordinates
        # These are non-dimension coordinates that should be included
        for coord_name in zarr_coords:
            if coord_name not in coords:
                coord_info = zarr_coords[coord_name]
                # Handle new format with dtype and dims info
                if isinstance(coord_info, dict) and "data" in coord_info:
                    coord_data = coord_info["data"]
                    coord_dims = tuple(coord_info.get("dims", [coord_name]))
                    coord_attrs = coord_info.get("attrs", {})
                    coord_dtype = coord_info.get("dtype", None)
                    # Convert datetime if needed
                    coord_data = _convert_datetime_coordinate(
                        coord_data, coord_dtype
                    )

                    # Apply subsetting for 1D cell coordinates when we have indices
                    if (
                        len(coord_dims) == 1
                        and coord_dims[0] == "cell"
                        and self.indices is not None
                        and not (
                            isinstance(self.indices, slice)
                            and self.indices == slice(None)
                        )
                    ):
                        if isinstance(self.indices, (int, np.integer)):
                            coord_data = (
                                [coord_data[self.indices]]
                                if isinstance(coord_data, list)
                                else [coord_data[self.indices]]
                            )
                        else:
                            coord_data = np.array(coord_data)[
                                self.indices
                            ].tolist()

                    # Only add if its dimensions are present in this array's dims
                    if all(cd in dims for cd in coord_dims):
                        # Use 2-tuple format (dims, data) - xarray will handle attrs separately
                        coords[coord_name] = (coord_dims, coord_data)
                else:
                    # Old format (just data) - skip for now
                    pass

        # Normalize dimension names if requested (for individual variable access)
        if normalize_dims:
            normalized_dims = []
            normalized_coords = {}
            for i, dim in enumerate(dims):
                # Normalize 'band (...)' to just 'band'
                if dim.startswith("band (") and dim.endswith(")"):
                    normalized_dim = "band"
                    # Use the coordinate from the specific band dimension
                    if dim in coords:
                        normalized_coords[normalized_dim] = coords[dim]
                else:
                    normalized_dim = dim
                    if dim in coords:
                        normalized_coords[normalized_dim] = coords[dim]
                normalized_dims.append(normalized_dim)

            # Also copy non-dimension coordinates (like lon/lat)
            for coord_name, coord_data in coords.items():
                if coord_name not in dims:  # Non-dimension coordinate
                    normalized_coords[coord_name] = coord_data

            dims = tuple(normalized_dims)
            coords = normalized_coords

        da = xr.DataArray(
            data, dims=dims, coords=coords, attrs=attrs, name=array_name
        )

        # Use LPJmLData constructor for proper normalization (only when normalize_dims=True)
        if normalize_dims and LPJmLData is not xr.DataArray:
            try:
                # LPJmLData constructor normalizes 'band (hdate)' -> 'band' automatically
                return LPJmLData(da)
            except:
                # Fallback if constructor fails
                da.__class__ = LPJmLData
                return da
        elif LPJmLData is not xr.DataArray:
            # No normalization, just convert class
            da.__class__ = LPJmLData

        return da

    def isel(self, indexers=None, drop: bool = False, **kwargs):
        """Select by integer position."""
        # Handle both dict and keyword arguments
        if indexers is None:
            indexers = kwargs
        elif isinstance(indexers, dict):
            indexers.update(kwargs)
        else:
            # Handle positional arguments like isel(band=crop_idx)
            indexers = kwargs

        # Handle cell indexing
        if "cell" in indexers:
            cell_indexer = indexers["cell"]

            if isinstance(self.indices, slice) and self.indices == slice(None):
                new_indices = cell_indexer
            elif self._is_scalar:
                raise IndexError("Cannot index a scalar cell view")
            else:
                if isinstance(self.indices, np.ndarray):
                    new_indices = self.indices[cell_indexer]
                else:
                    new_indices = np.arange(
                        self.backend.root.attrs.get("n_cells", 1000)
                    )[self.indices][cell_indexer]

            return ZarrDataArrayView(self.backend, self.path, new_indices)

        # For non-cell dimensions, delegate to xarray
        xarray_obj = self.to_xarray()
        result = xarray_obj.isel(indexers, drop=drop)
        return result

    def __getattr__(self, name: str):
        """Delegate unknown attributes to xarray DataArray."""
        if name.startswith("_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        # Delegate to xarray DataArray for all methods like sum(), mean(), etc.
        xarray_obj = self.to_xarray()
        if hasattr(xarray_obj, name):
            return getattr(xarray_obj, name)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __array__(self, dtype=None):
        """Make this view compatible with numpy operations."""
        return np.array(self.values, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Support numpy ufuncs by converting to xarray."""
        return ufunc(self.to_xarray(), *inputs[1:], **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        """Support numpy array functions by converting to xarray."""
        if func.__module__ == "xarray.core.concat":
            return func(self.to_xarray(), *args[1:], **kwargs)
        return NotImplemented


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
    """

    def __init__(self, backend, group, coord_name, initial_values):
        self._backend = backend
        self._group = group
        self._coord_name = coord_name
        self._initial_values = np.array(initial_values)

    @property
    def values(self):
        """Get coordinate values as a writable numpy array."""
        # Get fresh values from dataset-level attrs
        zarr_group = self._backend.root[self._group]
        coords = zarr_group.attrs.get("coords", {})

        if self._coord_name in coords:
            coord_info = coords[self._coord_name]

            # Check if new format with dtype info
            if isinstance(coord_info, dict) and "data" in coord_info:
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
                    coords[self._coord_name] = {
                        "data": new_values.view("int64").tolist(),
                        "dtype": "datetime64[ns]",
                    }
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
