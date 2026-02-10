"""Output collection and writing system for pycopanlpjml.

This module provides a complete output system for collecting and writing model
outputs in multiple formats (NetCDF, Parquet, CSV). It uses efficient batch
collection with pre-allocated numpy arrays and incremental Zarr storage.

Classes
-------
Output
    Container for output variable metadata definitions.
OutputDefinitionMixin
    Mixin for entity classes to define output variables.
OutputCollectionMixin
    Mixin for component classes to collect outputs efficiently.

Functions
---------
write_outputs_netcdf
    Write cell-based outputs to individual NetCDF files.
write_outputs_parquet
    Write outputs to Parquet format (backward compatible).
write_outputs_csv
    Write outputs to CSV format (backward compatible).
write_outputs_tables
    Write outputs to multiple table formats in a single pass.
collect_variable_metadata_from_model
    Collect variable metadata from all model entity definitions.

The output system handles:
- Efficient batch collection using pre-allocated numpy arrays
- Incremental Zarr storage during simulation (memory-efficient)
- Post-simulation conversion to NetCDF/Parquet/CSV formats
- Variable metadata propagation (units, descriptions)
- Individual-to-cell aggregation for spatial outputs

Output access (output_array, output_table)
-----------------------------------------
Collected outputs can be accessed lazily on World, Region (Country), and Cell.
No extra work during simulation; computed only when the property is accessed.

- ``output_array``: xarray Dataset (raw format) for the last collected year.
- ``output_table``: long-format DataFrame (year, cell, entity, variable, value, unit).

On Region and Cell, both are filtered to that entity's cells.

Example
-------
>>> # Define output variables on an entity class
>>> class MyCell(Cell):
...     output_variables = Output(
...         soilc=Variable("soil carbon", unit=DAU.gC_per_m2),
...         yield_val=Variable("crop yield", unit=DAU.t_per_ha),
...     )
...
>>> # Outputs are collected automatically during simulation
>>> for year in model.lpjml.get_sim_years():
...     model.update(year)
...
>>> # Write outputs to files after simulation
>>> model.finalize_output_streams()
"""

from typing import List, Iterator, Tuple, Any, Optional, Dict, Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import re
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq
from pycoupler.data import LPJmLData

# Suppress Zarr format 3 warnings (safe to use but not yet in official spec)
warnings.filterwarnings(
    "ignore",
    message=r".*vlen-utf8.*is currently not part in the Zarr format 3.*",
    category=UserWarning,
    module="zarr",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Consolidated metadata is currently not part in the Zarr.*",
    category=UserWarning,
    module="zarr",
)

__all__ = [
    "Output",
    "OutputDefinitionMixin",
    "OutputCollectionMixin",
    "dataset_to_output_table",
    "write_outputs_netcdf",
    "write_outputs_parquet",
    "write_outputs_csv",
    "write_outputs_tables",
    "collect_variable_metadata_from_model",
]

# ============================================================================
# Constants
# ============================================================================

OUTPUT_COLUMNS = [
    "year", "cell", "lon", "lat", "country", "area [km2]",
    "class", "variable", "value", "unit",
]

METADATA_DATA_VARS = {
    "cell_lon", "cell_lat", "cell_area_km2", "cell_country",
    "individual_lon", "individual_lat", "individual_area_km2",
    "individual_country",
}

TABLE_WRITE_CHUNK_YEARS = 8  # Years per chunk when materializing tables


# ============================================================================
# Utility Functions
# ============================================================================

def _sanitize_prefix(value: Optional[str], default: str = "outputs") -> str:
    """Sanitize file prefix to create a safe filename.

    Removes special characters and ensures the result is a valid filename
    component.

    Parameters
    ----------
    value : str or None
        The prefix to sanitize.
    default : str, optional
        Default value if input is empty (default: "outputs").

    Returns
    -------
    str
        A sanitized prefix safe for use in filenames.
    """
    candidate = (value or "").strip() or default
    safe = re.sub(r"[^\w\-]+", "_", candidate).strip("_")
    return safe or default


def _normalize_year_array(values: np.ndarray) -> np.ndarray:
    """Convert various time representations to integer year array.

    Handles integer arrays, datetime64 arrays, and string dates.

    Parameters
    ----------
    values : numpy.ndarray
        Array of time values in various formats.

    Returns
    -------
    numpy.ndarray
        Array of years as int64 values.
    """
    if values.size == 0:
        return values
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.int64)
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.DatetimeIndex(values).year.values.astype(np.int64)
    try:
        return pd.to_datetime(values).year.values.astype(np.int64)
    except Exception:
        return values.astype(np.int64, copy=False)


def _normalize_country_value(value: Any) -> Optional[str]:
    """Convert various country code representations to a clean string.

    Handles arrays, bytes, quoted strings, and various other formats
    that country codes might be stored in.

    Parameters
    ----------
    value : Any
        Country code in various formats (string, bytes, array, etc.).

    Returns
    -------
    str or None
        Normalized country code string, or None if not extractable.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return _normalize_country_value(value[0]) if len(value) > 0 else None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        try:
            return _normalize_country_value(value.item())
        except ValueError:
            return _normalize_country_value(value.reshape(-1)[0])
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else str(value)
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return value.decode("utf-8")
        except Exception:
            return str(value)
    
    string_value = str(value).strip()
    # Remove quotes and brackets
    while string_value:
        if string_value.startswith("'") and string_value.endswith("'"):
            string_value = string_value[1:-1].strip()
        elif string_value.startswith('"') and string_value.endswith('"'):
            string_value = string_value[1:-1].strip()
        elif string_value.startswith("[") and string_value.endswith("]"):
            string_value = string_value[1:-1].strip()
        else:
            break
    return string_value or None


def _quote_country_array(values: np.ndarray) -> np.ndarray:
    """Wrap ISO country codes in single quotes for legacy CSV format.

    The legacy inseeds format expects country codes to be quoted to
    preserve them as strings when reading CSV files.

    Parameters
    ----------
    values : numpy.ndarray
        Array of country code strings.

    Returns
    -------
    numpy.ndarray or None
        Array with quoted country codes, or None if input is None.
    """
    if values is None:
        return None
    formatted = np.empty(len(values), dtype=object)
    for i, val in enumerate(values):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            formatted[i] = None
            continue
        val_str = str(val)
        formatted[i] = val_str if val_str.startswith("'") and val_str.endswith("'") else f"'{val_str}'"  # noqa: E501
    return formatted


def _extract_scalar_value(value: Any) -> float:
    """Extract a scalar float from various value types.

    Handles scalars, single-element arrays, and objects with item() method.

    Parameters
    ----------
    value : Any
        Value to extract scalar from.

    Returns
    -------
    float
        The extracted scalar value, or NaN if extraction fails.
    """
    if value is None:
        return np.nan
    if np.isscalar(value):
        return float(value)
    if hasattr(value, "item"):
        return float(value.item())
    if hasattr(value, "__len__") and len(value) == 1:
        return float(value[0])
    return np.nan


# ============================================================================
# Metadata Classes
# ============================================================================

@dataclass
class _CellMetadata:
    """Cell-level metadata for output processing."""
    ids: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    area_km2: np.ndarray
    country: np.ndarray
    country_raw: np.ndarray
    id_to_pos: Dict[int, int]


@dataclass
class _CountryMetadata:
    """Country-level metadata for output processing."""
    countries: List[Any]
    codes: np.ndarray
    names: np.ndarray


@dataclass
class _IndividualMetadata:
    """Individual-level metadata for output processing."""
    ids: np.ndarray
    cell: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    area_km2: np.ndarray
    country: np.ndarray
    classes: np.ndarray


# ============================================================================
# Metadata Preparation
# ============================================================================

def _prepare_cell_metadata(ds: xr.Dataset) -> _CellMetadata:
    """Extract cell-level metadata from dataset coordinates.

    Builds a metadata cache containing cell IDs, coordinates, area,
    and country information from the dataset's coordinates and data
    variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing cell dimension and associated coordinates.

    Returns
    -------
    _CellMetadata
        Dataclass containing all extracted cell metadata.
    """
    n_cells = ds.sizes.get("cell", 0)
    cell_ids = np.asarray(
        ds.coords["cell"].values if "cell" in ds.coords else np.arange(n_cells)
    ).astype(np.int64, copy=False)

    def _coord(name: str, fill_value: Any, dtype: type = float) -> np.ndarray:
        if name in ds.coords:
            return np.asarray(ds.coords[name].values).astype(dtype, copy=False)
        if name in ds.data_vars and "cell" in ds[name].dims:
            return np.asarray(ds[name].values).astype(dtype, copy=False)
        return np.full(n_cells, fill_value, dtype=dtype)

    lon = _coord("cell_lon", np.nan)
    lat = _coord("cell_lat", np.nan)
    area_km2 = _coord("cell_area_km2", np.nan)
    raw_country = _coord("cell_country", None, dtype=object)

    country_raw = np.empty_like(raw_country, dtype=object)
    country = np.empty_like(raw_country, dtype=object)
    for i, val in enumerate(raw_country):
        normalized = _normalize_country_value(val)
        country_raw[i] = normalized
        country[i] = (
            _quote_country_array(np.array([normalized], dtype=object))[0]
            if normalized else None
        )

    return _CellMetadata(
        ids=cell_ids,
        lon=lon,
        lat=lat,
        area_km2=area_km2,
        country=country,
        country_raw=country_raw,
        id_to_pos={int(cell_id): idx for idx, cell_id in enumerate(cell_ids)},  # noqa: E501
    )


def _prepare_individual_metadata(ds: xr.Dataset) -> Optional[_IndividualMetadata]:  # noqa: E501
    """Extract individual-level metadata from dataset coordinates.

    Builds a metadata cache for individual entities including their IDs,
    associated cell indices, coordinates, and class information.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing individual_id dimension and associated coordinates.

    Returns
    -------
    _IndividualMetadata or None
        Dataclass containing all extracted individual metadata, or None if
        no individual_id coordinate is present.
    """
    if "individual_id" not in ds.coords:
        return None

    ids = np.asarray(ds.coords["individual_id"].values)
    n = len(ids)
    if n == 0:
        return None

    def _coord(name: str, fill_value: Any, dtype: type = float) -> np.ndarray:
        if name in ds.coords:
            return np.asarray(ds.coords[name].values).astype(dtype, copy=False)
        if name in ds.data_vars and "individual_id" in ds[name].dims:
            return np.asarray(ds[name].values).astype(dtype, copy=False)
        return np.full(n, fill_value, dtype=dtype)

    return _IndividualMetadata(
        ids=ids.astype(np.int64, copy=False),
        cell=_coord("individual_cell", -1, dtype=np.int64),
        lon=_coord("individual_lon", np.nan),
        lat=_coord("individual_lat", np.nan),
        area_km2=_coord("individual_area_km2", np.nan),
        country=_coord("individual_country", None, dtype=object),
        classes=_coord("individual_class", "Individual", dtype=object),
    )


# ============================================================================
# Variable Metadata Collection
# ============================================================================

def _variable_attrs(entity_or_class: Any, var_name: str) -> Dict[str, str]:
    """Extract xarray-ready attributes from entity's Output definition.

    Retrieves variable metadata (long_name, description, units) from the
    entity's output_variables definition for use as xarray DataArray
    attributes.

    Parameters
    ----------
    entity_or_class : Any
        Entity instance or class with output_variables attribute.
    var_name : str
        Name of the output variable.

    Returns
    -------
    dict
        Dictionary containing 'long_name', optionally 'description'
        and 'units'.
    """
    attrs: Dict[str, str] = {}
    display_name = str(var_name)
    description = ""
    unit_symbol = ""

    entity_cls = (
        entity_or_class if isinstance(entity_or_class, type)
        else getattr(entity_or_class, "__class__", None)
    )
    output_vars = getattr(entity_cls, "output_variables", None)

    if output_vars is not None and hasattr(output_vars, "get"):
        var_obj = output_vars.get(var_name)
        if var_obj is not None:
            if getattr(var_obj, "name", None):
                display_name = str(var_obj.name)
            if getattr(var_obj, "desc", None):
                description = str(var_obj.desc)
            unit_obj = getattr(var_obj, "unit", None)
            if unit_obj is not None:
                if hasattr(unit_obj, "symbol") and unit_obj.symbol:
                    unit_symbol = str(unit_obj.symbol)
                elif hasattr(unit_obj, "name") and unit_obj.name:
                    unit_symbol = str(unit_obj.name)
                else:
                    unit_symbol = str(unit_obj)

    attrs["long_name"] = display_name
    if description:
        attrs["description"] = description
    if unit_symbol:
        attrs["units"] = unit_symbol
    return attrs


def _first_entity(collection: Any) -> Optional[Any]:
    """Return first non-null entity from a collection.

    Parameters
    ----------
    collection : Any
        Collection (list, set, dict) of entities.

    Returns
    -------
    Any or None
        First non-None entity, or None if collection is empty/None.
    """
    if collection is None:
        return None
    iterator = collection.values() if isinstance(collection, dict) else collection  # noqa: E501
    return next((item for item in iterator if item is not None), None)


def _collect_metadata_from_entities(entities: Iterable[Any]) -> Dict[str, Dict[str, str]]:  # noqa: E501
    """Collect variable metadata dictionaries from entity classes.

    Iterates through entities and extracts output variable definitions,
    building a combined metadata dictionary for all unique variables.

    Parameters
    ----------
    entities : Iterable[Any]
        Iterable of entity instances or classes.

    Returns
    -------
    dict
        Mapping of variable names to their attribute dictionaries.
    """
    metadata: Dict[str, Dict[str, str]] = {}
    for entity in entities:
        if entity is None:
            continue
        entity_cls = entity if isinstance(entity, type) else entity.__class__
        output_vars = getattr(entity_cls, "output_variables", None)
        if output_vars is None:
            continue
        for var_name in getattr(output_vars, "names", []):
            if var_name not in metadata:
                attrs = _variable_attrs(entity_cls, var_name)
                if attrs:
                    metadata[var_name] = attrs
    return metadata


def collect_variable_metadata_from_model(model: Any) -> Dict[str, Dict[str, str]]:  # noqa: E501
    """Collect variable metadata from all model entity definitions.

    Gathers output variable metadata from world, countries, regions,
    cells, and individuals in the model.

    Parameters
    ----------
    model : Model
        Model component instance with entity collections.

    Returns
    -------
    dict
        Mapping of variable names to their attribute dictionaries
        (long_name, description, units).
    """
    entities: List[Any] = []
    world = getattr(model, "world", None)
    if world is not None:
        entities.append(world)

    # Model-level collections
    for attr_name in ("countries", "regions", "cells", "individuals", 
                      "households", "farmers", "_farmers"):
        entities.append(_first_entity(getattr(model, attr_name, None)))

    # World-level collections
    if world is not None:
        for attr_name in ("countries", "cells", "individuals"):
            entities.append(_first_entity(getattr(world, attr_name, None)))

    return _collect_metadata_from_entities(entities)


# ============================================================================
# Individual-to-Cell Aggregation
# ============================================================================

def _individual_to_cell_dataarray(
    data: xr.DataArray,
    cell_meta: Optional[_CellMetadata],
    individual_meta: Optional[_IndividualMetadata],
) -> xr.DataArray:
    """Aggregate individual-indexed arrays onto LPJmL cell grid.

    Groups individual values by their associated cell and computes
    cell-level means, producing a cell-indexed DataArray suitable
    for NetCDF output.

    Parameters
    ----------
    data : xarray.DataArray
        Individual-indexed data with 'individual_id' dimension.
    cell_meta : _CellMetadata or None
        Cell metadata for coordinate assignment.
    individual_meta : _IndividualMetadata or None
        Individual metadata containing cell associations.

    Returns
    -------
    xarray.DataArray
        Cell-indexed array with aggregated values.

    Raises
    ------
    RuntimeError
        If aggregation fails due to missing metadata or coordinates.
    """
    if "individual_id" not in data.dims:
        return data

    have_cell_meta = cell_meta is not None and getattr(cell_meta, "ids", np.array([])).size > 0  # noqa: E501
    have_individual_meta = individual_meta is not None and getattr(individual_meta, "ids", np.array([])).size > 0  # noqa: E501

    result_attrs = dict(data.attrs)
    working = data.load()

    if not (have_cell_meta and have_individual_meta):
        return _individual_to_cell_fallback(working, result_attrs)

    # Get cell labels
    if "individual_cell" in working.coords:
        cell_labels = np.asarray(
            working.coords["individual_cell"].values,
            dtype=np.int64
        )
    else:
        cell_labels = np.asarray(individual_meta.cell, dtype=np.int64)
        working = working.assign_coords(
            individual_cell=("individual_id", cell_labels[:working.sizes["individual_id"]]),  # noqa: E501
        )

    valid_idx = np.where(cell_labels >= 0)[0]
    if valid_idx.size == 0:
        has_time = "time" in working.dims
        time_values = (
            np.asarray(working.coords["time"].values) if has_time
            else np.array([], dtype=np.int64)
        )
        empty_shape = (len(cell_meta.ids), time_values.size) if has_time else (len(cell_meta.ids),)  # noqa: E501
        empty = np.full(empty_shape, np.nan, dtype=np.float64)
        coords: Dict[str, Tuple[str, np.ndarray]] = {
            "cell": ("cell", cell_meta.ids),
            "lon": ("cell", cell_meta.lon),
            "lat": ("cell", cell_meta.lat),
        }
        if has_time:
            coords["time"] = ("time", time_values)
        return xr.DataArray(
            empty,
            dims=["cell", "time"] if has_time else ["cell"],
            coords=coords,
            attrs=result_attrs,
            name=data.name,
        )

    if valid_idx.size != working.sizes["individual_id"]:
        working = working.isel(individual_id=valid_idx)
        cell_labels = cell_labels[valid_idx]
        working = working.assign_coords(individual_cell=("individual_id", cell_labels))  # noqa: E501

    # Group by cell and aggregate
    grouped = (
        working.groupby("individual_cell")
        .mean(dim="individual_id", skipna=True, keep_attrs=True)
        .rename({"individual_cell": "cell"})
    )

    grouped = grouped.reindex(cell=cell_meta.ids)
    grouped = grouped.assign_coords(
        cell=("cell", cell_meta.ids),
        lon=("cell", cell_meta.lon),
        lat=("cell", cell_meta.lat),
        area_km2=("cell", cell_meta.area_km2),
        country=("cell", cell_meta.country),
    )

    # Update coordinates attribute
    coord_attr = result_attrs.get("coordinates")
    if coord_attr:
        replacements = {
            "individual_lon": "lon", "individual_lat": "lat",
            "individual_area_km2": "area_km2", "individual_country": "country",
            "individual_cell": "cell",
        }
        for old, new in replacements.items():
            coord_attr = coord_attr.replace(old, new)
        result_attrs["coordinates"] = coord_attr

    grouped.attrs.update(result_attrs)
    return grouped


def _individual_to_cell_fallback(
    data: xr.DataArray, attrs: Dict[str, Any]
) -> xr.DataArray:
    """Fallback aggregation using only lon/lat coordinates.

    Used when proper cell metadata is not available. Groups individuals
    by their rounded lon/lat coordinates and computes cell-level means.

    Parameters
    ----------
    data : xarray.DataArray
        Individual-indexed data.
    attrs : dict
        Attributes to preserve on the output array.

    Returns
    -------
    xarray.DataArray
        Cell-indexed array with aggregated values.

    Raises
    ------
    RuntimeError
        If neither cell metadata nor lon/lat coordinates are available.
    """
    lon_coord = data.coords.get("individual_lon")
    lat_coord = data.coords.get("individual_lat")
    if lon_coord is None or lat_coord is None:
        raise RuntimeError(
            "Cannot aggregate individual outputs without cell metadata or "
            "individual lon/lat coordinates."
        )

    lon = np.asarray(lon_coord.values, dtype=np.float64)
    lat = np.asarray(lat_coord.values, dtype=np.float64)
    values = data.values
    if values.ndim == 1:
        values = values[:, np.newaxis]

    time_vals = (
        np.asarray(data.coords["time"].values) if "time" in data.dims
        else np.array([], dtype=np.int64)
    )

    # Group by unique (lat, lon) pairs
    key = np.round(np.column_stack([lat, lon]), decimals=6)
    unique_coords, inverse = np.unique(key, axis=0, return_inverse=True)
    n_cells = unique_coords.shape[0]
    n_time = values.shape[1]

    grid = np.full((n_cells, n_time), np.nan, dtype=np.float64)
    for idx in range(n_cells):
        mask = inverse == idx
        if np.any(mask):
            grid[idx] = np.nanmean(values[mask], axis=0)

    # Sort by lat (descending), then lon
    order = np.lexsort((-unique_coords[:, 0], unique_coords[:, 1]))
    grid = grid[order]
    unique_coords = unique_coords[order]

    coords: Dict[str, Tuple[str, np.ndarray]] = {
        "cell": ("cell", np.arange(n_cells, dtype=np.int64)),
        "lat": ("cell", unique_coords[:, 0]),
        "lon": ("cell", unique_coords[:, 1]),
    }
    if time_vals.size:
        coords["time"] = ("time", time_vals)

    return xr.DataArray(
        grid,
        dims=["cell", "time"] if time_vals.size else ["cell"],
        coords=coords,
        attrs=attrs,
        name=data.name,
    )


# ============================================================================
# Table Writers (CSV/Parquet)
# ============================================================================

class _OutputTableWriter:
    """Incremental writer for CSV/Parquet output tables."""

    def __init__(self, output_path: str, file_format: str):
        self.output_path = output_path
        self.file_format = file_format.lower()
        self.file_name = os.path.join(
            output_path,
            "inseeds_data.parquet" if self.file_format == "parquet" else "inseeds_data.csv",  # noqa: E501
        )
        os.makedirs(output_path, exist_ok=True)

        if os.path.exists(self.file_name):
            os.remove(self.file_name)

        self._header_written = False
        self._parquet_writer = None
        self._wrote_data = False

    def write(self, df: pd.DataFrame) -> None:
        """Write DataFrame to file."""
        if df is None or df.empty:
            return

        # Ensure consistent column order
        missing_cols = [col for col in OUTPUT_COLUMNS if col not in df.columns]
        for col in missing_cols:
            df[col] = None if col in {"class", "country", "variable", "unit"} else np.nan  # noqa: E501
        df = df[OUTPUT_COLUMNS]

        # Convert numeric columns
        for col in ["cell", "lon", "lat", "area [km2]"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        if self.file_format == "csv":
            df.to_csv(
                self.file_name, mode="a", header=not self._header_written, index=False  # noqa: E501
            )
            self._header_written = True
        elif self.file_format == "parquet":
            table = pa.Table.from_pandas(df, preserve_index=False)
            if self._parquet_writer is None:
                self._parquet_writer = pq.ParquetWriter(self.file_name, table.schema)  # noqa: E501
            self._parquet_writer.write_table(table)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
        self._wrote_data = True

    def close(self) -> None:
        """Close writer and create empty file if no data written."""
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
        if not self._wrote_data:
            empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
            if self.file_format == "csv":
                empty.to_csv(self.file_name, index=False)
            elif self.file_format == "parquet":
                pq.write_table(pa.Table.from_pandas(empty, preserve_index=False), self.file_name)  # noqa: E501
            self._wrote_data = True


class _TableExportManager:
    """Manages streaming table outputs during simulation."""

    def __init__(
        self,
        output_path: str,
        formats: List[str],
        metadata_lookup: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        self.output_path = output_path
        self.formats = formats
        self.metadata_lookup = dict(metadata_lookup or {})
        self._writers = {
            fmt: _OutputTableWriter(output_path, fmt) for fmt in formats
        }
        self._cell_meta: Optional[_CellMetadata] = None
        self._individual_meta: Optional[_IndividualMetadata] = None
        self._finalized = False
        self._result_paths: Dict[str, str] = {}

    def _ensure_metadata(self, ds: xr.Dataset) -> None:
        """Lazy-load metadata from dataset."""
        if self._cell_meta is None:
            self._cell_meta = _prepare_cell_metadata(ds)
        if self._individual_meta is None:
            self._individual_meta = _prepare_individual_metadata(ds)

    def write_year(self, ds: xr.Dataset, year: int) -> None:
        """Write single-year dataset to all configured table sinks."""
        if self._finalized:
            return

        self._ensure_metadata(ds)
        ds_to_use = ds.drop_vars(
            [var for var in METADATA_DATA_VARS if var in ds.data_vars]
        )

        years = (
            _normalize_year_array(np.asarray(ds_to_use.time.values))
            if "time" in ds_to_use.coords
            else np.array([year], dtype=np.int64)
        )

        # Merge metadata
        metadata_lookup = dict(self.metadata_lookup)
        ds_meta = ds_to_use.attrs.get("variable_metadata")
        if isinstance(ds_meta, dict):
            metadata_lookup.update(ds_meta)

        # Process each variable
        for var_name, var_data in ds_to_use.data_vars.items():
            if var_name in ds_to_use.coords:
                continue

            var_attrs = getattr(var_data, "attrs", {})
            var_meta = metadata_lookup.get(var_name, {})
            var_unit = var_attrs.get("units") or var_meta.get("units", "")
            var_display_name = var_attrs.get("long_name") or var_meta.get("long_name", var_name)  # noqa: E501

            dims = set(var_data.dims)
            if "individual_id" in dims:
                df = _build_individual_dataframe(
                    var_data,
                    years,
                    var_display_name,
                    var_unit,
                    self._individual_meta
                )
            elif "cell" in dims:
                df = _build_cell_dataframe(
                    var_data,
                    years,
                    var_display_name,
                    var_unit,
                    self._cell_meta
                )
            elif "country" in dims:
                df = _build_country_dataframe(
                    var_data,
                    years,
                    var_display_name,
                    var_unit,
                    self._cell_meta
                )
            else:
                df = _build_world_dataframe(
                    var_data,
                    years,
                    var_display_name,
                    var_unit
                )

            if df is None or df.empty:
                continue
            for writer in self._writers.values():
                writer.write(df)

    def finalize(self) -> Dict[str, str]:
        """Close all writers and return file paths."""
        if self._finalized:
            return dict(self._result_paths)
        for fmt, writer in self._writers.items():
            writer.close()
            self._result_paths[fmt] = writer.file_name
        self._finalized = True
        return dict(self._result_paths)


# ============================================================================
# Dataset to Table Conversion
# ============================================================================


def dataset_to_output_table(
    ds: xr.Dataset,
    *,
    variable_metadata: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    """Convert output Dataset to long-format DataFrame (output table).

    Converts the xarray Dataset produced by collect_outputs into the legacy
    output table format: year, cell, lon, lat, country, area [km2], class,
    variable, value, unit.

    Parameters
    ----------
    ds : xarray.Dataset
        Output dataset from collect_outputs (with time, cell, individual_id, etc.).
    variable_metadata : dict, optional
        Additional variable metadata (long_name, units) to merge.

    Returns
    -------
    pandas.DataFrame
        Long-format table; empty if ds has no data vars.
    """
    if ds is None or not ds.data_vars:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    ds_to_use = ds.drop_vars(
        [var for var in METADATA_DATA_VARS if var in ds.data_vars]
    )

    years = (
        _normalize_year_array(np.asarray(ds_to_use.time.values))
        if "time" in ds_to_use.coords
        else np.array([0], dtype=np.int64)
    )

    metadata_lookup: Dict[str, Dict[str, str]] = {}
    if isinstance(ds_to_use.attrs.get("variable_metadata"), dict):
        metadata_lookup.update(ds_to_use.attrs["variable_metadata"])
    if variable_metadata:
        metadata_lookup.update(variable_metadata)

    cell_meta = _prepare_cell_metadata(ds_to_use)
    individual_meta = _prepare_individual_metadata(ds_to_use)

    dfs: List[pd.DataFrame] = []
    for var_name, var_data in ds_to_use.data_vars.items():
        if var_name in ds_to_use.coords:
            continue

        var_attrs = getattr(var_data, "attrs", {})
        var_meta = metadata_lookup.get(var_name, {})
        var_unit = var_attrs.get("units") or var_meta.get("units", "")
        var_display_name = var_attrs.get("long_name") or var_meta.get(
            "long_name", var_name
        )

        dims = set(var_data.dims)
        if "individual_id" in dims:
            df = _build_individual_dataframe(
                var_data,
                years,
                var_display_name,
                var_unit,
                individual_meta,
            )
        elif "cell" in dims:
            df = _build_cell_dataframe(
                var_data,
                years,
                var_display_name,
                var_unit,
                cell_meta,
            )
        elif "country" in dims:
            df = _build_country_dataframe(
                var_data,
                years,
                var_display_name,
                var_unit,
                cell_meta,
            )
        else:
            df = _build_world_dataframe(
                var_data,
                years,
                var_display_name,
                var_unit,
            )

        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    result = pd.concat(dfs, ignore_index=True)
    # Alias "class" -> "entity" for backward compatibility with inseeds
    if "class" in result.columns:
        result = result.rename(columns={"class": "entity"})
    return result


def read_output_table_from_zarr(
    store_path: str,
    *,
    variable_metadata: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    """Read full output table from Zarr store.

    Parameters
    ----------
    store_path : str
        Path to Zarr store (with model_outputs group).
    variable_metadata : dict, optional
        Additional variable metadata to merge.

    Returns
    -------
    pandas.DataFrame
        Long-format output table (all years).
    """
    try:
        ds = xr.open_zarr(
            store_path,
            group="model_outputs",
            consolidated=False,
        )
    except Exception:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if not ds.data_vars:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return dataset_to_output_table(ds, variable_metadata=variable_metadata)


# ============================================================================
# DataFrame Builders
# ============================================================================

def _entity_array_and_years(
    var_data: xr.DataArray, entity_dim: Optional[str], fallback_years: np.ndarray  # noqa: E501
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Extract 2D array (entity x time) and year vector from DataArray.

    Reshapes the input DataArray to a 2D matrix suitable for DataFrame
    construction, with entities along axis 0 and time along axis 1.

    Parameters
    ----------
    var_data : xarray.DataArray
        Input data array with entity and/or time dimensions.
    entity_dim : str or None
        Name of the entity dimension (e.g., 'cell', 'individual_id'),
        or None for scalar/world-level data.
    fallback_years : numpy.ndarray
        Years to use if time dimension is missing.

    Returns
    -------
    tuple
        (values, years) where values is a 2D array (n_entities, n_years)
        and years is a 1D array of integer years.
    """
    data = var_data
    years = fallback_years

    if "time" not in data.dims:
        data = data.expand_dims({"time": fallback_years})
        data = data.assign_coords(time=("time", fallback_years))

    if "time" in data.coords:
        years = _normalize_year_array(np.asarray(data.coords["time"].values))

    if entity_dim is None:
        arr = np.asarray(data.transpose("time").values)
        arr = arr[np.newaxis, :] if arr.ndim == 1 else arr.reshape(1, -1)
        return arr.astype(np.float64, copy=False), years

    if entity_dim not in data.dims:
        return None, years

    extra_dims = [dim for dim in data.dims if dim not in (entity_dim, "time")]
    if extra_dims:
        data = data.transpose(entity_dim, "time", *extra_dims)
    else:
        data = data.transpose(entity_dim, "time")

    arr = np.asarray(data.values)
    arr = arr[:, np.newaxis] if arr.ndim == 1 else arr.reshape(arr.shape[0], -1)  # noqa: E501
    return arr.astype(np.float64, copy=False), years


def _build_dataframe_base(
    matrix: np.ndarray,
    years: np.ndarray,
    var_display_name: str,
    var_unit: str,
    entity_class: str,
    cell: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None,
    lat: Optional[np.ndarray] = None,
    area: Optional[np.ndarray] = None,
    country: Optional[np.ndarray] = None,
) -> Optional[pd.DataFrame]:
    """Base function for building output DataFrames in legacy format.

    Constructs a DataFrame with the standard output table columns from
    the provided arrays. Filters out NaN values automatically.

    Parameters
    ----------
    matrix : numpy.ndarray
        2D value matrix (n_entities, n_years).
    years : numpy.ndarray
        1D array of years.
    var_display_name : str
        Human-readable variable name.
    var_unit : str
        Unit string for the variable.
    entity_class : str
        Entity class name (e.g., 'Cell', 'Country', 'Farmer').
    cell : numpy.ndarray, optional
        Cell indices (expanded to match matrix shape).
    lon : numpy.ndarray, optional
        Longitudes (expanded to match matrix shape).
    lat : numpy.ndarray, optional
        Latitudes (expanded to match matrix shape).
    area : numpy.ndarray, optional
        Areas in km² (expanded to match matrix shape).
    country : numpy.ndarray, optional
        Country codes (expanded to match matrix shape).

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with standard output columns, or None if no valid data.
    """
    if matrix is None or matrix.size == 0:
        return None

    n_entities, n_time = matrix.shape
    values = matrix.reshape(-1)
    mask = np.isfinite(values)
    if not mask.any():
        return None

    # Build column arrays
    year_col = np.tile(years[:n_time], n_entities)
    class_col = np.full(year_col.size, entity_class, dtype=object)
    
    df_dict = {
        "year": year_col[mask].astype(np.int64),
        "cell": cell[mask] if cell is not None else None,
        "lon": lon[mask] if lon is not None else None,
        "lat": lat[mask] if lat is not None else None,
        "country": country[mask] if country is not None else None,
        "area [km2]": area[mask] if area is not None else None,
        "class": class_col[mask],
        "variable": var_display_name,
        "value": values[mask],
        "unit": var_unit,
    }
    
    return pd.DataFrame(df_dict)


def _build_cell_dataframe(
    var_data: xr.DataArray,
    fallback_years: np.ndarray,
    var_display_name: str,
    var_unit: str,
    cell_meta: _CellMetadata,
) -> Optional[pd.DataFrame]:
    """Build DataFrame for cell-level variables.

    Parameters
    ----------
    var_data : xarray.DataArray
        Cell-indexed variable data.
    fallback_years : numpy.ndarray
        Years to use if time dimension is missing.
    var_display_name : str
        Human-readable variable name.
    var_unit : str
        Unit string for the variable.
    cell_meta : _CellMetadata
        Cell metadata for coordinates.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with cell-level data, or None if no valid data.
    """
    matrix, years = _entity_array_and_years(var_data, "cell", fallback_years)
    if matrix is None or matrix.size == 0:
        return None

    n_cells = min(matrix.shape[0], len(cell_meta.ids))
    matrix = matrix[:n_cells]
    n_time = matrix.shape[1]

    return _build_dataframe_base(
        matrix, years, var_display_name, var_unit, "Cell",
        cell=np.repeat(cell_meta.ids[:n_cells], n_time),
        lon=np.repeat(cell_meta.lon[:n_cells], n_time),
        lat=np.repeat(cell_meta.lat[:n_cells], n_time),
        area=np.round(np.repeat(cell_meta.area_km2[:n_cells], n_time), 4),
        country=np.repeat(cell_meta.country[:n_cells], n_time),
    )


def _build_individual_dataframe(
    var_data: xr.DataArray,
    fallback_years: np.ndarray,
    var_display_name: str,
    var_unit: str,
    individual_meta: Optional[_IndividualMetadata],
) -> Optional[pd.DataFrame]:
    """Build DataFrame for individual-level variables.

    Parameters
    ----------
    var_data : xarray.DataArray
        Individual-indexed variable data.
    fallback_years : numpy.ndarray
        Years to use if time dimension is missing.
    var_display_name : str
        Human-readable variable name.
    var_unit : str
        Unit string for the variable.
    individual_meta : _IndividualMetadata or None
        Individual metadata for coordinates.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with individual-level data, or None if metadata missing.
    """
    if individual_meta is None:
        return None

    matrix, years = _entity_array_and_years(
        var_data,
        "individual_id",
        fallback_years
    )
    if matrix is None or matrix.size == 0:
        return None

    n_individuals = min(matrix.shape[0], len(individual_meta.ids))
    matrix = matrix[:n_individuals]
    n_time = matrix.shape[1]

    # Get class name (use first individual's class or fallback)
    class_name = (
        individual_meta.classes[0] if len(individual_meta.classes) > 0 else "Individual"  # noqa: E501
    )
    
    return _build_dataframe_base(
        matrix, years, var_display_name, var_unit, class_name,
        cell=np.repeat(individual_meta.cell[:n_individuals], n_time),
        lon=np.repeat(individual_meta.lon[:n_individuals], n_time),
        lat=np.repeat(individual_meta.lat[:n_individuals], n_time),
        area=np.round(
            np.repeat(individual_meta.area_km2[:n_individuals], n_time),
            4
        ),
        country=np.repeat(individual_meta.country[:n_individuals], n_time),
    )


def _build_country_dataframe(
    var_data: xr.DataArray,
    fallback_years: np.ndarray,
    var_display_name: str,
    var_unit: str,
    cell_meta: _CellMetadata,
) -> Optional[pd.DataFrame]:
    """Build DataFrame for country-level variables.

    Expands country-level values to all cells within each country,
    maintaining compatibility with the cell-based output format.

    Parameters
    ----------
    var_data : xarray.DataArray
        Country-indexed variable data.
    fallback_years : numpy.ndarray
        Years to use if time dimension is missing.
    var_display_name : str
        Human-readable variable name.
    var_unit : str
        Unit string for the variable.
    cell_meta : _CellMetadata
        Cell metadata for mapping countries to cells.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with country data expanded to cells.
    """
    if "country" not in var_data.dims or len(cell_meta.ids) == 0:
        return None

    matrix, years = _entity_array_and_years(
        var_data,
        "country",
        fallback_years
    )
    if matrix is None or matrix.size == 0:
        return None

    country_codes = np.asarray(var_data.coords["country"].values, dtype=object)
    n_countries = min(matrix.shape[0], len(country_codes))
    matrix = matrix[:n_countries]
    n_time = matrix.shape[1]

    frames = []
    for idx in range(n_countries):
        code = _normalize_country_value(country_codes[idx])
        if code is None:
            continue
        values = matrix[idx]
        if np.all(np.isnan(values)):
            continue

        cell_mask = cell_meta.country_raw == code
        cell_indices = np.where(cell_mask)[0]
        if cell_indices.size == 0:
            continue

        value_matrix = np.broadcast_to(values, (cell_indices.size, n_time))
        flat_values = value_matrix.reshape(-1)
        mask = np.isfinite(flat_values)
        if not mask.any():
            continue

        year_col = np.tile(years[:n_time], cell_indices.size)
        cell_col = np.repeat(cell_meta.ids[cell_indices], n_time)
        lon_col = np.repeat(cell_meta.lon[cell_indices], n_time)
        lat_col = np.repeat(cell_meta.lat[cell_indices], n_time)
        area_col = np.repeat(cell_meta.area_km2[cell_indices], n_time)
        country_col = np.repeat(cell_meta.country[cell_indices], n_time)
        
        frames.append(_build_dataframe_base(
            value_matrix, years, var_display_name, var_unit, "Country",
            cell=cell_col, lon=lon_col, lat=lat_col,
            area=np.round(area_col, 4), country=country_col,
        ))

    return pd.concat(frames, ignore_index=True) if frames else None


def _build_world_dataframe(
    var_data: xr.DataArray,
    fallback_years: np.ndarray,
    var_display_name: str,
    var_unit: str,
) -> Optional[pd.DataFrame]:
    """Build DataFrame for world-level (scalar) variables.

    Creates a DataFrame with world-level aggregated values, with None
    values for cell-specific fields.

    Parameters
    ----------
    var_data : xarray.DataArray
        Scalar or time-indexed variable data.
    fallback_years : numpy.ndarray
        Years to use if time dimension is missing.
    var_display_name : str
        Human-readable variable name.
    var_unit : str
        Unit string for the variable.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with world-level data, or None if no valid data.
    """
    matrix, years = _entity_array_and_years(var_data, None, fallback_years)
    if matrix is None or matrix.size == 0:
        return None

    values = matrix.reshape(-1)
    mask = np.isfinite(values)
    if not mask.any():
        return None

    year_col = years[:matrix.shape[1]]
    none_array = np.full(year_col.size, None, dtype=object)
    
    return pd.DataFrame({
        "year": year_col[mask].astype(np.int64),
        "cell": none_array[mask],
        "lon": none_array[mask],
        "lat": none_array[mask],
        "country": none_array[mask],
        "area [km2]": none_array[mask],
        "class": np.full(np.count_nonzero(mask), "World", dtype=object),
        "variable": var_display_name,
        "value": values[mask],
        "unit": var_unit,
    })


# ============================================================================
# Output Classes
# ============================================================================

class Output:
    """Container for output variable metadata.

    Stores Variable objects as attributes, providing a simple way to define
    which attributes of an entity should be written to output.

    Examples
    --------
    >>> from pycopancore.data_model.variable import Variable
    >>> output_vars = Output(
    ...     soilc=Variable("soil carbon", "description", unit=DAU.gC_per_m2),
    ...     yield_var=Variable("crop yield", "description")
    ... )
    >>> output_vars.names
    ['soilc', 'yield_var']
    """

    def __init__(self, **kwargs):
        """Initialize with variable objects."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._invalidate_names_cache()

    @property
    def names(self) -> List[str]:
        """Return list of output variable names (excluding private
        attributes).
        """
        if not hasattr(self, '_cached_names'):
            self._cached_names = [
                k for k in self.__dict__.keys() if not k.startswith("_")
            ]
        return self._cached_names

    def _invalidate_names_cache(self):
        """Invalidate cached names list."""
        if hasattr(self, '_cached_names'):
            delattr(self, '_cached_names')

    def get(self, name: str, default=None):
        """Get variable object by name."""
        return getattr(self, name, default)

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Iterate over (name, variable) pairs."""
        for name in self.names:
            yield name, getattr(self, name)

    def __repr__(self) -> str:
        """String representation showing variable names."""
        return f"Output({', '.join(self.names)})"


class OutputDefinitionMixin:
    """Mixin for entity classes to define output variables.

    Entity classes (World, Cell, Region, Individual) inherit from this mixin
    and define output_variables to specify which attributes should be written.

    Required implementations:
    - output_variables: Output class attribute (models override this)
    - model: property returning Component model instance
    - get_defined_outputs(): method returning list of output variable names
    """

    output_variables = Output()

    @property
    def model(self):
        """Reference to Component model instance (subclasses must
        implement).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement 'model' property"
        )

    def get_defined_outputs(self) -> List[str]:
        """Get list of output variable names filtered by config.

        Reads from model.config.coupled_config.output using the entity type
        name (lowercased) as the config key.

        Returns
        -------
        List[str]
            List of variable names to output, empty if not configured.
        """
        if not hasattr(self, "model") or self.model is None:
            return []
        if not hasattr(self.model, "config"):
            return []

        try:
            entity_type = self.__class__.__name__.lower()
            output_dict = self.model.config.coupled_config.output.to_dict()
            # Try entity_type first, then check for aliases
            config_outputs = output_dict.get(entity_type, [])
            # Support backward compatibility: 'farmer' is alias for
            # 'individual'
            if not config_outputs and entity_type == "farmer":
                config_outputs = output_dict.get("individual", [])
            # Support backward compatibility: 'individual' can be used for
            # 'farmer'
            if not config_outputs and entity_type == "individual":
                config_outputs = output_dict.get("farmer", [])
            return [
                var for var in self.__class__.output_variables.names
                if var in config_outputs
            ]
        except Exception:
            return []


class OutputCollectionMixin:
    """Mixin for Component classes to efficiently collect outputs.

    Provides optimized, vectorized methods to collect outputs from all entity
    levels (world, country, cell, individual) and store them in xarray
    Datasets with Zarr backend.

    Performance optimizations:
    - Pre-allocated numpy arrays for batch collection
    - Vectorized operations where possible
    - Cached metadata to avoid recomputation
    - Single-pass collection loops

    Required attributes:
    - world: World instance with entities
    - config: Configuration object
    - pycopanlpjml_config: pycopanlpjml configuration
    """

    def collect_outputs(self, t: int) -> None:
        """Collect outputs from all entity levels for year t.

        Collects outputs from world, countries, cells, individuals, structures
        them into xarray Dataset, stores in world._output_data, and appends to
        Zarr store for historical data.

        Parameters
        ----------
        t : int
            Current simulation year
        """
        if not self._should_collect_outputs(t):
            return

        self._ensure_output_metadata()

        # Collect from each entity level
        datasets = []
        for collector in [
            self._collect_world_outputs,
            self._collect_country_outputs,
            self._collect_cell_outputs,
            self._collect_individual_outputs,
        ]:
            ds = collector(t)
            if ds is not None and len(ds.data_vars) > 0:
                datasets.append(ds)

        if not datasets:
            return

        # Merge and add time coordinate
        combined_ds = xr.merge(datasets).assign_coords(time=[t])

        # Attach variable metadata
        metadata_cache = getattr(self, "_variable_metadata_cache", None)
        if metadata_cache is None:
            try:
                metadata_cache = collect_variable_metadata_from_model(self)
            except Exception:
                metadata_cache = {}
            self._variable_metadata_cache = metadata_cache

        if metadata_cache:
            for var_name, attrs in metadata_cache.items():
                if var_name not in combined_ds:
                    continue
                current_attrs = dict(combined_ds[var_name].attrs)
                missing = {
                    k: v for k, v in attrs.items()
                    if k not in current_attrs or not current_attrs[k]
                }
                if missing:
                    combined_ds[var_name] = combined_ds[var_name].assign_attrs(
                        {**current_attrs, **missing}
                    )
            combined_ds.attrs["variable_metadata"] = metadata_cache

        # Store in world
        self.world._output_data = combined_ds

        # Stream to table writers if configured
        exporter = self._maybe_init_table_exporter()
        if exporter is not None:
            exporter.write_year(combined_ds, t)

        # Initialize and append to Zarr store
        if not hasattr(self, "_output_store_initialized"):
            self._output_store_initialized = False

        if not self._output_store_initialized:
            try:
                self._init_output_store()
                self._output_store_initialized = True
            except RuntimeError:
                pass  # World not yet created

        if (
            self._output_store_initialized
            and hasattr(self.world, "_output_store_path")
            and self.world._output_store_path
        ):
            self._append_to_zarr(combined_ds, t)
            # Keep _output_data so world.output_array / output_table remain available

    def _should_collect_outputs(self, t: int) -> bool:
        """Check if outputs should be collected for this year."""
        if not hasattr(self, "config") or self.config is None:
            return False

        def _formats_from_config(cfg) -> Optional[List[str]]:
            if cfg is None:
                return None
            if isinstance(cfg, Mapping):
                formats = cfg.get("format") or cfg.get("output_formats")
                return list(formats) if formats else None
            return (
                getattr(cfg, "format", None)
                or getattr(cfg, "output_formats", None)
            )

        try:
            # Check pycopanlpjml config - if format is empty, no outputs
            if hasattr(self, "pycopanlpjml_config"):
                output_config = getattr(
                    self.pycopanlpjml_config,
                    "output",
                    None
                )
                if output_config is not None:
                    enabled = getattr(output_config, "enabled", None)
                    if isinstance(output_config, Mapping):
                        if enabled is None:
                            enabled = output_config.get("enabled")
                    if enabled is False:
                        return False
                    formats = _formats_from_config(output_config) or []
                    if not formats:
                        return False

            # Check if any outputs defined
            coupled_output = getattr(
                getattr(self.config, "coupled_config", None),
                "output",
                None
            )
            if coupled_output is None:
                return False
            if isinstance(coupled_output, Mapping):
                return len(coupled_output) > 0
            if hasattr(coupled_output, "to_dict"):
                output_dict = coupled_output.to_dict()
                return output_dict is not None and len(output_dict) > 0
            return bool(coupled_output)
        except Exception:
            return False

    def _ensure_output_metadata(self) -> None:
        """Build static metadata once (cached for performance)."""
        if not hasattr(self, "_cell_metadata_ready"):
            self._cell_metadata_ready = False
        if not hasattr(self, "_country_metadata_ready"):
            self._country_metadata_ready = False
        if not hasattr(self, "_individual_metadata_ready"):
            self._individual_metadata_ready = False

        if not self._cell_metadata_ready:
            self._cell_metadata_cache = self._build_cell_metadata_cache()
            self._cell_metadata_ready = True

        if not self._country_metadata_ready:
            self._country_metadata_cache = self._build_country_metadata_cache()
            self._country_metadata_ready = True

        if not self._individual_metadata_ready:
            self._individual_metadata_cache = self._build_individual_metadata_cache()  # noqa: E501
            self._individual_metadata_ready = True

    def _table_output_formats(self) -> List[str]:
        """Get list of table output formats from config."""
        def _formats_from_config(cfg) -> Optional[List[str]]:
            if cfg is None:
                return None
            if isinstance(cfg, Mapping):
                formats = cfg.get("format") or cfg.get("output_formats")
                return list(formats) if formats else None
            return (
                getattr(cfg, "format", None)
                or getattr(cfg, "output_formats", None)
            )

        formats: Optional[List[str]] = None

        if hasattr(self, "config") and hasattr(self.config, "coupled_config"):
            config_output = getattr(self.config.coupled_config, "output", None)
            formats = _formats_from_config(config_output)

        if not formats and hasattr(self, "pycopanlpjml_config"):
            output_config = getattr(self.pycopanlpjml_config, "output", None)
            formats = _formats_from_config(output_config)

        if not formats:
            return []

        result: List[str] = []
        for fmt in formats:
            fmt_lower = str(fmt).lower()
            if fmt_lower in {"csv", "parquet"} and fmt_lower not in result:
                result.append(fmt_lower)
        return result

    def _resolve_table_output_dir(self) -> Optional[str]:
        """Resolve table output directory from config."""
        if not hasattr(self, "config"):
            return None
        sim_path = getattr(self.config, "sim_path", None)
        if not sim_path:
            return None
        sim_name = getattr(self.config, "sim_name", None) or "coupled_run"
        output_dir = os.path.join(sim_path, "output", sim_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _maybe_init_table_exporter(self) -> Optional[_TableExportManager]:
        """Lazy-initialize table exporter if formats configured."""
        if not hasattr(self, "_table_export_manager"):
            self._table_export_manager = None

        if self._table_export_manager is not None:
            return self._table_export_manager

        formats = self._table_output_formats()
        if not formats:
            return None

        output_dir = self._resolve_table_output_dir()
        if output_dir is None:
            return None

        metadata_cache = getattr(self, "_variable_metadata_cache", {}) or {}
        exporter = _TableExportManager(output_dir, formats, metadata_cache)
        self._table_export_manager = exporter

        if self.world is not None:
            self.world._table_export_manager = exporter
            self.world._table_export_formats = formats

        return exporter

    def _build_cell_metadata_cache(self) -> Optional[_CellMetadata]:
        """Build cell metadata cache (called once per simulation).

        Uses vectorized operations on world-level arrays for performance.
        Avoids per-cell xarray isel calls which are extremely slow for
        large grids (~67k cells would require ~268k isel calls).
        """
        world = getattr(self, "world", None)
        cells = list(world.cells) if world is not None and hasattr(world, "cells") else []  # noqa: E501
        self._cell_entities = cells

        if not cells:
            self._cell_output_vars = []
            return None

        output_vars = cells[0].get_defined_outputs()
        self._cell_output_vars = output_vars or []

        n_cells = len(cells)

        # Build cell index array from cell objects (fast: just attribute access)
        cell_indices = np.array(
            [getattr(c, "_cell_index", i) for i, c in enumerate(cells)],
            dtype=np.int64
        )
        cell_ids = cell_indices.copy()

        # Initialize output arrays
        lon = np.full(n_cells, np.nan, dtype=np.float64)
        lat = np.full(n_cells, np.nan, dtype=np.float64)
        area = np.full(n_cells, np.nan, dtype=np.float64)
        country_raw = np.empty(n_cells, dtype=object)

        # Extract metadata from world-level arrays VECTORIZED (single array access)
        if world is not None:
            grid = getattr(world, "grid", None)
            if grid is not None:
                # Get full lon/lat arrays once, then index (flatten to ensure 1D)
                if hasattr(grid, "lon"):
                    try:
                        full_lon = np.asarray(grid.lon.values).flatten()
                        lon = full_lon[cell_indices].astype(np.float64)
                    except Exception:
                        pass
                if hasattr(grid, "lat"):
                    try:
                        full_lat = np.asarray(grid.lat.values).flatten()
                        lat = full_lat[cell_indices].astype(np.float64)
                    except Exception:
                        pass

            # Get area from world-level array
            area_data = getattr(world, "area", None)
            if area_data is not None:
                try:
                    full_area = np.asarray(area_data.values).flatten()
                    # Convert m² to km²
                    area = full_area[cell_indices].astype(np.float64) * 1e-6
                except Exception:
                    pass

            # Get country codes from world-level array
            country_data = getattr(world, "country_code", None)
            if country_data is not None:
                try:
                    full_country = np.asarray(country_data.values).flatten()
                    raw_codes = full_country[cell_indices]
                    # Vectorized normalization
                    for i, code in enumerate(raw_codes):
                        country_raw[i] = _normalize_country_value(code)
                except Exception:
                    # Fallback to per-cell extraction
                    for i, cell in enumerate(cells):
                        code = getattr(cell, "country_code", None)
                        country_raw[i] = _normalize_country_value(code)
            else:
                # Fallback: get from cell attributes
                for i, cell in enumerate(cells):
                    code = getattr(cell, "country_code", None)
                    country_raw[i] = _normalize_country_value(code)

        # Vectorized country quoting
        country = _quote_country_array(country_raw)

        return _CellMetadata(
            ids=cell_ids,
            lon=lon,
            lat=lat,
            area_km2=area,
            country=country.astype(object),
            country_raw=country_raw,
            id_to_pos={int(cid): pos for pos, cid in enumerate(cell_ids)},
        )

    def _build_country_metadata_cache(self) -> Optional[_CountryMetadata]:
        """Build country metadata cache (called once per simulation)."""
        world = getattr(self, "world", None)
        countries = list(world.countries) if world is not None and hasattr(world, "countries") else []  # noqa: E501
        self._country_entities = countries

        if not countries:
            self._country_output_vars = []
            return None

        output_vars = countries[0].get_defined_outputs()
        self._country_output_vars = output_vars or []

        n_countries = len(countries)
        codes = np.empty(n_countries, dtype=object)
        names = np.empty(n_countries, dtype=object)

        for idx, country in enumerate(countries):
            raw_code = getattr(country, "code", f"country_{idx}")
            normalized_code = _normalize_country_value(raw_code) or raw_code
            codes[idx] = normalized_code
            names[idx] = getattr(country, "name", normalized_code)

        return _CountryMetadata(countries=countries, codes=codes, names=names)

    def _build_individual_metadata_cache(self) -> Optional[_IndividualMetadata]:  # noqa: E501
        """Build individual metadata cache (called once per simulation)."""
        cells = getattr(self, "_cell_entities", None) or []
        cell_meta = getattr(self, "_cell_metadata_cache", None)

        all_individuals = []
        cell_ids = []
        lon_values = []
        lat_values = []
        area_values = []
        country_values = []
        class_values = []

        def _safe_scalar(arr, i, default):
            """Extract scalar from array, handling 2D arrays or missing data."""
            if arr is None or i >= len(arr):
                return default
            val = arr[i]
            if np.ndim(val) > 0:
                return float(np.asarray(val).flat[0])
            return val if val is not None else default

        for idx, cell in enumerate(cells):
            individuals = getattr(cell, "individuals", None)
            if not individuals:
                continue

            cell_id = _safe_scalar(cell_meta.ids, idx, idx) if cell_meta else idx
            lon_val = _safe_scalar(cell_meta.lon, idx, np.nan) if cell_meta else np.nan
            lat_val = _safe_scalar(cell_meta.lat, idx, np.nan) if cell_meta else np.nan
            area_val = _safe_scalar(cell_meta.area_km2, idx, np.nan) if cell_meta else np.nan  # noqa: E501
            country_val = (
                cell_meta.country[idx] if cell_meta is not None and idx < len(cell_meta.country) else None  # noqa: E501
            )

            for individual in individuals:
                all_individuals.append(individual)
                cell_ids.append(cell_id if cell_id is not None else -1)
                lon_values.append(lon_val)
                lat_values.append(lat_val)
                area_values.append(area_val)
                country_values.append(country_val)
                class_values.append(individual.__class__.__name__)

        self._individual_entities = all_individuals

        if not all_individuals:
            self._individual_output_vars = []
            return None

        output_vars = all_individuals[0].get_defined_outputs()
        self._individual_output_vars = output_vars or []

        return _IndividualMetadata(
            ids=np.arange(len(all_individuals), dtype=np.int32),
            cell=np.array(cell_ids, dtype=np.int64),
            lon=np.array(lon_values, dtype=np.float64),
            lat=np.array(lat_values, dtype=np.float64),
            area_km2=np.array(area_values, dtype=np.float64),
            country=np.array(country_values, dtype=object),
            classes=np.array(class_values, dtype=object),
        )

    def _extract_cell_metadata(self, cell) -> Tuple[Optional[int], float, float, float, Optional[str]]:  # noqa: E501
        """Extract (cell_id, lon, lat, area_km2, country_code) for a cell."""
        world = getattr(self, "world", None)
        cell_idx = getattr(cell, "_cell_index", None)

        cell_id = cell_idx
        lon_val = lat_val = area_val = np.nan
        country_code = getattr(cell, "country_code", None)

        if world is not None and cell_idx is not None:
            grid = getattr(world, "grid", None)
            if grid is not None and hasattr(grid, "isel"):
                try:
                    grid_cell = grid.isel(cell=cell_idx)
                    if hasattr(grid_cell, "values"):
                        cell_id = int(grid_cell.values)
                except Exception:
                    cell_id = cell_idx

                if hasattr(grid, "lon"):
                    try:
                        lon_val = float(grid.lon.isel(cell=cell_idx).values)
                    except Exception:
                        pass
                if hasattr(grid, "lat"):
                    try:
                        lat_val = float(grid.lat.isel(cell=cell_idx).values)
                    except Exception:
                        pass

            area_data = getattr(world, "area", None)
            if area_data is not None and hasattr(area_data, "isel"):
                try:
                    area_val = float(
                        area_data.isel(cell=cell_idx).values
                    ) * 1e-6
                except Exception:
                    pass

            if country_code is None:
                country_data = getattr(world, "country_code", None)
                if country_data is not None and hasattr(country_data, "isel"):
                    try:
                        raw_code = country_data.isel(cell=cell_idx).values
                        country_code = _normalize_country_value(raw_code)
                    except Exception:
                        pass

        normalized_country = _normalize_country_value(country_code)
        quoted_country = (
            _quote_country_array(
                np.array([normalized_country],
                dtype=object)
            )[0]
            if normalized_country else None
        )

        return (cell_id, lon_val, lat_val, area_val, quoted_country)

    def _collect_world_outputs(self, t: int) -> Optional[xr.Dataset]:
        """Collect world-level outputs (scalar values, no spatial
        dimension).
        """
        if self.world is None:
            return None

        output_vars = self.world.get_defined_outputs()
        if not output_vars:
            return None

        n_vars = len(output_vars)
        values = np.full(n_vars, np.nan, dtype=np.float64)

        for i, var_name in enumerate(output_vars):
            try:
                value = getattr(self.world, var_name, None)
                if value is not None:
                    values[i] = _extract_scalar_value(value)
            except Exception:
                pass

        data_vars = {}
        for i, var_name in enumerate(output_vars):
            data_array = xr.DataArray(
                values[i:i+1], dims=["time"], name=var_name
            )
            attrs = _variable_attrs(self.world.__class__, var_name)
            if attrs:
                data_array = data_array.assign_attrs(attrs)
            data_vars[var_name] = data_array

        return xr.Dataset(data_vars)

    def _collect_country_outputs(self, t: int) -> Optional[xr.Dataset]:
        """Collect country-level outputs."""
        self._ensure_output_metadata()
        country_meta = getattr(self, "_country_metadata_cache", None)
        if country_meta is None or not country_meta.countries:
            return None

        countries = country_meta.countries
        output_vars = getattr(self, "_country_output_vars", None) or []
        if not output_vars:
            return None

        n_countries = len(countries)
        n_vars = len(output_vars)
        values = np.full((n_countries, n_vars), np.nan, dtype=np.float64)
        country_codes = np.empty(n_countries, dtype=object)
        country_names = np.empty(n_countries, dtype=object)

        for i, country in enumerate(countries):
            country_codes[i] = getattr(country, "code", f"country_{i}")
            country_names[i] = getattr(country, "name", country_codes[i])
            for j, var_name in enumerate(output_vars):
                try:
                    value = getattr(country, var_name, None)
                    if value is not None:
                        values[i, j] = _extract_scalar_value(value)
                except Exception:
                    pass

        data_vars = {}
        for j, var_name in enumerate(output_vars):
            data_array = xr.DataArray(
                values[:, j:j+1], dims=["country", "time"], name=var_name
            )
            attrs = _variable_attrs(countries[0].__class__, var_name)
            if attrs:
                data_array = data_array.assign_attrs(attrs)
            data_vars[var_name] = data_array

        return xr.Dataset(data_vars, coords={
            "country": (["country"], country_codes),
            "country_name": (["country"], country_names),
        })

    def _collect_cell_outputs(self, t: int) -> Optional[xr.Dataset]:
        """Collect cell-level outputs."""
        self._ensure_output_metadata()
        cell_meta = getattr(self, "_cell_metadata_cache", None)
        cells = getattr(self, "_cell_entities", None)
        if cell_meta is None or not cells:
            return None

        output_vars = getattr(self, "_cell_output_vars", None) or []
        if not output_vars:
            return None

        n_cells = len(cells)
        n_vars = len(output_vars)
        values = np.full((n_cells, n_vars), np.nan, dtype=np.float64)

        meta_values = getattr(cell_meta, "values", None)
        for i, cell in enumerate(cells):
            for j, var_name in enumerate(output_vars):
                try:
                    value = getattr(cell, var_name, None)
                    if value is not None:
                        scalar = _extract_scalar_value(value)
                        values[i, j] = scalar
                        if meta_values and var_name in meta_values:
                            try:
                                meta_values[var_name][i] = scalar
                            except Exception:
                                pass
                except Exception:
                    pass

        data_vars = {}
        for j, var_name in enumerate(output_vars):
            data_array = xr.DataArray(
                values[:, j:j+1], dims=["cell", "time"], name=var_name
            )
            attrs = _variable_attrs(cells[0].__class__, var_name)
            if attrs:
                data_array = data_array.assign_attrs(attrs)
            data_vars[var_name] = data_array

        return xr.Dataset(data_vars, coords={
            "cell": (["cell"], cell_meta.ids),
            "cell_lon": (["cell"], cell_meta.lon),
            "cell_lat": (["cell"], cell_meta.lat),
            "cell_area_km2": (["cell"], cell_meta.area_km2),
            "cell_country": (["cell"], cell_meta.country),
        })

    def _collect_individual_outputs(self, t: int) -> Optional[xr.Dataset]:
        """Collect individual-level outputs."""
        self._ensure_output_metadata()
        individual_meta = getattr(self, "_individual_metadata_cache", None)
        individuals = getattr(self, "_individual_entities", None)
        if individual_meta is None or not individuals:
            return None

        output_vars = getattr(self, "_individual_output_vars", None) or []
        if not output_vars:
            return None

        n_individuals = len(individuals)
        n_vars = len(output_vars)
        values = np.full((n_individuals, n_vars), np.nan, dtype=np.float64)

        for i, individual in enumerate(individuals):
            for j, var_name in enumerate(output_vars):
                try:
                    value = getattr(individual, var_name, None)
                    if value is not None:
                        scalar = _extract_scalar_value(value)
                        values[i, j] = scalar
                        meta = getattr(individual_meta, "values", None)
                        if meta and var_name in meta:
                            try:
                                meta[var_name][i] = scalar
                            except Exception:
                                pass
                except Exception:
                    pass

        data_vars = {}
        for j, var_name in enumerate(output_vars):
            data_array = xr.DataArray(
                values[:, j:j+1], dims=["individual_id", "time"], name=var_name
            )
            attrs = _variable_attrs(individuals[0].__class__, var_name)
            if attrs:
                data_array = data_array.assign_attrs(attrs)
            data_vars[var_name] = data_array

        return xr.Dataset(data_vars, coords={
            "individual_id": (["individual_id"], individual_meta.ids),
            "individual_cell": (["individual_id"], individual_meta.cell),
            "individual_class": (["individual_id"], individual_meta.classes),
            "individual_lon": (["individual_id"], individual_meta.lon),
            "individual_lat": (["individual_id"], individual_meta.lat),
            "individual_area_km2": (["individual_id"], individual_meta.area_km2),  # noqa: E501
            "individual_country": (["individual_id"], individual_meta.country),
        })

    def _init_output_store(self) -> None:
        """Initialize Zarr store for output data.

        Uses same Zarr store as LPJmL data, writing model outputs to
        "model_outputs" group. Store path auto-detected from config.sim_path.
        """
        try:
            import zarr
        except ImportError:
            raise ImportError(
                "zarr package is required for output storage. "
                "Install with: pip install zarr"
            )

        if not hasattr(self, "world") or self.world is None:
            raise RuntimeError("Cannot initialize output store: world not yet created")  # noqa: E501

        # Always use temporary storage for Zarr (outputs are written to final
        # formats after simulation)
        import tempfile
        store_path = os.path.join(
            tempfile.gettempdir(), f"inseeds_outputs_{os.getpid()}.zarr"
        )

        store = zarr.open(store_path, mode="a")
        self.world._output_store_path = store_path
        self.world._output_zarr_group = store
        self._zarr_pending = []
        # Default flush interval: 5 years (buffered writes for performance)
        self._zarr_flush_interval = 5

    def _append_to_zarr(self, year_data: xr.Dataset, t: int) -> None:
        """Append current year's data to Zarr store (buffered writes)."""
        if self.world._output_zarr_group is None:
            return

        if not hasattr(self, "_zarr_pending"):
            self._zarr_pending = []
        self._zarr_pending.append(year_data)

        if len(self._zarr_pending) >= self._zarr_flush_interval:
            self._flush_pending_zarr()

    def _flush_pending_zarr(self, force: bool = False) -> None:
        """Flush buffered Zarr writes."""
        if not hasattr(self, "_zarr_pending") or not self._zarr_pending:
            return
        if not force and len(self._zarr_pending) < self._zarr_flush_interval:
            return

        try:
            import zarr  # noqa: F401
        except ImportError:
            self._zarr_pending.clear()
            return

        try:
            chunk = xr.concat(
                self._zarr_pending, dim="time",
                data_vars="minimal", coords="minimal", compat="override"
            )
            chunk.to_zarr(
                self.world._output_store_path,
                group="model_outputs",
                mode="a",
                append_dim="time",
            )
        except Exception:
            try:
                chunk.to_zarr(
                    self.world._output_store_path,
                    group="model_outputs",
                    mode="a",
                )
            except Exception:
                pass
        finally:
            self._zarr_pending.clear()

    def finalize_output_streams(self) -> Dict[str, str]:
        """Flush pending Zarr buffers and finalize streaming table writers."""
        paths: Dict[str, str] = {}
        exporter = getattr(self, "_table_export_manager", None)
        if exporter is not None:
            try:
                paths = exporter.finalize()
            finally:
                self._table_export_manager = None
                if hasattr(self.world, "_table_export_manager"):
                    self.world._table_export_manager = None
        self._flush_pending_zarr(force=True)
        return paths


# ============================================================================
# Output Writers
# ============================================================================

def _build_global_cf_attrs(ds: xr.Dataset, prefix: str) -> Dict[str, str]:
    """Build CF-compliant global metadata for NetCDF outputs.

    Creates global attributes following CF-1.8 conventions for climate
    and forecast data.

    Parameters
    ----------
    ds : xarray.Dataset
        Source dataset containing optional existing attributes.
    prefix : str
        File prefix used for title generation.

    Returns
    -------
    dict
        CF-compliant global attributes including title, institution,
        source, history, and Conventions.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    history_line = f"{timestamp}: pycopanlpjml.write_outputs_netcdf"
    existing_history = ds.attrs.get("history")
    history = f"{history_line}\n{existing_history}" if existing_history else history_line  # noqa: E501

    attrs = {
        "title": ds.attrs.get("title") or f"{prefix} outputs",
        "institution": ds.attrs.get("institution") or "Potsdam Institute for Climate Impact Research",  # noqa: E501
        "source": ds.attrs.get("source") or "pycopanlpjml",
        "history": history,
        "references": ds.attrs.get("references"),
        "contact": ds.attrs.get("contact"),
        "Conventions": "CF-1.8",
    }
    return {k: v for k, v in attrs.items() if v}


def write_outputs_netcdf(
    output_store_path: str,
    output_path: str,
    start_year: int,
    end_year: int,
    *,
    file_prefix: Optional[str] = None,
    compression: bool = True,
    compression_level: int = 4,
) -> Dict[str, str]:
    """Write cell-based outputs to individual NetCDF files (gridded).

    Reads from Zarr store and writes one NetCDF file per variable.
    Individual-level variables are aggregated to cell-level before writing.

    Parameters
    ----------
    output_store_path : str
        Path to Zarr store containing collected outputs
    output_path : str
        Directory where NetCDF files should be placed
    start_year, end_year : int
        Inclusive year range to write
    file_prefix : str, optional
        Prefix prepended to every NetCDF file name
    compression : bool
        Whether to enable zlib compression (default: True)
    compression_level : int
        Compression level 1-9 when compression enabled (default: 4)

    Returns
    -------
    Dict[str, str]
        Mapping of variable name to written NetCDF path
    """
    try:
        import zarr  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "zarr package is required for reading output store. "
            "Install with: pip install zarr"
        ) from exc

    os.makedirs(output_path, exist_ok=True)

    try:
        ds = xr.open_zarr(
            output_store_path,
            group="model_outputs",
            consolidated=False
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to read Zarr store: {exc}") from exc

    # Filter by time range
    if "time" in ds.coords:
        time_mask = (ds.time >= start_year) & (ds.time <= end_year)
        ds = ds.isel(time=time_mask)

    # Prepare metadata
    cell_meta = _prepare_cell_metadata(ds)
    individual_meta = _prepare_individual_metadata(ds)

    # Clean up dataset
    if "variable_metadata" in ds.attrs:
        ds.attrs.pop("variable_metadata", None)

    coords_to_drop = [
        name for name, coord_var in ds.coords.items()
        if coord_var.dtype.kind in ["U", "S", "O"]
    ]
    if coords_to_drop:
        ds = ds.drop_vars(coords_to_drop)

    drop_candidates = [var for var in METADATA_DATA_VARS if var in ds.data_vars]  # noqa: E501
    if drop_candidates:
        ds = ds.drop_vars(drop_candidates)

    # Prepare for writing
    fallback = ds.attrs.get("sim_name") or "outputs"
    prefix = _sanitize_prefix(file_prefix or fallback)
    global_attrs = _build_global_cf_attrs(ds, prefix)
    written: Dict[str, str] = {}
    delayed_writes: List[Any] = []

    # Write each variable
    for var_name, var_data in ds.data_vars.items():
        if var_name in ds.coords or not np.issubdtype(var_data.dtype, np.number):  # noqa: E501
            continue

        target = var_data.copy(deep=False)
        
        # Aggregate individual-level to cell-level
        if "individual_id" in target.dims:
            target = _individual_to_cell_dataarray(
                target,
                cell_meta,
                individual_meta
            )
        
        # Assign cell coordinates if available
        if "cell" in target.dims:
            cell_meta_len = len(cell_meta.ids) if cell_meta is not None and cell_meta.ids is not None else 0  # noqa: E501
            cell_len = target.sizes.get("cell", 0)
            if cell_meta_len and cell_meta_len == cell_len:
                target = target.assign_coords(
                    cell=("cell", cell_meta.ids),
                    lon=("cell", cell_meta.lon),
                    lat=("cell", cell_meta.lat),
                )
            elif cell_meta_len and cell_meta_len != cell_len:
                if not {"lon", "lat"} <= set(target.coords):
                    raise RuntimeError(
                        f"Cell metadata size mismatch and target lacks lon/lat coords for '{var_name}'."  # noqa: E501
                    )
            else:
                if not {"lon", "lat"} <= set(target.coords):
                    raise RuntimeError(f"Cell metadata missing; cannot write variable '{var_name}'.")  # noqa: E501

        target.attrs["_global_attrs"] = dict(global_attrs)

        lpjml_array = LPJmLData(target)
        target_file = os.path.join(output_path, f"{prefix}_{var_name}.nc4")
        delayed = lpjml_array.to_netcdf(
            target_file,
            compression=compression,
            complevel=compression_level,
            engine="netcdf4",
            compute=False,
        )
        delayed_writes.append(delayed)
        written[var_name] = target_file

    if delayed_writes:
        try:
            import dask
        except ImportError as exc:
            raise RuntimeError(
                "dask is required for deferred NetCDF writes. "
                "Install with: pip install dask[complete]"
            ) from exc
        dask.compute(*delayed_writes)

    if not written:
        raise RuntimeError("No numeric output variables available for NetCDF writing.")  # noqa: E501

    return written


def write_outputs_tables(
    output_store_path: str,
    output_path: str,
    start_year: int,
    end_year: int,
    *,
    formats: List[str],
    variable_metadata: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, str]:
    """Write outputs to one or more tabular formats in a single pass.

    Converts xarray Dataset to pandas DataFrame format matching legacy inseeds:
    - Columns: year, cell, lon, lat, country, area [km2], class, variable,
    value, unit
    - For individuals: includes individual_id, individual_class

    Parameters
    ----------
    output_store_path : str
        Path to Zarr store containing output data
    output_path : str
        Directory path where files will be written
    start_year, end_year : int
        Inclusive year range to write
    formats : List[str]
        Formats to write (subset of {'csv', 'parquet'})
    variable_metadata : dict, optional
        Additional variable metadata to merge

    Returns
    -------
    Dict[str, str]
        Mapping of format name to written file path
    """
    if not formats:
        return {}

    try:
        import zarr
    except ImportError:
        raise ImportError(
            "zarr package is required for reading output store. "
            "Install with: pip install zarr"
        )

    supported = {"csv", "parquet"}
    normalized_formats: List[str] = []
    for fmt in formats:
        fmt_lower = fmt.lower()
        if fmt_lower not in supported:
            raise ValueError(f"Unsupported table format: {fmt}")
        if fmt_lower not in normalized_formats:
            normalized_formats.append(fmt_lower)

    os.makedirs(output_path, exist_ok=True)

    try:
        ds = xr.open_zarr(
            output_store_path,
            group="model_outputs",
            consolidated=False
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to read Zarr store: {exc}")

    # Filter by time range
    if "time" in ds.coords:
        time_mask = (ds.time >= start_year) & (ds.time <= end_year)
        ds = ds.isel(time=time_mask)
        years = _normalize_year_array(np.asarray(ds.time.values))
    else:
        years = np.arange(start_year, end_year + 1, dtype=np.int64)

    # Prepare metadata
    store_metadata = {}
    if isinstance(ds.attrs.get("variable_metadata"), dict):
        store_metadata = dict(ds.attrs["variable_metadata"])
    metadata_lookup: Dict[str, Dict[str, str]] = {}
    metadata_lookup.update(store_metadata)
    if variable_metadata:
        metadata_lookup.update(variable_metadata)

    cell_meta = _prepare_cell_metadata(ds)
    individual_meta = _prepare_individual_metadata(ds)
    ds = ds.drop_vars([var for var in METADATA_DATA_VARS if var in ds.data_vars])  # noqa: E501

    # Initialize writers
    writers = {
        fmt: _OutputTableWriter(output_path, fmt) for fmt in normalized_formats
    }

    # Cache variable metadata
    var_metadata_cache: Dict[str, Tuple[str, str]] = {}
    for var_name, var_data in ds.data_vars.items():
        if var_name in ds.coords:
            continue
        var_attrs = getattr(var_data, "attrs", {})
        var_meta = metadata_lookup.get(var_name, {})
        display_name = var_attrs.get("long_name") or var_meta.get("long_name", var_name)  # noqa: E501
        var_unit = var_attrs.get("units") or var_meta.get("units", "")
        var_metadata_cache[var_name] = (display_name, var_unit)

    # Process in chunks to avoid memory issues
    plans = []
    time_size = ds.sizes.get("time")
    if time_size is None:
        plans.append((None, years))
    else:
        chunk = max(1, min(TABLE_WRITE_CHUNK_YEARS, int(time_size)))
        for start in range(0, time_size, chunk):
            stop = min(start + chunk, time_size)
            plans.append(
                (slice(start, stop),
                years[start:stop])
            )

    try:
        for time_sel, chunk_years in plans:
            chunk_ds = ds if time_sel is None else ds.isel(time=time_sel)
            chunk_ds = chunk_ds.load()

            for var_name, var_data in chunk_ds.data_vars.items():
                if var_name in chunk_ds.coords:
                    continue

                var_display_name, var_unit = var_metadata_cache.get(var_name, (var_name, ""))  # noqa: E501

                dims = set(var_data.dims)
                if "individual_id" in dims:
                    df = _build_individual_dataframe(
                        var_data,
                        chunk_years,
                        var_display_name,
                        var_unit,
                        individual_meta
                    )
                elif "cell" in dims:
                    df = _build_cell_dataframe(
                        var_data,
                        chunk_years,
                        var_display_name,
                        var_unit,
                        cell_meta
                    )
                elif "country" in dims:
                    df = _build_country_dataframe(
                        var_data,
                        chunk_years,
                        var_display_name,
                        var_unit,
                        cell_meta
                    )
                else:
                    df = _build_world_dataframe(
                        var_data,
                        chunk_years,
                        var_display_name,
                        var_unit
                    )

                if df is None or df.empty:
                    continue
                for writer in writers.values():
                    writer.write(df)
    finally:
        for writer in writers.values():
            writer.close()

    return {fmt: writer.file_name for fmt, writer in writers.items()}


def write_outputs_parquet(
    output_store_path: str,
    output_path: str,
    start_year: int,
    end_year: int,
    variable_metadata: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """Write outputs to Parquet format (backward compatible with old
    inseeds).
    """
    results = write_outputs_tables(
        output_store_path, output_path, start_year, end_year,
        formats=["parquet"], variable_metadata=variable_metadata,
    )
    return results["parquet"]


def write_outputs_csv(
    output_store_path: str,
    output_path: str,
    start_year: int,
    end_year: int,
    variable_metadata: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """Write outputs to CSV format (backward compatible with old inseeds)."""
    results = write_outputs_tables(
        output_store_path, output_path, start_year, end_year,
        formats=["csv"], variable_metadata=variable_metadata,
    )
    return results["csv"]
