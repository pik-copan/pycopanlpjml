"""World entity for copan:LPJmL integration.

This module provides the World class and related utilities for managing
LPJmL earth system data within copan:CORE models.

Classes
-------
World
    The global container for LPJmL data and entities.
_LocalWorldView
    Lightweight world representation for worker processes.
_ModelConfigView
    Minimal wrapper for model config on workers.

The World class serves as the single source of truth for earth system data:

- to_earth: Data sent TO the earth system (LPJmL inputs)
- from_earth: Data received FROM the earth system (LPJmL outputs)
- grid: Cell coordinates and metadata
- country_code: ISO 3-letter country codes per cell
- area: Cell areas in square meters

All data is stored in-memory as xarray/LPJmL datasets. Region, Country, and
Cell entities access views/slices of this data, so writes at any level are
immediately visible at all other levels.

Example
-------
>>> from pycopanlpjml import World
>>> from pycoupler.coupler import LPJmLCoupler
>>>
>>> lpjml = LPJmLCoupler(config_file="config.json")
>>> world = World(
...     to_earth=lpjml.read_input(copy=False),
...     from_earth=lpjml.read_historic_output(),
...     grid=lpjml.grid,
...     country_code=lpjml.country,
... )
>>>
>>> # Access data at world level
>>> world.to_earth["irrig"].values[:] = 0.5
>>>
>>> # Data is also visible at country level
>>> germany = world.countries[0]
>>> print(germany.from_earth["yield"].mean())
"""

import networkx as nx
import numpy as np
import pandas as pd
import pycopancore.model_components.base.implementation as base
try:
    from pycoupler.utils import warn_deprecated_alias
except ImportError:

    def warn_deprecated_alias(instance, old_name: str, new_name: str) -> None:
        """No-op when pycoupler does not provide warn_deprecated_alias."""
        pass

from .mixin import AliasMixin
from .output import OutputDefinitionMixin, dataset_to_output_table


class World(base.World, AliasMixin, OutputDefinitionMixin):
    """The global container for LPJmL data and copan:CORE entities.

    World holds in-memory xarray/LPJmL datasets that serve as the single
    source of truth for earth system data. All entities (Region, Country,
    Cell) access views into this data.

    Parameters
    ----------
    to_earth : xarray.Dataset or LPJmLDataSet, optional
        Data sent to the earth system (LPJmL inputs).
        Formerly called 'input'.
    from_earth : xarray.Dataset or LPJmLDataSet, optional
        Data received from the earth system (LPJmL outputs).
        Formerly called 'output'. Read-only; updated by LPJmL.
    grid : xarray.DataArray or LPJmLData, optional
        Grid coordinates (lon, lat) for each cell.
    country_code : xarray.DataArray or LPJmLData, optional
        ISO 3-letter country code for each cell.
    area : xarray.DataArray or LPJmLData, optional
        Cell areas in square meters.
    chunk_size : int, optional
        If provided, chunk datasets along 'cell' dimension for Dask.
    input : xarray.Dataset, optional
        Deprecated alias for to_earth.
    output : xarray.Dataset, optional
        Deprecated alias for from_earth.
    **kwargs : dict
        Additional arguments passed to base.World.

    Attributes
    ----------
    to_earth : xarray.Dataset
        Data sent to earth system. Read/write.
    from_earth : xarray.Dataset
        Data from earth system. Read-only.
    output_array : xarray.Dataset or None
        Collected model outputs (xarray) for the last year. Lazy; see Notes.
    output_table : pandas.DataFrame
        Collected model outputs (long-format table) for the last year. Lazy;
        see Notes.
    grid : xarray.DataArray
        Cell coordinates. Read-only.
    country_code : xarray.DataArray
        Country codes per cell. Read-only.
    area : xarray.DataArray
        Cell areas. Read-only.
    cell_neighbourhood : networkx.Graph
        Graph of cell connectivity.
    country_neighbourhood : networkx.Graph
        Graph of country connectivity.
    cells : set
        All Cell entities (from pycopancore).
    countries : set
        All Country entities (from pycopancore).

    Example
    -------
    >>> world = World(
    ...     to_earth=lpjml.read_input(copy=False),
    ...     from_earth=lpjml.read_historic_output(),
    ...     grid=lpjml.grid,
    ...     country_code=lpjml.country,
    ... )
    >>>
    >>> # Modify inputs
    >>> world.to_earth["fertilization"].values[:] *= 0.5
    >>>
    >>> # Read outputs
    >>> mean_yield = world.from_earth["yield"].mean()

    See Also
    --------
    Region : Base class for spatial aggregations; also has output_array,
    output_table.
    Country : Country-level entity; also has output_array, output_table.
    Cell : Grid cell entity; also has output_array, output_table.

    Notes
    -----
    **Output access (output_array, output_table)**: Available on World, Region
    (Country), and Cell. Both are lazy—no extra work during simulation.
    ``output_array`` returns the raw xarray Dataset; ``output_table`` returns
    a long-format DataFrame (year, cell, entity, variable, value, unit).
    """

    def __init__(
        self,
        to_earth=None,
        from_earth=None,
        grid=None,
        country_code=None,
        area=None,
        chunk_size=None,
        input=None,  # Deprecated
        output=None,  # Deprecated
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Handle deprecated parameter names
        if input is not None and to_earth is None:
            to_earth = input
        if output is not None and from_earth is None:
            from_earth = output

        # Store configuration
        self.chunk_size = chunk_size
        self.cell_neighbourhood = nx.Graph()
        self.country_neighbourhood = nx.Graph()
        self._model = kwargs.get("model", None)

        # Store and materialize data as in-memory datasets
        # These are the single source of truth; all entities use views
        self._to_earth_data = self._ensure_eager_dataset(
            self._maybe_chunk_dataset(to_earth)
        )
        self._from_earth_data = self._ensure_eager_dataset(
            self._maybe_chunk_dataset(from_earth)
        )
        self._grid_data = self._ensure_eager_array(
            self._maybe_chunk_array(grid)
        )
        self._country_data = self._ensure_eager_array(
            self._maybe_chunk_array(country_code)
        )
        self._area_data = self._ensure_eager_array(
            self._maybe_chunk_array(area)
        )

        # Output storage (for model outputs, separate from from_earth)
        self._output_data = None
        self._output_store_path = None
        self._output_zarr_group = None

    # -------------------------------------------------------------------------
    # Primary data properties
    # -------------------------------------------------------------------------

    @property
    def to_earth(self):
        """Get data sent to earth system (LPJmL inputs).

        This is read/write. Changes are immediately visible to all entities
        that have views into this dataset.

        Returns
        -------
        xarray.Dataset or LPJmLDataSet
            The input dataset.
        """
        return self._to_earth_data

    @to_earth.setter
    def to_earth(self, value):
        """Set data sent to earth system."""
        self._to_earth_data = value

    @property
    def from_earth(self):
        """Get data received from earth system (LPJmL outputs).

        This is read-only. Only LPJmL updates this data via
        Model.update_lpjml().

        Returns
        -------
        xarray.Dataset or LPJmLDataSet
            The output dataset.
        """
        return self._from_earth_data

    @property
    def grid(self):
        """Get grid coordinates (lon, lat) for each cell.

        Read-only; grid is a fixed physical property.

        Returns
        -------
        xarray.DataArray or LPJmLData
            Grid coordinates.
        """
        return self._grid_data

    @property
    def country_code(self):
        """Get country code (ISO 3-letter) for each cell.

        Returns
        -------
        xarray.DataArray or LPJmLData
            Country codes.
        """
        return self._country_data

    @country_code.setter
    def country_code(self, value):
        """Set country codes."""
        self._country_data = value

    @property
    def area(self):
        """Get cell areas in square meters.

        Read-only; area is a fixed physical property.

        Returns
        -------
        xarray.DataArray or LPJmLData
            Cell areas.
        """
        return self._area_data

    # -------------------------------------------------------------------------
    # Backward compatibility aliases
    # -------------------------------------------------------------------------

    @property
    def input(self):
        """Deprecated alias for to_earth."""
        warn_deprecated_alias(self, "input", "to_earth")
        return self.to_earth

    @input.setter
    def input(self, value):
        warn_deprecated_alias(self, "input", "to_earth")
        self.to_earth = value

    @property
    def output(self):
        """Get model output dataset.

        If model outputs have been collected, returns those.
        Otherwise returns from_earth for backward compatibility.

        Returns
        -------
        xarray.Dataset
            Model outputs or from_earth.
        """
        if self._output_data is None:
            return self.from_earth
        return self._output_data

    @property
    def output_array(self):
        """Get model output as xarray Dataset for the last collected year
        (lazy).

        Returns the raw xarray Dataset—no conversion is done. None if no
        outputs collected yet. Updated each time collect_outputs runs. No extra
        work during simulation; computed only when this property is accessed.

        Returns
        -------
        xarray.Dataset or None
            Output dataset for the last collected year.
        """
        return getattr(self, "_output_data", None)

    @property
    def output_table(self):
        """Get model output as long-format DataFrame for the last collected
        year (lazy).

        Converts output_array to legacy table format (year, cell, lon, lat,
        country, area [km2], entity, variable, value, unit) only when accessed.
        Empty DataFrame if no outputs collected yet. No extra work during
        simulation.

        Returns
        -------
        pandas.DataFrame
            Long-format output table.
        """
        data = getattr(self, "_output_data", None)
        if data is None:
            return pd.DataFrame()
        try:
            return dataset_to_output_table(data)
        except Exception:
            return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Output support
    # -------------------------------------------------------------------------

    @property
    def model(self):
        """Get model reference (for OutputDefinitionMixin)."""
        if hasattr(self, "_model"):
            return self._model
        return None

    def get_defined_outputs(self):
        """Get output variable names enabled in config.

        Returns
        -------
        list of str
            Variable names to collect.
        """
        if not hasattr(self, "model") or self.model is None:
            return []
        if not hasattr(self.model, "config"):
            return []

        try:
            config_outputs = (
                self.model.config.coupled_config.output.to_dict().get(
                    "world", []
                )
            )
            return [
                var
                for var in self.__class__.output_variables.names
                if var in config_outputs
            ]
        except Exception:
            return []

    # -------------------------------------------------------------------------
    # Local view support
    # -------------------------------------------------------------------------

    def build_local_view(self, cell_indices):
        """Create a lightweight world for a subset of cells.

        Used for serializing country data to Dask workers. The local view
        contains only the data needed for the specified cells.

        Parameters
        ----------
        cell_indices : array-like
            Indices of cells to include in the view.

        Returns
        -------
        _LocalWorldView
            Lightweight world representation.
        """
        return _LocalWorldView(self, cell_indices)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _maybe_chunk_dataset(self, ds):
        """Chunk dataset along 'cell' dimension if chunk_size is set."""
        if ds is None or self.chunk_size is None:
            return ds
        try:
            if (
                hasattr(ds, "dims")
                and "cell" in ds.dims
                and hasattr(ds, "chunk")
            ):  # noqa: E501
                cell_dim = ds.dims["cell"]
                chunk = min(self.chunk_size, cell_dim)
                return ds.chunk({"cell": chunk})
        except Exception:
            return ds
        return ds

    def _maybe_chunk_array(self, da):
        """Chunk data array along 'cell' dimension if chunk_size is set."""
        if da is None or self.chunk_size is None:
            return da
        try:
            if (
                hasattr(da, "dims")
                and "cell" in da.dims
                and hasattr(da, "chunk")
            ):  # noqa: E501
                cell_dim = da.sizes["cell"]
                chunk = min(self.chunk_size, cell_dim)
                return da.chunk({"cell": chunk})
        except Exception:
            return da
        return da

    def _ensure_eager_dataset(self, ds):
        """Materialize dataset to numpy-backed arrays."""
        if ds is None:
            return None
        loader = getattr(ds, "load", None) or getattr(ds, "compute", None)
        if callable(loader):
            try:
                return loader()
            except Exception:
                return ds
        return ds

    def _ensure_eager_array(self, da):
        """Materialize data array to numpy-backed array."""
        if da is None:
            return None
        loader = getattr(da, "load", None) or getattr(da, "compute", None)
        if callable(loader):
            try:
                return loader()
            except Exception:
                return da
        return da


# =============================================================================
# WORKER-SIDE SUPPORT CLASSES
# =============================================================================


class _ModelConfigView:
    """Minimal wrapper to expose model config on workers.

    This avoids serializing the full model which would create
    circular references.
    """

    __slots__ = ("config",)

    def __init__(self, model):
        self.config = getattr(model, "config", None)


class _LocalWorldView:
    """Lightweight world representation for worker processes.

    Contains only the data needed for a subset of cells, without
    references to the full world or other entities that would
    cause serialization issues.

    Parameters
    ----------
    parent_world : World
        The full world to create a view from.
    cell_indices : array-like
        Global cell indices to include.

    Attributes
    ----------
    to_earth : xarray.Dataset
        Input data for the subset of cells.
    from_earth : xarray.Dataset
        Output data for the subset of cells.
    grid : xarray.DataArray
        Grid coordinates for the subset.
    country_code : xarray.DataArray
        Country codes for the subset.
    area : xarray.DataArray
        Areas for the subset.

    Notes
    -----
    Neighbourhood graphs are set to None because they reference
    entities that would cause recursion during serialization.
    Workers don't need them for country.update().
    """

    def __init__(self, parent_world, cell_indices):
        # Get model config without back-references
        parent_model = getattr(parent_world, "_model", None)
        if parent_model is not None:
            self._model = _ModelConfigView(parent_model)
        else:
            self._model = None

        # No neighbourhood graphs on workers (would cause recursion)
        self.cell_neighbourhood = None
        self.country_neighbourhood = None
        self.chunk_size = None

        # Build index mapping
        self._global_cell_indices = np.asarray(cell_indices, dtype=int)
        self._global_to_local = {
            int(idx): pos
            for pos, idx in enumerate(self._global_cell_indices)  # noqa: E501
        }

        # Slice and materialize data
        self._to_earth_data = self._slice_dataset(parent_world.to_earth)
        self._from_earth_data = self._slice_dataset(parent_world.from_earth)
        self._grid_data = self._slice_array(parent_world.grid)
        self._country_data = self._slice_array(parent_world.country_code)
        self._area_data = self._slice_array(parent_world.area)

    def _slice_dataset(self, dataset):
        """Slice dataset to local cell indices and materialize."""
        if dataset is None or not hasattr(dataset, "isel"):
            return dataset
        sliced = dataset.isel(cell=self._global_cell_indices)
        sliced = self._materialize_dataset(sliced)
        return sliced.copy(deep=True) if hasattr(sliced, "copy") else sliced

    def _slice_array(self, array):
        """Slice data array to local cell indices and materialize."""
        if array is None or not hasattr(array, "isel"):
            return array
        sliced = array.isel(cell=self._global_cell_indices)
        sliced = self._materialize_array(sliced)
        return sliced.copy(deep=True) if hasattr(sliced, "copy") else sliced

    def _materialize_dataset(self, dataset):
        """Force all lazy arrays in dataset to numpy.

        Preserves the original structure (important for LPJmL datasets
        where different variables have different band dimensions).
        """
        if dataset is None:
            return None

        # Try load() first (computes in-place, preserves structure)
        if hasattr(dataset, "load"):
            try:
                return dataset.load()
            except Exception:
                pass

        # Try compute() (returns new computed dataset)
        if hasattr(dataset, "compute"):
            try:
                return dataset.compute()
            except Exception:
                pass

        # Fallback: force each variable individually
        for name in list(dataset.data_vars):
            var = dataset[name]
            if hasattr(var.data, "compute"):
                dataset[name].data = np.asarray(var.values)

        for name in list(dataset.coords):
            coord = dataset.coords[name]
            if hasattr(coord.data, "compute"):
                dataset.coords[name].data = np.asarray(coord.values)

        return dataset

    def _materialize_array(self, array):
        """Force lazy data array to numpy."""
        if array is None:
            return None

        if hasattr(array, "load"):
            try:
                return array.load()
            except Exception:
                pass

        if hasattr(array, "compute"):
            try:
                return array.compute()
            except Exception:
                pass

        if hasattr(array.data, "compute"):
            array.data = np.asarray(array.values)

        return array

    # -------------------------------------------------------------------------
    # Data properties (mirror World interface)
    # -------------------------------------------------------------------------

    @property
    def to_earth(self):
        """Input data for this subset of cells."""
        return self._to_earth_data

    @property
    def from_earth(self):
        """Output data for this subset of cells."""
        return self._from_earth_data

    @property
    def input(self):
        """Deprecated alias for to_earth."""
        warn_deprecated_alias(self, "input", "to_earth")
        return self.to_earth

    @property
    def output(self):
        """Deprecated alias for from_earth."""
        warn_deprecated_alias(self, "output", "from_earth")
        return self.from_earth

    @property
    def grid(self):
        """Grid coordinates for this subset."""
        return self._grid_data

    @property
    def country_code(self):
        """Country codes for this subset."""
        return self._country_data

    @property
    def area(self):
        """Cell areas for this subset."""
        return self._area_data

    @property
    def model(self):
        """Model config view."""
        return self._model

    # -------------------------------------------------------------------------
    # Index mapping
    # -------------------------------------------------------------------------

    def map_global_to_local(self, indices):
        """Convert global cell indices to local indices.

        Parameters
        ----------
        indices : array-like
            Global cell indices.

        Returns
        -------
        list of int
            Local indices within this view.
        """
        return [self._global_to_local[int(idx)] for idx in indices]

    def build_local_view(self, cell_indices):
        """Return self if indices match, otherwise raise.

        A local view cannot create a different local view.

        Parameters
        ----------
        cell_indices : array-like
            Cell indices for the new view.

        Returns
        -------
        _LocalWorldView
            Self if indices match.

        Raises
        ------
        ValueError
            If indices don't match this view's cells.
        """
        candidate = np.asarray(cell_indices, dtype=int)
        if np.array_equal(candidate, self._global_cell_indices):
            return self
        raise ValueError(
            "Cannot build a different local view from an already local world"
        )
