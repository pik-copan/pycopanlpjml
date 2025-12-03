"""World entity type mixin class for copan:LPJmL component.

This implementation keeps LPJmL input/output/grid/country/area as
in-memory LPJmLDataSet/LPJmLData (xarray) objects during the simulation.

All model levels (World, Region, Country, Cell) take views/slices of
these arrays, so writes at any level are immediately visible at all
other levels without going through Zarr in the hot path. Zarr can still
be used at the boundary (e.g. LPJmL coupling or persistence), but the
core dynamics operate purely in memory.
"""

import networkx as nx
import numpy as np
import pycopancore.model_components.base.implementation as base
from .mixin import AliasMixin
from .output import OutputTableMixin
from .output import Output


class World(base.World, AliasMixin, OutputTableMixin):
    """An LPJmL-integrating world entity.

    World entity type (mixin) class for copan:LPJmL component. A world
    instance holds data attributes as pycoupler.LPJmLData or
    pycoupler.LPJmLDataSet that are received and send via the lpjml
    instance of the pycoupler.LPJmLCoupler class.

    Output Variables
    ----------------
    Models can override the `output_variables` class attribute to define
    which world attributes should be written to output tables.

    Parameters
    ----------
    input : pycoupler.LPJmLDataSet
        Coupled LPJmL model inputs.
    output : pycoupler.LPJmLDataSet
        Coupled LPJmL model outputs.
    grid : pycoupler.LPJmLData
        Grid of the LPJmL model.
    country_code : pycoupler.LPJmLData
        Country code of each cell (ISO 3-letter codes).
    area : pycoupler.LPJmLData
        Area of each cell in square meters.
    kwargs : dict, optional
        Additional keyword arguments.

    Returns
    -------
    World
        An instance of the LPJmL World.


    Examples
    --------
    In this example, we will demonstate an exammplaric initialization of a
    `pycopanlpjml.Cell` instance independent of the `pycopanlpjml.Component`
    that automatizes the initializtion of all cells belonging to a world.

    A prerequisite is the start of an LPJmL simulation in coupled mode
    described in ...
    To connect to the LPJmL simulation we use the `pycoupler.LPJmLCoupler`
    class.

    >>> from pycoupler.coupler import LPJmLCoupler
    >>> from pycopanlpjml import World


    The configuration file is a json file that holds the configuration for
    the integrated copan:LPJmL model simulation.

    >>> config_file = "path/to/config_file.json"
    >>> lpjml = LPJmLCoupler(
    ...     config_file=config_file,
    ...     host="localhost",
    ...     port=2042,
    ... )

    Initialize LPJmL world, all data is read/send from and to the LPJmL model
    >>> world = World(
    ...     input=lpjml.read_input(copy=False),
    ...     output=lpjml.read_historic_output(),
    ...     grid=lpjml.grid,
    ...     country_code=lpjml.country,  # ISO 3-letter country codes per cell
    ... )

    """

    def __init__(
        self,
        input=None,
        output=None,
        grid=None,
        country_code=None,
        area=None,
        chunk_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Chunk size is currently unused but kept for API compatibility
        self.chunk_size = chunk_size
        self.cell_neighbourhood = nx.Graph()
        self.country_neighbourhood = nx.Graph()

        # Store model reference for OutputTableMixin
        self._model = kwargs.get("model", None)

        # Store the in-memory data; these are the single source of truth.
        # Optionally chunk along 'cell' if a chunk_size is provided so that
        # xarray can leverage Dask-native parallelism efficiently.
        self._input_data = self._ensure_eager_dataset(
            self._maybe_chunk_dataset(input)
        )
        self._output_data = self._ensure_eager_dataset(
            self._maybe_chunk_dataset(output)
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

    @property
    def input(self):
        """Get input dataset for the world (LPJmLDataSet/xarray Dataset)."""
        return self._input_data

    @input.setter
    def input(self, value):
        """Set input dataset values."""
        self._input_data = value

    @property
    def output(self):
        """Get output dataset for the world (LPJmLDataSet/xarray Dataset)."""
        return self._output_data

    @output.setter
    def output(self, value):
        """Set output dataset values."""
        self._output_data = value

    @property
    def grid(self):
        """Get grid data array for the world (LPJmLData/xarray DataArray)."""
        return self._grid_data

    @grid.setter
    def grid(self, value):
        """Set grid values."""
        self._grid_data = value

    @property
    def country_code(self):
        """Get country code data array (ISO 3-letter codes per cell)."""
        return self._country_data

    @country_code.setter
    def country_code(self, value):
        """Set country code values."""
        self._country_data = value

    @property
    def area(self):
        """Get area data array (square meters per cell)."""
        return self._area_data

    @area.setter
    def area(self, value):
        """Set area values."""
        self._area_data = value

    def build_local_view(self, cell_indices):
        """Create a lightweight world representation for given cell indices."""
        return _LocalWorldView(self, cell_indices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_chunk_dataset(self, ds):
        """Chunk an xarray/LPJmL dataset along 'cell' if chunk_size is set."""
        if ds is None or self.chunk_size is None:
            return ds
        try:
            # LPJmLDataSet often wraps an xarray.Dataset and forwards .chunk
            if (
                hasattr(ds, "dims")
                and "cell" in ds.dims
                and hasattr(ds, "chunk")
            ):
                cell_dim = ds.dims["cell"]
                chunk = min(self.chunk_size, cell_dim)
                return ds.chunk({"cell": chunk})
        except Exception:
            # If anything goes wrong, fall back to the original dataset
            return ds
        return ds

    def _maybe_chunk_array(self, da):
        """Chunk LPJmL/xarray data array along 'cell' if chunk_size is set."""
        if da is None or self.chunk_size is None:
            return da
        try:
            if hasattr(da, "dims") and "cell" in da.dims and hasattr(
                da, "chunk"
            ):
                cell_dim = da.sizes["cell"]
                chunk = min(self.chunk_size, cell_dim)
                return da.chunk({"cell": chunk})
        except Exception:
            return da
        return da

    def _ensure_eager_dataset(self, ds):
        """Materialize datasets once so they only hold NumPy buffers."""
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
        """Materialize data arrays once so they carry NumPy buffers."""
        if da is None:
            return None
        loader = getattr(da, "load", None) or getattr(da, "compute", None)
        if callable(loader):
            try:
                return loader()
            except Exception:
                return da
        return da

    @property
    def model(self):
        """Get model reference (required by OutputTableMixin)."""
        # World is typically accessed via self.model.world, so we need to find
        # the model that owns this world
        # Check if model was passed during initialization
        if hasattr(self, "_model"):
            return self._model
        # Try to find model through common patterns
        # This is a fallback - ideally model should be set explicitly
        return None

    def get_defined_outputs(self):
        """Get list of output variable names based on config.

        Returns
        -------
        List[str]
            List of variable names to output (filtered by config)
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


class _ModelConfigView:
    """Minimal wrapper to expose model config on workers without backrefs."""

    __slots__ = ("config",)

    def __init__(self, model):
        self.config = getattr(model, "config", None)


class _LocalWorldView:
    """Minimal world-like container scoped to a subset of cells."""

    def __init__(self, parent_world, cell_indices):
        parent_model = getattr(parent_world, "_model", None)
        self._model = (
            _ModelConfigView(parent_model) if parent_model is not None else None
        )
        # Worker-local worlds should not retain references to the full
        # neighbourhood graphs because those graphs are made of Cell/Country
        # objects, which point back to the original world and trigger recursion
        # during cloudpickle serialization. Workers currently don't rely on
        # these graphs, so we drop them here.
        self.cell_neighbourhood = None
        self.country_neighbourhood = None
        self.chunk_size = None
        self._global_cell_indices = np.asarray(cell_indices, dtype=int)
        self._global_to_local = {
            int(idx): pos for pos, idx in enumerate(self._global_cell_indices)
        }
        self._input_data = self._slice_dataset(parent_world.input)
        self._output_data = self._slice_dataset(parent_world.output)
        self._grid_data = self._slice_array(parent_world.grid)
        self._country_data = self._slice_array(parent_world.country_code)
        self._area_data = self._slice_array(parent_world.area)

    def _slice_dataset(self, dataset):
        if dataset is None or not hasattr(dataset, "isel"):
            return dataset
        sliced = dataset.isel(cell=self._global_cell_indices)
        sliced = self._materialize_dataset(sliced)
        return sliced.copy(deep=True) if hasattr(sliced, "copy") else sliced

    def _slice_array(self, array):
        if array is None or not hasattr(array, "isel"):
            return array
        sliced = array.isel(cell=self._global_cell_indices)
        sliced = self._materialize_array(sliced)
        return sliced.copy(deep=True) if hasattr(sliced, "copy") else sliced

    def _materialize_dataset(self, dataset):
        loader = getattr(dataset, "load", None) or getattr(
            dataset, "compute", None
        )
        if callable(loader):
            try:
                return loader()
            except Exception:
                return dataset
        return dataset

    def _materialize_array(self, array):
        loader = getattr(array, "load", None) or getattr(
            array, "compute", None
        )
        if callable(loader):
            try:
                return loader()
            except Exception:
                return array
        return array

    @property
    def input(self):
        return self._input_data

    @property
    def output(self):
        return self._output_data

    @property
    def grid(self):
        return self._grid_data

    @property
    def country_code(self):
        return self._country_data

    @property
    def area(self):
        return self._area_data

    @property
    def model(self):
        return self._model

    def map_global_to_local(self, indices):
        return [self._global_to_local[int(idx)] for idx in indices]

    def build_local_view(self, cell_indices):
        candidate = np.asarray(cell_indices, dtype=int)
        if np.array_equal(candidate, self._global_cell_indices):
            return self
        raise ValueError(
            "Cannot build a different local view from an already local world"
        )
