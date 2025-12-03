"""Cell entity type (mixin) class for copan:LPJmL component."""

import pycopancore.model_components.base.implementation as base
from .mixin import AliasMixin
from .output import OutputTableMixin
from .output import Output
import numpy as np


class Cell(base.Cell, AliasMixin, OutputTableMixin):
    """An LPJmL-integrating cell entity.

    Cell entity type (mixin) class for copan:LPJmL component. It inherits the
    copan:CORE cell entity and structure and integrates LPJmL input and output
    data (attributes) as well as grid, country and area information as
    `pycoupler.LPJmLData` and `pycoupler.LPJmLDataSet` instances.
    A cell instance should hold (numpy) views of attributes to the
    corresponding cell in the copan:LPJmL world instance.

    Parameters
    ----------
    input : pycoupler.LPJmLDataSet
        Coupled LPJmL model input.
    output : pycoupler.LPJmLDataSet
        Coupled LPJmL model output.
    grid : pycoupler.LPJmLData
        Grid of the LPJmL model.
    country : str
        Country of the cell as a country code.
    area : float
        Area of the cell in square meters.
    kwargs : dict, optional
        Additional keyword arguments.

    Returns
    -------
    Cell
        An instance of the copan:LPJmL Cell.

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
    ...     country_code=lpjml.country,  # ISO 3-letter country codes
    ... )

    Initialize a (first) cell instance
    >>> cell_id = 0
    >>> cell = Cell(
    ...     world=world,
    ...     cell_index=cell_id,
    ...     grid=world.grid.isel({"cell": cell_id}),
    ... )

    Access the cell's country code
    >>> cell.country_code  # Returns the ISO 3-letter code (e.g., 'DEU')

    >>> cell
    """

    # Output variables (models can override this)
    output_variables = Output()

    _entity_alias = "cell"

    def __init__(
        self,
        world=None,
        country=None,
        cell_index=None,
        input=None,
        output=None,
        grid=None,
        **kwargs,
    ):
        """Initialize a Cell instance.

        Parameters
        ----------
        world : World
            Reference to World instance (required for Zarr backend access)
        country : Country, optional
            Country instance this cell belongs to
        cell_index : int
            Global cell index in the world
        input : xr.Dataset or ZarrDatasetView
            Input data (used to extract cell index if cell_index not provided)
        output : xr.Dataset or ZarrDatasetView
            Output data (used to extract cell index if cell_index not provided)
        grid : xr.DataArray or ZarrDataArrayView
            Grid data (used to extract cell index if cell_index not provided)
        **kwargs : dict
            Additional keyword arguments
        """
        # Pass world to base class (pycopancore handles world property)
        if "world" not in kwargs:
            kwargs["world"] = world

        # Map country to social_system for pycopancore
        if country is not None and "social_system" not in kwargs:
            kwargs["social_system"] = country

        super().__init__(**kwargs)

        # Note: self.country is now a property that returns self.social_system

        # Determine cell index
        if cell_index is not None:
            self._cell_index = cell_index
        elif grid is not None:
            # Extract cell index from grid
            if hasattr(grid, "cell"):
                # xarray DataArray - get the cell value
                cell_coord = grid.cell
                if hasattr(cell_coord, "values"):
                    # Handle both scalar and array cases
                    cell_vals = cell_coord.values
                    self._cell_index = int(
                        cell_vals.item()
                        if cell_vals.shape == ()
                        else cell_vals[0]
                    )
                else:
                    self._cell_index = int(cell_coord)
            elif hasattr(grid, "coords") and "cell" in grid.coords:
                # ZarrDataArrayView
                cell_vals = grid.coords["cell"]
                self._cell_index = int(
                    cell_vals if np.isscalar(cell_vals) else cell_vals[0]
                )
            else:
                raise ValueError("Cannot determine cell index from grid")
        else:
            raise ValueError("Either cell_index or grid must be provided")

        # Initialize neighbourhood
        self.neighbourhood = list()

    @property
    def country(self):
        """Get the country instance this cell belongs to.

        This is just an alias for social_system for LPJmL compatibility.
        """
        return self.social_system

    @property
    def country_code(self):
        """Get the country code (ISO 3-letter code) for this cell.

        Returns
        -------
        str or None
            ISO 3-letter country code (e.g., 'DEU', 'FRA') or None if not
            available
        """
        if self._cell_index is None:
            return None
        # Prefer in-memory world.country_code (xarray/LPJmLData)
        world = getattr(self, "_world", None)
        if world is None or world.country_code is None:
            return None
        local_idx = self._resolve_local_index(world)
        if local_idx is None:
            return None
        if not hasattr(world.country_code, "values"):
            return None
        return str(world.country_code.values[local_idx])

    @country_code.setter
    def country_code(self, value):
        """Set the country code for this cell.

        Country codes can change due to political boundaries or administrative
        changes.

        Parameters
        ----------
        value : str
            ISO 3-letter country code (e.g., 'DEU', 'FRA')
        """
        if self._cell_index is None:
            raise ValueError(
                "Cannot set country_code: cell index not initialized"
            )
        world = getattr(self, "_world", None)
        if world is None or world.country_code is None:
            raise ValueError("Country data not available")
        if not hasattr(world.country_code, "values"):
            raise ValueError(
                "world.country_code must be an array-like with values"
            )
        local_idx = self._resolve_local_index(world)
        if local_idx is None:
            raise ValueError(
                "Cannot resolve cell index for world country data"
            )
        world.country_code.values[local_idx] = value

    @property
    def cell_index(self):
        """Get the global cell index."""
        return self._cell_index

    def _resolve_local_index(self, world):
        """Map global cell index to world-specific index."""
        if world is None:
            return None
        if hasattr(world, "_global_to_local"):
            mapping = getattr(world, "_global_to_local", None)
            if mapping is None:
                return None
            if self._cell_index not in mapping:
                raise ValueError(
                    (
                        f"Cell index {self._cell_index} not present "
                        "in this world view"
                    )
                )
            return mapping[self._cell_index]
        return self._cell_index

    def _selector_for_world(self, world):
        idx = self._resolve_local_index(world)
        if idx is None:
            return None
        return [idx]

    def _selector_for_world_drop_dim(self, world):
        """Get selector that drops the cell dimension (for single cell
        access)."""
        idx = self._resolve_local_index(world)
        if idx is None:
            return None
        return idx  # Return integer, not list, to drop dimension

    @property
    def input(self):
        """Get input dataset view for this cell.

        Returns a view on the world's input dataset restricted to this cell.
        The cell dimension is dropped, so indexing is (band, time) not (cell,
        band, time).
        """
        if self._cell_index is None:
            return None
        world = getattr(self, "_world", None)
        if world is None or world.input is None:
            return None
        selector = self._selector_for_world_drop_dim(world)
        if selector is None:
            return None
        # xarray/LPJmLDataSet: use isel to get a view for this cell
        # Using integer selector drops the cell dimension
        if hasattr(world.input, "isel"):
            return world.input.isel(cell=selector)
        return world.input

    @input.setter
    def input(self, value):
        """Set input values for this cell.

        Note: Writes directly into the world's in-memory input dataset.
        """
        if self._cell_index is None:
            raise ValueError("Cannot set input: cell index not initialized")
        world = getattr(self, "_world", None)
        if world is None or world.input is None:
            raise ValueError("Cannot set input: world input not available")
        if not hasattr(world.input, "data_vars"):
            raise ValueError("World input must be an xarray/LPJmL Dataset")
        if not hasattr(value, "data_vars"):
            raise ValueError("Assigned value must be an xarray/LPJmL Dataset")
        for var_name, var_data in value.data_vars.items():
            if var_name in world.input.data_vars:
                local_idx = self._resolve_local_index(world)
                if local_idx is None:
                    raise ValueError(
                        "Cannot resolve cell index for writing world input"
                    )
                world.input[var_name].values[local_idx] = var_data.values

    @property
    def output(self):
        """Get output dataset view for this cell.

        The cell dimension is dropped, so indexing is (band, time) not (cell,
        band, time).
        """
        if self._cell_index is None:
            return None
        world = getattr(self, "_world", None)
        if world is None or world.output is None:
            return None
        selector = self._selector_for_world_drop_dim(world)
        if selector is None:
            return None
        if hasattr(world.output, "isel"):
            return world.output.isel(cell=selector)
        return world.output

    @output.setter
    def output(self, value):
        """Set output values for this cell."""
        if self._cell_index is None:
            raise ValueError(
                "Cannot set output: cell index not initialized"
            )
        world = getattr(self, "_world", None)
        if world is None or world.output is None:
            raise ValueError("Cannot set output: world output not available")
        if not hasattr(world.output, "data_vars"):
            raise ValueError("World output must be an xarray/LPJmL Dataset")
        if not hasattr(value, "data_vars"):
            raise ValueError("Assigned value must be an xarray/LPJmL Dataset")
        for var_name, var_data in value.data_vars.items():
            if var_name in world.output.data_vars:
                local_idx = self._resolve_local_index(world)
                if local_idx is None:
                    raise ValueError(
                        "Cannot resolve cell index for writing world output"
                    )
                world.output[var_name].values[local_idx] = (
                    var_data.values
                )

    @property
    def grid(self):
        """Get grid data array view for this cell (read-only).

        Grid coordinates are fixed geographical properties and cannot be
        changed.
        The cell dimension is dropped for single cell access.
        """
        if self._cell_index is None:
            return None
        world = getattr(self, "_world", None)
        if world is None or world.grid is None:
            return None
        selector = self._selector_for_world_drop_dim(world)
        if selector is None:
            return None
        if hasattr(world.grid, "isel"):
            return world.grid.isel(cell=selector)
        return world.grid

    @property
    def area(self):
        """Get area for this cell (read-only).

        Area is a fixed geographical property and cannot be changed.
        The cell dimension is dropped, returning a scalar.
        """
        if self._cell_index is None:
            return None
        world = getattr(self, "_world", None)
        if world is None or world.area is None:
            return None
        selector = self._selector_for_world_drop_dim(world)
        if selector is None:
            return None
        if hasattr(world.area, "isel"):
            return world.area.isel(cell=selector)
        return world.area

    @property
    def model(self):
        """Get model reference (required by OutputTableMixin).

        Accesses _world directly to avoid triggering world property
        which could cause restoration during output writing.
        """
        # Access _world directly to avoid property access overhead
        # This is safe because Cell.world property just returns self._world
        world = getattr(self, "_world", None)
        if world is not None:
            # Check if world has _model attribute directly (World instances)
            if hasattr(world, "_model") and world._model is not None:
                return world._model
            # Fallback to world.model property (for Region/Country)
            if hasattr(world, "model"):
                return world.model
        return None

    def get_defined_outputs(self):
        """Get list of output variable names based on config.

        Returns
        -------
        List[str]
            List of variable names to output (filtered by config)
        """
        if self.model is None or not hasattr(self.model, "config"):
            return []

        try:
            config_outputs = (
                self.model.config.coupled_config.output.to_dict().get(
                    "cell", []
                )
            )
            return [
                var
                for var in self.__class__.output_variables.names
                if var in config_outputs
            ]
        except Exception:
            return []
