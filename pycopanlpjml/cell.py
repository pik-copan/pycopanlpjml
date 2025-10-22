"""Cell entity type (mixin) class for copan:LPJmL component."""

import pycopancore.model_components.base.implementation as base
from .mixin import AliasMixin
import numpy as np


class Cell(base.Cell, AliasMixin):
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

        Returns the country code from the world's country_code data array
        for this cell's index.

        Returns
        -------
        str or None
            ISO 3-letter country code (e.g., 'DEU', 'FRA') or None if not available
        """
        if self.world is None or not hasattr(self.world, "country_code"):
            return None
        if self.world.country_code is None:
            return None
        # Get the country code for this cell's index
        country_data = self.world.country_code
        if hasattr(country_data, "values"):
            return str(country_data.values[self._cell_index])
        return None

    @country_code.setter
    def country_code(self, value):
        """Set the country code for this cell.

        Country codes can change due to political boundaries or administrative changes.

        Parameters
        ----------
        value : str
            ISO 3-letter country code (e.g., 'DEU', 'FRA')
        """
        if self.world is None or self._cell_index is None:
            raise ValueError(
                "Cannot set country_code: world or cell index not initialized"
            )
        if "country" not in self.world._zarr_backend.root:
            raise ValueError("Country data not available in world")

        self.world._zarr_backend.root["country"][self._cell_index] = value

    @property
    def cell_index(self):
        """Get the global cell index."""
        return self._cell_index

    @property
    def input(self):
        """Get input dataset view for this cell.

        Returns a ZarrDatasetView that provides xarray-like access to only
        this cell's data. Changes made through this view are immediately
        visible to World and Country instances.
        """
        if self.world is None or self._cell_index is None:
            return None
        # Use world.input property to ensure Zarr backend is initialized
        world_input = self.world.input
        return world_input.isel({"cell": self._cell_index})

    @input.setter
    def input(self, value):
        """Set input values for this cell.

        Note: Writes to the underlying Zarr store, immediately synchronized
        with all other views.
        """
        if self.world is None or self._cell_index is None:
            raise ValueError(
                "Cannot set input: world or cell index not initialized"
            )

        # Write to Zarr backend at specific index
        if hasattr(value, "data_vars"):
            for var_name, var_data in value.data_vars.items():
                self.world._zarr_backend.root["input"][var_name][
                    self._cell_index
                ] = var_data.values

    @property
    def output(self):
        """Get output dataset view for this cell."""
        if self.world is None or self._cell_index is None:
            return None
        # Use world.output property to ensure Zarr backend is initialized
        world_output = self.world.output
        return world_output.isel({"cell": self._cell_index})

    @output.setter
    def output(self, value):
        """Set output values for this cell."""
        if self.world is None or self._cell_index is None:
            raise ValueError(
                "Cannot set output: world or cell index not initialized"
            )

        if hasattr(value, "data_vars"):
            for var_name, var_data in value.data_vars.items():
                self.world._zarr_backend.root["output"][var_name][
                    self._cell_index
                ] = var_data.values

    @property
    def grid(self):
        """Get grid data array view for this cell (read-only).

        Grid coordinates are fixed geographical properties and cannot be changed.
        """
        if self.world is None or self._cell_index is None:
            return None
        # Use world.grid property to ensure Zarr backend is initialized
        world_grid = self.world.grid
        return world_grid.isel({"cell": self._cell_index})

    @property
    def area(self):
        """Get area data array view for this cell (read-only).

        Area is a fixed geographical property and cannot be changed.
        """
        if self.world is None or self._cell_index is None:
            return None
        # Use world.area property to ensure Zarr backend is initialized
        world_area = self.world.area
        if world_area is None:
            return None
        return world_area.isel({"cell": self._cell_index})
