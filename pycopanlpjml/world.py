"""World entity type for copan:LPJmL component."""

import numpy as np
import networkx as nx
import xarray as xr
import pycopancore.model_components.base.implementation as base


class World(base.World):
    """An LPJmL-integrating world entity.

    World holds the LPJmL arrays (source of truth). Cells store scalar
    ``isel`` views. Countries re-isel a copy of the current slice on access.

    Parameters
    ----------
    input : LPJmLDataSet
        Coupled LPJmL model inputs (to_earth).
    output : LPJmLDataSet
        Coupled LPJmL model outputs (from_earth).
    grid : LPJmLData
        Grid of the LPJmL model (lon, lat per cell).
    country_code : LPJmLData
        Country code array from LPJmL (ISO codes per cell).
    area : LPJmLData
        Area of each cell in square meters.
    kwargs : dict, optional
        Additional keyword arguments.

    Attributes
    ----------
    input : LPJmLDataSet
        Data sent to LPJmL.
    output : LPJmLDataSet
        Data from LPJmL.
    to_earth : LPJmLDataSet
        Alias for input.
    from_earth : LPJmLDataSet
        Alias for output.
    grid : LPJmLData
        Cell coordinates (lon, lat).
    country_code : LPJmLData
        Country code array (ISO codes per cell).
    area : LPJmLData
        Cell areas in m².
    countries : set
        Country entities (not the country_code array).

    Examples
    --------
    >>> world = World(
    ...     input=lpjml.read_input(copy=False),
    ...     output=lpjml.read_historic_output(),
    ...     grid=lpjml.grid,
    ...     country_code=lpjml.country,
    ...     area=lpjml.terr_area,
    ... )
    """

    def __init__(
        self,
        input=None,
        output=None,
        grid=None,
        country_code=None,
        area=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store model reference if provided
        self._model = kwargs.get("model", None)

        # Internal data storage
        self._to_earth_data = None
        self._from_earth_data = None
        self._grid_data = None
        self._country_code_data = None
        self._area_data = None

        # Direct data storage
        if input is not None:
            self._to_earth_data = input
            self._sync_input_time_from_lpjml()

        if output is not None:
            self._from_earth_data = output

        if grid is not None:
            self._grid_data = grid
            self.neighbourhood = nx.Graph()

        if country_code is not None:
            self._country_code_data = country_code

        self._area_data = area
        self.country_neighbourhood = nx.Graph()

    def _sync_input_time_from_lpjml(self):
        """Align input time to LPJmL sim year when the coord is datetime."""
        lpjml = getattr(self._model, "lpjml", None)
        sim_year = getattr(lpjml, "sim_year", None)
        if sim_year is None or self._to_earth_data is None:
            return
        if "time" not in self._to_earth_data.coords:
            return
        time = self._to_earth_data.time
        if not np.issubdtype(time.dtype, np.datetime64):
            return
        time.values[0] = np.datetime64(f"{sim_year}-12-31")

    @property
    def input(self):
        """Data sent to LPJmL."""
        return self._to_earth_data

    @input.setter
    def input(self, value):
        """Set input data."""
        self._to_earth_data = value

    @property
    def output(self):
        """Data from LPJmL."""
        return self._from_earth_data

    @output.setter
    def output(self, value):
        """Set output data."""
        self._from_earth_data = value

    @property
    def to_earth(self):
        """Alias for input (data sent to LPJmL)."""
        return self._to_earth_data

    @property
    def from_earth(self):
        """Alias for output (data from LPJmL)."""
        return self._from_earth_data

    @property
    def grid(self):
        """Grid coordinates."""
        return self._grid_data

    @grid.setter
    def grid(self, value):
        """Set grid data."""
        self._grid_data = value

    @property
    def country_code(self):
        """Country code array (ISO codes per cell)."""
        return self._country_code_data

    @country_code.setter
    def country_code(self, value):
        """Set country code data."""
        self._country_code_data = value

    @property
    def area(self):
        """Cell areas."""
        return self._area_data

    @area.setter
    def area(self, value):
        """Set area data."""
        self._area_data = value

    @property
    def model(self):
        """Get model reference."""
        return self._model

    @property
    def countries(self):
        """Get countries (social systems)."""
        return getattr(self, "_social_systems", set())

    @property
    def statistic(self):
        """Global statistics/metrics accessible from all levels.

        Provides aggregated global values that individuals/cells/countries
        can access without needing full world data. Updated once per year.

        Returns
        -------
        WorldStatistic
            Container with global metrics.
        """
        if not hasattr(self, "_statistic"):
            self._statistic = WorldStatistic(self)
        return self._statistic


class WorldStatistic:
    """Global statistics/metrics for the world.

    Computed lazily from world data. Available at all levels:
    - world.statistic
    - country.world.statistic
    - cell.world.statistic

    Add metrics as needed (e.g., mean_temperature, total_population).
    """

    def __init__(self, world):
        self._world = world
        self._cache = {}

    def _get_or_compute(self, key, compute_fn):
        """Get cached value or compute it."""
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]

    def clear_cache(self):
        """Clear cached statistics (call after world data updates)."""
        self._cache.clear()

    @property
    def ncell(self):
        """Total number of cells globally."""

        def compute():
            return (
                len(self._world.cells) if hasattr(self._world, "cells") else 0
            )

        return self._get_or_compute("ncell", compute)

    @property
    def ncountry(self):
        """Total number of countries."""

        def compute():
            return len(self._world.countries)

        return self._get_or_compute("ncountry", compute)

    def get(self, key, default=None):
        """Get a custom statistic by key."""
        return self._cache.get(key, default)

    def set(self, key, value):
        """Set a custom statistic (for model-specific global metrics)."""
        self._cache[key] = value
