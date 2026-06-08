"""World entity type for copan:LPJmL component."""

import numpy as np
import networkx as nx
import xarray as xr
import pycopancore.model_components.base.implementation as base


class World(base.World):
    """An LPJmL-integrating world entity.

    World entity holds LPJmL data as the global source of truth.
    Countries get copies of relevant data, cells get views into countries.

    Parameters
    ----------
    input : LPJmLDataSet
        Coupled LPJmL model inputs.
    output : LPJmLDataSet
        Coupled LPJmL model outputs.
    grid : LPJmLData
        Grid of the LPJmL model.
    country : LPJmLData
        Countries of each cell as country code.
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
    country : LPJmLData
        Country codes per cell.
    area : LPJmLData
        Cell areas in m².
    neighbourhood : nx.Graph
        Cell neighbourhood graph.
    country_neighbourhood : nx.Graph
        Country neighbourhood graph.

    Examples
    --------
    >>> world = World(
    ...     input=lpjml.read_input(copy=False),
    ...     output=lpjml.read_historic_output(),
    ...     grid=lpjml.grid,
    ...     country=lpjml.country,
    ...     area=lpjml.terr_area,
    ... )
    """

    def __init__(
        self,
        input=None,
        output=None,
        grid=None,
        country=None,
        country_code=None,
        area=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store model reference if provided
        self._model = kwargs.get("model", None)

        # Internal data storage (model.py accesses these directly)
        self._to_earth_data = None
        self._from_earth_data = None
        self._grid_data = None
        self._country_data = None
        self._area_data = None

        # Direct data storage
        if input is not None:
            self._to_earth_data = input
            # Set initial time if model available
            if self._model and hasattr(self._model, "lpjml"):
                self._to_earth_data.time.values[0] = np.datetime64(
                    f"{self._model.lpjml.sim_year}-12-31"
                )

        if output is not None:
            self._from_earth_data = output

        if grid is not None:
            self._grid_data = grid
            # Initialize neighbourhood graph
            self.neighbourhood = nx.Graph()

        # Accept either country or country_code (backward compatibility)
        if country is not None:
            self._country_data = country
        elif country_code is not None:
            self._country_data = country_code

        self._area_data = area

        # Country neighbourhood graph
        self.country_neighbourhood = nx.Graph()

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
    def country(self):
        """Country codes per cell."""
        return self._country_data

    @country.setter
    def country(self, value):
        """Set country data."""
        self._country_data = value

    @property
    def country_code(self):
        """Alias for country (backward compatibility)."""
        return self._country_data

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
    - country._world.statistic (via _LocalWorldView)
    - cell.model.world.statistic

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
            return len(self._world.cells) if hasattr(self._world, 'cells') else 0
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


class _StatisticSnapshot:
    """Snapshot of WorldStatistic for workers.

    Captures current values at serialization time. Workers see
    slightly stale global stats (one sync cycle lag) but this is
    acceptable for most use cases (social norms, global context).
    """

    def __init__(self, world_statistic=None):
        if world_statistic is None:
            self._cache = {}
        else:
            # Copy all cached values
            self._cache = dict(world_statistic._cache)
            # Also capture computed properties
            self._cache["ncell"] = world_statistic.ncell
            self._cache["ncountry"] = world_statistic.ncountry

    @property
    def ncell(self):
        """Total number of cells globally."""
        return self._cache.get("ncell", 0)

    @property
    def ncountry(self):
        """Total number of countries."""
        return self._cache.get("ncountry", 0)

    def get(self, key, default=None):
        """Get a custom statistic by key."""
        return self._cache.get(key, default)


class _ConfigView:
    """Minimal config view for workers (no model back-references).

    Only contains coupled_config for output variable definitions.
    """

    def __init__(self, config):
        self.sim_path = getattr(config, "sim_path", None)
        self.sim_name = getattr(config, "sim_name", None)
        self.coupled_config = getattr(config, "coupled_config", None)
