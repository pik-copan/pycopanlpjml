"""Lightweight region implementations for copan:LPJmL.

Architecture:
- World owns global data
- Countries COPY their slice from World (sync once per year)
- Cells have VIEWS into their Country's data (shared memory within country)
"""

import numpy as np
import pycopancore.model_components.base.implementation as base
from pycopancore.private._simple_expressions import unknown

from pycopanlpjml.output import OutputDefinitionMixin
from pycopanlpjml.serial import (
    STATE_ALLOWLIST,
    is_non_serializable,
    contains_non_serializable_references,
    create_clone_structure,
    update_cached_clones,
    rebuild_neighbourhood_graphs,
    fix_agent_back_references,
    restore_individual_cell,
)
from pycopanlpjml.world import _StatisticSnapshot, _ConfigView


class _ModelView:
    """Lightweight model view for workers providing lpjml-like access.
    
    Mirrors the LPJmLCoupler interface for code that uses model.lpjml.*
    """

    def __init__(self, config=None, sim_year=None, ncell=None,
                 firstyear=None, lastyear=None, start_coupling=None):
        self.config = config
        self._sim_year = sim_year
        self._ncell = ncell
        self._firstyear = firstyear
        self._lastyear = lastyear
        self._start_coupling = start_coupling
        # Provide lpjml-like access: model.lpjml.sim_year works
        self.lpjml = self

    @property
    def sim_year(self):
        """Current simulation year."""
        return self._sim_year

    @property
    def ncell(self):
        """Number of LPJmL cells."""
        return self._ncell

    @property
    def sim_years(self):
        """List of all simulation years."""
        return list(self.get_sim_years())

    @property
    def historic_years(self):
        """List of all historic years."""
        return list(self.get_historic_years(match_period=False))

    @property
    def coupled_years(self):
        """List of all coupled years."""
        return list(self.get_coupled_years(match_period=False))

    def get_sim_years(self):
        """Generator for all simulation years."""
        if self._sim_year is None or self._lastyear is None:
            return
        current = self._sim_year
        while current <= self._lastyear:
            yield current
            current += 1

    def get_historic_years(self, match_period=True):
        """Generator for all historic years."""
        if self._firstyear is None or self._start_coupling is None:
            return
        if match_period and self._firstyear >= self._start_coupling:
            raise ValueError(
                f"No historic years available. Simulated year {self._firstyear} "
                f"is >= coupled year {self._start_coupling}."
            )
        current = self._firstyear
        while current < self._start_coupling:
            yield current
            current += 1

    def get_coupled_years(self, match_period=True):
        """Generator for all coupled years."""
        if self._sim_year is None or self._lastyear is None or self._start_coupling is None:
            return
        if match_period and self._sim_year < self._start_coupling:
            raise ValueError(
                f"Historic years left. Simulated year {self._sim_year} "
                f"is < coupled_year {self._start_coupling}."
            )
        current = self._sim_year
        while current <= self._lastyear:
            yield current
            current += 1


class _LocalWorldView:
    """Lightweight world view for workers.

    Contains ONLY global statistics and config - NO data arrays.
    Country owns its data arrays directly; cells access them through country.
    This avoids serializing the full World with all its large arrays.
    """

    def __init__(self, world=None, model_config=None, lpjml_snapshot=None):
        # Snapshot statistics (detached from World) - always create valid object
        world_statistic = getattr(world, "statistic", None)
        self._statistic = _StatisticSnapshot(world_statistic)

        # Config view (minimal copy, no back-references)
        world_config = getattr(world, "config", None) if world else None
        self.config = _ConfigView(world_config) if world_config else None

        # Model view for code that accesses model.lpjml.* 
        lpjml = lpjml_snapshot or {}
        self._model = _ModelView(
            config=model_config,
            sim_year=lpjml.get("sim_year"),
            ncell=lpjml.get("ncell"),
            firstyear=lpjml.get("firstyear"),
            lastyear=lpjml.get("lastyear"),
            start_coupling=lpjml.get("start_coupling"),
        )

    @property
    def statistic(self):
        """Global statistics snapshot (no World reference)."""
        return self._statistic

    @property
    def model(self):
        """Provide model access for entities."""
        return self._model


class Region(base.SocialSystem, OutputDefinitionMixin):
    """Base class for LPJmL-integrated regions.

    Countries COPY data from World. Cells get VIEWS into Country data.
    
    Inherits from OutputDefinitionMixin to support output variable collection
    for country-level outputs.
    """

    def __init__(
        self,
        world=None,
        indices=None,
        **kwargs,
    ):
        super().__init__(world=world, **kwargs)
        self._world = world
        self.indices = np.asarray(indices) if indices is not None else np.array([])
        self._cells = []
        self.neighbourhood = set()

        # Country's OWN COPY of data (sliced from world)
        self._input = None
        self._output = None
        self._grid = None
        self._area = None

    @property
    def world(self):
        """World reference (real World on driver, _LocalWorldView on workers)."""
        return self._world

    @world.setter
    def world(self, value):
        """Set world reference (used by base class __init__)."""
        self._world = value

    @property
    def model(self):
        """Access to Model instance for output collection.

        Traverses world -> model hierarchy to provide model access
        needed by OutputDefinitionMixin.get_defined_outputs().
        """
        if hasattr(self, "_model") and self._model is not None:
            return self._model
        if self._world is not None:
            return getattr(self._world, "_model", None) or getattr(self._world, "model", None)
        return None

    def init_data(self):
        """Copy data slice from world. Call after country is created."""
        if self._world is None or len(self.indices) == 0:
            return
            
        # COPY (not view) the data slices from world
        self._input = self._world.input.isel(cell=self.indices).copy()
        self._output = self._world.output.isel(cell=self.indices).copy()
        self._grid = self._world.grid.isel(cell=self.indices).copy()
        self._area = self._world.area.isel(cell=self.indices).copy()

    def sync_from_world(self):
        """Sync data FROM world TO country (start of year). Driver-only."""
        if self._world is None:
            raise RuntimeError("sync_from_world requires world reference (driver-only)")
        if len(self.indices) == 0:
            raise RuntimeError("sync_from_world requires cell indices")

        # Copy output from world (LPJmL results)
        if self._world.output is not None and self._output is not None:
            for var in self._output.data_vars:
                if var in self._world.output.data_vars:
                    self._output[var].values[:] = self._world.output[var].values[self.indices]

    def sync_to_world(self):
        """Sync data FROM country TO world (end of year). Driver-only."""
        if self._world is None:
            raise RuntimeError("sync_to_world requires world reference (driver-only)")
        if len(self.indices) == 0:
            raise RuntimeError("sync_to_world requires cell indices")

        # Copy input changes back to world (farmer decisions)
        if self._input is not None and self._world.input is not None:
            for var in self._input.data_vars:
                if var in self._world.input.data_vars:
                    self._world.input[var].values[self.indices] = self._input[var].values

    @property
    def input(self):
        """Country's copy of input data."""
        return self._input

    @property
    def output(self):
        """Country's copy of output data."""
        return self._output

    @property
    def to_earth(self):
        """Alias for input."""
        return self._input

    @property
    def from_earth(self):
        """Alias for output."""
        return self._output

    @property
    def grid(self):
        """Country's copy of grid data."""
        return self._grid

    @property
    def area(self):
        """Country's copy of area data."""
        return self._area

    @property
    def cells(self):
        """Cells in this region."""
        return self._cells

    @cells.setter
    def cells(self, value):
        """Set cells for this region (called by pycopancore)."""
        # pycopancore passes 'unknown' (_Unknown object) during initialization
        if value == "unknown" or (hasattr(value, "__class__") and "Unknown" in value.__class__.__name__):
            return
        if isinstance(value, (list, tuple, set)):
            self._cells = list(value)
        else:
            self._cells.append(value) if value not in self._cells else None

    @property
    def individuals(self):
        """All individuals in this region."""
        inds = set()
        for cell in self._cells:
            inds.update(getattr(cell, "_individuals", set()))
        return inds

    @individuals.setter
    def individuals(self, value):
        """Set individuals for this region (called by pycopancore).
        
        Since individuals are stored per-cell, this is a no-op for 'unknown'.
        For actual values, we store them in _direct_individuals.
        """
        # pycopancore passes 'unknown' (_Unknown object) during initialization
        if value == "unknown" or (hasattr(value, "__class__") and "Unknown" in value.__class__.__name__):
            return
        if not hasattr(self, "_direct_individuals"):
            self._direct_individuals = set()
        if isinstance(value, (list, tuple, set)):
            self._direct_individuals.update(value)
        else:
            self._direct_individuals.add(value)

    def _get_or_create_serialization_cache(self):
        """Get or create cached clone structure for fast serialization.

        On first call, creates cloned cells/individuals and caches the structure.
        On subsequent calls, returns the cached structure for fast updates.
        """
        cache = getattr(self, "_serialization_cache", None)
        if cache is not None:
            return cache

        # First time: build full clone structure
        original_cells = set(self._cells or [])

        # Get individuals from cells
        original_individuals = set()
        for cell in original_cells:
            original_individuals.update(getattr(cell, "_individuals", set()))

        # Create clone structure using serialization module
        cell_map, individual_map, cloned_cells, cloned_individuals = (
            create_clone_structure(original_cells, original_individuals)
        )

        cache = {
            "cell_map": cell_map,
            "individual_map": individual_map,
            "cloned_cells": cloned_cells,
            "cloned_individuals": cloned_individuals,
        }
        self._serialization_cache = cache
        return cache

    def __getstate__(self):
        """Serialize region for Dask workers.

        Uses cached clone structure for performance. On first call, creates
        full clones. On subsequent calls, only updates dynamic attributes.
        """
        # Get model config and lpjml snapshot for workers
        model_config = None
        lpjml_snapshot = {}
        if self._world is not None:
            model = getattr(self._world, "_model", None) or getattr(self._world, "model", None)
            if model is not None:
                model_config = getattr(model, "config", None)
                lpjml = getattr(model, "lpjml", None)
                if lpjml is not None:
                    lpjml_snapshot["sim_year"] = getattr(lpjml, "sim_year", None)
                    lpjml_snapshot["ncell"] = getattr(lpjml, "ncell", None)
                    # Get config values for year generators
                    lpjml_config = getattr(lpjml, "config", None)
                    if lpjml_config is not None:
                        lpjml_snapshot["firstyear"] = getattr(lpjml_config, "firstyear", None)
                        lpjml_snapshot["lastyear"] = getattr(lpjml_config, "lastyear", None)
                        lpjml_snapshot["start_coupling"] = getattr(lpjml_config, "start_coupling", None)

        # Create lightweight world view with snapshotted statistic/config only
        local_world = _LocalWorldView(world=self._world, model_config=model_config, lpjml_snapshot=lpjml_snapshot)

        # Get model view for setting on cloned entities
        model_view = local_world._model

        # Get or create cached clone structure
        cache = self._get_or_create_serialization_cache()
        cell_map = cache["cell_map"]
        individual_map = cache["individual_map"]

        # Update cached clones with current dynamic values
        update_cached_clones(cell_map, individual_map, local_world, model_view)

        # Build state dict (excluding circular ref collections)
        exclude_keys = {
            "_world", "world", "neighbourhood", "_cells", "_direct_cells",
            "_next_lower_social_systems", "_higher_social_systems",
            "_next_higher_social_system", "_direct_individuals",
            "_individuals", "_model", "model", "_config", "config",
            "_serialization_cache",  # Don't serialize the cache itself
            "_farmers_cache",  # Points to original farmers, not clones
        }

        state = {}
        for key, value in self.__dict__.items():
            if key in exclude_keys:
                continue
            if is_non_serializable(value):
                continue
            state[key] = value

        # Use cached clones
        cloned_cells = cache["cloned_cells"]
        cloned_individuals = cache["cloned_individuals"]

        state["_world"] = local_world
        state["_cells"] = list(cloned_cells)
        state["_direct_cells"] = set(cloned_cells)
        state["_next_lower_social_systems"] = set(cloned_cells)
        state["_direct_individuals"] = cloned_individuals
        state["_individuals"] = set(cloned_individuals)

        # Country's data arrays are serialized directly (they're xarray copies)
        # Note: _input, _output, _grid, _area are NOT in exclude_keys
        # so they get copied above from self.__dict__

        # Convert numpy arrays to lists (faster serialization)
        if "indices" in state and isinstance(state["indices"], np.ndarray):
            state["indices"] = state["indices"].tolist()

        # Sanitize state: replace non-serializable values with 'unknown'
        for key, value in list(state.items()):
            if key in STATE_ALLOWLIST:
                continue
            if contains_non_serializable_references(value):
                state[key] = unknown

        return state

    def __setstate__(self, state):
        """Restore region after unpickling.

        Rebuilds object references and neighbourhoods that were converted
        to indices during serialization.
        """
        import numpy as np

        self.__dict__.update(state)

        if hasattr(self, "indices") and isinstance(self.indices, list):
            self.indices = np.array(self.indices)

        # Data arrays (_input, _output, _grid, _area) are restored directly
        # from state dict above - they're serialized on the country, not _LocalWorldView

        # Get local_world for statistic/config access
        local_world = getattr(self, "_world", None)

        defaults = {
            "neighbourhood": set(),
            "_direct_cells": set(),
            "_next_lower_social_systems": set(),
            "_higher_social_systems": None,
            "_next_higher_social_system": None,
            "_direct_individuals": set(),
            "_individuals": set(),
        }
        for key, default in defaults.items():
            if not hasattr(self, key):
                setattr(self, key, default)

        # Get model view for config access
        model_view = getattr(local_world, "_model", None) if local_world else None

        # Build cell lookup for relinking individuals
        cells = set(getattr(self, "_cells", []))
        cell_lookup = {}
        for cell in cells:
            key = (getattr(cell, "_uid", None), getattr(cell, "_cell_index", None))
            cell_lookup[key] = cell

        # Rebind cells to country
        all_individuals = set()
        for idx, cell in enumerate(cells):
            cell._social_system = self
            cell._social_systems = [self]
            cell._world = local_world
            if model_view is not None:
                cell._model = model_view

            # CRITICAL: Reconnect cell views to country's data arrays
            # Without this, cells have stale views from the driver
            # Use int index - view connection works with .values[()] assignment
            local_idx = getattr(cell, "_local_index", idx)
            if self._input is not None and hasattr(self._input, "isel"):
                cell.input = self._input.isel(cell=local_idx)
            if self._output is not None and hasattr(self._output, "isel"):
                cell.output = self._output.isel(cell=local_idx)
            if self._grid is not None and hasattr(self._grid, "isel"):
                cell.grid = self._grid.isel(cell=local_idx)
            if self._area is not None and hasattr(self._area, "isel"):
                cell.area = self._area.isel(cell=local_idx)

            if not hasattr(cell, "_individuals") or cell._individuals is None:
                cell._individuals = set()

            individuals = set(cell._individuals)
            for individual in individuals:
                restore_individual_cell(
                    individual, fallback_cell=cell, cell_lookup=cell_lookup
                )
                individual._world = local_world
                individual._social_system = self
                individual._social_systems = [self]
                if model_view is not None:
                    individual._model = model_view
                if not hasattr(individual, "neighbourhood") or individual.neighbourhood is None:
                    individual.neighbourhood = []
                # Fix behaviour.agent back-reference (crucial for TPB!)
                fix_agent_back_references(individual)

            cell._individuals = individuals
            cell._direct_individuals = set(individuals)
            all_individuals.update(individuals)

        # Update country-level individual caches
        self._direct_individuals = set(all_individuals)
        self._individuals = set(all_individuals)
        self._direct_cells = cells
        self._next_lower_social_systems = set()

        if model_view is not None:
            self._model = model_view

        # Rebuild neighbourhood graphs from stored indices
        rebuild_neighbourhood_graphs(cells, all_individuals)


class Country(Region):
    """A region representing a single country.

    Parameters
    ----------
    name : str
        Country name.
    code : str
        ISO 3-letter country code.
    world : World
        World instance.
    indices : array-like
        Cell indices for this country.
    """

    def __init__(
        self,
        name=None,
        code=None,
        world=None,
        indices=None,
        **kwargs,
    ):
        super().__init__(world=world, indices=indices, **kwargs)
        self.name = name
        self.code = code

    @property
    def country_code(self):
        """ISO 3-letter country code."""
        return self.code

    @property
    def _cell_indices(self):
        """Backward compatibility alias for indices."""
        return self.indices

    @property
    def statistic(self):
        """Country-level statistics/metrics accessible from all levels.

        Provides aggregated country-level values that individuals/cells
        can access without needing full country data. Updated once per year.

        Returns
        -------
        CountryStatistic
            Container with global metrics.
        """
        if not hasattr(self, "_statistic"):
            self._statistic = CountryStatistic(self)
        return self._statistic

class CountryStatistic:
    """Country-level statistics/metrics .

    Computed lazily from country data. Available at all levels:
    - country.statistic
    - cell.country.statistic
    - world.countries[country_code].statistic

    Add metrics as needed (e.g., mean_temperature, total_population).
    """

    def __init__(self, country):
        self._cache = {}

    def _get_or_compute(self, key, compute_fn):
        """Get cached value or compute it."""
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]

    def clear_cache(self):
        """Clear cached statistics (call after world data updates)."""
        self._cache.clear()

    def get(self, key, default=None):
        """Get a custom statistic by key."""
        return self._cache.get(key, default)

    def set(self, key, value):
        """Set a custom statistic (for country-level metrics)."""
        self._cache[key] = value

class WorldRegion(Region):
    """A region representing a group of countries (e.g., EU, G7)."""

    def __init__(self, name=None, code=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.code = code
