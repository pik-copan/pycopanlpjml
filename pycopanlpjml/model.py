"""Model for building copan:LPJmL integrated models.

This module provides the Model class, which serves as the foundation
for integrating LPJmL (Lund-Potsdam-Jena managed Land) terrestrial earth system
model with copan:CORE based social-ecological models.

Classes
-------
Model
    Main class for building LPJmL-integrated models.

The Model handles:
- LPJmL coupler connection and configuration
- Initialization of World, Country, and Cell entities
- Parallel execution of country updates (Dask, MPI, or serial)
- Synchronization of state between workers and driver
- Neighbourhood graph construction

Example
-------
>>> import pycopanlpjml as lpjml
>>>
>>> class MyModel(lpjml.Model):
...     def __init__(self, **kwargs):
...         super().__init__(**kwargs)
...         self.world = lpjml.World(
...             to_earth=self.lpjml.read_input(copy=False),
...             from_earth=self.lpjml.read_historic_output(),
...             grid=self.lpjml.grid,
...             country_code=self.lpjml.country,
...         )
...         self.init_countries(country_class=lpjml.Country)
...         self.init_cells(cell_class=lpjml.Cell)
...
...     def update(self, t):
...         self.update_countries(t)
...         self.update_lpjml(t)
...
>>> model = MyModel(config_file="config.json")
>>> for year in model.lpjml.get_sim_years():
...     model.update(year)
"""

import os
import sys
from typing import Any, Sequence

import pandas as pd
import networkx as nx
import numpy as np
from pycoupler.config import CoupledConfig, read_config, read_yaml
from pycoupler.coupler import LPJmLCoupler
from pycoupler.utils import get_countries
from pycopancore.private._simple_expressions import unknown as _UNKNOWN

from .output import OutputCollectionMixin, read_output_table_from_zarr
from .parallel import get_executor, ActorManager
from .serial import set_dotted_path


class Model(OutputCollectionMixin):
    """Main component for building LPJmL-integrated copan:CORE models.

    This mixin class provides the infrastructure for coupling copan:CORE
    models with the LPJmL earth system model. It handles:

    - Connection to LPJmL via the pycoupler library
    - Entity initialization (World, Countries, Cells)
    - Parallel execution of country updates
    - Data exchange with LPJmL (to_earth/from_earth)
    - Neighbourhood graph construction

    Parameters
    ----------
    config_file : str, optional
        Path to the integrated model configuration file.
        Either config_file or lpjml must be provided.
    lpjml : LPJmLCoupler, optional
        Pre-configured LPJmL coupler instance.
        Either config_file or lpjml must be provided.
    lpjml_couplerversion : int, default=3
        LPJmL coupler protocol version.
    lpjml_host : str, default="localhost"
        Hostname where LPJmL is running.
    lpjml_port : int, default=2042
        Port for LPJmL communication.
    **kwargs : dict
        Additional arguments passed to parent classes.

    Attributes
    ----------
    lpjml : LPJmLCoupler
        The LPJmL coupler instance for data exchange.
    config : LPJmLConfig
        LPJmL configuration object.
    pycopanlpjml_config : CoupledConfig
        pycopanlpjml-specific configuration.
    world : World
        The World entity (set by subclass __init__).
    countries : list
        List of Country entities (set by init_countries).

    Example
    -------
    A minimal model that stops fertilization after a certain year:

    >>> class StopFertilizationModel(lpjml.Model):
    ...     def __init__(self, stop_year, **kwargs):
    ...         super().__init__(**kwargs)
    ...         self.stop_year = stop_year
    ...
    ...         self.world = lpjml.World(
    ...             to_earth=self.lpjml.read_input(copy=False),
    ...             from_earth=self.lpjml.read_historic_output(),
    ...             grid=self.lpjml.grid,
    ...             country_code=self.lpjml.country,
    ...         )
    ...         self.init_cells(cell_class=lpjml.Cell)
    ...
    ...     def update(self, t):
    ...         if t == self.stop_year:
    ...             self.world.to_earth["fertilization"].values[:] = 0
    ...         self.update_lpjml(t)

    See Also
    --------
    World : The world entity holding global data.
    Country : Country-level entities with parallel update support.
    Cell : Cell-level entities for grid-based processing.
    """

    def __init__(
        self,
        config_file=None,
        lpjml=None,
        lpjml_couplerversion=3,
        lpjml_host="localhost",
        lpjml_port=2224,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Initialize LPJmL coupler connection
        if config_file is not None:
            self.pycopanlpjml_config = self._load_pycopanlpjml_config(
                config_file
            )
            self.lpjml = LPJmLCoupler(
                config_file=config_file,
                version=lpjml_couplerversion,
                host=lpjml_host,
                port=lpjml_port,
            )
        elif lpjml is not None:
            self.lpjml = lpjml
            self.pycopanlpjml_config = self._load_pycopanlpjml_config()
        else:
            raise ValueError("Either config_file or lpjml must be provided")

        # Apply country code conversion if configured
        self._countries_as_names()
        self.config = self.lpjml.config

        # Initialize parallel executor (auto-detects environment)
        self._parallel_executor = get_executor(config=self.pycopanlpjml_config)

        # Check if actors are enabled in config (default: True)
        if hasattr(self.pycopanlpjml_config, "get"):
            parallel_cfg = self.pycopanlpjml_config.get("parallelization", {})
        elif hasattr(self.pycopanlpjml_config, "parallelization"):
            parallel_cfg = self.pycopanlpjml_config.parallelization or {}
        else:
            parallel_cfg = {}
        if hasattr(parallel_cfg, "get"):
            self._use_dask_actors = parallel_cfg.get("use_actors", True)
        elif hasattr(parallel_cfg, "use_actors"):
            self._use_dask_actors = getattr(parallel_cfg, "use_actors", True)
        else:
            self._use_dask_actors = True
        self._actor_manager = None  # Lazy initialization

        # Output store lazy initialization flag
        self._output_store_initialized = False

    # -------------------------------------------------------------------------
    # Serialization support
    # -------------------------------------------------------------------------

    def __getstate__(self):
        """Prepare component for pickling (Dask workers).

        Removes parallel executor which should not be shared across workers.
        """
        state = self.__dict__.copy()
        state["_parallel_executor"] = None
        return state

    def __setstate__(self, state):
        """Restore component after unpickling."""
        self.__dict__.update(state)
        if "_parallel_executor" not in self.__dict__:
            self._parallel_executor = None

    # -------------------------------------------------------------------------
    # Entity initialization
    # -------------------------------------------------------------------------

    def init_countries(self, country_class, world_views=None, **kwargs):
        """Initialize country entities from LPJmL grid.

        Creates a Country instance for each unique country code in the grid.
        Countries are automatically registered with the World via pycopancore.

        Parameters
        ----------
        country_class : type
            Country class to instantiate (e.g., pycopanlpjml.Country or
            subclass).
        world_views : list of str, optional
            Additional world attributes to expose as views on each country.
        **kwargs : dict
            Additional keyword arguments passed to country constructor.

        Example
        -------
        >>> self.init_countries(
        ...     country_class=MyCountry,
        ...     world_views=["to_earth"]
        ... )
        >>> print(f"Initialized {len(self.countries)} countries")
        """
        # Ensure world has a reference to this model (for config access during serialization)
        if self.world is not None:
            self.world._model = self
        
        countries = []
        country_names = _get_country_names()

        # Get unique country codes - ensure conversion to ISO if lpjml has code_to_name
        country_values = self.world.country_code.values
        if hasattr(country_values, "compute"):
            country_values = country_values.compute()
        
        # Flatten and remove NaN
        country_flat = country_values.flatten()
        valid_mask = ~np.isnan(country_flat) if np.issubdtype(country_flat.dtype, np.floating) else np.ones(len(country_flat), dtype=bool)
        unique_countries = np.unique(country_flat[valid_mask])

        for country_code in unique_countries:
            country_indices = np.where(country_values == country_code)[0]
            
            # Try to get country info - handle both ISO codes (str) and numeric codes
            country_info = None
            iso_code = country_code
            
            # If it's already an ISO code string
            if isinstance(country_code, str):
                country_info = country_names.get(country_code)
                iso_code = country_code
            # If it's numeric, it might be an LPJmL internal code - use fallback
            elif isinstance(country_code, (int, float, np.integer, np.floating)):
                # Try looking up by numeric code stringified
                iso_code = str(int(country_code))
                country_info = country_names.get(iso_code)
            
            if country_info is None:
                # Fallback: use the code itself as both name and code
                country_info = {"name": str(iso_code), "code": str(iso_code)}

            country = country_class(
                name=country_info["name"],
                code=country_info["code"],
                world=self.world,
                indices=country_indices,
                **kwargs,
            )
            # Initialize country's data copy from world
            country.init_data()
            countries.append(country)

        self.countries = countries
        # Also register with world so world.countries works
        self.world._social_systems = set(countries)

    def init_cells(self, cell_class, world_views=None, **kwargs):
        """Initialize cell entities from LPJmL grid.

        Creates a Cell instance for each grid cell. If countries are
        initialized first, cells are assigned to their respective countries.

        Parameters
        ----------
        cell_class : type
            Cell class to instantiate (e.g., pycopanlpjml.Cell or subclass).
        world_views : list of str, optional
            Additional world attributes to expose as views on each cell.
        **kwargs : dict
            Additional keyword arguments passed to cell constructor.

        Example
        -------
        >>> self.init_countries(country_class=MyCountry)
        >>> self.init_cells(cell_class=MyCell)
        >>> print(f"Total cells: {len(list(self.world.cells))}")
        """
        # Ensure world has a reference to this model (for config access during serialization)
        if self.world is not None and getattr(self.world, "_model", None) is None:
            self.world._model = self
        
        # Get neighbourhood matrix for cell connectivity
        neighbour_matrix = self.lpjml.grid.get_neighbourhood(id=False)
        world_cells = []

        # Determine country list
        if hasattr(self, "countries"):
            country_list = list(self.countries or [])
        else:
            country_list = list(self.world.countries)

        if country_list:
            # Normal case: cells grouped by country
            for country in country_list:
                if not len(country.indices):
                    continue
                cells = [
                    cell_class(
                        world=self.world,
                        country=country,
                        cell_index=global_idx,
                        local_index=local_idx,
                        # Use int index - view connection works with .values[()] assignment
                        input=country._input.isel(cell=local_idx),
                        output=country._output.isel(cell=local_idx),
                        grid=country._grid.isel(cell=local_idx),
                        area=country._area.isel(cell=local_idx),
                        **kwargs,
                    )
                    for local_idx, global_idx in enumerate(country.indices)
                ]
                country._cells = cells
                world_cells.extend(cells)
        else:
            # Fallback: cells without countries (use world directly)
            total_cells = self.lpjml.grid.shape[0]
            cells = [
                cell_class(
                    world=self.world,
                    country=None,
                    cell_index=cell_idx,
                    local_index=cell_idx,
                    # Use int index - view connection works with .values[()] assignment
                    input=self.world.input.isel(cell=cell_idx),
                    output=self.world.output.isel(cell=cell_idx),
                    grid=self.world.grid.isel(cell=cell_idx),
                    area=self.world.area.isel(cell=cell_idx),
                    **kwargs,
                )
                for cell_idx in range(total_cells)
            ]
            world_cells.extend(cells)

        # Build neighbourhood graphs
        self._assign_cell_neighbourhood(world_cells, neighbour_matrix)
        self._assign_country_neighbourhood()

    def init_worldregions(self, worldregion_class, world_views=None, **kwargs):
        """Initialize world region entities (placeholder).

        Parameters
        ----------
        worldregion_class : type
            WorldRegion class to instantiate.
        world_views : list of str, optional
            Additional world attributes to expose as views.
        **kwargs : dict
            Additional keyword arguments.
        """
        pass  # To be implemented

    # -------------------------------------------------------------------------
    # Update methods
    # -------------------------------------------------------------------------

    def update_countries(self, t):
        """Update all countries for one simulation step.

        Automatically chooses execution mode based on configuration:
        - Dask: For HPC clusters with dask-mpi
        - MPI: For pure MPI environments
        - Serial: For single-country runs or debugging

        For Dask mode, countries are serialized, sent to workers, updated,
        and state changes are synchronized back to the driver.

        Parameters
        ----------
        t : int
            Current simulation time step (year).

        Notes
        -----
        State synchronization:

        - to_earth changes are merged back into world.to_earth
        - Individual attribute changes are applied to original entities
        - from_earth is read-only (updated only by LPJmL in update_lpjml)

        Example
        -------
        >>> for year in range(2020, 2100):
        ...     self.update_countries(year)
        ...     self.update_lpjml(year)
        """
        # Optional profiling context
        try:
            from .profiling_utils import timed_context
        except ImportError:
            from contextlib import contextmanager

            @contextmanager
            def timed_context(name):
                yield

        # Get country list
        if hasattr(self, "countries"):
            countries = self.countries
        else:
            countries = list(self.world.countries)
        countries = list(countries)

        # Ensure individual indices are assigned
        self._ensure_individual_indices()

        # Execute based on parallelization mode
        mode = self._parallel_executor.config.mode
        if mode == "dask":
            # Dask actors receive from_earth data directly - no sync needed here
            self._update_countries_dask_actors(countries, t, timed_context)
        else:
            # Serial mode: sync data to countries, update, sync back
            for country in countries:
                if hasattr(country, "sync_from_world"):
                    country.sync_from_world()
            with timed_context(f"update_countries_serial_year_{t}"):
                for country in countries:
                    country.update(t)
            for country in countries:
                if hasattr(country, "sync_to_world"):
                    country.sync_to_world()

        # Synchronization barrier (MPI only)
        self._parallel_executor.barrier()

    def _update_countries_dask_actors(self, countries, t, timed_context):
        """Execute country updates using Dask Actors (persistent workers).

        Actors persist across years, eliminating re-serialization overhead.
        Only from_earth data is sent each year, and only deltas are returned.

        This is significantly faster than the task-based approach for:
        - Large countries (many cells/farmers)
        - Long simulations (many years)
        - Complex models (large state per entity)

        Parameters
        ----------
        countries : list
            List of Country instances to update.
        t : int
            Current simulation year.
        timed_context : callable
            Context manager for profiling.
        """

        # Initialize actor manager on first call
        if not hasattr(self, "_actor_manager") or self._actor_manager is None:
            client = self._parallel_executor.config._client
            self._actor_manager = ActorManager(client, countries)
            with timed_context("deploy_actors"):
                self._actor_manager.deploy()

        # Get from_earth data to send to workers
        from_earth = getattr(self.world, "from_earth", None)

        # Prepare world stats update for workers
        # Sync all world.statistic entries to workers for cross-entity learning
        world_stats_update = None
        if hasattr(self.world, "statistic"):
            stats_cache = getattr(self.world.statistic, "_cache", {})
            if stats_cache:
                world_stats_update = dict(stats_cache)

        # Update all actors (minimal data transfer)
        with timed_context(f"actor_update_year_{t}"):
            results = self._actor_manager.update_all(t, from_earth, world_stats_update)

        # Merge results back into world (skip empty updates)
        # Build lookup for country stats sync
        countries_by_code = {c.country_code: c for c in countries}

        for result in results:
            if not result:
                continue

            # Handle both old (3-tuple) and new (4-tuple) return formats
            if len(result) == 4:
                cell_indices, updated_to_earth, updated_individuals, country_stats = result
            else:
                cell_indices, updated_to_earth, updated_individuals = result
                country_stats = None

            # Skip if no actual changes
            has_to_earth = (
                updated_to_earth is not None and len(updated_to_earth) > 0
            )  # noqa: E501
            has_individuals = (
                updated_individuals is not None
                and updated_individuals.get("values")
            )

            if not has_to_earth and not has_individuals and not country_stats:
                continue

            # Only convert cell_indices if we have to_earth updates
            if has_to_earth:
                cell_indices = np.asarray(cell_indices, dtype=int)
                for var_name, values in updated_to_earth.items():
                    if var_name in self.world.to_earth.data_vars:
                        self.world.to_earth[var_name].values[
                            cell_indices
                        ] = values  # noqa: E501

            # Apply individual updates (generic for any Individual type)
            if has_individuals:
                indices = updated_individuals.get("indices")
                values_dict = updated_individuals.get("values", {})
                if indices is not None and values_dict:
                    self._apply_individual_updates(indices, values_dict)

            # Sync country stats back to driver's country.statistic
            if country_stats:
                country_code = country_stats.get("country_code")
                cache = country_stats.get("cache", {})
                if country_code and cache:
                    driver_country = countries_by_code.get(country_code)
                    if driver_country and hasattr(driver_country, "statistic"):
                        for key, value in cache.items():
                            driver_country.statistic.set(key, value)

    def update_lpjml(self, t):
        """Exchange data with LPJmL for one simulation step.

        Sends to_earth data to LPJmL, receives from_earth data back,
        and updates time coordinates.

        Parameters
        ----------
        t : int
            Current simulation time step (year).

        Notes
        -----
        - to_earth is sent to LPJmL as input for the next step
        - from_earth is updated with LPJmL output
        - Time coordinates are updated on both datasets
        - Connection is closed after the last year

        Example
        -------
        >>> def update(self, t):
        ...     self.update_countries(t)  # Run social dynamics
        ...     self.update_lpjml(t)      # Exchange with LPJmL
        """
        # Update to_earth time coordinate
        self.world.to_earth.time.values[0] = np.datetime64(f"{t+1}", "Y")

        if not hasattr(sys, "_called_from_test"):
            # Send to_earth to LPJmL
            to_earth_data = (
                self.world.to_earth.to_xarray()
                if hasattr(self.world.to_earth, "to_xarray")
                else self.world.to_earth
            )
            self.lpjml.send_input(to_earth_data, t)

            # Receive from_earth from LPJmL
            for name, from_earth_var in self.lpjml.read_output(t).items():
                world_from_earth = self.world.from_earth[name]
                if hasattr(world_from_earth, "to_xarray"):
                    world_from_earth = world_from_earth.to_xarray()
                new_data = from_earth_var[:].values
                self.world._from_earth_data[name].values[:] = new_data

            # Update from_earth time coordinate
            self.world._from_earth_data.time.values[:] = np.array(
                [
                    np.datetime64(f"{year}-12-31")
                    for year in range(
                        t + 1 - len(self.world.from_earth.time), t + 1
                    )  # noqa: E501
                ]
            )

            # Close connection after last year
            if t == self.lpjml.config.lastyear:
                self.lpjml.close()
        else:
            # Test mode: only update time coordinates
            self.world._from_earth_data.time.values[:] = np.array(
                [
                    np.datetime64(f"{year}-12-31")
                    for year in range(
                        t + 1 - len(self.world.from_earth.time), t + 1
                    )  # noqa: E501
                ]
            )

    # -------------------------------------------------------------------------
    # Individual synchronization
    # -------------------------------------------------------------------------

    def _apply_individual_updates(self, indices, values_dict):
        """Apply individual-level updates returned from workers."""
        if not hasattr(self, "_individual_index_map"):
            self._individual_index_map = self._build_individual_index_map()

        individual_map = self._individual_index_map
        indices = np.asarray(indices, dtype=np.int64)

        for attr_name, attr_values in values_dict.items():
            for idx, value in zip(indices, attr_values):
                individual = individual_map.get(int(idx))
                if individual is not None:
                    set_dotted_path(individual, attr_name, value)

    def _ensure_individual_indices(self) -> None:
        """Ensure every individual has a stable index for synchronization."""
        self._individual_index_map = self._build_individual_index_map()

    def _build_individual_index_map(self):
        """Build lookup from stable individual index to entity instance.

        Returns
        -------
        dict
            {int: Individual} mapping from index to entity.
        """
        individual_map = {}
        next_index = getattr(self, "_next_individual_index", 0)

        def _as_list(value):
            if value is None or value is _UNKNOWN:
                return []
            try:
                return list(value)
            except TypeError:
                return []

        seen_ids: set[int] = set()
        collected: list[Any] = []

        def _add_candidates(candidates):
            for individual in _as_list(candidates):
                if individual is _UNKNOWN or individual is None:
                    continue
                ident = id(individual)
                if ident in seen_ids:
                    continue
                seen_ids.add(ident)
                collected.append(individual)

        # Collect from world
        world = getattr(self, "world", None)
        if world is not None:
            _add_candidates(getattr(world, "_individuals", None))
            _add_candidates(getattr(world, "_direct_individuals", None))
            try:
                _add_candidates(getattr(world, "individuals", None))
            except Exception:
                pass

        # Collect from countries and cells
        containers = []
        if hasattr(self, "countries"):
            containers = _as_list(self.countries)
        elif world is not None:
            containers = _as_list(getattr(world, "countries", None))

        for country in containers:
            _add_candidates(getattr(country, "_individuals", None))
            _add_candidates(getattr(country, "_direct_individuals", None))
            try:
                _add_candidates(country.get_individuals())
            except Exception:
                _add_candidates(getattr(country, "individuals", None))

            cell_candidates = getattr(country, "_direct_cells", None)
            if not cell_candidates:
                cell_candidates = getattr(country, "cells", None)

            for cell in _as_list(cell_candidates):
                _add_candidates(getattr(cell, "_individuals", None))
                _add_candidates(getattr(cell, "_direct_individuals", None))
                try:
                    _add_candidates(cell.get_individuals())
                except Exception:
                    _add_candidates(getattr(cell, "individuals", None))

        # Collect from model-level caches
        _add_candidates(getattr(self, "_individuals", None))
        _add_candidates(getattr(self, "individuals", None))
        _add_candidates(getattr(self, "_individual_entities", None))

        # Assign indices
        for individual in collected:
            idx = getattr(individual, "_individual_index", None)
            if idx is None:
                idx = next_index
                next_index += 1
                try:
                    setattr(individual, "_individual_index", int(idx))
                except Exception:
                    continue
            individual_map[int(idx)] = individual

        self._next_individual_index = next_index
        return individual_map

    # -------------------------------------------------------------------------
    # Neighbourhood graph construction
    # -------------------------------------------------------------------------

    def _assign_cell_neighbourhood(self, cells, neighbour_matrix):
        """Build cell neighbourhood graph from LPJmL grid.

        Parameters
        ----------
        cells : list
            List of Cell entities.
        neighbour_matrix : xarray.DataArray
            Neighbour indices from lpjml.grid.get_neighbourhood().
            Uses GLOBAL cell indices (not local list positions).
        """
        self.world.cell_neighbourhood = nx.Graph()
        self.world.cell_neighbourhood.add_nodes_from(cells)

        # Build lookup: global_cell_index -> cell object
        cell_by_global_index = {
            getattr(cell, "_cell_index", None): cell
            for cell in cells
            if getattr(cell, "_cell_index", None) is not None
        }

        # Build edges using GLOBAL indices (not enumeration position!)
        for cell in cells:
            global_idx = getattr(cell, "_cell_index", None)
            if global_idx is None:
                continue
            
            # Look up neighbours using cell's GLOBAL index
            neighbour_indices = neighbour_matrix.isel({"cell": global_idx}).values
            
            for neighbour_idx in neighbour_indices:
                if neighbour_idx >= 0:
                    neighbour_cell = cell_by_global_index.get(neighbour_idx)
                    if neighbour_cell is not None:
                        self.world.cell_neighbourhood.add_edge(cell, neighbour_cell)

        # Clear lookup to free memory
        del cell_by_global_index

        for cell in cells:
            cell.neighbourhood = set(
                self.world.cell_neighbourhood.neighbors(cell)
            )

    def _assign_country_neighbourhood(self):
        """Build country neighbourhood graph from cell connections."""
        self.world.country_neighbourhood = nx.Graph()
        self.world.country_neighbourhood.add_nodes_from(self.world.countries)

        for cell1, cell2 in self.world.cell_neighbourhood.edges:
            country1 = getattr(cell1, "country", None)
            country2 = getattr(cell2, "country", None)
            if country1 and country2 and country1 is not country2:
                self.world.country_neighbourhood.add_edge(country1, country2)

        for country in self.world.countries:
            country.neighbourhood = set(
                self.world.country_neighbourhood.neighbors(country)
            )

    # -------------------------------------------------------------------------
    # Output table (delegates to world)
    # -------------------------------------------------------------------------

    @property
    def output_table(self):
        """Get output table (all years from Zarr, or last year from memory).

        Returns long-format DataFrame (year, cell, variable, value, ...).
        When Zarr store exists, flushes pending writes and reads full table.
        Otherwise returns last collected year from world.output_table.
        """
        world = getattr(self, "world", None)
        if world is None:
            return pd.DataFrame()

        store_path = getattr(world, "_output_store_path", None)
        if store_path:
            try:
                self._flush_pending_zarr(force=True)
                return read_output_table_from_zarr(store_path)
            except Exception:
                pass

        return world.output_table

    # -------------------------------------------------------------------------
    # Configuration helpers
    # -------------------------------------------------------------------------

    # Path to default pycopanlpjml config
    _DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

    def _load_pycopanlpjml_config(self, config_file=None):
        """Load pycopanlpjml configuration.

        Priority:
        1. JSON (pycoupler): If has pycopanlpjml sections, validate them
        2. YAML fallback: Load from pycopanlpjml/config.yaml

        If JSON config doesn't have pycopanlpjml sections (parallelization,
        output.format), falls back to YAML defaults.

        Parameters
        ----------
        config_file : str, optional
            Path to main config file (JSON from pycoupler or YAML).

        Returns
        -------
        CoupledConfig
            Configuration object.
        """
        # Try JSON first (from pycoupler)
        if config_file and os.path.exists(config_file):
            lower = config_file.lower()
            if lower.endswith(".json") or lower.endswith(".cjson"):
                try:
                    lpjml_config = read_config(config_file, to_dict=False)
                    config = getattr(lpjml_config, "coupled_config", None)
                    if config is not None:
                        config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
                        if self._has_pycopanlpjml_sections(config_dict):
                            self._validate_pycopanlpjml_config(config_dict)
                            return config
                except Exception:
                    pass

        # Fall back to YAML (pycopanlpjml defaults)
        return self._load_config_from_yaml(config_file)

    def _get_required_config_structure(self):
        """Get required config structure from pycopanlpjml/config.yaml."""
        default_config = read_yaml(self._DEFAULT_CONFIG_PATH, CoupledConfig)
        config_dict = default_config.to_dict()
        required = {}
        for section, content in config_dict.items():
            if isinstance(content, dict):
                required[section] = list(content.keys())
        return required

    def _has_pycopanlpjml_sections(self, config_dict):
        """Check if config has pycopanlpjml-specific sections."""
        required = self._get_required_config_structure()
        return any(section in config_dict for section in required)

    def _load_config_from_yaml(self, config_file=None):
        """Load configuration from YAML file."""
        config_paths = []
        if config_file:
            config_dir = os.path.dirname(config_file)
            config_paths.append(os.path.join(config_dir, "config.yaml"))

        config_paths.extend([self._DEFAULT_CONFIG_PATH, "config.yaml"])

        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    return read_yaml(config_path, CoupledConfig)
                except Exception:
                    continue

        raise FileNotFoundError(
            f"No pycopanlpjml config.yaml found. Searched: {config_paths}"
        )

    def _validate_pycopanlpjml_config(self, config_dict):
        """Validate pycopanlpjml config has required elements from config.yaml."""
        required = self._get_required_config_structure()
        missing = []
        for section, required_keys in required.items():
            if section not in config_dict:
                missing.append(f"'{section}' section")
            elif isinstance(config_dict[section], dict):
                for key in required_keys:
                    if key not in config_dict[section]:
                        missing.append(f"'{section}.{key}'")

        if missing:
            raise ValueError(
                f"Invalid pycopanlpjml config - missing: {', '.join(missing)}. "
                f"See {self._DEFAULT_CONFIG_PATH} for required elements."
            )

    def _countries_as_names(self):
        """Convert country codes to names if configured."""
        if (
            self.lpjml.config.coupled_config.lpjml_settings.country_code_to_name  # noqa: E501
        ):  # noqa: E501
            self.lpjml.code_to_name(True)

    def _create_views_dict(self, source, views, indices):
        """Create views dictionary from source entity.

        Parameters
        ----------
        source : World or Country
            Source entity with xarray data.
        views : list of str
            Attribute names to create views from.
        indices : array-like
            Cell indices for the view.

        Returns
        -------
        dict
            {view_name: view_object} mapping.
        """
        if not views:
            return {}

        result = {}
        for view_name in views:
            attr = getattr(source, view_name, None)
            if attr is not None and hasattr(attr, "isel"):
                result[view_name] = attr.isel({"cell": indices}, drop=False)
        return result


# =============================================================================
# MODULE-LEVEL HELPERS
# =============================================================================


def _get_country_names():
    """Get mapping of country codes to name/code dicts.

    Returns
    -------
    dict
        {code: {"name": str, "code": str}} mapping.
    """
    return {
        value["code"]: {"name": value["name"], "code": value["code"]}
        for key, value in get_countries().items()
    }
