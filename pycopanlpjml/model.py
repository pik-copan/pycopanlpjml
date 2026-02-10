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
from .parallelization import get_executor
from .serialization import (
    serialize_country_for_worker,
    sync_world,
    sync_world_batch,
)


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
        lpjml_port=2042,
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
        countries = []
        country_names = _get_country_names()

        # Get unique country codes
        country_values = self.world.country_code.values
        if hasattr(country_values, "compute"):
            country_values = country_values.compute()

        unique_countries = np.unique(country_values)

        for country_code in unique_countries:
            country_indices = np.where(country_values == country_code)[0]

            countries.append(
                country_class(
                    name=country_names[country_code]["name"],
                    code=country_names[country_code]["code"],
                    world=self.world,
                    grid=country_indices,
                    **self._create_views_dict(self.world, world_views, country_indices),  # noqa: E501
                    **kwargs,
                )
            )

        self.countries = countries

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
                cell_indices = getattr(country, "_cell_indices", None)
                if cell_indices is None:
                    continue

                cells = [
                    cell_class(
                        world=self.world,
                        country=country,
                        cell_index=cell_idx,
                        **self._create_views_dict(country, world_views, icell),
                        **kwargs,
                    )
                    for icell, cell_idx in enumerate(cell_indices)
                ]
                world_cells.extend(cells)
        else:
            # Fallback: cells without countries
            total_cells = self.lpjml.grid.shape[0]
            cells = [
                cell_class(
                    world=self.world,
                    country=None,
                    cell_index=cell_idx,
                    **self._create_views_dict(self.world, world_views, cell_idx),  # noqa: E501
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
        if self._parallel_executor.config.mode == "dask":
            # Check if actors are enabled (faster for repeated updates)
            use_actors = getattr(self, "_use_dask_actors", True)
            if use_actors:
                self._update_countries_dask_actors(countries, t, timed_context)
            else:
                self._update_countries_dask(countries, t, timed_context)
        else:
            with timed_context(f"update_countries_serial_year_{t}"):
                self._parallel_executor.map(
                    lambda country: country.update(t), countries
                )

        # Synchronization barrier (MPI only)
        self._parallel_executor.barrier()

    def _update_countries_dask(self, countries, t, timed_context):
        """Execute country updates using Dask.

        Handles serialization, worker dispatch, and result merging.
        Uses parallel serialization and smart batching for performance.
        """
        from concurrent.futures import ThreadPoolExecutor
        import os

        client = self._parallel_executor.config._client

        # Get batching config (default: batch countries with < 100 cells)
        batch_threshold = 100  # Countries with fewer cells get batched
        batch_size = 10  # Max countries per batch

        # Parallel serialization using threads (GIL-released during pickle)
        n_threads = min(8, len(countries), os.cpu_count() or 4)

        if n_threads > 1 and len(countries) > 4:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                serialized_countries = list(executor.map(
                    serialize_country_for_worker, countries
                ))
        else:
            serialized_countries = [
                serialize_country_for_worker(country) for country in countries
            ]

        # Separate large and small countries for batching
        large_payloads = []
        small_payloads = []

        for i, country in enumerate(countries):
            cell_indices = getattr(country, "_cell_indices", None)
            if cell_indices is None:
                n_cells = 0
            elif hasattr(cell_indices, "__len__"):
                n_cells = len(cell_indices)
            else:
                n_cells = 0
            payload = serialized_countries[i]
            if n_cells >= batch_threshold:
                large_payloads.append(payload)
            else:
                small_payloads.append(payload)

        # Submit large countries individually
        futures = []
        if large_payloads:
            futures.extend(client.map(
                sync_world,
                large_payloads,
                [t] * len(large_payloads),
                pure=False,
            ))

        # Batch small countries together
        if small_payloads:
            batches = [
                small_payloads[i:i + batch_size]
                for i in range(0, len(small_payloads), batch_size)
            ]
            batch_futures = client.map(
                sync_world_batch,
                batches,
                [t] * len(batches),
                pure=False,
            )
            futures.extend(batch_futures)

        # Gather all results
        raw_results = client.gather(futures)

        # Flatten batch results
        results = []
        batch_idx = 0
        for i, result in enumerate(raw_results):
            if i < len(large_payloads):
                # Individual result
                results.append(result)
            else:
                # Batch result - flatten
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)

        # Merge results back into world
        for result in results:
            if not result:
                continue

            cell_indices, updated_to_earth, updated_individuals = result
            cell_indices = np.asarray(cell_indices, dtype=int)

            # Apply to_earth updates
            if updated_to_earth is not None:
                for var_name, values in updated_to_earth.items():
                    if var_name in self.world.to_earth.data_vars:
                        self.world.to_earth[var_name].values[cell_indices] = values  # noqa: E501

            # Apply individual updates
            if updated_individuals is not None:
                indices = updated_individuals.get("indices")
                values_dict = updated_individuals.get("values", {})
                if indices is not None:
                    self._apply_individual_updates(indices, values_dict)

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
        from .dask_actors import ActorManager

        # Initialize actor manager on first call
        if not hasattr(self, "_actor_manager") or self._actor_manager is None:
            client = self._parallel_executor.config._client
            self._actor_manager = ActorManager(client, countries)
            # Deploy actors (one-time cost)
            with timed_context("deploy_actors"):
                self._actor_manager.deploy()

        # Get from_earth data to send to workers
        from_earth = getattr(self.world, "from_earth", None)

        # Update all actors (minimal data transfer)
        with timed_context(f"actor_update_year_{t}"):
            results = self._actor_manager.update_all(t, from_earth)

        # Merge results back into world (OPTIMIZATION 3: skip empty updates)
        for result in results:
            if not result:
                continue

            cell_indices, updated_to_earth, updated_individuals = result
            
            # Skip if no actual changes
            has_to_earth = updated_to_earth is not None and len(updated_to_earth) > 0  # noqa: E501
            has_individuals = (
                updated_individuals is not None 
                and updated_individuals.get("values")
            )
            
            if not has_to_earth and not has_individuals:
                continue
            
            # Only convert cell_indices if we have to_earth updates
            if has_to_earth:
                cell_indices = np.asarray(cell_indices, dtype=int)
                for var_name, values in updated_to_earth.items():
                    if var_name in self.world.to_earth.data_vars:
                        self.world.to_earth[var_name].values[cell_indices] = values  # noqa: E501

            # Apply individual updates (generic for any Individual type)
            if has_individuals:
                indices = updated_individuals.get("indices")
                values_dict = updated_individuals.get("values", {})
                if indices is not None and values_dict:
                    self._apply_individual_updates(indices, values_dict)

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
            self.world._from_earth_data.time.values[:] = np.array([
                np.datetime64(f"{year}-12-31")
                for year in range(t + 1 - len(self.world.from_earth.time), t + 1)  # noqa: E501
            ])

            # Close connection after last year
            if t == self.lpjml.config.lastyear:
                self.lpjml.close()
        else:
            # Test mode: only update time coordinates
            self.world._from_earth_data.time.values[:] = np.array([
                np.datetime64(f"{year}-12-31")
                for year in range(t + 1 - len(self.world.from_earth.time), t + 1)  # noqa: E501
            ])

    # -------------------------------------------------------------------------
    # Individual synchronization
    # -------------------------------------------------------------------------

    def _apply_individual_updates(self, indices, values_dict):
        """Apply individual-level updates returned from workers.

        Parameters
        ----------
        indices : np.ndarray
            Individual indices that were updated.
        values_dict : dict
            {attr_name: np.ndarray} of updated values.
        """
        if not hasattr(self, "_individual_index_map"):
            self._individual_index_map = self._build_individual_index_map()

        individual_map = self._individual_index_map
        indices = np.asarray(indices, dtype=np.int64)

        for attr_name, attr_values in values_dict.items():
            for idx, value in zip(indices, attr_values):
                individual = individual_map.get(int(idx))
                if individual is None:
                    continue
                try:
                    setattr(individual, attr_name, value)
                except AttributeError:
                    # Handle read-only properties
                    cache = getattr(individual, "_synced_read_only_values", None)  # noqa: E501
                    if cache is None:
                        cache = {}
                        setattr(individual, "_synced_read_only_values", cache)
                    cache[attr_name] = value

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
        """
        self.world.cell_neighbourhood = nx.Graph()
        self.world.cell_neighbourhood.add_nodes_from(cells)

        for icell, cell in enumerate(cells):
            for neighbour in neighbour_matrix.isel({"cell": icell}).values:
                if neighbour >= 0:
                    self.world.cell_neighbourhood.add_edge(
                        cell,
                        cells[neighbour]
                    )

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

    def _load_pycopanlpjml_config(self, config_file=None):
        """Load pycopanlpjml configuration.

        **JSON first**: When config_file is a JSON file (e.g. from pycoupler
        to_json()), the coupled_config from that JSON is the primary source.
        Run-script overrides (e.g. parallelization.mode = "serial") are
        respected.

        **YAML fallback**: Only when the JSON cannot be read or has no
        coupled_config, configuration is loaded from YAML files.

        Parameters
        ----------
        config_file : str, optional
            Path to main config file (JSON from pycoupler or YAML).

        Returns
        -------
        CoupledConfig
            Configuration object.
        """
        # 1. JSON first: prefer coupled_config from the JSON file
        if config_file and os.path.exists(config_file):
            lower = config_file.lower()
            if lower.endswith(".json") or lower.endswith(".cjson"):
                try:
                    lpjml_config = read_config(config_file, to_dict=False)
                    coupled = getattr(lpjml_config, "coupled_config", None)
                    if coupled is not None:
                        # Merge with YAML defaults for keys missing in JSON
                        return self._merge_config_with_yaml_defaults(
                            coupled, config_file
                        )
                except Exception:
                    pass  # Fall through to YAML

        # 2. YAML fallback
        return self._load_pycopanlpjml_config_from_yaml(config_file)

    def _merge_config_with_yaml_defaults(
        self, from_json: CoupledConfig, config_file: str
    ) -> CoupledConfig:
        """Merge JSON config with YAML defaults; JSON values take precedence."""
        yaml_cfg = self._load_pycopanlpjml_config_from_yaml(config_file)
        if yaml_cfg is None:
            return from_json
        return self._deep_merge_config(yaml_cfg, from_json)

    def _deep_merge_config(
        self, base: CoupledConfig, override: CoupledConfig
    ) -> CoupledConfig:
        """Merge override into base; override values take precedence."""
        base_d = base.to_dict()
        override_d = override.to_dict()

        def merge_dicts(d_base: dict, d_override: dict) -> dict:
            out = dict(d_base)
            for k, v in d_override.items():
                if v is None:
                    continue
                if isinstance(v, dict) and k in out and isinstance(out[k], dict):
                    out[k] = merge_dicts(out[k], v)
                else:
                    out[k] = v
            return out

        merged = merge_dicts(base_d, override_d)
        return self._dict_to_coupled_config(merged)

    def _dict_to_coupled_config(self, d: dict) -> CoupledConfig:
        """Recursively convert dict to CoupledConfig."""
        return CoupledConfig({
            k: self._dict_to_coupled_config(v) if isinstance(v, dict) else v
            for k, v in d.items()
        })

    def _load_pycopanlpjml_config_from_yaml(self, config_file=None):
        """Load pycopanlpjml configuration from YAML files (fallback)."""
        config_paths = []

        if config_file:
            config_dir = os.path.dirname(config_file)
            config_paths.append(
                os.path.join(config_dir, "pycopanlpjml_config.yaml")
            )
            config_paths.append(os.path.join(config_dir, "config.yaml"))

        config_paths.extend([
            os.path.join(os.path.dirname(__file__), "config.yaml"),
            "pycopanlpjml_config.yaml",
            "config.yaml",
        ])

        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    return read_yaml(config_path, CoupledConfig)
                except Exception as e:
                    print(f"Warning: Could not load config from {config_path}: {e}")  # noqa: E501
                    continue

        default_config_path = os.path.join(
            os.path.dirname(__file__),
            "config.yaml"
        )
        print(f"Warning: No config found, loading defaults from {default_config_path}")  # noqa: E501
        return read_yaml(default_config_path, CoupledConfig)

    def _countries_as_names(self):
        """Convert country codes to names if configured."""
        if self.lpjml.config.coupled_config.lpjml_settings.country_code_to_name:  # noqa: E501
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
