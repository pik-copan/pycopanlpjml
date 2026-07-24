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
- Execution of country updates
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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import pandas as pd
import networkx as nx
import numpy as np
from pycoupler.config import CoupledConfig, read_config, read_yaml
from pycoupler.coupler import LPJmLCoupler
from pycoupler.utils import get_countries

from .output import OutputCollectionMixin, read_output_table_from_zarr


class Model(OutputCollectionMixin):
    """Main component for building LPJmL-integrated copan:CORE models.

    This mixin class provides the infrastructure for coupling copan:CORE
    models with the LPJmL earth system model. It handles:

    - Connection to LPJmL via the pycoupler library
    - Entity initialization (World, Countries, Cells)
    - Execution of country updates
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
    Country : Country-level entities for regional processing.
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

        # Convert lpjml internal country id to ISO alpha-3 codes
        self.lpjml.country_id_to_code()

        # Output store lazy initialization flag
        self._output_store_initialized = False


    @property
    def config(self):
        return self.lpjml.config


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
        # Get unique country codes - ensure conversion to ISO if lpjml has code_to_name
        country_values = self.world.country_code.values
        unique_countries = np.unique(country_values)

        for country_code in unique_countries:
            country_indices = np.where(country_values == country_code)[0]
            
            country = country_class(
                code=country_code,
                world=self.world,
                indices=country_indices,
                **kwargs,
            )
            countries.append(country)

        self.countries = countries
        # Also register with world so world.countries works
        self.world._social_systems = set(countries)

    def init_cells(self, cell_class, **kwargs):
        """Initialize cell entities from LPJmL grid.

        Creates a Cell instance for each grid cell. If countries are
        initialized first, cells are assigned to their respective countries.

        Parameters
        ----------
        cell_class : type
            Cell class to instantiate (e.g., pycopanlpjml.Cell or subclass).
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

        if hasattr(self, "countries"):
            # Normal case: cells grouped by country
            for country in self.countries:
                if not len(country.indices):
                    continue
                cells = [
                    cell_class(
                        world=self.world,
                        country=country,
                        cell_index=global_idx,
                        local_index=local_idx,
                        # Use global index to get views directly from world
                        input=self.world.input.isel(cell=global_idx),
                        output=self.world.output.isel(cell=global_idx),
                        grid=self.world.grid.isel(cell=global_idx),
                        area=self.world.area.isel(cell=global_idx),
                        **kwargs,
                    )
                    for local_idx, global_idx in enumerate(country.indices)
                ]
                country._cells = cells
                world_cells.extend(cells)

            # Build cell neighbourhood links
            self._assign_cell_neighbourhood(world_cells, neighbour_matrix)
            # Build country neighbourhood links
            self._assign_country_neighbourhood()

        else:
            # Fallback: cells without countries (use world directly)
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
                for cell_idx in range(self.lpjml.ncell)
            ]
            world_cells.extend(cells)

        # Build cell neighbourhood links
        self._assign_cell_neighbourhood(world_cells, neighbour_matrix)

    def init_worldregions(self, worldregion_class, **kwargs):
        """Initialize world region entities (placeholder).

        Parameters
        ----------
        worldregion_class : type
            WorldRegion class to instantiate.
        **kwargs : dict
            Additional keyword arguments.
        """
        pass  # To be implemented

    def refresh_cell_views(self):
        """Refresh all cell views to point to current world data.

        Call this after replacing world.input or world.output (e.g., after
        cutoff_historical_data()) to ensure cells see the updated data.

        This updates the stored isel views on each cell to point to the
        current world data arrays.
        """
        for cell in self.world.cells:
            idx = cell.cell_index
            if self.world.input is not None:
                cell.input = self.world.input.isel(cell=idx)
            if self.world.output is not None:
                cell.output = self.world.output.isel(cell=idx)
            # grid and area typically don't change, but refresh if needed
            if self.world.grid is not None:
                cell.grid = self.world.grid.isel(cell=idx)
            if self.world.area is not None:
                cell.area = self.world.area.isel(cell=idx)

    # -------------------------------------------------------------------------
    # Update methods
    # -------------------------------------------------------------------------

    def update_countries(self, t):
        """Update all countries for one simulation step.

        Supports parallel execution via threading (for Python 3.14+ free-threaded).
        Configure via config.parallelization settings.

        Parameters
        ----------
        t : int
            Current simulation time step (year).

        Example
        -------
        >>> for year in range(2020, 2100):
        ...     self.update_countries(year)
        ...     self.update_lpjml(year)
        """
        # Get country list
        if hasattr(self, "countries"):
            countries = self.countries
        else:
            countries = list(self.world.countries)

        # Check parallelization mode
        parallelization_mode = self.config.coupled_config.parallelization.mode

        if parallelization_mode == "threaded": # and len(countries) > 1:
            self._update_countries_threaded(countries, t)
        else:
            # Serial execution
            for country in countries:
                country.update(t)

    def _update_countries_threaded(self, countries, t):
        """Update countries in parallel using ThreadPoolExecutor.

        Designed for Python 3.14+ free-threaded mode where threads can
        truly run in parallel without GIL constraints.

        Parameters
        ----------
        countries : list
            List of Country objects to update.
        t : int
            Current simulation time step (year).
        """
        parallel_config = self.config.coupled_config.parallelization
        max_workers = parallel_config.max_workers
        debug = parallel_config.debug

        # Check if GIL is enabled (Python 3.13+ only)
        gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
        if debug and gil_enabled:
            print("[parallel] WARNING: GIL is enabled - threading won't provide speedup")
            print("[parallel] Run with: python3.14t (free-threaded build)")

        # Auto-detect workers from SLURM or CPU count
        if max_workers <= 0:
            max_workers = int(os.environ.get("SLURM_CPUS_ON_NODE", 0))
            if max_workers <= 0:
                max_workers = os.cpu_count() or 1

        # Cap at number of countries
        max_workers = min(max_workers, len(countries))

        if debug:
            gil_status = "disabled" if not gil_enabled else "enabled"
            print(f"[parallel] Updating {len(countries)} countries with {max_workers} threads (GIL {gil_status})")

        def update_country(country):
            """Thread worker function."""
            try:
                country.update(t)
                return country.code, None
            except Exception as e:
                return country.code, e

        # Execute updates in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(update_country, countries))

        # Check for errors
        errors = [(code, err) for code, err in results if err is not None]
        if errors:
            error_msgs = [f"{code}: {err}" for code, err in errors]
            raise RuntimeError(f"Errors during parallel country update:\n" + "\n".join(error_msgs))

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
                    )
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
                    )
                ]
            )

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
