"""LPJmL-specific region implementations."""

from pycopancore.private._simple_expressions import unknown
import pycopancore.model_components.base.implementation as base
from pycopanlpjml.mixin import AliasMixin
import numpy as np


class Region(base.SocialSystem, AliasMixin):
    """An LPJmL-integrating region entity.

    Region entity type (mixin) class for copan:LPJmL component. It inherits the
    copan:CORE region entity and structure and integrates LPJmL input and
    output data (attributes) as well as grid, country and area information as
    `pycoupler.LPJmLData` and `pycoupler.LPJmLDataSet` instances.

    This class serves as the base for both Country and WorldRegion implementations
    in the LPJmL context. It provides the core functionality for handling LPJmL
    data structures and grid-based information.

    Parameters
    ----------
    input : pycoupler.LPJmLDataSet
        Coupled LPJmL model input data for the region
    output : pycoupler.LPJmLData
        Coupled LPJmL model output data for the region
    grid : pycoupler.LPJmLData
        Grid information for the region from LPJmL model
    area : float
        Area of the region in square meters
    world : World
        The world instance this region belongs to
    metabolism : Metabolism, optional
        The metabolism process taxon for this region
    **kwargs : dict
        Additional keyword arguments passed to base Region

    Returns
    -------
    Region
        An instance of the copan:LPJmL Region

    Examples
    --------
    Here's an example demonstrating the initialization of a Region instance
    in the context of an LPJmL simulation:

    First, set up the LPJmL coupling:

    >>> from pycoupler.coupler import LPJmLCoupler
    >>> from pycopanlpjml import World
    >>> from pycopanlpjml.region import Region, Country, WorldRegion

    Configure and connect to LPJmL:

    >>> config_file = "path/to/config_file.json"
    >>> lpjml = LPJmLCoupler(
    ...     config_file=config_file,
    ...     host="localhost",
    ...     port=2042
    ... )

    Initialize the world with LPJmL data:

    >>> world = World(
    ...     input=lpjml.read_input(copy=False),
    ...     output=lpjml.read_historic_output(),
    ...     grid=lpjml.grid,
    ...     country_code=lpjml.country  # ISO 3-letter country codes
    ... )

    Create a region for a specific set of cells:

    >>> cell_indices = [0, 1, 2]  # Example cell indices
    >>> region = Region(
    ...     world=world,
    ...     input=world.input.isel(cell=cell_indices),
    ...     output=world.output.isel(cell=cell_indices),
    ...     grid=world.grid.isel(cell=cell_indices)
    ... )

    For country-specific regions, use the Country class:

    >>> germany = Country(
    ...     world=world,
    ...     country_code='DEU',
    ...     input=world.input.sel(country='DEU'),
    ...     output=world.output.sel(country='DEU'),
    ...     grid=world.grid.sel(country='DEU')
    ... )

    For dynamic region groups like the EU, use WorldRegion:

    >>> from country_converter import CountryConverter
    >>> cc = CountryConverter()
    >>> eu = WorldRegion(
    ...     world=world,
    ...     region_name='EU',
    ...     country_converter=cc
    ... )

    Region Hierarchy
    ----------------
    Regions support hierarchical relationships through pycopancore's social system
    hierarchy. Use the following to manage region hierarchies:

    **Setting Hierarchy:**

    - `upper_region`: Set the parent region when creating a region
    - `next_higher_social_system`: Alternative way to set parent (same as upper_region)
    - Cannot set both `upper_region` and `next_higher_social_system` simultaneously

    **Accessing Hierarchy:**

    - `next_higher_region`: Get the immediate parent region (read/write)
    - `higher_regions`: Get all ancestor regions recursively (read-only)
    - `next_lower_regions`: Get immediate child regions (read-only)
    - `lower_regions`: Get all descendant regions recursively (read-only)

    **Example:**

    >>> # Create a parent-child hierarchy
    >>> eu = WorldRegion(name="EU", world=world, grid=eu_cells)
    >>> germany = Country(
    ...     name="Germany",
    ...     code="DEU",
    ...     world=world,
    ...     grid=germany_cells,
    ...     upper_region=eu  # Set EU as parent
    ... )
    >>>
    >>> # Access hierarchy
    >>> germany.next_higher_region  # Returns eu
    >>> eu.next_lower_regions  # Returns [germany, ...]
    >>>
    >>> # Modify hierarchy
    >>> germany.next_higher_region = None  # Remove from hierarchy

    Data Access
    -----------
    All region data (input, output, grid, area) are Zarr-backed views that
    automatically synchronize across World, Country, and Cell levels:

    - Changes made at any level are immediately visible at all other levels
    - Views are filtered to only include cells belonging to the region
    - Data is stored once in a centralized Zarr store for efficiency

    Notes
    -----
    - The Region class handles LPJmL-specific data structures and provides
      the foundation for country and region-based analysis
    - All data attributes (input, output, grid) are expected to be compatible
      with the LPJmL data structure
    - The neighbourhood attribute is automatically initialized when grid
      information is provided
    - Use `world.country_code` to access country codes (ISO 3-letter strings)
    - Use `world.countries` to access Country entity instances
    """

    type = "region"

    def __init__(
        self,
        name=None,
        code=None,
        world=None,
        input=None,
        output=None,
        grid=None,
        upper_region=None,
        **kwargs,
    ):
        """Initialize an LPJmL region.

        Parameters
        ----------
        name : str
            Region name
        code : str
            Region code
        world : World
            Reference to the World instance (required for Zarr backend access)
        input : pycoupler.LPJmLDataSet or ZarrDatasetView
            Coupled LPJmL model input (used to extract cell indices)
        output : pycoupler.LPJmLData or ZarrDatasetView
            Coupled LPJmL model output (used to extract cell indices)
        grid : pycoupler.LPJmLData or ZarrDataArrayView
            Grid of the LPJmL model (used to extract cell indices)
        upper_region : Region, optional
            Parent region in hierarchy
        **kwargs : dict
            Additional keyword arguments passed to super()
        """
        # Ensure world is passed to base class
        if "world" not in kwargs:
            kwargs["world"] = world

        super().__init__(**kwargs)

        self.type = "region"
        self.code = code
        self.name = name

        # mapping for social_system hierarchy
        if upper_region is not None and self.next_higher_social_system is None:
            self.next_higher_social_system = upper_region
        elif (
            upper_region is not None
            and self.next_higher_social_system is not None
        ):  # noqa: E501
            raise ValueError(
                "upper_region and next_higher_social_system cannot be set at the same time"  # noqa: E501
            )

        # Extract and store cell indices from grid
        # These indices are the key to accessing the right data from Zarr
        if grid is not None:
            # grid could be xarray DataArray or ZarrDataArrayView
            if hasattr(grid, "cell"):
                # xarray DataArray
                self._cell_indices = grid.cell.values
            elif hasattr(grid, "coords") and "cell" in grid.coords:
                # ZarrDataArrayView
                self._cell_indices = np.array(grid.coords["cell"])
            else:
                # Assume it's already cell indices
                self._cell_indices = np.array(grid)

            self.neighbourhood = list()
        else:
            self._cell_indices = None

    @property
    def input(self):
        """Get input dataset view for this region's cells.

        Returns a ZarrDatasetView that provides xarray-like access to only
        the cells belonging to this region. Changes made through this view
        are immediately visible to World and Cell instances.
        """
        if self.world is None or self._cell_indices is None:
            return None
        return self.world._zarr_backend.get_view("input", self._cell_indices)

    @input.setter
    def input(self, value):
        """Set input values for this region's cells.

        Note: Writes to the underlying Zarr store, immediately synchronized
        with all other views.
        """
        if self.world is None or self._cell_indices is None:
            raise ValueError(
                "Cannot set input: world or cell indices not initialized"
            )

        # Write to Zarr backend at specific indices
        if hasattr(value, "data_vars"):
            for var_name, var_data in value.data_vars.items():
                self.world._zarr_backend.root["input"][var_name][
                    self._cell_indices
                ] = var_data.values

    @property
    def output(self):
        """Get output dataset view for this region's cells."""
        if self.world is None or self._cell_indices is None:
            return None
        return self.world._zarr_backend.get_view("output", self._cell_indices)

    @output.setter
    def output(self, value):
        """Set output values for this region's cells."""
        if self.world is None or self._cell_indices is None:
            raise ValueError(
                "Cannot set output: world or cell indices not initialized"
            )

        if hasattr(value, "data_vars"):
            for var_name, var_data in value.data_vars.items():
                self.world._zarr_backend.root["output"][var_name][
                    self._cell_indices
                ] = var_data.values

    @property
    def grid(self):
        """Get grid data array view for this region's cells (read-only).

        Grid coordinates are fixed geographical properties and cannot be changed.
        """
        if self.world is None or self._cell_indices is None:
            return None
        return self.world._zarr_backend.get_array_view(
            "grid", self._cell_indices
        )

    @property
    def area(self):
        """Get area data array view for this region's cells (read-only).

        Area is a fixed geographical property and cannot be changed.
        """
        if self.world is None or self._cell_indices is None:
            return None
        if "area" not in self.world._zarr_backend.root:
            return None
        return self.world._zarr_backend.get_array_view(
            "area", self._cell_indices
        )

    @property
    def next_higher_region(self):
        """Get next higher region."""
        return self.next_higher_social_system

    @next_higher_region.setter
    def next_higher_region(self, s):
        """Set next higher region."""
        self.next_higher_social_system = s

    @property  # read-only
    def higher_regions(self):
        """Get higher regions regions recursively."""
        return self.higher_social_systems

    @higher_regions.setter
    def higher_regions(self, u):
        """Set higher regions regions recursively."""
        self.higher_social_systems = u

    @property  # read-only
    def next_lower_regions(self):
        """Get next lower regions."""
        return self.next_lower_social_systems

    @property  # read-only
    def lower_regions(self):
        """Get next lower regions recursively."""
        return self.lower_social_systems


class Country(Region):
    """A Region representing a country based on LPJmL country codes.

    The Country class extends Region to represent a specific country. Countries
    are typically identified by their ISO 3-letter country code (e.g., 'DEU', 'FRA').

    Examples
    --------
    >>> germany = Country(
    ...     name="Germany",
    ...     code="DEU",  # ISO 3-letter country code
    ...     world=world,
    ...     grid=germany_cell_indices
    ... )
    >>>
    >>> # Access the country code
    >>> germany.country_code  # Returns 'DEU'
    >>> germany.code  # Also returns 'DEU' (same thing)
    """

    type = "country"

    def __init__(self, **kwargs):
        """Initialize a country region.

        Parameters
        ----------
        code : str
            The ISO 3-letter country code (e.g., 'DEU', 'FRA', 'USA')
        name : str, optional
            Human-readable country name (e.g., 'Germany', 'France')
        world : World
            Reference to the World instance
        grid : array-like or ZarrDataArrayView
            Cell indices or grid data for cells in this country
        **kwargs : dict
            Additional keyword arguments passed to super()
        """
        super().__init__(**kwargs)

        self.type = "country"

    def __getstate__(self):
        """Break circular reference during pickling for Dask serialization.
        
        Country objects have a circular reference: Country → world → countries → Country.
        This breaks the cycle by storing the world's Zarr store path instead of the world object.
        """
        import os
        import sys
        
        state = self.__dict__.copy()
        
        # Store world's Zarr store path if available, otherwise None
        # Note: base class uses self._world, not self.world
        world_obj = state.get("_world")
        if world_obj is not None and hasattr(world_obj, "_zarr_store_path"):
            state["_world_zarr_path"] = world_obj._zarr_store_path
            if os.environ.get("PYCOPANLPJML_DEBUG_PICKLE") == "1":
                print(
                    f"DEBUG Country.__getstate__: stripping world reference "
                    f"(storing Zarr path: {world_obj._zarr_store_path})",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            state["_world_zarr_path"] = None
        
        # Remove world reference to break circular dependency
        state["_world"] = None
        
        return state

    def __setstate__(self, state):
        """Restore country state after unpickling.
        
        The world reference is set to None and will be restored lazily when accessed.
        """
        self.__dict__.update(state)
        # Ensure _world is initialized (base class expects it)
        if "_world" not in self.__dict__:
            self._world = None
        # World reference will be restored lazily via @world property when needed
    
    @property
    def world(self):
        """Get world reference, restoring it lazily from Zarr store if needed."""
        # If world is None but we have a Zarr path, restore it
        if self._world is None and hasattr(self, '_world_zarr_path') and self._world_zarr_path:
            self._restore_world_from_zarr()
        return self._world
    
    @world.setter
    def world(self, value):
        """Set world reference."""
        self._world = value
    
    def _restore_world_from_zarr(self):
        """Restore a minimal World object from Zarr store path for accessing backend.
        
        This creates a minimal World wrapper that provides access to the Zarr backend
        without needing the full World object state. This is sufficient for Region
        properties (input, output, grid, area) which only need world._zarr_backend.
        """
        import os
        import sys
        from .zarr_backend import ZarrBackend
        
        if os.environ.get("PYCOPANLPJML_DEBUG_PICKLE") == "1":
            print(
                f"DEBUG Country._restore_world_from_zarr: restoring world from "
                f"Zarr path: {self._world_zarr_path}",
                file=sys.stderr,
                flush=True,
            )
        
        # Create a minimal World-like object that provides access to Zarr backend
        # We can't fully reconstruct World (it needs model, lpjml, etc.), but we
        # only need the Zarr backend for Region properties
        class MinimalWorld:
            def __init__(self, zarr_path):
                self._zarr_store_path = zarr_path
                self._zarr_backend = ZarrBackend(zarr_path, overwrite=False)
                self._zarr_initialized = True
        
        self._world = MinimalWorld(self._world_zarr_path)

    @property
    def country_code(self):
        """Get the country code (ISO 3-letter code) for this country.

        This is an alias for the `code` attribute, provided for clarity
        when working with country codes.

        Returns
        -------
        str or None
            ISO 3-letter country code (e.g., 'DEU', 'FRA') or None
        """
        return self.code


class WorldRegion(Region):
    """A Region representing a group of countries (e.g. EU, G7) with dynamic
    membership."""

    type = "world_region"

    def __init__(self, **kwargs):
        """Initialize a world region.

        Parameters
        ----------
        region_name : str
            Name of the region (e.g. 'EU', 'G7')
        country_converter : CountryConverter, optional
            Instance of country-converter for dynamic region resolution
        **kwargs : dict
            Additional keyword arguments passed to super()
        """
        super().__init__(**kwargs)

        self.type = "world_region"
