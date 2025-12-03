"""LPJmL-specific region implementations."""

from pycopancore.private._simple_expressions import unknown
from pycopancore.private._mixin import _Mixin
import pycopancore.model_components.base.implementation as base
from pycopanlpjml.mixin import AliasMixin
from .output import OutputTableMixin
from .output import Output
import numpy as np
try:  # Optional dependency during documentation builds
    import xarray as xr  # type: ignore
except ImportError:  # pragma: no cover
    xr = None
try:  # Sympy is required by pycopancore but guard just in case
    from sympy import Basic as _SympyBasic  # type: ignore
except ImportError:  # pragma: no cover
    class _SympyBasic:  # type: ignore
        pass


def _is_local_world_view(value):
    cls = value.__class__
    return cls.__name__ == "_LocalWorldView" and cls.__module__.endswith(".world")


def _contains_non_serializable_references(value, visited=None):
    """Return True if value (recursively) references mixin/expr objects."""
    if visited is None:
        visited = set()
    if isinstance(value, (int, float, bool, str, type(None), np.generic)):
        return False
    if isinstance(value, np.ndarray):
        return False
    if xr is not None and isinstance(value, (xr.Dataset, xr.DataArray)):
        return False
    if _is_local_world_view(value):
        return False
    obj_id = id(value)
    if obj_id in visited:
        return False
    visited.add(obj_id)
    # Drop direct references to other mixin instances or sympy expressions
    if isinstance(value, (_Mixin, _SympyBasic)):
        return True
    cls = value.__class__
    module_name = getattr(cls, "__module__", "")
    class_name = getattr(cls, "__name__", "")
    if module_name.startswith("pycopancore.private._expressions") or "LLGExpr" in class_name:
        return True
    if isinstance(value, dict):
        return any(
            _contains_non_serializable_references(item, visited)
            for item in value.values()
        )
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if _contains_non_serializable_references(item, visited):
                return True
        return False
    return False


class Region(base.SocialSystem, AliasMixin, OutputTableMixin):
    """An LPJmL-integrating region entity.

    Region entity type (mixin) class for copan:LPJmL component. It inherits the
    copan:CORE region entity and structure and integrates LPJmL input and
    output data (attributes) as well as grid, country and area information as
    `pycoupler.LPJmLData` and `pycoupler.LPJmLDataSet` instances.

    This class serves as the base for both Country and WorldRegion
    implementations in the LPJmL context. It provides the core functionality
    for handling LPJmL data structures and grid-based information.

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
    Regions support hierarchical relationships through pycopancore's social
    system hierarchy. Use the following to manage region hierarchies:

    **Setting Hierarchy:**

    - `upper_region`: Set the parent region when creating a region
    - `next_higher_social_system`: Alternative way to set parent (same as
    upper_region)
    - Cannot set both `upper_region` and `next_higher_social_system`
    simultaneously

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

    # Output variables (models can override this)
    output_variables = Output()

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
    def world(self):
        """Public access to world; blocked when only a local view is available."""
        world = getattr(self, "_world", None)
        if world is None:
            return None
        try:
            from .world import _LocalWorldView
        except ImportError:  # pragma: no cover - circular import guard
            _LocalWorldView = ()
        if isinstance(world, _LocalWorldView):
            raise RuntimeError(
                "region.world.* is unavailable during parallel country updates. "
                "Aggregate the desired metric on the driver (e.g., store it on "
                "next_higher_region or use region.output) before dispatch."
            )
        return world

    @world.setter
    def world(self, value):
        self._world = value

    def _ensure_backend(self):
        """Compatibility stub for removed backend (no-op)."""
        return None

    @property
    def input(self):
        """Get input dataset view for this region's cells.

        Returns a view on the world's input dataset restricted to this
        region's cells.
        """
        if self._cell_indices is None:
            return None
        world = getattr(self, "_world", None)
        if world is None or world.input is None:
            return None
        if hasattr(world.input, "isel"):
            return world.input.isel(cell=self._cell_indices)
        return world.input

    @input.setter
    def input(self, value):
        """Set input values for this region's cells.

        Note: Writes to the underlying Zarr store, immediately synchronized
        with all other views.
        """
        if self._cell_indices is None:
            raise ValueError("Cannot set input: cell indices not initialized")
        world = getattr(self, "_world", None)
        if world is None or world.input is None:
            raise ValueError("Cannot set input: world input not available")
        if not hasattr(world.input, "data_vars"):
            raise ValueError("World input must be an xarray/LPJmL Dataset")
        if not hasattr(value, "data_vars"):
            raise ValueError("Assigned value must be an xarray/LPJmL Dataset")
        for var_name, var_data in value.data_vars.items():
            if var_name in world.input.data_vars:
                world.input[var_name].values[self._cell_indices] = (
                    var_data.values
                )

    @property
    def output(self):
        """Get output dataset view for this region's cells."""
        if self._cell_indices is None:
            return None
        world = getattr(self, "_world", None)
        if world is None or world.output is None:
            return None
        if hasattr(world.output, "isel"):
            return world.output.isel(cell=self._cell_indices)
        return world.output

    @output.setter
    def output(self, value):
        """Set output values for this region's cells."""
        if self._cell_indices is None:
            raise ValueError("Cannot set output: cell indices not initialized")
        world = getattr(self, "_world", None)
        if world is None or world.output is None:
            raise ValueError("Cannot set output: world output not available")
        if not hasattr(world.output, "data_vars"):
            raise ValueError("World output must be an xarray/LPJmL Dataset")
        if not hasattr(value, "data_vars"):
            raise ValueError("Assigned value must be an xarray/LPJmL Dataset")
        for var_name, var_data in value.data_vars.items():
            if var_name in world.output.data_vars:
                world.output[var_name].values[self._cell_indices] = (
                    var_data.values
                )

    @property
    def grid(self):
        """Get grid data array view for this region's cells (read-only).

        Grid coordinates are fixed geographical properties and cannot be
        changed.
        """
        if self._cell_indices is None:
            return None
        world = getattr(self, "_world", None)
        if world is None or world.grid is None:
            return None
        if hasattr(world.grid, "isel"):
            return world.grid.isel(cell=self._cell_indices)
        return world.grid

    @property
    def area(self):
        """Get area data array view for this region's cells (read-only).

        Area is a fixed geographical property and cannot be changed.
        """
        if self._cell_indices is None:
            return None
        world = getattr(self, "_world", None)
        if world is None or world.area is None:
            return None
        if hasattr(world.area, "isel"):
            return world.area.isel(cell=self._cell_indices)
        return world.area

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

    @property
    def model(self):
        """Get model reference (required by OutputTableMixin).

        Accesses _world directly to avoid the worker guard on the
        public world property.
        """
        # Access _world directly to avoid property guard overhead
        world = getattr(self, "_world", None)
        if world is not None:
            # Check if world has _model attribute directly (World instances)
            if hasattr(world, "_model") and world._model is not None:
                return world._model
            # Fallback to world.model property (for nested regions)
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
            # Use 'region' as the config key for Region class
            config_outputs = (
                self.model.config.coupled_config.output.to_dict().get(
                    "region", []
                )
            )
            return [
                var
                for var in self.__class__.output_variables.names
                if var in config_outputs
            ]
        except Exception:
            return []


class Country(Region):
    """A Region representing a country based on LPJmL country codes.

    The Country class extends Region to represent a specific country. Countries
    are typically identified by their ISO 3-letter country code (e.g., 'DEU',
    'FRA').

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

    # Output variables (models can override this)
    output_variables = Output()

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

        Country objects have a circular reference:
        Country → world → countries → Country.
        This implementation breaks the cycle by temporarily removing
        large object-reference attributes during pickling, while keeping
        the world reference so that in-memory world state is available
        on Dask workers.
        """
        # Pre-compute output_variables.names to cache it before __dict__.copy()
        try:
            _ = self.__class__.output_variables.names
        except Exception:
            pass

        # Exclude object reference sets/lists BEFORE __dict__.copy().
        # These contain object references that cause cloudpickle to traverse
        # the object graph during copy. Workers don't need these during
        # country.update().
        temp_excluded = {}
        for key in [
            "neighbourhood",
            "_direct_cells",
            "_next_lower_social_systems",
            "_higher_social_systems",
            "_next_higher_social_system",
        ]:
            if key in self.__dict__:
                temp_excluded[key] = self.__dict__[key]
                del self.__dict__[key]

        state = self.__dict__.copy()

        world = state.get("_world")
        cell_indices = state.get("_cell_indices")
        if (
            world is not None
            and cell_indices is not None
            and hasattr(world, "build_local_view")
        ):
            state["_world"] = world.build_local_view(cell_indices)

        # Restore excluded attributes on the object (do not modify permanently)
        for key, val in temp_excluded.items():
            self.__dict__[key] = val

        # Convert _cell_indices numpy array to list BEFORE serialization.
        # numpy arrays can cause slow serialization, especially if they're
        # views or have complex metadata. Converting to list is faster and
        # workers can reconstruct.
        if "_cell_indices" in state and state["_cell_indices"] is not None:
            import numpy as np
            if isinstance(state["_cell_indices"], np.ndarray):
                state["_cell_indices"] = state["_cell_indices"].tolist()

        # Exclude any model/config references that might have been cached
        # (These should be accessed via properties, not pickled)
        excluded_keys = []
        for key in ["_model", "model", "_config", "config"]:
            if key in state:
                excluded_keys.append(key)
                del state[key]

        # Strip remaining entries that reference other entities or expression
        # graphs (e.g., LLGExpr). Those objects trigger recursion in cloudpickle
        # and are caches that can be recomputed when needed.
        for key, value in list(state.items()):
            if _contains_non_serializable_references(value):
                state[key] = unknown

        return state

    def __setstate__(self, state):
        """Restore country state after unpickling.

        The world reference is set to None if missing. The neighbourhood is
        excluded from pickling (contains Country references that would cause
        recursive serialization), so we initialize it to an empty set. It can
        be reconstructed from world.country_neighbourhood if needed, but
        workers don't need it during update().
        """
        self.__dict__.update(state)
        # Ensure _world attribute exists (base class expects it)
        if "_world" not in self.__dict__:
            self._world = None

        # Initialize excluded object reference attributes to empty
        # sets/lists/None (excluded from pickling to avoid recursive
        # serialization and slow __dict__.copy()). Workers don't need
        # these during update(), but we initialize them to prevent
        # AttributeError if code tries to access them.
        if "neighbourhood" not in self.__dict__:
            self.neighbourhood = set()
        if "_direct_cells" not in self.__dict__:
            self._direct_cells = set()
        if "_next_lower_social_systems" not in self.__dict__:
            self._next_lower_social_systems = set()
        if "_higher_social_systems" not in self.__dict__:
            self._higher_social_systems = None
        if "_next_higher_social_system" not in self.__dict__:
            self._next_higher_social_system = None

        # Convert _cell_indices list back to numpy array after unpickling
        # (We converted it to list in __getstate__ for faster serialization)
        if "_cell_indices" in self.__dict__ and isinstance(
            self._cell_indices, list
        ):
            import numpy as np

            self._cell_indices = np.array(self._cell_indices)

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

    def get_defined_outputs(self):
        """Get list of output variable names based on config.

        Uses 'country' as the config key for Country class.

        Returns
        -------
        List[str]
            List of variable names to output (filtered by config)
        """
        if self.model is None or not hasattr(self.model, "config"):
            return []

        try:
            # Use 'country' as the config key for Country class
            config_outputs = (
                self.model.config.coupled_config.output.to_dict().get(
                    "country", []
                )
            )
            return [
                var
                for var in self.__class__.output_variables.names
                if var in config_outputs
            ]
        except Exception:
            return []


class WorldRegion(Region):
    """A Region representing a group of countries (e.g. EU, G7) with dynamic
    membership."""

    type = "world_region"

    # Output variables (models can override this)
    output_variables = Output()

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

        # AttributeError if code tries to access them.
        if "neighbourhood" not in self.__dict__:
            self.neighbourhood = set()
        if "_direct_cells" not in self.__dict__:
            self._direct_cells = set()
        if "_next_lower_social_systems" not in self.__dict__:
            self._next_lower_social_systems = set()
        if "_higher_social_systems" not in self.__dict__:
            self._higher_social_systems = None
        if "_next_higher_social_system" not in self.__dict__:
            self._next_higher_social_system = None

        # Convert _cell_indices list back to numpy array after unpickling
        # (We converted it to list in __getstate__ for faster serialization)
        if "_cell_indices" in self.__dict__ and isinstance(
            self._cell_indices, list
        ):
            import numpy as np

            self._cell_indices = np.array(self._cell_indices)

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

    def get_defined_outputs(self):
        """Get list of output variable names based on config.

        Uses 'country' as the config key for Country class.

        Returns
        -------
        List[str]
            List of variable names to output (filtered by config)
        """
        if self.model is None or not hasattr(self.model, "config"):
            return []

        try:
            # Use 'country' as the config key for Country class
            config_outputs = (
                self.model.config.coupled_config.output.to_dict().get(
                    "country", []
                )
            )
            return [
                var
                for var in self.__class__.output_variables.names
                if var in config_outputs
            ]
        except Exception:
            return []


class WorldRegion(Region):
    """A Region representing a group of countries (e.g. EU, G7) with dynamic
    membership."""

    type = "world_region"

    # Output variables (models can override this)
    output_variables = Output()

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
