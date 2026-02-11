"""LPJmL-specific region implementations extending copan:CORE.

This module provides the region entity hierarchy for integrating LPJmL
(Lund-Potsdam-Jena managed Land) earth system model data with copan:CORE based
social-ecological models.

Classes
-------
Region
    Base class for LPJmL-integrated regions with access to earth system data.
Country
    A region representing a single country, identified by ISO 3-letter code.
WorldRegion
    A region representing a group of countries (e.g., EU, G7).

The classes support:
- Access to LPJmL input/output data via xarray views
- Hierarchical region relationships (parent/child)
- Parallel processing via Dask (with custom serialization)
- Dynamic individual access (e.g., ``country.individuals``)

Cells and Individuals
--------------------
Regions contain **cells** (spatial units) and **individuals** (agents such as
farmers). Each cell has a ``neighbourhood`` list of neighbouring cells.
Individuals typically live in cells and have a ``neighbourhood`` of
neighbouring individuals (often derived from cell neighbours). Neighbourhoods
enable local interactions (e.g., social learning, adoption of practices).

Cross-Border Neighbour Buffer (Parallel Mode)
----------------------------------------
When countries are processed in parallel (Dask), each worker receives only its
country's cells and individuals. Border cells/individuals may have neighbours
in *other* countries. To enable cross-border interactions (e.g., spreading of
practices), the serialization captures an **cross-border neighbour buffer**:

- At serialization (``__getstate__``): Cells and individuals that neighbour
  internal entities but belong to other countries are cloned as read-only
  snapshots (all attributes copied, circular refs excluded).
- At deserialization (``__setstate__``): These snapshots are reconstructed as
  cross-border neighbour objects and added to each internal entity's
  ``neighbourhood``.

Cross-border neighbours have ``_is_external = True`` and carry one timestep of
lag (since they are snapshots from serialization time). This allows models like
inseeds to compute social norms and attitude from neighbours across borders.

Example
-------
>>> from pycopanlpjml import World
>>> from pycopanlpjml.region import Country
>>>
>>> # Create a country with LPJmL data
>>> germany = Country(
...     name="Germany",
...     code="DEU",
...     world=world,
...     grid=world.grid.sel(country="DEU")
... )
>>>
>>> # Access LPJmL data for this country
>>> germany.from_earth["yield"]  # Output from LPJmL
>>> germany.to_earth["irrig"]    # Input to LPJmL
"""

import numpy as np
import pandas as pd
import xarray as xr
from sympy import Basic as _SympyBasic

import pycopancore.model_components.base.implementation as base
from pycopancore.private._mixin import _Mixin
from pycopancore.private._simple_expressions import unknown
try:
    from pycoupler.utils import warn_deprecated_alias
except ImportError:

    def warn_deprecated_alias(instance, old_name: str, new_name: str) -> None:
        """No-op when pycoupler does not provide warn_deprecated_alias."""
        pass

from pycopanlpjml.mixin import AliasMixin
from pycopanlpjml.output import Output, OutputDefinitionMixin


class Region(base.SocialSystem, AliasMixin, OutputDefinitionMixin):
    """Base class for LPJmL-integrated regions.

    A Region is a spatial entity that aggregates cells from the LPJmL grid.
    It provides access to earth system data (inputs/outputs) and supports
    hierarchical relationships with other regions.

    This class serves as the foundation for Country and WorldRegion, providing:

    - **Data access**: Views into world-level xarray datasets filtered by cell
    indices
    - **Hierarchy**: Parent/child relationships via ``next_higher_region``
    - **Individuals**: Dynamic access to individuals (e.g.,
    ``region.get_individuals("farmer")``)
    - **Output collection**: Integration with the output system

    Parameters
    ----------
    name : str, optional
        Human-readable name for the region.
    code : str, optional
        Short identifier code (e.g., ISO country code).
    world : World
        The World instance this region belongs to. Required for data access.
    to_earth : xarray.Dataset or LPJmLDataSet, optional
        Initial input data (deprecated, use ``world.to_earth`` instead).
    from_earth : xarray.Dataset or LPJmLData, optional
        Initial output data (deprecated, use ``world.from_earth`` instead).
    grid : xarray.DataArray or array-like, optional
        Cell indices or grid data defining which cells belong to this region.
    upper_region : Region, optional
        Parent region in the hierarchy.
    **kwargs : dict
        Additional arguments passed to the base SocialSystem.

    Attributes
    ----------
    to_earth : xarray.Dataset
        Data sent to the earth system (LPJmL inputs) for this region's cells.
    from_earth : xarray.Dataset
        Data received from the earth system (LPJmL outputs) for this region's
        cells.
    output_array : xarray.Dataset or None
        Collected model outputs (xarray) for this region's cells. Lazy.
    output_table : pandas.DataFrame
        Collected model outputs (long-format table) for this region's cells.
        Lazy.
    grid : xarray.DataArray
        Grid coordinates for this region's cells.
    area : xarray.DataArray
        Cell areas for this region.
    cells : set
        Set of Cell entities belonging to this region.
    individuals : set
        Set of Individual entities in this region.

    Examples
    --------
    Create a region from a subset of cells:

    >>> region = Region(
    ...     name="Study Area",
    ...     world=world,
    ...     grid=np.array([0, 1, 2, 3])  # Cell indices
    ... )

    Access earth system data:

    >>> # Get LPJmL outputs for this region
    >>> yields = region.from_earth["yield"]
    >>>
    >>> # Modify LPJmL inputs for this region
    >>> region.to_earth["irrig"].values[:] = 0.5

    Work with region hierarchy:

    >>> # Set parent region
    >>> region.next_higher_region = parent_region
    >>>
    >>> # Get all ancestor regions
    >>> ancestors = region.higher_regions

    Access individuals dynamically:

    >>> # Get all individuals of a specific type
    >>> farmers = region.get_individuals("farmer")  # Model-specific type

    See Also
    --------
    Country : A region representing a single country.
    WorldRegion : A region representing a group of countries.
    World : ``output_array`` and ``output_table`` also available on World,
    Cell.

    Notes
    -----
    The data properties (``to_earth``, ``from_earth``, ``grid``, ``area``)
    return views into the world-level datasets, filtered by this region's
    cell indices. Changes to ``to_earth`` are immediately visible at all
    levels (world, region, cell).

    **Cells and individuals**
    - Cells are spatial units with ``neighbourhood`` (list of neighbouring
      cells from grid topology). They link to LPJmL via ``from_earth`` /
      ``to_earth``.
    - Individuals (e.g., farmers) live in cells and have ``neighbourhood``
      (neighbouring individuals, often derived from cell neighbours).
    - Models use neighbourhoods for local interactions (social learning,
      adoption dynamics).
    """

    # Class-level output variable definitions (models can override)
    output_variables = Output()

    # Entity type identifier
    type = "region"

    def __init__(
        self,
        name=None,
        code=None,
        world=None,
        to_earth=None,
        from_earth=None,
        grid=None,
        upper_region=None,
        input=None,
        output=None,
        **kwargs,
    ):
        # Handle deprecated parameter names
        if to_earth is None and input is not None:
            to_earth = input
        if from_earth is None and output is not None:
            from_earth = output

        # Pass world to base class
        if "world" not in kwargs:
            kwargs["world"] = world

        super().__init__(**kwargs)

        # Store basic identifiers
        self.type = "region"
        self.code = code
        self.name = name

        # Set up hierarchy relationship
        if upper_region is not None and self.next_higher_social_system is None:
            self.next_higher_social_system = upper_region
        elif (
            upper_region is not None
            and self.next_higher_social_system is not None
        ):  # noqa: E501
            raise ValueError(
                "Cannot set both 'upper_region' and 'next_higher_social_system'"  # noqa: E501
            )

        # Extract cell indices from grid data
        # These indices are the key to accessing data from world-level datasets
        if grid is not None:
            if hasattr(grid, "cell"):
                # xarray DataArray with 'cell' coordinate
                self._cell_indices = grid.cell.values
            elif hasattr(grid, "coords") and "cell" in grid.coords:
                # ZarrDataArrayView or similar
                self._cell_indices = np.array(grid.coords["cell"])
            else:
                # Assume it's already an array of cell indices
                self._cell_indices = np.array(grid)
            self.neighbourhood = list()
        else:
            self._cell_indices = None

    # -------------------------------------------------------------------------
    # World property with worker guard
    # -------------------------------------------------------------------------

    @property
    def world(self):
        """Get the World instance this region belongs to.

        Returns
        -------
        World or None
            The world instance, or None if not set.

        Raises
        ------
        RuntimeError
            If accessed on a worker during parallel processing, where only
            a lightweight _LocalWorldView is available.
        """
        world = getattr(self, "_world", None)
        if world is None:
            return None

        # Guard against accessing full world on workers
        try:
            from .world import _LocalWorldView
        except ImportError:
            _LocalWorldView = ()

        if isinstance(world, _LocalWorldView):
            raise RuntimeError(
                "region.world.* is unavailable during parallel country updates. "  # noqa: E501
                "Aggregate the desired metric on the driver (e.g., store it on "  # noqa: E501
                "next_higher_region or use region.output) before dispatch."
            )
        return world

    @world.setter
    def world(self, value):
        self._world = value

    # -------------------------------------------------------------------------
    # Earth system data properties
    # -------------------------------------------------------------------------

    @property
    def to_earth(self):
        """Get data sent to the earth system (LPJmL inputs).

        Returns a view on the world's ``to_earth`` dataset, filtered to
        include only this region's cells. Changes are immediately visible
        at all levels.

        Returns
        -------
        xarray.Dataset or None
            Input data for this region's cells, or None if unavailable.
        """
        if self._cell_indices is None:
            return None
        world = getattr(self, "_world", None)
        if world is None or world.to_earth is None:
            return None
        if hasattr(world.to_earth, "isel"):
            return world.to_earth.isel(cell=self._cell_indices)
        return world.to_earth

    @to_earth.setter
    def to_earth(self, value):
        """Set data sent to the earth system for this region's cells.

        Writes directly to the world-level dataset, immediately visible
        at all levels.

        Parameters
        ----------
        value : xarray.Dataset
            New input data for this region's cells.

        Raises
        ------
        ValueError
            If cell indices are not initialized or world data unavailable.
        """
        if self._cell_indices is None:
            raise ValueError(
                "Cannot set to_earth: cell indices not initialized"
            )  # noqa: E501
        world = getattr(self, "_world", None)
        if world is None or world.to_earth is None:
            raise ValueError(
                "Cannot set to_earth: world.to_earth not available"
            )  # noqa: E501
        if not hasattr(world.to_earth, "data_vars"):
            raise ValueError("World to_earth must be an xarray Dataset")
        if not hasattr(value, "data_vars"):
            raise ValueError("Assigned value must be an xarray Dataset")

        # Copy each variable's data to the appropriate cell indices
        for var_name, var_data in value.data_vars.items():
            if var_name in world.to_earth.data_vars:
                world.to_earth[var_name].values[
                    self._cell_indices
                ] = var_data.values  # noqa: E501

    @property
    def from_earth(self):
        """Get data received from the earth system (LPJmL outputs).

        Returns a view on the world's ``from_earth`` dataset, filtered to
        include only this region's cells. This is read-only; only LPJmL
        updates this data via ``Component.update_lpjml()``.

        Returns
        -------
        xarray.Dataset or None
            Output data for this region's cells, or None if unavailable.
        """
        if self._cell_indices is None:
            return None
        world = getattr(self, "_world", None)
        if world is None or world.from_earth is None:
            return None
        if hasattr(world.from_earth, "isel"):
            # Select each variable individually to handle datasets with
            # conflicting dimension sizes (e.g., different 'band' sizes)
            try:
                return world.from_earth.isel(cell=self._cell_indices)
            except ValueError:
                # Fallback: select each variable individually
                import xarray as xr

                data_vars = {}
                for var_name in world.from_earth.data_vars:
                    var = world.from_earth[var_name]
                    if "cell" in var.dims:
                        data_vars[var_name] = var.isel(cell=self._cell_indices)
                    else:
                        data_vars[var_name] = var
                return xr.Dataset(data_vars)
        return world.from_earth

    @property
    def grid(self):
        """Get grid coordinates for this region's cells.

        Returns
        -------
        xarray.DataArray or None
            Grid coordinates (lon/lat), or None if unavailable.
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
        """Get cell areas for this region.

        Returns
        -------
        xarray.DataArray or None
            Cell areas in square meters, or None if unavailable.
        """
        if self._cell_indices is None:
            return None
        world = getattr(self, "_world", None)
        if world is None or world.area is None:
            return None
        if hasattr(world.area, "isel"):
            return world.area.isel(cell=self._cell_indices)
        return world.area

    # -------------------------------------------------------------------------
    # Backward compatibility aliases
    # -------------------------------------------------------------------------

    @property
    def input(self):
        """Backward compatibility alias for ``to_earth``."""
        warn_deprecated_alias(self, "input", "to_earth")
        return self.to_earth

    @input.setter
    def input(self, value):
        warn_deprecated_alias(self, "input", "to_earth")
        self.to_earth = value

    @property
    def output(self):
        """Backward compatibility alias for ``from_earth`` (read-only)."""
        warn_deprecated_alias(self, "output", "from_earth")
        return self.from_earth

    # -------------------------------------------------------------------------
    # Collected model output (lazy; delegates to world)
    # -------------------------------------------------------------------------

    @property
    def output_array(self):
        """Get model output as xarray Dataset for this region's cells (lazy).

        Delegates to ``world.output_array``, filtered to this region's cells.
        None if unavailable. No extra work during simulation.

        Returns
        -------
        xarray.Dataset or None
        """
        world = getattr(self, "_world", None)
        if world is None:
            return None
        ds = getattr(world, "output_array", None)
        if ds is None or self._cell_indices is None:
            return ds
        try:
            if "cell" in ds.dims:
                return ds.isel(cell=self._cell_indices)
        except (ValueError, KeyError):
            pass
        return ds

    @property
    def output_table(self):
        """Get model output as long-format DataFrame for this region's cells
        (lazy).

        Delegates to ``world.output_table``, filtered to rows where cell is in
        this region. Empty DataFrame if unavailable. No extra work during
        simulation.

        Returns
        -------
        pandas.DataFrame
        """
        world = getattr(self, "_world", None)
        if world is None:
            return pd.DataFrame()
        table = getattr(world, "output_table", None)
        if table is None or table.empty or self._cell_indices is None:
            return table if table is not None else pd.DataFrame()
        if "cell" not in table.columns:
            return table
        return table[table["cell"].isin(self._cell_indices)].copy()

    # -------------------------------------------------------------------------
    # Hierarchy properties
    # -------------------------------------------------------------------------

    @property
    def next_higher_region(self):
        """Get the immediate parent region in the hierarchy.

        Returns
        -------
        Region or None
            The parent region, or None if this is a top-level region.
        """
        return self.next_higher_social_system

    @next_higher_region.setter
    def next_higher_region(self, value):
        """Set the immediate parent region."""
        self.next_higher_social_system = value

    @property
    def higher_regions(self):
        """Get all ancestor regions recursively.

        Returns
        -------
        list
            List of ancestor regions, from immediate parent to root.
        """
        return self.higher_social_systems

    @higher_regions.setter
    def higher_regions(self, value):
        self.higher_social_systems = value

    @property
    def next_lower_regions(self):
        """Get immediate child regions.

        Returns
        -------
        set
            Set of regions that have this region as their parent.
        """
        return self.next_lower_social_systems

    @property
    def lower_regions(self):
        """Get all descendant regions recursively.

        Returns
        -------
        set
            Set of all descendant regions.
        """
        return self.lower_social_systems

    # -------------------------------------------------------------------------
    # Individual access
    # -------------------------------------------------------------------------

    def _current_individuals(self):
        """Return a concrete set of individuals without cache errors."""
        individuals = getattr(self, "_individuals", None)
        if individuals is unknown or individuals is None:
            try:
                individuals = super().individuals
            except AttributeError:
                individuals = set()
        if individuals is unknown or individuals is None:
            individuals = set()
        return set(individuals)

    def get_individuals(self, type_name=None):
        """Get individuals filtered by type name.

        Parameters
        ----------
        type_name : str, optional
            Type name to filter by (e.g., 'farmer', 'consumer').
            Case-insensitive; singular and plural forms accepted.
            If None, returns all individuals.

        Returns
        -------
        set
            Set of matching Individual entities.

        Examples
        --------
        >>> region.get_individuals()  # All individuals
        >>> region.get_individuals("farmer")  # Only farmers
        >>> region.get_individuals("consumers")  # Only consumers
        """
        individuals = self._current_individuals()
        if type_name is None:
            return individuals

        normalized = _normalize_label(type_name)
        if normalized is None:
            return set()
        if normalized in {"individual", "individuals"}:
            return individuals

        matched = set()
        for individual in individuals:
            labels = _individual_type_labels(individual)
            if normalized in labels:
                matched.add(individual)
        return matched

    def _available_individual_labels(self):
        """Return all labels that can be used for dynamic individual access."""
        labels = {"individual", "individuals"}
        for individual in self._current_individuals():
            labels.update(_individual_type_labels(individual))
        return labels

    def __getattr__(self, name):
        """Enable dynamic access to individuals by type.

        This allows accessing individuals by type using attribute syntax:
        ``region.farmers`` (if "farmer" is a valid type) instead of
        ``region.get_individuals("farmer")``.
        """
        if name.startswith("__"):
            raise AttributeError(name)

        normalized = _normalize_label(name)
        if normalized in {"individual", "individuals"}:
            return self.get_individuals()

        available = self._available_individual_labels()
        if normalized in available:
            return self.get_individuals(normalized)

        raise AttributeError(name)

    # -------------------------------------------------------------------------
    # Model and output support
    # -------------------------------------------------------------------------

    @property
    def model(self):
        """Get reference to the model component.

        Used by OutputDefinitionMixin to access configuration.

        Returns
        -------
        Model or None
            The model component, or None if unavailable.
        """
        world = getattr(self, "_world", None)
        if world is not None:
            if hasattr(world, "_model") and world._model is not None:
                return world._model
            if hasattr(world, "model"):
                return world.model
        return None

    def get_defined_outputs(self):
        """Get output variable names enabled in config.

        Returns
        -------
        list of str
            Variable names that should be collected for output.
        """
        if self.model is None or not hasattr(self.model, "config"):
            return []

        try:
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

    def _ensure_backend(self):
        """Compatibility stub for removed backend (no-op)."""
        return None


class Country(Region):
    """A region representing a single country.

    Country extends Region with country-specific functionality:

    - ISO 3-letter country code identification
    - Custom serialization for parallel processing via Dask
    - Country-specific output configuration

    Parameters
    ----------
    code : str
        ISO 3-letter country code (e.g., 'DEU', 'FRA', 'USA').
    name : str, optional
        Human-readable country name (e.g., 'Germany').
    world : World
        The World instance this country belongs to.
    grid : array-like
        Cell indices for cells in this country.
    **kwargs : dict
        Additional arguments passed to Region.

    Attributes
    ----------
    country_code : str
        Alias for ``code``, the ISO 3-letter country code.
    cells : set
        Set of Cell entities in this country.
    individuals : set
        Set of Individual entities (dynamic access by type).

    Examples
    --------
    Create a country:

    >>> germany = Country(
    ...     name="Germany",
    ...     code="DEU",
    ...     world=world,
    ...     grid=world.grid.sel(country="DEU")
    ... )

    Access country data:

    >>> germany.country_code  # 'DEU'
    >>> germany.from_earth["yield"].mean()  # Average yield

    Access individuals:

    >>> individuals = germany.get_individuals("farmer")  # Model-specific
    >>> for individual in individuals:
    ...     print(individual.income)

    Notes
    -----
    Country objects support parallel processing via Dask. The ``__getstate__``
    and ``__setstate__`` methods handle serialization by:

    1. Creating lightweight clones of cells and individuals
    2. Breaking circular references that would cause recursion
    3. Converting neighbourhood graphs to index-based representation
    4. Rebuilding object references after deserialization

    **Cross-border neighbour buffer**
    Border cells/individuals may have neighbours in other countries. To enable
    cross-border interactions (e.g., spreading of practices), the serialization
    includes an cross-border neighbour buffer:
    - At ``__getstate__``: Neighbours in other countries are captured as
      read-only snapshots (all attributes copied).
    - At ``__setstate__``: These snapshots are reconstructed as external
      neighbour objects (``_CrossBorderNeighbour``) and added to each entity's
      ``neighbourhood``.
    - Cross-border neighbours have ``_is_external = True`` and carry one
     timestep of lag. Use them for read-only access (e.g., ``n.tillage``,
     ``n.soilc``).

    See Also
    --------
    Region : Base class with data access methods.
    WorldRegion : For groups of countries.
    """

    type = "country"
    output_variables = Output()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "country"

    @property
    def country_code(self):
        """Get the ISO 3-letter country code.

        Returns
        -------
        str or None
            Country code (e.g., 'DEU', 'FRA'), or None if not set.
        """
        return self.code

    def get_defined_outputs(self):
        """Get output variable names enabled in config.

        Uses 'country' as the config key.

        Returns
        -------
        list of str
            Variable names to collect for output.
        """
        if self.model is None or not hasattr(self.model, "config"):
            return []

        try:
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

    # -------------------------------------------------------------------------
    # Serialization for parallel processing
    # -------------------------------------------------------------------------
    #
    # Serialization sends cloned cells/individuals to Dask workers. The
    # external Cross-border neighbour buffer captures neighbours in other
    # countries as read-only snapshots so border entities retain cross-country
    # neighbourhood links.
    # -------------------------------------------------------------------------

    def _get_or_create_serialization_cache(self):
        """Get or create cached clone structure for fast serialization.

        On first call, creates cloned cells/individuals and caches the
        structure.
        On subsequent calls, returns the cached structure for fast updates.

        Returns
        -------
        dict
            Cache containing cell_map, individual_map, and cloned entities.
        """
        cache = getattr(self, "_serialization_cache", None)
        if cache is not None:
            return cache

        # First time: build full clone structure
        original_cells = (
            getattr(self, "_direct_cells", set()) or set()
        )  # noqa: E501

        # Get individuals from country level (may not be in cells)
        original_individuals = getattr(self, "_direct_individuals", set())
        if not original_individuals:
            original_individuals = (
                getattr(self, "_individuals", set()) or set()
            )

        # Create clone structure
        cell_map, individual_map, cloned_cells, cloned_individuals = (
            _create_clone_structure(
                original_cells,
                original_individuals,  # country-level individuals
            )
        )

        cache = {
            "cell_map": cell_map,  # original -> clone
            "individual_map": individual_map,  # original -> clone
            "cloned_cells": cloned_cells,
            "cloned_individuals": cloned_individuals,
        }
        self._serialization_cache = cache
        return cache

    def __getstate__(self):
        """Prepare country for serialization (Dask workers).

        Uses cached clone structure for performance. On first call, creates
        full clones. On subsequent calls, only updates dynamic attributes.

        Also builds an cross-border neighbour buffer: neighbours in other
        countries are captured as read-only snapshots (all attributes) so
        border entities can access cross-country status after deserialization.

        Returns
        -------
        dict
            Serializable state dictionary including
            ``_cross_border_neighbour_buffer``.
        """
        # Pre-compute output_variables.names to cache it
        try:
            _ = self.__class__.output_variables.names
        except Exception:
            pass

        # Materialize individual caches
        try:
            if getattr(self, "_direct_individuals", unknown) is unknown:
                _ = self.direct_individuals
            if getattr(self, "_individuals", unknown) is unknown:
                _ = self.individuals
        except Exception:
            pass

        # Get or create cached clone structure
        cache = self._get_or_create_serialization_cache()
        cell_map = cache["cell_map"]
        individual_map = cache["individual_map"]

        # Build lightweight local view for this serialization
        world = getattr(self, "_world", None)
        cell_indices = getattr(self, "_cell_indices", None)
        if (
            world is not None
            and cell_indices is not None
            and hasattr(world, "build_local_view")
        ):
            local_world = world.build_local_view(cell_indices)
        else:
            local_world = world

        model_view = (
            getattr(local_world, "_model", None) if local_world else None
        )

        # Update cached clones with current dynamic values
        _update_cached_clones(
            cell_map, individual_map, local_world, model_view
        )

        # Build state dict (excluding circular ref collections)
        exclude_keys = {
            "neighbourhood",
            "_cells",
            "_direct_cells",
            "_next_lower_social_systems",
            "_higher_social_systems",
            "_next_higher_social_system",
            "_direct_individuals",
            "_individuals",
            "_serialization_cache",  # Don't serialize the cache itself
        }

        state = {
            k: v for k, v in self.__dict__.items() if k not in exclude_keys
        }

        # Use cached clones
        cloned_cells = cache["cloned_cells"]
        cloned_individuals = cache["cloned_individuals"]

        state["_world"] = local_world
        state["_direct_cells"] = cloned_cells
        state["_cells"] = set(cloned_cells)
        state["_next_lower_social_systems"] = set(cloned_cells)
        state["_direct_individuals"] = cloned_individuals
        state["_individuals"] = set(cloned_individuals)

        # Convert numpy arrays to lists (faster serialization)
        if "_cell_indices" in state and state["_cell_indices"] is not None:
            if isinstance(state["_cell_indices"], np.ndarray):
                state["_cell_indices"] = state["_cell_indices"].tolist()

        # Remove model/config references (accessed via properties)
        for key in ["_model", "model", "_config", "config"]:
            state.pop(key, None)

        # Strip non-serializable expressions
        for key, value in list(state.items()):
            if key in _STATE_ALLOWLIST:
                continue
            if _contains_non_serializable_references(value):
                state[key] = unknown

        # Build cross-border neighbour buffer for cross-country spreading
        external_buffer = _build_cross_border_neighbour_buffer(
            cache["cell_map"],
            cache["individual_map"],
            local_world,
            model_view,
        )
        state["_cross_border_neighbour_buffer"] = external_buffer

        return state

    def __setstate__(self, state):
        """Restore country state after deserialization.

        Rebuilds object references and neighbourhoods that were
        converted to indices during serialization. Reconstructs external
        neighbours from the buffer and adds them to each
        border entity's ``neighbourhood``.

        Parameters
        ----------
        state : dict
            State dictionary from ``__getstate__``.
        """
        # Extract external buffer before updating __dict__
        external_buffer = state.pop("_cross_border_neighbour_buffer", None)

        self.__dict__.update(state)

        # Ensure required attributes exist
        if "_world" not in self.__dict__:
            self._world = None

        # Initialize collections that were excluded from serialization
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
            if key not in self.__dict__:
                setattr(self, key, default)

        # Convert cell indices back to numpy array
        if "_cell_indices" in self.__dict__ and isinstance(
            self._cell_indices, list
        ):  # noqa: E501
            self._cell_indices = np.array(self._cell_indices)

        # Store external buffer for neighbourhood reconstruction
        self._cross_border_neighbour_buffer = external_buffer

        # Rebuild object references and neighbourhoods
        _assign_country_references(self)


class WorldRegion(Region):
    """A region representing a group of countries.

    WorldRegion is used for multi-country regions like the European Union,
    G7, or custom groupings. It inherits all Region functionality.

    Parameters
    ----------
    region_name : str
        Name of the region (e.g., 'EU', 'G7').
    country_converter : CountryConverter, optional
        Instance for dynamic country resolution.
    **kwargs : dict
        Additional arguments passed to Region.

    Examples
    --------
    >>> from country_converter import CountryConverter
    >>> cc = CountryConverter()
    >>>
    >>> eu = WorldRegion(
    ...     name="European Union",
    ...     region_name="EU",
    ...     world=world,
    ...     country_converter=cc
    ... )
    """

    type = "world_region"
    output_variables = Output()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "world_region"


# =============================================================================
# SERIALIZATION HELPERS
# =============================================================================
#
# These functions support Country serialization for Dask parallel processing.
# They handle:
# - Detecting non-serializable references (expressions, mixin instances)
# - Cloning cells and individuals with broken circular references
# - Converting neighbourhoods to index-based representation
# - Rebuilding object references after deserialization
# - cross-border neighbour buffer: cloning neighbours in other countries as
#   read-only snapshots so border entities can access cross-country status
#   (one timestep lag) for spreading dynamics
#
# =============================================================================

# Attributes to skip when cloning cells (contain circular references)
_CELL_ATTR_SKIP = frozenset(
    {
        "_world",
        "world",
        "social_system",
        "_social_system",
        "_social_systems",
        "social_systems",
        "neighbourhood",
        "_individuals",
        "_next_lower_social_systems",
        "_higher_social_systems",
        "_next_higher_social_system",
        "_direct_cells",
        "_direct_individuals",
        "_higher_culture_systems",
        "_higher_environment_systems",
        "_higher_metabolism_systems",
        "model",
        "_model",
    }
)

# Attributes to skip when cloning individuals
_INDIVIDUAL_ATTR_SKIP = frozenset(
    {
        "_world",
        "world",
        "_cell",
        "cell",
        "social_system",
        "_social_system",
        "_social_systems",
        "social_systems",
        "neighbourhood",
        "model",
        "_model",
    }
)

# State keys that are allowed to contain object references
_STATE_ALLOWLIST = frozenset(
    {
        "_cells",
        "_direct_cells",
        "_next_lower_social_systems",
        "_direct_individuals",
        "_individuals",
    }
)


def _is_local_world_view(value):
    """Check if value is a _LocalWorldView instance."""
    cls = value.__class__
    return cls.__name__ == "_LocalWorldView" and cls.__module__.endswith(
        ".world"
    )  # noqa: E501


def _is_dask_expression(value):
    """Check if value is a Dask expression graph."""
    cls = value.__class__
    name = getattr(cls, "__name__", "")
    if name == "LLGExpr":
        return True
    module = getattr(cls, "__module__", "")
    return module.startswith("dask.") and name.lower().endswith("expr")


def _contains_non_serializable_references(value, visited=None):
    """Check if value contains references that would break serialization.

    Detects mixin instances, sympy expressions, and Dask expression graphs
    that would cause infinite recursion or fail during cloudpickle.

    Parameters
    ----------
    value : Any
        The value to check.
    visited : set, optional
        Set of already-visited object IDs (for cycle detection).

    Returns
    -------
    bool
        True if non-serializable references are found.
    """
    if visited is None:
        visited = set()

    # Fast path: primitives are always safe
    if isinstance(value, (int, float, bool, str, type(None), np.generic)):
        return False

    # Dask expressions cause recursion
    if _is_dask_expression(value):
        return True

    # Check numpy arrays with object dtype
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            try:
                for _, item in np.ndenumerate(value):
                    if _contains_non_serializable_references(item, visited):
                        return True
            except RecursionError:
                return True
        return False

    # Check xarray DataArray
    if isinstance(value, xr.DataArray):
        if getattr(value, "dtype", None) == object:
            return _contains_non_serializable_references(value.values, visited)
        return False

    # Check xarray Dataset
    if isinstance(value, xr.Dataset):
        try:
            for var in value.data_vars.values():
                if _contains_non_serializable_references(var, visited):
                    return True
            for coord in value.coords.values():
                if _contains_non_serializable_references(coord, visited):
                    return True
        except RecursionError:
            return True
        return False

    # Handle _LocalWorldView specially
    if _is_local_world_view(value):
        obj_id = id(value)
        if obj_id in visited:
            return False
        visited.add(obj_id)
        for attr in (
            "_to_earth_data",
            "_from_earth_data",
            "_grid_data",
            "_country_data",
            "_area_data",
        ):
            if hasattr(value, attr):
                if _contains_non_serializable_references(
                    getattr(value, attr), visited
                ):  # noqa: E501
                    return True
        return False

    # Cycle detection for general objects
    obj_id = id(value)
    if obj_id in visited:
        return False
    visited.add(obj_id)

    # Mixin instances and sympy expressions cause recursion
    if isinstance(value, (_Mixin, _SympyBasic)):
        return True

    # Check for pycopancore expression objects
    cls = value.__class__
    module_name = getattr(cls, "__module__", "")
    class_name = getattr(cls, "__name__", "")
    if (
        module_name.startswith("pycopancore.private._expressions")
        or "LLGExpr" in class_name
    ):  # noqa: E501
        return True

    # Recursively check containers
    try:
        if isinstance(value, dict):
            for item in value.values():
                if _contains_non_serializable_references(item, visited):
                    return True
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                if _contains_non_serializable_references(item, visited):
                    return True
    except RecursionError:
        return True

    return False


def _create_clone_structure(cells, individuals):
    """Create clone structure for cached serialization.

    Creates empty clone objects with static attributes (indices, structure).
    Dynamic attributes are set later by _update_cached_clones.

    Parameters
    ----------
    cells : iterable
        Original Cell entities to clone.
    individuals : iterable
        Original Individual entities to clone.

    Returns
    -------
    tuple
        (cell_map, individual_map, cloned_cells, cloned_individuals)
        Maps are original -> clone for fast lookup.
    """
    cell_map = {}  # original -> clone
    individual_map = {}  # original -> clone
    cloned_cells = set()
    cloned_individuals = set()

    iterable_cells = list(cells or [])

    # Create cell clones with static structure
    for cell in iterable_cells:
        clone = cell.__class__.__new__(cell.__class__)
        clone.__dict__ = {}

        # Copy STATIC attributes only (indices, etc.)
        clone._cell_index = getattr(cell, "_cell_index", None)
        clone._entity_alias = getattr(cell, "_entity_alias", "cell")

        # Initialize empty collections
        clone.neighbourhood = []
        clone._individuals = set()
        clone._direct_cells = set()
        clone._direct_individuals = set()
        clone._next_lower_social_systems = set()

        # Clear social links
        clone._social_system = None
        clone._social_systems = []
        clone._next_higher_social_system = None
        clone._higher_social_systems = None

        cell_map[cell] = clone
        cloned_cells.add(clone)

    # Create individual clones with static structure (from cell._individuals)
    for cell in iterable_cells:
        clone_cell = cell_map.get(cell)
        if clone_cell is None:
            continue

        original_individuals = (
            getattr(cell, "_individuals", None) or set()
        )  # noqa: E501
        for individual in original_individuals:
            clone_individual = individual.__class__.__new__(
                individual.__class__
            )
            clone_individual.__dict__ = {}

            # Copy STATIC attributes
            clone_individual._individual_index = getattr(
                individual, "_individual_index", None
            )
            clone_individual._entity_alias = getattr(
                individual, "_entity_alias", "individual"
            )

            # Clear social links
            clone_individual._social_system = None
            clone_individual._social_systems = []

            # Detach from cell but store index for reconstruction
            # (matches behavior expected by tests and __setstate__)
            clone_individual._cell = None
            clone_individual._cell_index = getattr(cell, "_cell_index", None)
            clone_individual._cell_uid = getattr(clone_cell, "_uid", None)
            clone_individual.neighbourhood = []

            # Still add to clone cell's _individuals collection
            clone_cell._individuals.add(clone_individual)

            individual_map[individual] = clone_individual
            cloned_individuals.add(clone_individual)

    # Clone individuals stored at country level (not in cells)
    for individual in list(individuals or []):
        if individual in individual_map:
            continue  # Already cloned from cell

        # Find the clone cell for this individual's original cell
        original_cell = getattr(individual, "_cell", None)
        clone_cell = cell_map.get(original_cell)
        if clone_cell is None:
            continue  # No matching cell clone

        clone_individual = individual.__class__.__new__(individual.__class__)
        clone_individual.__dict__ = {}

        # Copy STATIC attributes
        clone_individual._individual_index = getattr(
            individual, "_individual_index", None
        )
        clone_individual._entity_alias = getattr(
            individual, "_entity_alias", "individual"
        )

        # Clear social links
        clone_individual._social_system = None
        clone_individual._social_systems = []

        # Detach from cell but store index
        clone_individual._cell = None
        clone_individual._cell_index = getattr(
            original_cell, "_cell_index", None
        )
        clone_individual._cell_uid = getattr(clone_cell, "_uid", None)
        clone_individual.neighbourhood = []

        # Add to clone cell's _individuals collection
        clone_cell._individuals.add(clone_individual)

        individual_map[individual] = clone_individual
        cloned_individuals.add(clone_individual)

    # Store neighbourhood as INDICES (static)
    for original_cell in iterable_cells:
        clone_cell = cell_map.get(original_cell)
        if clone_cell is None:
            continue

        neighbours = getattr(original_cell, "neighbourhood", None)
        if neighbours:
            clone_cell._neighbourhood_indices = [
                getattr(n, "_cell_index", None)
                for n in neighbours
                if cell_map.get(n) is not None
            ]
        else:
            clone_cell._neighbourhood_indices = []

    for original_individual, clone_individual in individual_map.items():
        neighbours = getattr(original_individual, "neighbourhood", None)
        if neighbours:
            clone_individual._neighbourhood_indices = [
                getattr(n, "_individual_index", None)
                for n in neighbours
                if individual_map.get(n) is not None
            ]
        else:
            clone_individual._neighbourhood_indices = []

    return cell_map, individual_map, cloned_cells, cloned_individuals


def _update_cached_clones(cell_map, individual_map, local_world, model_view):
    """Update cached clones with current dynamic attribute values.

    This is the fast path for serialization after the first call.
    Only copies dynamic attributes, not static structure.

    Parameters
    ----------
    cell_map : dict
        Mapping of original cells to their cached clones.
    individual_map : dict
        Mapping of original individuals to their cached clones.
    local_world : _LocalWorldView
        Current year's lightweight world view.
    model_view : _ModelConfigView
        Current model config view.
    """
    # Update cells with current dynamic values
    for original_cell, clone_cell in cell_map.items():
        # Update world reference
        clone_cell._world = local_world
        if model_view is not None:
            clone_cell._model = model_view

        # Copy dynamic attributes from original
        for attr, value in original_cell.__dict__.items():
            if attr in _CELL_ATTR_SKIP:
                continue
            # Skip static attrs we already set
            if attr in {"_cell_index", "_entity_alias"}:
                continue
            # Check for non-serializable
            if _contains_non_serializable_references(value):
                clone_cell.__dict__[attr] = unknown
            else:
                clone_cell.__dict__[attr] = value

    # Update individuals with current dynamic values
    for original_individual, clone_individual in individual_map.items():
        # Update world reference
        clone_individual._world = local_world
        if model_view is not None:
            clone_individual._model = model_view

        # Copy dynamic attributes from original
        for attr, value in original_individual.__dict__.items():
            if attr in _INDIVIDUAL_ATTR_SKIP:
                continue
            # Skip static attrs
            if attr in {"_individual_index", "_entity_alias"}:
                continue
            # Check for non-serializable
            if _contains_non_serializable_references(value):
                clone_individual.__dict__[attr] = unknown
            else:
                clone_individual.__dict__[attr] = value


def _build_cross_border_neighbour_buffer(
    cell_map, individual_map, local_world, model_view
):
    """Build buffer of cross-border neighbours for cross-country spreading.

    Identifies cells and individuals that are neighbours of internal entities
    but belong to other countries. Creates read-only clones of these external
    entities to enable cross-border interactions during parallel processing.

    Parameters
    ----------
    cell_map : dict
        Mapping of original internal cells to their clones.
    individual_map : dict
        Mapping of original internal individuals to their clones.
    local_world : _LocalWorldView
        Current year's lightweight world view.
    model_view : _ModelConfigView
        Current model config view.

    Returns
    -------
    dict
        Buffer containing:
        - 'external_cells': list of cloned external cell dicts
        - 'external_individuals': list of cloned external individual dicts
        - 'cell_external_neighbours': dict mapping internal cell_index to list
          of external cell_indices
        - 'individual_external_neighbours': dict mapping internal
          individual_index to list of external individual_indices
    """
    # Collect cross-border neighbours (cells not in this country)
    external_cells = {}  # cell_index -> original cell
    external_individuals = {}  # individual_index -> original individual

    # Maps for internal -> cross-border neighbour indices
    cell_external_neighbours = {}  # internal cell_index -> [ext cell_indices]
    individual_external_neighbours = {}  # internal ind_index -> [ext ind_idx]

    # Find external cell neighbours
    for original_cell in cell_map.keys():
        cell_idx = getattr(original_cell, "_cell_index", None)
        neighbours = getattr(original_cell, "neighbourhood", None) or []

        ext_neighbour_indices = []
        for neighbour in neighbours:
            if neighbour not in cell_map:
                # This is an external cell
                ext_idx = getattr(neighbour, "_cell_index", None)
                if ext_idx is not None:
                    external_cells[ext_idx] = neighbour
                    ext_neighbour_indices.append(ext_idx)

        if ext_neighbour_indices and cell_idx is not None:
            cell_external_neighbours[cell_idx] = ext_neighbour_indices

    # Find external individual neighbours
    for original_individual in individual_map.keys():
        ind_idx = getattr(original_individual, "_individual_index", None)
        neighbours = getattr(original_individual, "neighbourhood", None) or []

        ext_neighbour_indices = []
        for neighbour in neighbours:
            if neighbour not in individual_map:
                # This is an external individual
                ext_idx = getattr(neighbour, "_individual_index", None)
                if ext_idx is not None:
                    external_individuals[ext_idx] = neighbour
                    ext_neighbour_indices.append(ext_idx)

        if ext_neighbour_indices and ind_idx is not None:
            individual_external_neighbours[ind_idx] = ext_neighbour_indices

    # Clone external cells (read-only snapshots)
    cloned_external_cells = []
    for ext_idx, ext_cell in external_cells.items():
        cell_state = _clone_external_entity(
            ext_cell, _CELL_ATTR_SKIP, local_world, model_view
        )
        cell_state["_cell_index"] = ext_idx
        cell_state["_is_external"] = True
        cloned_external_cells.append(cell_state)

    # Clone external individuals (read-only snapshots)
    cloned_external_individuals = []
    for ext_idx, ext_individual in external_individuals.items():
        ind_state = _clone_external_entity(
            ext_individual, _INDIVIDUAL_ATTR_SKIP, local_world, model_view
        )
        ind_state["_individual_index"] = ext_idx
        ind_state["_is_external"] = True
        # Store cell index for reconstruction
        ext_cell = getattr(ext_individual, "_cell", None)
        if ext_cell is not None:
            ind_state["_cell_index"] = getattr(ext_cell, "_cell_index", None)
        cloned_external_individuals.append(ind_state)

    return {
        "external_cells": cloned_external_cells,
        "external_individuals": cloned_external_individuals,
        "cell_external_neighbours": cell_external_neighbours,
        "individual_external_neighbours": individual_external_neighbours,
    }


def _clone_external_entity(entity, skip_attrs, local_world, model_view):
    """Create a serializable state dict from an external entity.

    Parameters
    ----------
    entity : Cell or Individual
        The external entity to clone.
    skip_attrs : frozenset
        Attributes to skip (contain circular references).
    local_world : _LocalWorldView
        Current year's lightweight world view.
    model_view : _ModelConfigView
        Current model config view.

    Returns
    -------
    dict
        Serializable state dictionary with entity's dynamic attributes.
    """
    state = {
        "_entity_class_name": entity.__class__.__name__,
        "_entity_class_module": entity.__class__.__module__,
    }

    # Copy dynamic attributes
    for attr, value in entity.__dict__.items():
        if attr in skip_attrs:
            continue
        # Skip static attrs that we handle separately
        if attr in {"_cell_index", "_individual_index", "_entity_alias"}:
            continue
        # Check for non-serializable
        if _contains_non_serializable_references(value):
            state[attr] = unknown
        else:
            state[attr] = value

    return state


def _assign_country_references(country):
    """Rebuild object references after deserialization.

    Re-binds cloned cells and individuals to the country, sets up
    social system links, and reconstructs neighbourhood graphs.

    Parameters
    ----------
    country : Country
        The deserialized country to fix up.
    """
    cells = getattr(country, "_direct_cells", None) or set()
    world = getattr(country, "_world", None)

    # Get model reference
    model_view = None
    if world is not None:
        model_view = getattr(world, "_model", None)
        if model_view is None and hasattr(world, "model"):
            try:
                model_view = world.model
            except Exception:
                model_view = None

    def _attach_model_reference(entity):
        """Attach model reference to entity."""
        if entity is None or model_view is None:
            return
        entity._model = model_view
        try:
            setattr(entity, "model", model_view)
        except (AttributeError, TypeError):
            pass

    _attach_model_reference(country)

    # Ensure cells is a set
    cells = set(cells)
    country._direct_cells = cells
    country._next_lower_social_systems = set()

    # Build cell lookup for relinking individuals
    cell_lookup = {}
    for cell in cells:
        key = (getattr(cell, "_uid", None), getattr(cell, "_cell_index", None))
        cell_lookup[key] = cell

    # Rebind cells to country
    for cell in cells:
        cell._social_system = country
        cell._social_systems = [country]
        cell._world = world
        _attach_model_reference(cell)

        if not hasattr(cell, "_individuals") or cell._individuals is None:
            cell._individuals = set()

        individuals = set(cell._individuals)
        for individual in individuals:
            _restore_individual_cell(
                individual, fallback_cell=cell, cell_lookup=cell_lookup
            )
            individual._world = world
            individual._social_system = country
            individual._social_systems = [country]
            _attach_model_reference(individual)
            if (
                not hasattr(individual, "neighbourhood")
                or individual.neighbourhood is None
            ):  # noqa: E501
                individual.neighbourhood = []

        cell._individuals = individuals
        cell._direct_individuals = set(individuals)

    # Update country-level individual caches
    all_individuals = set()
    for cell in cells:
        all_individuals.update(getattr(cell, "_individuals", set()))
    country._direct_individuals = set(all_individuals)
    country._individuals = set(all_individuals)

    # Relink any orphaned individuals
    for individual in list(
        getattr(country, "_direct_individuals", set())
    ):  # noqa: E501
        cell = getattr(individual, "_cell", None)
        if cell is None:
            relinked = _restore_individual_cell(
                individual, fallback_cell=None, cell_lookup=cell_lookup
            )
            if relinked is not None:
                relinked._individuals.add(individual)
                relinked._direct_individuals.add(individual)
                _attach_model_reference(relinked)
        _attach_model_reference(individual)

    # Rebuild neighbourhood graphs from stored indices
    external_buffer = getattr(country, "_cross_border_neighbour_buffer", None)
    _rebuild_neighbourhood_graphs(cells, all_individuals, external_buffer)

    # Clean up temporary buffer attribute
    if hasattr(country, "_cross_border_neighbour_buffer"):
        del country._cross_border_neighbour_buffer


def _rebuild_neighbourhood_graphs(cells, individuals, external_buffer=None):
    """Rebuild neighbourhood graphs from stored indices.

    During cloning, neighbourhoods are stored as indices to avoid
    recursion. This function converts them back to object references.
    Also reconstructs cross-border neighbours from the buffer for cross-country
    interactions.

    Parameters
    ----------
    cells : set
        Cell entities with ``_neighbourhood_indices``.
    individuals : set
        Individual entities with ``_neighbourhood_indices``.
    external_buffer : dict, optional
        Buffer containing cross-border neighbour data for cross-country
        spreading.
    """
    # Build lookup tables for internal entities
    cell_by_index = {}
    for cell in cells:
        idx = getattr(cell, "_cell_index", None)
        if idx is not None:
            cell_by_index[idx] = cell

    individual_by_index = {}
    for individual in individuals:
        idx = getattr(individual, "_individual_index", None)
        if idx is not None:
            individual_by_index[idx] = individual

    # Reconstruct external entities from buffer
    external_cell_by_index = {}
    external_individual_by_index = {}
    cell_external_neighbours = {}
    individual_external_neighbours = {}

    if external_buffer is not None:
        # Reconstruct external cells
        for cell_state in external_buffer.get("external_cells", []):
            ext_cell = _reconstruct_external_cell(cell_state)
            if ext_cell is not None:
                idx = getattr(ext_cell, "_cell_index", None)
                if idx is not None:
                    external_cell_by_index[idx] = ext_cell

        # Reconstruct external individuals
        for ind_state in external_buffer.get("external_individuals", []):
            ext_ind = _reconstruct_external_individual(
                ind_state, external_cell_by_index
            )
            if ext_ind is not None:
                idx = getattr(ext_ind, "_individual_index", None)
                if idx is not None:
                    external_individual_by_index[idx] = ext_ind

        cell_external_neighbours = external_buffer.get(
            "cell_external_neighbours", {}
        )
        individual_external_neighbours = external_buffer.get(
            "individual_external_neighbours", {}
        )

    # Rebuild cell neighbourhoods (internal + external)
    for cell in cells:
        cell_idx = getattr(cell, "_cell_index", None)
        indices = getattr(cell, "_neighbourhood_indices", None)

        # Start with internal neighbours
        if indices:
            cell.neighbourhood = [
                cell_by_index[i] for i in indices if i in cell_by_index
            ]
        elif not hasattr(cell, "neighbourhood") or cell.neighbourhood is None:
            cell.neighbourhood = []

        # Add external neighbours
        if cell_idx is not None and cell_idx in cell_external_neighbours:
            for ext_idx in cell_external_neighbours[cell_idx]:
                if ext_idx in external_cell_by_index:
                    cell.neighbourhood.append(external_cell_by_index[ext_idx])

        # Clean up temporary attribute
        if hasattr(cell, "_neighbourhood_indices"):
            del cell._neighbourhood_indices

    # Rebuild individual neighbourhoods (internal + external)
    for individual in individuals:
        ind_idx = getattr(individual, "_individual_index", None)
        indices = getattr(individual, "_neighbourhood_indices", None)

        # Start with internal neighbours
        if indices:
            individual.neighbourhood = [
                individual_by_index[i]
                for i in indices
                if i in individual_by_index
            ]
        elif (
            not hasattr(individual, "neighbourhood")
            or individual.neighbourhood is None
        ):
            individual.neighbourhood = []

        # Add external neighbours
        if ind_idx is not None and ind_idx in individual_external_neighbours:
            for ext_idx in individual_external_neighbours[ind_idx]:
                if ext_idx in external_individual_by_index:
                    individual.neighbourhood.append(
                        external_individual_by_index[ext_idx]
                    )

        if hasattr(individual, "_neighbourhood_indices"):
            del individual._neighbourhood_indices


def _reconstruct_external_cell(cell_state):
    """Reconstruct an external cell from its serialized state.

    Creates a lightweight proxy cell object with the buffered attributes.
    The cell is marked as external and read-only.

    Parameters
    ----------
    cell_state : dict
        Serialized cell state from the external buffer.

    Returns
    -------
    object or None
        Reconstructed cell proxy, or None if reconstruction fails.
    """
    if cell_state is None:
        return None

    # Create a simple proxy object
    proxy = _CrossBorderNeighbour()
    proxy._is_external = True
    proxy._entity_alias = "cell"

    # Restore attributes from state
    for key, value in cell_state.items():
        if key.startswith("_entity_class"):
            continue
        setattr(proxy, key, value)

    # Initialize empty collections (external cells don't have their own
    # individuals)
    proxy.neighbourhood = []
    proxy._individuals = set()

    return proxy


def _reconstruct_external_individual(ind_state, external_cell_by_index):
    """Reconstruct an external individual from its serialized state.

    Creates a lightweight proxy individual object with the buffered attributes.
    The individual is marked as external and read-only.

    Parameters
    ----------
    ind_state : dict
        Serialized individual state from the external buffer.
    external_cell_by_index : dict
        Lookup table for external cells by index.

    Returns
    -------
    object or None
        Reconstructed individual proxy, or None if reconstruction fails.
    """
    if ind_state is None:
        return None

    # Create a simple proxy object
    proxy = _CrossBorderNeighbour()
    proxy._is_external = True
    proxy._entity_alias = "individual"

    # Restore attributes from state
    cell_idx = None
    for key, value in ind_state.items():
        if key.startswith("_entity_class"):
            continue
        if key == "_cell_index":
            cell_idx = value
            continue
        setattr(proxy, key, value)

    # Link to external cell if available
    if cell_idx is not None and cell_idx in external_cell_by_index:
        proxy._cell = external_cell_by_index[cell_idx]
    else:
        proxy._cell = None

    # Initialize empty neighbourhood (external individuals don't have
    # their own neighbours in this context)
    proxy.neighbourhood = []

    return proxy


class _CrossBorderNeighbour:
    """Cross-border neighbour (cell or individual) from another country.

    When a country is deserialized on a Dask worker, neighbours in other
    countries are reconstructed as these objects and added to each border
    entity's ``neighbourhood``. Use them for read-only access to neighbour
    status (e.g., ``n.tillage``, ``n.soilc``, ``n.cropyield``).

    Cross-border neighbours have:
    - All dynamic attributes from the original entity (one timestep behind)
    - ``_is_external = True`` flag
    - Empty neighbourhood (they don't have their own neighbours in this
    context)

    They do NOT have:
    - Working model/world references
    - Ability to modify LPJmL data
    - Social system links
    """

    def __init__(self):
        """Initialize empty cross-border neighbour."""
        pass

    def __repr__(self):
        """Return string representation."""
        entity_type = getattr(self, "_entity_alias", "entity")
        idx = getattr(
            self,
            "_individual_index",
            getattr(self, "_cell_index", "?"),
        )
        return f"<CrossBorderNeighbour {entity_type}[{idx}]>"


# =============================================================================
# INDIVIDUAL TYPE HELPERS
# =============================================================================
#
# These functions support dynamic individual access by type.
#
# =============================================================================


def _normalize_label(value):
    """Normalize a label for case-insensitive comparison.

    Parameters
    ----------
    value : str or None
        Label to normalize.

    Returns
    -------
    str or None
        Lowercase, stripped label, or None.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _individual_type_labels(individual):
    """Get all labels for an individual type.

    Returns both singular and plural forms of class name,
    entity alias, and type attribute.

    Parameters
    ----------
    individual : Individual
        The individual to get labels for.

    Returns
    -------
    set of str
        All applicable labels (e.g., {'farmer', 'farmers'} for Farmer class).
    """
    labels = set()
    candidates = [
        getattr(individual.__class__, "__name__", None),
        getattr(individual.__class__, "_entity_alias", None),
    ]

    type_attr = getattr(individual, "type", None)
    if isinstance(type_attr, str):
        candidates.append(type_attr)

    for candidate in candidates:
        norm = _normalize_label(candidate)
        if not norm:
            continue
        labels.add(norm)
        if not norm.endswith("s"):
            labels.add(f"{norm}s")

    return labels


def _restore_individual_cell(individual, *, fallback_cell, cell_lookup):
    """Recreate individual->cell link using stored metadata.

    Parameters
    ----------
    individual : Individual
        Individual to relink.
    fallback_cell : Cell
        Cell to use if lookup fails.
    cell_lookup : dict
        Mapping from (uid, index) to Cell.

    Returns
    -------
    Cell or None
        The cell the individual was linked to.
    """
    target = fallback_cell
    if target is None:
        key = (
            getattr(individual, "_cell_uid", None),
            getattr(individual, "_cell_index", None),
        )
        target = cell_lookup.get(key)

    if target is None:
        return None

    individual._cell = target
    return target
