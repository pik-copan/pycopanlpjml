"""Lightweight region implementations for copan:LPJmL.

Architecture:
- World owns global data
- Countries have VIEWS into World data (via isel)
- Cells have VIEWS into World data (via isel, through country indices)

All data modifications propagate automatically through the view chain.
"""

import numpy as np
import pycopancore.model_components.base.implementation as base
from pycopancore.private._simple_expressions import unknown

from pycopanlpjml.output import OutputDefinitionMixin
from pycoupler.coupler import get_countries

class Region(base.SocialSystem, OutputDefinitionMixin):
    """Base class for LPJmL-integrated regions.

    Countries have VIEWS into World data. Cells get VIEWS into the same data.
    All writes propagate through the view chain automatically.
    
    Inherits from OutputDefinitionMixin to support output variable collection
    for country-level outputs.
    """

    def __init__(
        self,
        code=None,
        name=None,
        world=None,
        indices=None,
        **kwargs,
    ):
        super().__init__(world=world, **kwargs)
        self._world = world
        self.indices = np.asarray(indices) if indices is not None else np.array([])
        self._cells = []
        self.neighbourhood = set()
        self.code = code
        self.name = name
        self.entity = "region"

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

    @property
    def input(self):
        """View into World's input data for this region's cells."""
        if self._world is None or len(self.indices) == 0:
            return None
        return self._world.input.isel(cell=self.indices)

    @input.setter
    def input(self, value):
        """Write input data back to World for this region's cells.

        Use this after modifying the subset to propagate changes to world:
            region_input = country.input
            region_input.some_var.values[:] = new_values
            country.input = region_input  # propagates to world
        """
        if self._world is None or len(self.indices) == 0:
            return
        for var_name in value.data_vars:
            self._world.input[var_name].values[self.indices] = value[var_name].values

    @property
    def output(self):
        """View into World's output data for this region's cells."""
        if self._world is None or len(self.indices) == 0:
            return None
        return self._world.output.isel(cell=self.indices)

    @output.setter
    def output(self, value):
        """Write output data back to World for this region's cells.

        Use this after modifying the subset to propagate changes to world:
            region_output = country.output
            region_output.some_var.values[:] = new_values
            country.output = region_output  # propagates to world
        """
        if self._world is None or len(self.indices) == 0:
            return
        for var_name in value.data_vars:
            self._world.output[var_name].values[self.indices] = value[var_name].values

    @property
    def to_earth(self):
        """Alias for input."""
        return self.input

    @property
    def from_earth(self):
        """Alias for output."""
        return self.output

    @property
    def grid(self):
        """View into World's grid data for this region's cells."""
        if self._world is None or len(self.indices) == 0:
            return None
        return self._world.grid.isel(cell=self.indices)

    @property
    def area(self):
        """View into World's area data for this region's cells."""
        if self._world is None or len(self.indices) == 0:
            return None
        return self._world.area.isel(cell=self.indices)

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
        code,
        world=None,
        indices=None,
        **kwargs,
    ):
        super().__init__(world=world, code=code, indices=indices, **kwargs)
        self.name = get_country_names()[self.code]
        self.entity = "country"

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
    """Country-level statistics/metrics.

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

    def __init__(self, code, name, **kwargs):
        super().__init__(code=code, name=name, **kwargs)
        self.entity = "worldregion"


def get_country_names():
    """Get mapping of country codes to names.

    Returns
    -------
    dict
        {code: {"name": str, "code": str}} mapping.
    """
    return {
        value["code"]: value["name"]
        for key, value in get_countries().items()
    }
