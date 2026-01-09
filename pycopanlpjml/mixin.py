"""Mixin classes for entity aliasing in pycopanlpjml.

This module provides mixins that enable semantic aliases for pycopancore
entity relationships. These aliases allow more intuitive access to entities
(e.g., `cell.country` instead of `cell.social_system`).

Key Features
------------
- Automatic pluralization of entity names
- Dynamic property creation for aliases
- Support for both singular and plural relationships
- Configurable alias mappings

Classes
-------
AliasMixin
    Mixin providing semantic alias support for pycopancore entities.

Functions
---------
pluralize
    Convert singular English words to their plural form.

Examples
--------
>>> class MyCell(AliasMixin, base.Cell):
...     pass
>>> cell = MyCell(country=some_country)
>>> cell.country  # Returns the same as cell.social_system
"""

from typing import Any, Dict, List


# ============================================================================
# Helper Functions
# ============================================================================

def pluralize(word: str) -> str:
    """Convert a singular word to its plural form using common English rules.

    Applies standard English pluralization rules to handle common cases.
    This is used internally to generate plural alias names from entity
    class names.

    Parameters
    ----------
    word : str
        The singular word to pluralize.

    Returns
    -------
    str
        The pluralized word.

    Examples
    --------
    >>> pluralize("country")
    'countries'
    >>> pluralize("cell")
    'cells'
    >>> pluralize("class")
    'classes'
    >>> pluralize("box")
    'boxes'
    """
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        # country -> countries, city -> cities
        return word[:-1] + "ies"
    elif word.endswith(("s", "x", "z", "ch", "sh")):
        # class -> classes, box -> boxes
        return word + "es"
    else:
        # region -> regions, world -> worlds
        return word + "s"


# ============================================================================
# Mixin Classes
# ============================================================================

class AliasMixin:
    """Mixin providing semantic aliases for pycopancore entity relationships.

    This mixin enables more intuitive attribute access by mapping semantic
    names (like 'country', 'region') to pycopancore's internal naming
    conventions (like 'social_system'). It supports both singular and
    plural relationships.

    The mixin works by:
    1. Accepting alias names in constructor arguments
    2. Dynamically creating properties that delegate to the real attributes
    3. Supporting automatic pluralization for collection relationships

    Attributes
    ----------
    _alias_map : dict
        Class-level mapping from canonical attribute names to their aliases.
        Keys are the internal pycopancore attribute names, values are lists
        of acceptable alias names.

    Notes
    -----
    - Aliases are case-sensitive and must match exactly
    - Properties are created at the class level for efficiency
    - Existing attributes are never overwritten

    Examples
    --------
    Using in a class definition:

    >>> class MyCell(AliasMixin, base.Cell):
    ...     _entity_alias = "cell"
    ...
    >>> cell = MyCell(country=some_country)  # Uses alias
    >>> cell.country  # Same as cell.social_system
    >>> cell.social_system  # Original attribute also works

    Using plural aliases:

    >>> world.countries  # Returns list of Country instances
    >>> world.regions    # Returns list of Region instances
    """

    # Maps destination attribute names to a list of aliases for incoming kwargs
    _alias_map: Dict[str, List[str]] = {
        "social_system": ["region", "country", "worldregion"],
        "social_systems": ["regions", "countries", "worldregions"],
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the mixin and set up aliases.

        Processes constructor arguments to redirect aliases to their
        canonical attribute names, then creates dynamic properties
        for attribute access.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments that may include aliased names.
            Recognized aliases are remapped to their canonical names
            before being passed to the parent class.

        Notes
        -----
        This method should be called as part of a cooperative multiple
        inheritance chain using `super().__init__(**kwargs)`.
        """
        # Accept alias keys and redirect them to real internal keys
        for true_name, aliases in self._alias_map.items():
            for alias in aliases:
                if alias in kwargs:
                    kwargs[true_name] = kwargs.pop(alias)

            # Assign if present in kwargs (does not override existing init
            # logic)
            if true_name in kwargs:
                setattr(self, true_name, kwargs[true_name])

        # After init, inject dynamic aliases
        self._add_all_alias_properties()

    def _add_all_alias_properties(self) -> None:
        """Create dynamic properties for all configured aliases.

        Iterates through `_alias_map` and creates getter/setter properties
        on the class that delegate to the canonical attribute names.

        Notes
        -----
        - Properties are added at the class level, not the instance level
        - Existing attributes are not overwritten to avoid conflicts
        - Uses lambda closures to capture the correct attribute name
        """
        cls = self.__class__
        for true_name, aliases in self._alias_map.items():
            for alias in aliases:
                # Avoid overwriting existing attributes
                if not hasattr(cls, alias):
                    setattr(
                        cls,
                        alias,
                        property(
                            lambda self, tn=true_name: getattr(self, tn, None),
                            lambda self, v, tn=true_name: setattr(self, tn, v),
                        ),
                    )

    def _add_entity_aliases(self, attr_name: str) -> None:
        """Create type-based aliases for an attribute's value(s).

        Examines the attribute's value to determine its entity type(s),
        then creates appropriately named aliases. For singular values,
        creates a singular alias. For collections, creates plural aliases
        that filter by entity type.

        Parameters
        ----------
        attr_name : str
            The name of the attribute to create aliases for.

        Notes
        -----
        This method handles both singular relationships (e.g., a cell's
        country) and plural relationships (e.g., a world's social_systems
        containing both countries and regions).

        For plural relationships, each entity type gets its own filtered
        property. For example, if `social_systems` contains both Country
        and Region instances, both `.countries` and `.regions` properties
        are created.

        Examples
        --------
        For a singular relationship:

        >>> cell._add_entity_aliases("social_system")
        >>> cell.country  # Now available if social_system is a Country

        For a plural relationship with mixed types:

        >>> world._add_entity_aliases("social_systems")
        >>> world.countries  # Filters to Country instances only
        >>> world.regions    # Filters to Region instances only
        """
        if not hasattr(self, attr_name):
            return

        val = getattr(self, attr_name)
        cls = self.__class__

        # Singular case: create alias based on entity type
        if not isinstance(val, (list, set, tuple, dict)):
            entity = val
            if entity is None:
                return
            # Get the entity's type name (lowercase class name)
            alias = getattr(
                entity.__class__, "type", entity.__class__.__name__.lower()
            )
            if not hasattr(cls, alias):
                setattr(
                    cls,
                    alias,
                    property(
                        lambda self, a=attr_name: getattr(self, a),
                        lambda self, v, a=attr_name: setattr(self, a, v),
                    ),
                )
        else:
            # Plural case: group by entity type and create filtered properties
            items = val.values() if isinstance(val, dict) else val
            alias_groups: Dict[str, List[Any]] = {}

            for item in items:
                if item is None:
                    continue
                singular = getattr(
                    item.__class__, "type", item.__class__.__name__.lower()
                )
                alias = pluralize(singular)
                alias_groups.setdefault(alias, []).append(item)

            # Add one property per plural alias that filters by type
            # Always create these to ensure they filter correctly
            for alias_name in alias_groups:
                setattr(
                    cls,
                    alias_name,
                    property(
                        lambda self, a=attr_name, an=alias_name: [
                            x
                            for x in (
                                getattr(self, a).values()
                                if isinstance(getattr(self, a), dict)
                                else getattr(self, a)
                            )
                            if pluralize(
                                getattr(
                                    x.__class__,
                                    "type",
                                    x.__class__.__name__.lower(),
                                )
                            )
                            == an
                        ],
                        lambda self, v, a=attr_name: setattr(self, a, v),
                    ),
                )

            # Add canonical alias for social_systems -> regions
            if attr_name == "social_systems" and not hasattr(cls, "regions"):
                setattr(
                    cls,
                    "regions",
                    property(
                        lambda self, a=attr_name: getattr(self, a),
                        lambda self, v, a=attr_name: setattr(self, a, v),
                    ),
                )
