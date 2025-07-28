# entity_aliasing.py

class AliasMixin:
    """
    A generic mixin to:
    1. Accept semantic alias names (e.g., country=...) as constructor args.
    2. Expose attribute aliases (e.g., .country, .countries) based on instance types.
    3. Work with both singular and plural relationships.
    4. Be reusable across all pycopancore entity types.
    """

    # Maps destination attribute names to a list of aliases for incoming kwargs
    _alias_map = {
        "social_system": ["region", "country", "worldregion"],
        "social_systems": ["regions", "countries", "worldregion"],
    }

    def __init__(self, **kwargs):
        # Accept alias keys and redirect them to real internal keys
        for true_name, aliases in self._alias_map.items():
            for alias in aliases:
                if alias in kwargs:
                    kwargs[true_name] = kwargs.pop(alias)

            # Assign if present in kwargs (does not override existing init logic)
            if true_name in kwargs:
                setattr(self, true_name, kwargs[true_name])

        # After init, inject dynamic aliases
        self._add_all_alias_properties()

    def _add_all_alias_properties(self):
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
                            lambda self, v, tn=true_name: setattr(self, tn, v)
                        )
                    )

    def _add_entity_aliases(self, attr_name):
        if not hasattr(self, attr_name):
            return

        val = getattr(self, attr_name)
        cls = self.__class__

        # Singular case
        if not isinstance(val, (list, set, tuple, dict)):
            entity = val
            if entity is None:
                return
            alias = getattr(
                entity.__class__, "type", entity.__class__.__name__.lower()
            )
            if not hasattr(cls, alias):
                setattr(
                    cls,
                    alias,
                    property(
                        lambda self, a=attr_name: getattr(self, a),
                        lambda self, v, a=attr_name: setattr(self, a, v)
                    )
                )

        else:
            items = val.values() if isinstance(val, dict) else val
            alias_groups = {}
            for item in items:
                if item is None:
                    continue
                alias = getattr(item.__class__, "type", item.__class__.__name__.lower()) + "s"
                alias_groups.setdefault(alias, []).append(item)

            # Add one property per plural alias
            for alias_name in alias_groups:
                if not hasattr(cls, alias_name):
                    setattr(
                        cls,
                        alias_name,
                        property(
                            lambda self, a=attr_name, an=alias_name: [
                                x for x in getattr(self, a)
                                if getattr(
                                    x.__class__,
                                    "type",
                                    x.__class__.__name__.lower(),
                                ) + "s" == an
                            ],
                            lambda self, v, a=attr_name: setattr(self, a, v)
                        )
                    )

            # Optionally add canonical alias (e.g. worldregions for social_systems)
            if attr_name == "social_systems" and not hasattr(cls, "regions"):
                setattr(
                    cls,
                    "regions",
                    property(
                        lambda self, a=attr_name: getattr(self, a),
                        lambda self, v, a=attr_name: setattr(self, a, v)
                    )
                )
