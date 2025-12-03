"""Output writing classes for defining output variables.

This module provides:
- Output: Container class for output variable metadata
- OutputTableMixin: Mixin for entity classes that can define output variables

Entity classes inherit from OutputTableMixin and define output_variables
to specify which attributes should be written to output.
"""

from typing import List, Iterator, Tuple, Any


class Output:
    """Container class for output variable metadata.

    This class stores Variable objects as attributes, providing a simple
    way to define which attributes of an entity should be written to output.
    Models define output variables like:

    .. code-block:: python

        output_variables = Output(
            var1=Variable("Variable Name", "description", unit=DAU.unit),
            var2=Variable("Another Variable", "description")
        )

    Parameters
    ----------
    **kwargs
        Variable objects from pycopancore.data_model.variable.Variable
        Each keyword becomes an attribute storing the Variable object.

    Attributes
    ----------
    names : List[str]
        List of output variable names (property)

    Examples
    --------
    >>> from pycopancore.data_model.variable import Variable
    >>> from pycopancore.data_model.master_data_model import (
    ...     dimensions_and_units as DAU
    ... )
    >>>
    >>> output_vars = Output(
    ...     soilc=Variable(
    ...         "soil organic carbon", "description", unit=DAU.gC_per_m2
    ...     ),
    ...     yield_var=Variable("crop yield", "description", unit=DAU.gC_per_m2)
    ... )
    >>> output_vars.names
    ['soilc', 'yield_var']
    >>> output_vars.get('soilc').name
    'soil organic carbon'
    """

    def __init__(self, **kwargs):
        """Initialize the Output container with variable objects.

        Parameters
        ----------
        **kwargs
            Variable objects from pycopancore.data_model.variable.Variable.
            Each keyword argument becomes an attribute storing the Variable
            object.

        Examples
        --------
        >>> from pycopancore.data_model.variable import Variable
        >>> output_vars = Output(soilc=Variable("soil carbon", "desc"))
        >>> output_vars.soilc.name
        'soil carbon'
        """
        # Use explicit setattr instead of __dict__.update for clarity
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Invalidate names cache after adding variables
        self._invalidate_names_cache()

    @property
    def names(self) -> List[str]:
        """Return list of output variable names.

        Returns
        -------
        List[str]
            List of variable names (attribute names), excluding private
            attributes that start with underscore.

        Examples
        --------
        >>> output_vars = Output(var1=Variable("Var1", "desc"))
        >>> output_vars.names
        ['var1']
        """
        # Cache the result to avoid repeated __dict__.keys() calls
        # This is critical for performance during pickling
        if not hasattr(self, '_cached_names'):
            # Filter out private attributes (starting with _) to only return
            # public variable names
            self._cached_names = [
                k for k in self.__dict__.keys()
                if not k.startswith("_")
            ]
        return self._cached_names

    def _invalidate_names_cache(self):
        """Invalidate the cached names list (call after adding variables)."""
        if hasattr(self, '_cached_names'):
            delattr(self, '_cached_names')

    def get(self, name: str, default=None):
        """Get variable object by name.

        Parameters
        ----------
        name : str
            Variable name to retrieve.
        default : any, optional
            Default value to return if variable not found.
            Default is None.

        Returns
        -------
        Variable or any
            Variable object if found, otherwise returns `default`.

        Examples
        --------
        >>> output_vars = Output(soilc=Variable("soil carbon", "desc"))
        >>> var = output_vars.get('soilc')
        >>> var.name
        'soil carbon'
        >>> output_vars.get('nonexistent', 'not found')
        'not found'
        """
        return getattr(self, name, default)

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Iterate over (name, variable) pairs.

        Yields
        ------
        Tuple[str, Variable]
            Tuples of (variable_name, variable_object) for each output
            variable.

        Examples
        --------
        >>> output_vars = Output(var1=Variable("Var1", "desc"))
        >>> for name, var in output_vars:
        ...     print(f"{name}: {var.name}")
        var1: Var1
        """
        for name in self.names:
            yield name, getattr(self, name)

    def __repr__(self) -> str:
        """Return string representation of Output object.

        Returns
        -------
        str
            String representation showing all variable names.

        Examples
        --------
        >>> output_vars = Output(var1=Variable("Var1", "desc"))
        >>> repr(output_vars)
        'Output(var1)'
        """
        var_names = ", ".join(self.names)
        return f"Output({var_names})"


class OutputTableMixin:
    """Mixin for entity classes that can define output variables.

    Entity classes (World, Cell, Region, Individual) should inherit from this
    mixin and define output_variables to specify which attributes should be
    written to output.

    Required implementations:
    - output_variables: Output class attribute (models override this)
    - model: property returning Component model instance
    - get_defined_outputs(): method returning list of output variable names

    Attributes
    ----------
    output_variables : Output
        Class attribute defining which variables to output (models override
        this)
    model : Component
        Instance attribute reference to the model component
    """

    # Class attribute - models should override this
    output_variables = Output()

    @property
    def model(self):
        """Reference to the Component model instance.

        Subclasses must implement this property.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement 'model' property"
        )

    def get_defined_outputs(self) -> List[str]:
        """Get list of output variable names based on config.

        Default implementation reads from model.config.coupled_config.output
        using the entity type name (lowercased) as the config key.

        Returns
        -------
        List[str]
            List of variable names to output (filtered by config).
            Returns empty list if model/config not available or on error.

        Notes
        -----
        The config key is determined by lowercasing the class name.
        For example, 'Farmer' -> 'farmer', 'World' -> 'world'.
        """
        # Early return if model reference is not available
        if not hasattr(self, "model") or self.model is None:
            return []
        if not hasattr(self.model, "config"):
            return []

        try:
            # Determine config key from entity type name
            # (e.g., 'Farmer' -> 'farmer')
            entity_type = self.__class__.__name__.lower()

            # Get list of enabled outputs from config for this entity type
            config_outputs = (
                self.model.config.coupled_config.output.to_dict().get(
                    entity_type, []
                )
            )

            # Filter class-level output_variables to only include those
            # enabled in config
            result = [
                var
                for var in self.__class__.output_variables.names
                if var in config_outputs
            ]

            return result
        except Exception:
            # Return empty list on any error (config access, missing keys,
            # etc.)
            return []
