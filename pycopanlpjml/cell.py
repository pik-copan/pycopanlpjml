"""Cell entity type for copan:LPJmL component.

This module defines the Cell class, which represents a spatial unit (grid cell)
in the copan:LPJmL coupled model. Each cell corresponds to a grid cell in the
LPJmL land surface model and provides access to earth system data (input/output)
as well as spatial metadata (coordinates, area, country).

Key Features
------------
- Integration with LPJmL via pycoupler
- View-based access to world-level xarray datasets
- Automatic cell index resolution
- Output variable support for simulation tracking
- Backward compatibility with legacy 'input'/'output' naming

Classes
-------
Cell
    LPJmL-integrating cell entity with earth system data access.

Examples
--------
>>> from pycoupler.coupler import LPJmLCoupler
>>> from pycopanlpjml import World, Cell
>>>
>>> # Initialize world with LPJmL data
>>> world = World(
...     to_earth=lpjml.read_input(copy=False),
...     from_earth=lpjml.read_historic_output(),
...     grid=lpjml.grid,
... )
>>>
>>> # Create a cell for a specific grid index
>>> cell = Cell(world=world, cell_index=0)
>>> cell.from_earth  # Access LPJmL output for this cell
>>> cell.country_code  # Get ISO 3-letter country code
"""

from typing import Any, List, Optional

import numpy as np
import pycopancore.model_components.base.implementation as base
from pycoupler.utils import warn_deprecated_alias

from .mixin import AliasMixin
from .output import Output, OutputDefinitionMixin


# ============================================================================
# Cell Entity Class
# ============================================================================

class Cell(base.Cell, AliasMixin, OutputDefinitionMixin):
    """An LPJmL-integrating cell entity.

    Cell entity type (mixin) class for the copan:LPJmL component. It inherits
    the copan:CORE cell entity and integrates LPJmL input/output data, grid
    information, and country metadata via pycoupler data structures.

    Each cell instance holds views (not copies) of data from the world-level
    datasets, enabling efficient memory usage and automatic synchronization.

    Parameters
    ----------
    world : World
        Reference to the World instance containing global datasets.
    country : Country, optional
        Country instance this cell belongs to.
    cell_index : int, optional
        Global cell index in the world. Either this or `grid` must be provided.
    to_earth : xarray.Dataset, optional
        Data sent to Earth system (LPJmL). Used to extract cell index if
        `cell_index` is not provided.
    from_earth : xarray.Dataset, optional
        Data received from Earth system (LPJmL). Used to extract cell index
        if `cell_index` is not provided.
    grid : xarray.DataArray, optional
        Grid data. Used to extract cell index if `cell_index` is not provided.
    input : xarray.Dataset, optional
        Deprecated alias for `to_earth`.
    output : xarray.Dataset, optional
        Deprecated alias for `from_earth`.
    **kwargs : dict
        Additional keyword arguments passed to parent classes.

    Attributes
    ----------
    output_variables : Output
        Class-level container for output variable definitions. Models can
        override this to specify which attributes to track during simulation.
    _entity_alias : str
        Entity type identifier used by AliasMixin.
    neighbourhood : list
        List of neighboring Cell instances for spatial interactions.

    Notes
    -----
    - Cell index is required and can be provided directly or extracted from
      the grid data.
    - Properties like `to_earth`, `from_earth`, and `grid` return views on
      world-level data, not copies.
    - The `country` property is an alias for `social_system` from pycopancore.

    Examples
    --------
    Creating a cell with explicit index:

    >>> cell = Cell(world=world, cell_index=42)

    Creating a cell from grid data:

    >>> cell = Cell(world=world, grid=world.grid.isel(cell=42))

    Accessing earth system data:

    >>> cell.to_earth["pft_harvest_frac"]  # Data sent to LPJmL
    >>> cell.from_earth["soilc"]  # Data received from LPJmL

    Accessing spatial metadata:

    >>> cell.country_code  # e.g., 'DEU'
    >>> cell.area  # Cell area in m²
    >>> cell.grid  # Grid coordinates
    """

    # Output variables (models can override this)
    output_variables = Output()

    # Entity type alias for AliasMixin
    _entity_alias = "cell"

    # ========================================================================
    # Initialization
    # ========================================================================

    def __init__(
        self,
        world: Optional[Any] = None,
        country: Optional[Any] = None,
        cell_index: Optional[int] = None,
        to_earth: Optional[Any] = None,
        from_earth: Optional[Any] = None,
        grid: Optional[Any] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a Cell instance.

        Parameters
        ----------
        world : World, optional
            Reference to World instance (required for data access).
        country : Country, optional
            Country instance this cell belongs to.
        cell_index : int, optional
            Global cell index. Either this or `grid` must be provided.
        to_earth : xarray.Dataset, optional
            Data sent to Earth system (used to extract cell index if
            `cell_index` not provided).
        from_earth : xarray.Dataset, optional
            Data received from Earth system (used to extract cell index if
            `cell_index` not provided).
        grid : xarray.DataArray, optional
            Grid data (used to extract cell index if `cell_index` not
            provided).
        input : xarray.Dataset, optional
            Deprecated alias for `to_earth`.
        output : xarray.Dataset, optional
            Deprecated alias for `from_earth`.
        **kwargs : dict
            Additional keyword arguments passed to parent classes.

        Raises
        ------
        ValueError
            If neither `cell_index` nor `grid` is provided, or if cell index
            cannot be determined from the grid.
        """
        # Backward compatibility: accept legacy parameter names
        if to_earth is None and input is not None:
            to_earth = input
        if from_earth is None and output is not None:
            from_earth = output

        # Pass world to base class (pycopancore handles world property)
        if "world" not in kwargs:
            kwargs["world"] = world

        # Map country to social_system for pycopancore compatibility
        if country is not None and "social_system" not in kwargs:
            kwargs["social_system"] = country

        super().__init__(**kwargs)

        # Determine cell index from various sources
        self._cell_index = self._determine_cell_index(cell_index, grid)

        # Initialize neighbourhood for spatial interactions
        self.neighbourhood: List[Any] = []

    def _determine_cell_index(
        self,
        cell_index: Optional[int],
        grid: Optional[Any],
    ) -> int:
        """Determine the cell index from available sources.

        Parameters
        ----------
        cell_index : int, optional
            Explicitly provided cell index.
        grid : xarray.DataArray, optional
            Grid data containing cell coordinate.

        Returns
        -------
        int
            The determined cell index.

        Raises
        ------
        ValueError
            If cell index cannot be determined from the provided arguments.
        """
        if cell_index is not None:
            return cell_index

        if grid is not None:
            return self._extract_cell_index_from_grid(grid)

        raise ValueError("Either cell_index or grid must be provided")

    def _extract_cell_index_from_grid(self, grid: Any) -> int:
        """Extract cell index from grid data.

        Parameters
        ----------
        grid : xarray.DataArray or similar
            Grid data containing cell coordinate information.

        Returns
        -------
        int
            The cell index extracted from the grid.

        Raises
        ------
        ValueError
            If cell index cannot be extracted from the grid structure.
        """
        # xarray DataArray with cell attribute
        if hasattr(grid, "cell"):
            cell_coord = grid.cell
            if hasattr(cell_coord, "values"):
                cell_vals = cell_coord.values
                return int(
                    cell_vals.item() if cell_vals.shape == () else cell_vals[0]
                )
            return int(cell_coord)

        # Object with coords dictionary containing 'cell'
        if hasattr(grid, "coords") and "cell" in grid.coords:
            cell_vals = grid.coords["cell"]
            return int(cell_vals if np.isscalar(cell_vals) else cell_vals[0])

        raise ValueError("Cannot determine cell index from grid")

    # ========================================================================
    # Country Properties
    # ========================================================================

    @property
    def country(self) -> Optional[Any]:
        """Get the country instance this cell belongs to.

        This is an alias for `social_system` for LPJmL compatibility.

        Returns
        -------
        Country or None
            The Country instance, or None if not assigned.
        """
        return self.social_system

    @property
    def country_code(self) -> Optional[str]:
        """Get the ISO 3-letter country code for this cell.

        Retrieves the country code from the world's country_code array
        using this cell's index.

        Returns
        -------
        str or None
            ISO 3-letter country code (e.g., 'DEU', 'FRA'), or None if
            not available.
        """
        if self._cell_index is None:
            return None

        world = getattr(self, "_world", None)
        if world is None or world.country_code is None:
            return None

        local_idx = self._resolve_local_index(world)
        if local_idx is None:
            return None

        if not hasattr(world.country_code, "values"):
            return None

        return str(world.country_code.values[local_idx])

    @country_code.setter
    def country_code(self, value: str) -> None:
        """Set the country code for this cell.

        Country codes can change due to political boundaries or administrative
        changes.

        Parameters
        ----------
        value : str
            ISO 3-letter country code (e.g., 'DEU', 'FRA').

        Raises
        ------
        ValueError
            If cell index is not initialized, world data is not available,
            or the local index cannot be resolved.
        """
        if self._cell_index is None:
            raise ValueError("Cannot set country_code: cell index not initialized")  # noqa: E501

        world = getattr(self, "_world", None)
        if world is None or world.country_code is None:
            raise ValueError("Country data not available")

        if not hasattr(world.country_code, "values"):
            raise ValueError("world.country_code must be an array-like with values")  # noqa: E501

        local_idx = self._resolve_local_index(world)
        if local_idx is None:
            raise ValueError("Cannot resolve cell index for world country data")  # noqa: E501

        world.country_code.values[local_idx] = value

    # ========================================================================
    # Cell Index and Resolution
    # ========================================================================

    @property
    def cell_index(self) -> Optional[int]:
        """Get the global cell index.

        Returns
        -------
        int or None
            The global cell index, or None if not set.
        """
        return self._cell_index

    def _resolve_local_index(self, world: Any) -> Optional[int]:
        """Map global cell index to world-specific local index.

        This method handles the case where a cell's global index needs to
        be translated to a position in a subset world view (e.g., for
        country-specific operations).

        Parameters
        ----------
        world : World or _LocalWorldView
            The world or world view to resolve the index for.

        Returns
        -------
        int or None
            The local index in the world's arrays, or None if the world
            is None.

        Raises
        ------
        ValueError
            If the cell index is not present in the world view's mapping.
        """
        if world is None:
            return None

        if hasattr(world, "_global_to_local"):
            mapping = getattr(world, "_global_to_local", None)
            if mapping is None:
                return None
            if self._cell_index not in mapping:
                raise ValueError(
                    f"Cell index {self._cell_index} not present in this world view"  # noqa: E501
                )
            return mapping[self._cell_index]

        return self._cell_index

    def _selector_for_world(self, world: Any) -> Optional[List[int]]:
        """Get xarray-compatible selector for this cell (preserves dimension).

        Parameters
        ----------
        world : World
            The world to get the selector for.

        Returns
        -------
        list of int or None
            A single-element list containing the local index, suitable for
            xarray's isel() with cell dimension preserved.
        """
        idx = self._resolve_local_index(world)
        if idx is None:
            return None
        return [idx]

    def _selector_for_world_drop_dim(self, world: Any) -> Optional[int]:
        """Get xarray-compatible selector that drops the cell dimension.

        Parameters
        ----------
        world : World
            The world to get the selector for.

        Returns
        -------
        int or None
            The local index as an integer, suitable for xarray's isel()
            which will drop the cell dimension.
        """
        idx = self._resolve_local_index(world)
        if idx is None:
            return None
        return idx

    # ========================================================================
    # Earth System Data Properties
    # ========================================================================

    @property
    def to_earth(self) -> Optional[Any]:
        """Get data sent to Earth system (LPJmL) for this cell.

        Returns a view on the world's to_earth dataset restricted to this
        cell. The cell dimension is dropped, so indexing is (band, time)
        not (cell, band, time).

        Formerly called 'input'.

        Returns
        -------
        xarray.Dataset or None
            Cell-specific view of the to_earth dataset, or None if not
            available.
        """
        if self._cell_index is None:
            return None

        world = getattr(self, "_world", None)
        if world is None or world.to_earth is None:
            return None

        selector = self._selector_for_world_drop_dim(world)
        if selector is None:
            return None

        if hasattr(world.to_earth, "isel"):
            return world.to_earth.isel(cell=selector)

        return world.to_earth

    @to_earth.setter
    def to_earth(self, value: Any) -> None:
        """Set data sent to Earth system (LPJmL) for this cell.

        Writes directly into the world's in-memory to_earth dataset.

        Parameters
        ----------
        value : xarray.Dataset
            Dataset containing variables to write.

        Raises
        ------
        ValueError
            If cell index is not initialized, world data is not available,
            or the value format is incorrect.
        """
        if self._cell_index is None:
            raise ValueError("Cannot set to_earth: cell index not initialized")  # noqa: E501

        world = getattr(self, "_world", None)
        if world is None or world.to_earth is None:
            raise ValueError("Cannot set to_earth: world to_earth not available")  # noqa: E501

        if not hasattr(world.to_earth, "data_vars"):
            raise ValueError("World to_earth must be an xarray/LPJmL Dataset")

        if not hasattr(value, "data_vars"):
            raise ValueError("Assigned value must be an xarray/LPJmL Dataset")

        for var_name, var_data in value.data_vars.items():
            if var_name in world.to_earth.data_vars:
                local_idx = self._resolve_local_index(world)
                if local_idx is None:
                    raise ValueError(
                        "Cannot resolve cell index for writing world to_earth"
                    )
                world.to_earth[var_name].values[local_idx] = var_data.values

    @property
    def from_earth(self) -> Optional[Any]:
        """Get data received from Earth system (LPJmL) for this cell.

        Returns a view on the world's from_earth dataset restricted to this
        cell. The cell dimension is dropped, so indexing is (band, time)
        not (cell, band, time).

        Formerly called 'output'. This property is read-only - only LPJmL
        updates this data via Component.update_lpjml().

        Returns
        -------
        xarray.Dataset or None
            Cell-specific view of the from_earth dataset, or None if not
            available.
        """
        if self._cell_index is None:
            return None

        world = getattr(self, "_world", None)
        if world is None or world.from_earth is None:
            return None

        selector = self._selector_for_world_drop_dim(world)
        if selector is None:
            return None

        if hasattr(world.from_earth, "isel"):
            return world.from_earth.isel(cell=selector)

        return world.from_earth

    # ========================================================================
    # Backward Compatibility Aliases
    # ========================================================================

    @property
    def input(self) -> Optional[Any]:
        """Backward compatibility alias for to_earth.

        .. deprecated::
            Use `to_earth` instead.
        """
        warn_deprecated_alias(self, "input", "to_earth")
        return self.to_earth

    @input.setter
    def input(self, value: Any) -> None:
        """Backward compatibility alias for to_earth.

        .. deprecated::
            Use `to_earth` instead.
        """
        warn_deprecated_alias(self, "input", "to_earth")
        self.to_earth = value

    @property
    def output(self) -> Optional[Any]:
        """Backward compatibility alias for from_earth (read-only).

        .. deprecated::
            Use `from_earth` instead.
        """
        warn_deprecated_alias(self, "output", "from_earth")
        return self.from_earth

    # ========================================================================
    # Grid and Area Properties
    # ========================================================================

    @property
    def grid(self) -> Optional[Any]:
        """Get grid data for this cell (read-only).

        Grid coordinates are fixed geographical properties and cannot be
        changed. The cell dimension is dropped for single cell access.

        Returns
        -------
        xarray.DataArray or None
            Cell-specific grid data, or None if not available.
        """
        if self._cell_index is None:
            return None

        world = getattr(self, "_world", None)
        if world is None or world.grid is None:
            return None

        selector = self._selector_for_world_drop_dim(world)
        if selector is None:
            return None

        if hasattr(world.grid, "isel"):
            return world.grid.isel(cell=selector)

        return world.grid

    @property
    def area(self) -> Optional[Any]:
        """Get area for this cell (read-only).

        Area is a fixed geographical property and cannot be changed.
        The cell dimension is dropped, returning a scalar.

        Returns
        -------
        xarray.DataArray or float or None
            Cell area (typically in m²), or None if not available.
        """
        if self._cell_index is None:
            return None

        world = getattr(self, "_world", None)
        if world is None or world.area is None:
            return None

        selector = self._selector_for_world_drop_dim(world)
        if selector is None:
            return None

        if hasattr(world.area, "isel"):
            return world.area.isel(cell=selector)

        return world.area

    # ========================================================================
    # Output Collection Support
    # ========================================================================

    @property
    def model(self) -> Optional[Any]:
        """Get model reference (required by OutputDefinitionMixin).

        Accesses _world directly to avoid triggering world property
        which could cause restoration during output writing.

        Returns
        -------
        ModelComponent or None
            The model component instance, or None if not available.
        """
        world = getattr(self, "_world", None)
        if world is not None:
            # Check if world has _model attribute directly (World instances)
            if hasattr(world, "_model") and world._model is not None:
                return world._model
            # Fallback to world.model property (for Region/Country)
            if hasattr(world, "model"):
                return world.model
        return None

    def get_defined_outputs(self) -> List[str]:
        """Get list of output variable names based on config.

        Filters the class-level output_variables by the 'cell' section
        of the model's output configuration.

        Returns
        -------
        list of str
            List of variable names to output for this cell, filtered by
            configuration.
        """
        if self.model is None or not hasattr(self.model, "config"):
            return []

        try:
            config_outputs = (
                self.model.config.coupled_config.output.to_dict().get("cell", [])  # noqa: E501
            )
            return [
                var
                for var in self.__class__.output_variables.names
                if var in config_outputs
            ]
        except Exception:
            return []
