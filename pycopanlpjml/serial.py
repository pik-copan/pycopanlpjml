"""Serialization utilities for parallel country processing.

This module provides functions for serializing Country objects and their
state across Dask worker boundaries. It handles:

- Detecting non-serializable references (mixin instances, sympy, asyncio)
- Cloning cells and individuals with broken circular references
- Converting neighbourhoods to index-based representation
- Rebuilding object references after deserialization

The serialization strategy avoids cloudpickle's recursive tokenization issues
by explicitly controlling what gets serialized via Country.__getstate__.

Architecture
------------
- Country.__getstate__ uses cached clone structures for performance
- First call: creates full clones with static structure
- Subsequent calls: only updates dynamic attributes
- Neighbourhoods stored as indices during serialization
- __setstate__ rebuilds object graphs from indices

Example
-------
>>> from pycopanlpjml.serial import (
...     serialize_country_for_worker,
...     deserialize_country,
... )
>>>
>>> # On driver: serialize country for worker
>>> payload = serialize_country_for_worker(country)
>>>
>>> # On worker: deserialize
>>> country = deserialize_country(payload)
>>> country.update(t=2025)
"""

from __future__ import annotations

import copy
import importlib
from typing import Any, Dict, Mapping, Optional, Set, Tuple, FrozenSet

import numpy as np
import xarray as xr
from sympy import Basic as _SympyBasic

from pycopancore.private._mixin import _Mixin
from pycopancore.private._simple_expressions import unknown


# Type alias for serialized country payload
CountryPayload = Tuple[str, str, Mapping[str, Any]]

# Cache for country class lookups (avoids repeated imports)
_COUNTRY_CLASS_CACHE: Dict[Tuple[str, str], type] = {}

# Cache for type-level serialization checks (True=serializable, False=not, None=needs value check)
_TYPE_SERIALIZABLE_CACHE: Dict[type, Optional[bool]] = {}

# Safe primitive types that are always serializable (checked first for speed)
_SAFE_PRIMITIVE_TYPES: Tuple[type, ...] = (
    int, float, bool, str, bytes, type(None),
)


def _check_type_serializable(t: type) -> Optional[bool]:
    """Check if a type is known to be serializable or not.
    
    Returns:
        True: Type is always serializable (primitives, numpy scalars)
        False: Type is never serializable (asyncio, mixin, sympy)
        None: Need to check value-level (containers, custom objects)
    """
    if t in _TYPE_SERIALIZABLE_CACHE:
        return _TYPE_SERIALIZABLE_CACHE[t]
    
    # Primitives are always safe
    if t in _SAFE_PRIMITIVE_TYPES:
        _TYPE_SERIALIZABLE_CACHE[t] = True
        return True
    
    # Check module-based patterns
    module_name = getattr(t, "__module__", "")
    class_name = getattr(t, "__name__", "")
    
    # Asyncio types - never serializable
    if module_name.startswith(("asyncio", "_asyncio")):
        _TYPE_SERIALIZABLE_CACHE[t] = False
        return False
    
    # Dask expressions - never serializable
    if module_name.startswith("dask.") and class_name.lower().endswith("expr"):
        _TYPE_SERIALIZABLE_CACHE[t] = False
        return False
    
    if class_name == "LLGExpr":
        _TYPE_SERIALIZABLE_CACHE[t] = False
        return False
    
    # pycopancore expressions - never serializable
    if module_name.startswith("pycopancore.private._expressions"):
        _TYPE_SERIALIZABLE_CACHE[t] = False
        return False
    
    # numpy scalar types - always safe
    if module_name.startswith("numpy") and issubclass(t, np.generic):
        _TYPE_SERIALIZABLE_CACHE[t] = True
        return True
    
    # Check for Mixin and Sympy (imported at module level)
    try:
        if issubclass(t, _Mixin):
            _TYPE_SERIALIZABLE_CACHE[t] = False
            return False
        if issubclass(t, _SympyBasic):
            _TYPE_SERIALIZABLE_CACHE[t] = False
            return False
    except TypeError:
        # issubclass fails for some types
        pass
    
    # numpy arrays need value-level check (dtype matters)
    if t is np.ndarray:
        _TYPE_SERIALIZABLE_CACHE[t] = None
        return None
    
    # xarray types need value-level check
    if module_name.startswith("xarray"):
        _TYPE_SERIALIZABLE_CACHE[t] = None
        return None
    
    # Container types need value-level check
    if t in (list, tuple, dict, set, frozenset):
        _TYPE_SERIALIZABLE_CACHE[t] = None
        return None
    
    # _LocalWorldView needs special handling
    if class_name == "_LocalWorldView":
        _TYPE_SERIALIZABLE_CACHE[t] = None
        return None
    
    # Default: assume needs value check for safety
    _TYPE_SERIALIZABLE_CACHE[t] = None
    return None


def clear_serialization_caches() -> None:
    """Clear all serialization-related caches.
    
    Call this if types are reloaded or for testing purposes.
    """
    _TYPE_SERIALIZABLE_CACHE.clear()
    _COUNTRY_CLASS_CACHE.clear()


# =============================================================================
# ATTRIBUTE SKIP LISTS
# =============================================================================

# Attributes to skip when cloning cells (contain circular references)
CELL_ATTR_SKIP: FrozenSet[str] = frozenset({
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
})

# Attributes to skip when cloning individuals
INDIVIDUAL_ATTR_SKIP: FrozenSet[str] = frozenset({
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
})

# State keys that are allowed to contain object references
STATE_ALLOWLIST: FrozenSet[str] = frozenset({
    "_cells",
    "_direct_cells",
    "_next_lower_social_systems",
    "_direct_individuals",
    "_individuals",
})



# =============================================================================
# NON-SERIALIZABLE DETECTION
# =============================================================================


def is_dask_expression(value: Any) -> bool:
    """Check if value is a Dask expression graph."""
    cls = value.__class__
    name = getattr(cls, "__name__", "")
    if name == "LLGExpr":
        return True
    module = getattr(cls, "__module__", "")
    return module.startswith("dask.") and name.lower().endswith("expr")


def is_local_world_view(value: Any) -> bool:
    """Check if value is a _LocalWorldView instance."""
    cls = value.__class__
    return cls.__name__ == "_LocalWorldView" and cls.__module__.endswith(
        (".world", ".region")
    )


def is_non_serializable(value: Any) -> bool:
    """Simple check for non-serializable objects (used in cloning)."""
    if value is None:
        return False
    if isinstance(value, _SympyBasic):
        return True
    if isinstance(value, _Mixin):
        return True
    if hasattr(value, "__module__"):
        mod = getattr(value, "__module__", "")
        if "asyncio" in mod or "dask" in mod:
            return True
    return False


def contains_non_serializable_references(value: Any, visited: Optional[Set[int]] = None) -> bool:
    """Check if value contains references that would break serialization.

    Detects mixin instances, sympy expressions, Dask expression graphs,
    and asyncio objects that would cause infinite recursion or fail
    during cloudpickle.

    Uses type-level caching for fast rejection of known safe/unsafe types.

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
    # Fast path: check type cache first
    cls = value.__class__
    type_result = _check_type_serializable(cls)
    if type_result is True:
        return False  # Type is always serializable
    if type_result is False:
        return True   # Type is never serializable
    
    # type_result is None - need value-level check
    if visited is None:
        visited = set()

    # Check numpy arrays with object dtype
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            try:
                for _, item in np.ndenumerate(value):
                    if contains_non_serializable_references(item, visited):
                        return True
            except RecursionError:
                return True
        return False

    # Check xarray DataArray
    if isinstance(value, xr.DataArray):
        if getattr(value, "dtype", None) == object:
            return contains_non_serializable_references(value.values, visited)
        return False

    # Check xarray Dataset
    if isinstance(value, xr.Dataset):
        try:
            for var in value.data_vars.values():
                if contains_non_serializable_references(var, visited):
                    return True
            for coord in value.coords.values():
                if contains_non_serializable_references(coord, visited):
                    return True
        except RecursionError:
            return True
        return False

    # Handle _LocalWorldView specially
    if is_local_world_view(value):
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
            "input",
            "output",
            "to_earth",
            "from_earth",
            "grid",
            "area",
        ):
            if hasattr(value, attr):
                attr_val = getattr(value, attr, None)
                if attr_val is not None and contains_non_serializable_references(
                    attr_val, visited
                ):
                    return True
        return False

    # Cycle detection for general objects
    obj_id = id(value)
    if obj_id in visited:
        return False
    visited.add(obj_id)

    # Mixin instances and sympy expressions cause recursion (fallback for uncached types)
    if isinstance(value, (_Mixin, _SympyBasic)):
        return True

    # Check for pycopancore expression objects (fallback)
    module_name = getattr(cls, "__module__", "")
    class_name = getattr(cls, "__name__", "")
    if (
        module_name.startswith("pycopancore.private._expressions")
        or "LLGExpr" in class_name
    ):
        return True

    # Recursively check containers
    try:
        if isinstance(value, dict):
            for item in value.values():
                if contains_non_serializable_references(item, visited):
                    return True
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                if contains_non_serializable_references(item, visited):
                    return True
    except RecursionError:
        return True

    return False


# =============================================================================
# CLONE STRUCTURE CREATION
# =============================================================================


def create_clone_structure(
    cells: Any, individuals: Any
) -> Tuple[Dict[Any, Any], Dict[Any, Any], Set[Any], Set[Any]]:
    """Create clone structure for cached serialization.

    Creates empty clone objects with static attributes (indices, structure).
    Dynamic attributes are set later by update_cached_clones.

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
    cell_map: Dict[Any, Any] = {}
    individual_map: Dict[Any, Any] = {}
    cloned_cells: Set[Any] = set()
    cloned_individuals: Set[Any] = set()

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

        original_individuals = getattr(cell, "_individuals", None) or set()
        for individual in original_individuals:
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

            # Detach from cell but store index for reconstruction
            clone_individual._cell = None
            clone_individual._cell_index = getattr(cell, "_cell_index", None)
            clone_individual._cell_uid = getattr(clone_cell, "_uid", None)
            clone_individual.neighbourhood = []

            # Add to clone cell's _individuals collection
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
        clone_individual._cell_index = getattr(original_cell, "_cell_index", None)
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


def update_cached_clones(
    cell_map: Dict[Any, Any],
    individual_map: Dict[Any, Any],
    local_world: Any,
    model_view: Any,
) -> None:
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
            if attr in CELL_ATTR_SKIP:
                continue
            # Skip static attrs we already set
            if attr in {"_cell_index", "_entity_alias"}:
                continue
            # Check for non-serializable
            if contains_non_serializable_references(value):
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
            if attr in INDIVIDUAL_ATTR_SKIP:
                continue
            # Skip static attrs
            if attr in {"_individual_index", "_entity_alias"}:
                continue
            # Check for non-serializable
            if contains_non_serializable_references(value):
                clone_individual.__dict__[attr] = unknown
            # Check if this attribute has a back-reference to the individual
            elif _has_back_reference(value, original_individual):
                clone_individual.__dict__[attr] = _copy_with_back_reference(
                    value, original_individual, clone_individual
                )
            else:
                clone_individual.__dict__[attr] = value


def _has_back_reference(obj: Any, parent: Any) -> bool:
    """Check if obj has any attribute that references parent."""
    if obj is None or not hasattr(obj, "__dict__"):
        return False
    obj_dict = getattr(obj, "__dict__", {})
    # Check if ANY attribute references the parent
    for value in obj_dict.values():
        if value is parent:
            return True
    return False


def _copy_with_back_reference(obj: Any, old_parent: Any, new_parent: Any) -> Any:
    """Copy an object that has a back-reference and update the reference.

    Used during serialization when an attribute (like a behaviour/decision
    model) has an attribute pointing back to its owner (the individual).
    We need to copy it and update any reference to old_parent → new_parent.

    Parameters
    ----------
    obj : object
        The object with a back-reference to copy.
    old_parent : object
        The original parent (original individual) to detect.
    new_parent : object
        The new parent (cloned individual) to reference.

    Returns
    -------
    object
        A copy with updated back-reference.
    """
    if obj is None:
        return None

    try:
        cls = obj.__class__
        copied = object.__new__(cls)

        src_dict = obj.__dict__
        dst_dict = copied.__dict__

        for attr, value in src_dict.items():
            if value is old_parent:
                # Update back-reference to new parent
                dst_dict[attr] = new_parent
            elif attr in ("_cache", "__dict__"):
                # Skip caches
                continue
            elif isinstance(value, dict):
                # Shallow copy dicts (values are typically scalars)
                dst_dict[attr] = value.copy() if value else {}
            elif isinstance(value, list):
                # Deep copy lists (may contain nested mutable state)
                dst_dict[attr] = copy.deepcopy(value)
            elif isinstance(value, set):
                dst_dict[attr] = value.copy() if value else set()
            else:
                # Scalars, tuples (immutable), numpy arrays - direct assign
                dst_dict[attr] = value

        return copied
    except Exception:
        # If copy fails, return original
        return obj


# =============================================================================
# NEIGHBOURHOOD GRAPH REBUILDING
# =============================================================================


def rebuild_neighbourhood_graphs(
    cells: Set[Any],
    individuals: Set[Any],
) -> None:
    """Rebuild neighbourhood graphs from stored indices.

    During cloning, neighbourhoods are stored as indices to avoid
    recursion. This function converts them back to object references.

    Parameters
    ----------
    cells : set
        Cell entities with ``_neighbourhood_indices``.
    individuals : set
        Individual entities with ``_neighbourhood_indices``.
    """
    # Build lookup tables
    cell_by_index: Dict[int, Any] = {}
    for cell in cells:
        idx = getattr(cell, "_cell_index", None)
        if idx is not None:
            cell_by_index[idx] = cell

    individual_by_index: Dict[int, Any] = {}
    for individual in individuals:
        idx = getattr(individual, "_individual_index", None)
        if idx is not None:
            individual_by_index[idx] = individual

    # Rebuild cell neighbourhoods
    for cell in cells:
        indices = getattr(cell, "_neighbourhood_indices", None)

        if indices:
            cell.neighbourhood = [
                cell_by_index[i] for i in indices if i in cell_by_index
            ]
        elif not hasattr(cell, "neighbourhood") or cell.neighbourhood is None:
            cell.neighbourhood = []

        # Clean up temporary attribute
        if hasattr(cell, "_neighbourhood_indices"):
            del cell._neighbourhood_indices

    # Rebuild individual neighbourhoods
    for individual in individuals:
        indices = getattr(individual, "_neighbourhood_indices", None)

        if indices:
            individual.neighbourhood = [
                individual_by_index[i] for i in indices if i in individual_by_index
            ]
        elif (
            not hasattr(individual, "neighbourhood") or individual.neighbourhood is None
        ):
            individual.neighbourhood = []

        if hasattr(individual, "_neighbourhood_indices"):
            del individual._neighbourhood_indices


# =============================================================================
# DESERIALIZATION HELPERS
# =============================================================================


def fix_agent_back_references(individual: Any) -> None:
    """Fix back-references from individual's sub-objects.

    After deserialization, sub-objects (like behaviour/decision models) may
    have back-references that point to a stale copy of the individual.
    This function updates those references to point to the deserialized
    individual.

    Generic approach: For each sub-object in individual's __dict__, check if
    any of its attributes point to an object of the EXACT SAME CLASS as
    individual but is NOT the individual itself. Such references are stale
    and must be updated.

    Uses `type(value) is individual_class` (exact class match) instead of
    `isinstance(value, individual_types)` to avoid matching base classes
    like `object` which would corrupt all attributes.

    Parameters
    ----------
    individual : Individual
        The deserialized individual to fix.
    """
    if individual is None:
        return

    individual_class = type(individual)  # Exact class, not MRO

    # Check all sub-objects for stale back-references
    for attr_name, obj in individual.__dict__.items():
        if obj is None or not hasattr(obj, "__dict__"):
            continue

        obj_dict = getattr(obj, "__dict__", None)
        if obj_dict is None:
            continue

        # Check for stale back-references: same exact class but different object
        for key, value in list(obj_dict.items()):
            if value is None or value is individual:
                continue  # Skip None or already correct
            # Exact class match - not isinstance which includes base classes
            if type(value) is individual_class:
                obj_dict[key] = individual


def restore_individual_cell(
    individual: Any,
    *,
    fallback_cell: Any,
    cell_lookup: Dict[Tuple[Any, Any], Any],
) -> Optional[Any]:
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


# =============================================================================
# MAIN SERIALIZATION FUNCTIONS
# =============================================================================


def serialize_country_for_worker(country: Any) -> bytes:
    """Convert a Country instance into a serialized payload for workers.

    Uses standard pickle (not cloudpickle) to avoid metaclass inspection
    that triggers deep recursion with pycopancore's mixin classes.

    Parameters
    ----------
    country : Country
        A Country instance to serialize.

    Returns
    -------
    bytes
        Pickle-serialized payload ready for transmission to workers.
    """
    import pickle

    cls = country.__class__
    state = country.__getstate__()
    payload = (cls.__module__, cls.__qualname__, state)

    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_country(pickled_payload: bytes) -> Any:
    """Reconstruct a Country instance from a serialized payload.

    Parameters
    ----------
    pickled_payload : bytes
        Pickle-serialized payload from serialize_country_for_worker.

    Returns
    -------
    Country
        A reconstructed Country instance.
    """
    import pickle

    module_name, qualname, state = pickle.loads(pickled_payload)
    cache_key = (module_name, qualname)

    # Look up class in cache or import it
    country_cls = _COUNTRY_CLASS_CACHE.get(cache_key)
    if country_cls is None:
        module = importlib.import_module(module_name)
        country_cls = module
        for attr in qualname.split("."):
            country_cls = getattr(country_cls, attr)
        _COUNTRY_CLASS_CACHE[cache_key] = country_cls

    # Create instance without calling __init__
    country = country_cls.__new__(country_cls)
    country.__setstate__(state)
    return country


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Type alias
    "CountryPayload",
    # Attribute constants
    "CELL_ATTR_SKIP",
    "INDIVIDUAL_ATTR_SKIP",
    "STATE_ALLOWLIST",
    # Detection functions
    "is_dask_expression",
    "is_local_world_view",
    "is_non_serializable",
    "contains_non_serializable_references",
    # Cache management
    "clear_serialization_caches",
    # Clone structure
    "create_clone_structure",
    "update_cached_clones",
    # Neighbourhood rebuilding
    "rebuild_neighbourhood_graphs",
    # Deserialization helpers
    "fix_agent_back_references",
    "restore_individual_cell",
    # Main serialization functions
    "serialize_country_for_worker",
    "deserialize_country",
    # Sync helpers
    "resolve_dotted_path",
    "set_dotted_path",
    "extract_output_scalar",
    "get_sync_attributes",
]


# ---------------------------------------------------------------------------
# Sync helpers (propagating individual state from workers to driver)
# ---------------------------------------------------------------------------

def resolve_dotted_path(obj: Any, path: str) -> Any:
    """Resolve dotted path to access nested object attributes.

    Checks _synced_values_cache first for values synced from workers.
    """
    if "." not in path:
        cache = getattr(obj, "_synced_values_cache", None)
        if cache is not None and path in cache:
            return cache[path]
        return getattr(obj, path, None)

    parts = path.split(".")
    value = obj

    for part in parts[:-1]:
        if value is None:
            return None
        value = getattr(value, part, None)

    if value is None:
        return None

    final_attr = parts[-1]
    cache = getattr(value, "_synced_values_cache", None)
    if cache is not None and final_attr in cache:
        return cache[final_attr]
    return getattr(value, final_attr, None)


def set_dotted_path(obj: Any, path: str, value: Any) -> None:
    """Set value on object using dotted path notation.

    For read-only properties, stores in _synced_values_cache.
    """
    if "." not in path:
        try:
            setattr(obj, path, value)
        except AttributeError:
            cache = getattr(obj, "_synced_values_cache", None)
            if cache is None:
                cache = {}
                object.__setattr__(obj, "_synced_values_cache", cache)
            cache[path] = value
        return

    parts = path.split(".")
    target = obj
    for part in parts[:-1]:
        if target is None:
            return
        target = getattr(target, part, None)

    if target is None:
        return

    final_attr = parts[-1]
    try:
        setattr(target, final_attr, value)
    except AttributeError:
        cache = getattr(target, "_synced_values_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(target, "_synced_values_cache", cache)
        cache[final_attr] = value


def extract_output_scalar(value: Any) -> float:
    """Convert value to float for synchronization."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def get_sync_attributes(individual: Any) -> list[str]:
    """Get output variable names to sync from worker to driver."""
    output_vars = getattr(individual.__class__, "output_variables", None)
    if output_vars is not None:
        names = getattr(output_vars, "names", None)
        if names:
            return list(names)
    return []
