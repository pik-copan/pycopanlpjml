"""Serialization utilities for parallel country processing.

This module provides functions for serializing Country objects and their
state across Dask worker boundaries. It handles:

- Converting Country instances to lightweight payloads
- Deserializing payloads back to Country instances on workers
- Synchronizing state changes (to_earth, individual attributes) back to driver

The serialization strategy avoids cloudpickle's recursive tokenization issues
by explicitly controlling what gets serialized via Country.__getstate__.

Functions
---------
serialize_country_for_worker
    Convert a Country to a serializable payload.
deserialize_country
    Reconstruct a Country from a payload on a worker.
sync_world
    Run country.update() on a worker and return state changes.
dataset_to_numpy_dict
    Convert xarray Dataset to plain numpy dict.
extract_world_slice
    Extract cells from a dataset using global/local index mapping.
get_sync_attributes
    Get list of individual attributes to synchronize.
extract_output_scalar
    Convert values to float for synchronization.

Example
-------
>>> from pycopanlpjml.serialization import (
...     serialize_country_for_worker,
...     deserialize_country,
...     sync_world,
... )
>>>
>>> # On driver: serialize country for worker
>>> payload = serialize_country_for_worker(country)
>>>
>>> # On worker: deserialize and update
>>> result = sync_world(payload, t=2025)
>>>
>>> # On driver: apply results
>>> cell_indices, to_earth_updates, individual_updates = result
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

# Type alias for serialized country payload
CountryPayload = Tuple[str, str, Mapping[str, Any]]

# Cache for country class lookups (avoids repeated imports)
_COUNTRY_CLASS_CACHE: Dict[Tuple[str, str], type] = {}


# =============================================================================
# MAIN SERIALIZATION FUNCTIONS
# =============================================================================


def serialize_country_for_worker(country: Any) -> CountryPayload:
    """Convert a Country instance into a lightweight payload for workers.

    The payload contains only the information needed to reconstruct the
    country on a worker, without circular references that would cause
    infinite recursion during cloudpickle serialization.

    Parameters
    ----------
    country : Country
        A Country instance to serialize.

    Returns
    -------
    CountryPayload
        A tuple of (module_name, qualname, state_dict) suitable for
        transmission to workers via Dask.

    See Also
    --------
    deserialize_country : Reconstruct a Country from the payload.
    Country.__getstate__ : Creates the serializable state dict.

    Example
    -------
    >>> payload = serialize_country_for_worker(germany)
    >>> module, qualname, state = payload
    >>> print(f"Serializing {qualname} with {len(state)} attributes")
    """
    cls = country.__class__
    state = country.__getstate__()
    return (cls.__module__, cls.__qualname__, state)


def deserialize_country(payload: CountryPayload) -> Any:
    """Reconstruct a Country instance from a serialized payload.

    Uses a class cache to avoid repeated module imports for the same
    country class type.

    Parameters
    ----------
    payload : CountryPayload
        The (module_name, qualname, state_dict) tuple from
        serialize_country_for_worker.

    Returns
    -------
    Country
        A reconstructed Country instance with cells and individuals
        re-attached via __setstate__.

    See Also
    --------
    serialize_country_for_worker : Creates the payload.
    Country.__setstate__ : Restores object references.

    Example
    -------
    >>> country = deserialize_country(payload)
    >>> country.update(t=2025)
    """
    module_name, qualname, state = payload
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


def sync_world(
    country_payload: CountryPayload,
    t: int,
) -> Optional[Tuple[np.ndarray, Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]]:
    """Run a country update on a worker and return state changes.

    This is the main worker-side function. It:
    1. Deserializes the country from the payload
    2. Runs country.update(t)
    3. Collects changes to to_earth data
    4. Collects changes to individual attributes
    5. Returns all changes as numpy arrays for merging on driver

    Parameters
    ----------
    country_payload : CountryPayload
        Serialized country from serialize_country_for_worker.
    t : int
        Current simulation time step (year).

    Returns
    -------
    tuple or None
        If successful, returns (cell_indices, to_earth_dict, individual_dict):

        - cell_indices : np.ndarray
            Global cell indices for this country
        - to_earth_dict : dict or None
            {var_name: np.ndarray} for updated to_earth variables
        - individual_dict : dict or None
            {"indices": np.ndarray, "values": {attr: np.ndarray}}

        Returns None if country has no world or cell indices.

    See Also
    --------
    ModelComponent.update_countries : Driver-side orchestration.
    """
    country = deserialize_country(country_payload)
    world = getattr(country, "_world", None)
    cell_indices = getattr(country, "_cell_indices", None)

    # Early return if no world data to sync
    if world is None or cell_indices is None:
        country.update(t)
        return None

    cell_indices = np.asarray(cell_indices)

    # Run the model-specific update logic
    country.update(t)

    # Collect to_earth changes
    updated_to_earth = dataset_to_numpy_dict(
        extract_world_slice(world.to_earth, world, cell_indices)
    )

    # Collect all individuals from country and cells
    individuals = set(getattr(country, "individuals", None) or [])
    direct_cells = getattr(country, "_direct_cells", None) or set()
    for cell in direct_cells:
        individuals.update(getattr(cell, "_individuals", None) or set())

    # Collect individual attribute changes
    updated_individuals: Optional[Dict[str, np.ndarray]] = None
    if individuals:
        # Find all attributes that need syncing
        attr_names = set()
        for individual in individuals:
            attr_names.update(get_sync_attributes(individual))
        attr_names = {attr for attr in attr_names if attr}

        if attr_names:
            indices = []
            values: Dict[str, list[float]] = {attr: [] for attr in attr_names}

            for individual in individuals:
                idx = getattr(individual, "_individual_index", None)
                if idx is None:
                    continue
                indices.append(int(idx))
                for attr in attr_names:
                    values[attr].append(
                        extract_output_scalar(getattr(individual, attr, None))
                    )

            if indices:
                updated_individuals = {
                    "indices": np.asarray(indices, dtype=np.int64),
                    "values": {
                        attr: np.asarray(vals, dtype=np.float64)
                        for attr, vals in values.items()
                    },
                }

    return (cell_indices, updated_to_earth, updated_individuals)


def sync_world_batch(
    country_payloads: List[CountryPayload],
    t: int,
) -> List[Optional[Tuple[np.ndarray, Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]]]:
    """Process multiple countries in a single worker task.

    This reduces Dask task scheduling overhead by batching small countries
    together. Useful when you have many small countries (few cells each).

    Parameters
    ----------
    country_payloads : list of CountryPayload
        List of serialized countries to process together.
    t : int
        Current simulation time step (year).

    Returns
    -------
    list
        List of results from sync_world for each country.

    Notes
    -----
    This is more efficient than submitting many small tasks because:
    - Reduces Dask scheduler overhead
    - Better memory locality on worker
    - Fewer network round-trips for results
    """
    return [sync_world(payload, t) for payload in country_payloads]


# =============================================================================
# DATA EXTRACTION HELPERS
# =============================================================================


def extract_world_slice(dataset: Any, world: Any, cell_indices: np.ndarray):
    """Extract cells from a dataset using global-to-local index mapping.

    Handles the case where the worker's _LocalWorldView has a different
    cell index space than the global world.

    Parameters
    ----------
    dataset : xarray.Dataset or LPJmLDataSet
        The dataset to slice (typically world.to_earth).
    world : World or _LocalWorldView
        The world object with optional index mapping.
    cell_indices : np.ndarray
        Global cell indices to extract.

    Returns
    -------
    xarray.Dataset or None
        Sliced dataset, or None if dataset is invalid.
    """
    if dataset is None or not hasattr(dataset, "isel"):
        return None

    selector = cell_indices

    # Handle local world views with remapped indices
    if hasattr(world, "map_global_to_local"):
        selector = world.map_global_to_local(cell_indices)
    elif hasattr(world, "_global_to_local"):
        selector = [world._global_to_local[int(idx)] for idx in cell_indices]

    return dataset.isel(cell=selector)


def dataset_to_numpy_dict(dataset: Any) -> Optional[Dict[str, np.ndarray]]:
    """Convert an xarray Dataset to a dict of numpy arrays.

    Creates deep copies of data to ensure values are independent of
    the original dataset.

    Parameters
    ----------
    dataset : xarray.Dataset or LPJmLDataSet
        Dataset to convert.

    Returns
    -------
    dict or None
        {var_name: np.ndarray} mapping, or None if dataset is invalid.
    """
    if dataset is None or not hasattr(dataset, "data_vars"):
        return None

    result: Dict[str, np.ndarray] = {}
    for var_name, var_data in dataset.data_vars.items():
        result[var_name] = np.array(var_data.values, copy=True)

    return result


# =============================================================================
# INDIVIDUAL ATTRIBUTE HELPERS
# =============================================================================


def is_sync_scalar(value: Any) -> bool:
    """Check if a value is suitable for synchronization.

    Only simple scalar values (numbers, strings) are synchronized
    between workers and driver.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        True if value is a sync-able scalar.
    """
    if value is None:
        return False
    if isinstance(value, (bool, int, float, str, np.generic)):
        return True
    try:
        return np.isscalar(value)
    except Exception:  # pragma: no cover
        return False


def get_sync_attributes(individual: Any) -> list[str]:
    """Get list of attribute names that should be synchronized.

    Determines which attributes to sync back from worker to driver.
    Priority order:

    1. individual.get_sync_attributes() method
    2. individual.__class__.sync_attributes class attribute
    3. individual.get_defined_outputs() method
    4. All public scalar attributes in __dict__

    Parameters
    ----------
    individual : Individual
        The individual entity to inspect.

    Returns
    -------
    list of str
        Attribute names to synchronize.
    """
    # Priority 1: Method that returns sync attributes
    if hasattr(individual, "get_sync_attributes"):
        attrs = individual.get_sync_attributes()
        if attrs:
            return list(attrs)

    # Priority 2: Class-level sync_attributes definition
    attrs = getattr(individual.__class__, "sync_attributes", None)
    if attrs:
        return list(attrs)

    # Priority 3 & 4: Combine defined outputs with public scalars
    public_attrs = [
        attr
        for attr, value in individual.__dict__.items()
        if not attr.startswith("_") and is_sync_scalar(value)
    ]

    getter = getattr(individual, "get_defined_outputs", None)
    if callable(getter):
        outputs = getter()
        if outputs:
            return sorted(set(outputs) | set(public_attrs))

    return public_attrs


def extract_output_scalar(value: Any) -> float:
    """Convert an attribute value to float for synchronization.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    float
        The value as float, or NaN if conversion fails.
    """
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "CountryPayload",
    "serialize_country_for_worker",
    "deserialize_country",
    "sync_world",
    "dataset_to_numpy_dict",
    "extract_world_slice",
    "get_sync_attributes",
    "extract_output_scalar",
]
