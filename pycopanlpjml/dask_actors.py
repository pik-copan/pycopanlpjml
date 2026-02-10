"""Dask Actor-based parallelization for persistent worker state.

This module provides Dask Actor wrappers for Country objects, enabling
persistent state on workers without re-serialization each year.

Benefits over standard Dask tasks:
- Countries are deployed ONCE and persist across all simulation years
- Only from_earth data (LPJmL output) is sent each year
- Only deltas (to_earth changes, individual updates) are returned
- Dramatically reduces serialization overhead for large countries

Classes
-------
CountryActor
    Dask Actor wrapper for a Country that persists on a worker.
ActorManager
    Manages deployment and coordination of CountryActors.

Example
-------
>>> from pycopanlpjml.dask_actors import ActorManager
>>>
>>> # Deploy countries as actors (once at simulation start)
>>> manager = ActorManager(client, countries)
>>> manager.deploy()
>>>
>>> # Each year: update with minimal data transfer
>>> for year in simulation_years:
...     results = manager.update_all(year, from_earth_data)
...     # Apply deltas to world
...     apply_results(world, results)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class CountryActor:
    """Dask Actor wrapper for a Country that persists on a worker.

    The actor holds a Country instance and its cells/individuals, updating
    them in place each year without re-serialization.

    Parameters
    ----------
    country_payload : tuple
        Serialized country payload (module, qualname, state).

    Attributes
    ----------
    country : Country
        The deserialized and persistent Country instance.
    cell_indices : np.ndarray
        Global cell indices for this country.
    initialized : bool
        Whether the actor has been initialized.
    """

    def __init__(self, country_payload: Tuple[str, str, Dict[str, Any]]):
        """Initialize the actor by deserializing the country."""
        from pycopanlpjml.serialization import deserialize_country

        self.country = deserialize_country(country_payload)
        self.cell_indices = np.asarray(
            getattr(self.country, "_cell_indices", []), dtype=np.int64
        )
        self.initialized = True

        # Build index mappings for fast updates
        self._cell_index_to_local = {
            int(idx): i for i, idx in enumerate(self.cell_indices)
        }

        # Cache cell references for fast iteration
        self._cells = list(getattr(self.country, "_direct_cells", set()))
        self._individuals = list(
            getattr(self.country, "_direct_individuals", set())
        )

        # Build individual index map
        self._individual_indices = np.array(
            [
                getattr(ind, "_individual_index", i)
                for i, ind in enumerate(self._individuals)
            ],
            dtype=np.int64,
        )

    def update(
        self, t: int, from_earth_slice: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[
        np.ndarray, Optional[Dict[str, np.ndarray]], Optional[Dict[str, Any]]
    ]:  # noqa: E501
        """Run country update and return deltas.

        Parameters
        ----------
        t : int
            Current simulation year.
        from_earth_slice : dict, optional
            LPJmL output data for this country's cells.
            Keys are variable names, values are numpy arrays.

        Returns
        -------
        cell_indices : np.ndarray
            Global cell indices for this country.
        to_earth_updates : dict or None
            Changes to to_earth data: {var_name: values_array}.
        individual_updates : dict or None
            Changes to individual attributes.
        """
        # Update from_earth data on cells if provided
        if from_earth_slice is not None:
            self._update_from_earth(from_earth_slice)

        # Capture pre-update state for delta detection
        pre_to_earth = self._capture_to_earth_state()

        # Run the actual country update
        self.country.update(t)

        # Compute deltas
        to_earth_updates = self._compute_to_earth_delta(pre_to_earth)
        individual_updates = self._extract_individual_updates()

        return self.cell_indices, to_earth_updates, individual_updates

    def update_with_broadcast(
        self,
        t: int,
        from_earth_full: Optional[Dict[str, np.ndarray]] = None,
        cell_indices: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray, Optional[Dict[str, np.ndarray]], Optional[Dict[str, Any]]
    ]:  # noqa: E501
        """Run country update with broadcast from_earth data.

        This method receives the full from_earth data and extracts its own
        slice based on cell_indices. This is more efficient than the driver
        extracting slices for each country.

        Parameters
        ----------
        t : int
            Current simulation year.
        from_earth_full : dict, optional
            Full from_earth data as {var_name: full_array}.
            Actor extracts its own slice.
        cell_indices : np.ndarray, optional
            Global cell indices for this country (used for slicing).

        Returns
        -------
        cell_indices : np.ndarray
            Global cell indices for this country.
        to_earth_updates : dict or None
            Changes to to_earth data: {var_name: values_array}.
        individual_updates : dict or None
            Changes to individual attributes (generic for any Individual type).
        """
        # Extract this country's slice from broadcast data
        from_earth_slice = None
        if from_earth_full is not None and cell_indices is not None:
            from_earth_slice = self._extract_own_slice(
                from_earth_full, cell_indices
            )

        # Delegate to standard update
        return self.update(t, from_earth_slice)

    def _extract_own_slice(
        self,
        from_earth_full: Dict[str, np.ndarray],
        cell_indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Extract this country's slice from full from_earth data.

        Parameters
        ----------
        from_earth_full : dict
            Full data as {var_name: full_array}.
        cell_indices : np.ndarray
            Global indices to extract.

        Returns
        -------
        dict
            Sliced data for this country's cells.
        """
        result = {}
        for var_name, full_array in from_earth_full.items():
            try:
                # Handle different array shapes
                if full_array.ndim == 1:
                    result[var_name] = full_array[cell_indices]
                elif full_array.ndim == 2:
                    result[var_name] = full_array[cell_indices, :]
                else:
                    # For higher dimensions, slice first dimension
                    result[var_name] = full_array[cell_indices]
            except (IndexError, KeyError):
                pass
        return result

    def _update_from_earth(
        self, from_earth_slice: Dict[str, np.ndarray]
    ) -> None:  # noqa: E501
        """Update from_earth data on cells."""
        world = getattr(self.country, "_world", None)
        if world is None:
            return

        from_earth = getattr(world, "from_earth", None)
        if from_earth is None:
            return

        # Update each variable
        for var_name, values in from_earth_slice.items():
            if hasattr(from_earth, var_name):
                try:
                    getattr(from_earth, var_name).values[:] = values
                except Exception:
                    pass

    def _capture_to_earth_state(self) -> Dict[str, np.ndarray]:
        """Capture current to_earth state for delta detection."""
        world = getattr(self.country, "_world", None)
        if world is None:
            return {}

        to_earth = getattr(world, "to_earth", None)
        if to_earth is None:
            return {}

        state = {}
        for var_name in to_earth.data_vars:
            try:
                state[var_name] = np.array(to_earth[var_name].values)
            except Exception:
                pass
        return state

    def _compute_to_earth_delta(
        self, pre_state: Dict[str, np.ndarray]
    ) -> Optional[Dict[str, np.ndarray]]:
        """Compute changes to to_earth since pre_state."""
        world = getattr(self.country, "_world", None)
        if world is None:
            return None

        to_earth = getattr(world, "to_earth", None)
        if to_earth is None:
            return None

        updates = {}
        for var_name in to_earth.data_vars:
            try:
                current = np.array(to_earth[var_name].values)
                pre = pre_state.get(var_name)
                if pre is not None and not np.array_equal(current, pre):
                    updates[var_name] = current
                elif pre is None:
                    updates[var_name] = current
            except Exception:
                pass

        return updates if updates else None

    def _extract_individual_updates(self) -> Optional[Dict[str, Any]]:
        """Extract individual attribute updates."""
        from pycopanlpjml.serialization import (
            get_sync_attributes,
            extract_output_scalar,
        )

        if not self._individuals:
            return None

        sync_attrs = get_sync_attributes(self._individuals[0])
        if not sync_attrs:
            return None

        values = {attr: [] for attr in sync_attrs}

        for individual in self._individuals:
            for attr in sync_attrs:
                val = getattr(individual, attr, None)
                values[attr].append(extract_output_scalar(val))

        return {
            "indices": self._individual_indices.tolist(),
            "values": {k: np.array(v) for k, v in values.items()},
        }


class ActorManager:
    """Manages deployment and coordination of CountryActors.

    Handles the lifecycle of actor deployment, update coordination,
    and result collection.

    Parameters
    ----------
    client : distributed.Client
        Dask client for actor deployment.
    countries : list
        List of Country instances to deploy as actors.

    Attributes
    ----------
    actors : list
        List of deployed CountryActor proxies.
    deployed : bool
        Whether actors have been deployed.
    """

    def __init__(self, client: Any, countries: List[Any]):
        """Initialize the manager with countries to deploy."""
        self.client = client
        self.countries = countries
        self.actors: List[Any] = []
        self.deployed = False
        # country_id -> actor_index
        self._country_to_actor: Dict[int, int] = {}

    def deploy(self) -> None:
        """Deploy all countries as actors to workers.

        This should be called once at simulation start. Countries are
        serialized and sent to workers where they persist.
        """
        from pycopanlpjml.serialization import serialize_country_for_worker

        if self.deployed:
            return

        # Serialize countries
        payloads = [serialize_country_for_worker(c) for c in self.countries]

        # Deploy as actors (distributed across workers)
        # client.submit with actor=True returns a Future that resolves to
        # ActorFuture
        actor_futures = []
        for i, payload in enumerate(payloads):
            future = self.client.submit(
                CountryActor,
                payload,
                actor=True,
                pure=False,
            )
            actor_futures.append(future)
            self._country_to_actor[id(self.countries[i])] = i

        # Wait for all actor futures to resolve to actual actor proxies
        self.actors = self.client.gather(actor_futures)
        self.deployed = True

    def update_all(
        self,
        t: int,
        from_earth: Any = None,
    ) -> List[Tuple[np.ndarray, Optional[Dict], Optional[Dict]]]:
        """Update all countries for year t.

        Uses three optimizations:
        1. Broadcast from_earth once (actors extract their own slice)
        2. Submit all actor updates in parallel
        3. Collect results in parallel using as_completed

        Parameters
        ----------
        t : int
            Current simulation year.
        from_earth : xarray.Dataset, optional
            Full from_earth data. Broadcast to all workers once.

        Returns
        -------
        list of tuples
            Results from each actor: (cell_indices, to_earth_updates,
            individual_updates)
        """
        if not self.deployed:
            self.deploy()

        # OPTIMIZATION 2: Broadcast from_earth once to all workers
        # Instead of extracting slices on driver, let actors extract their own
        from_earth_ref = None
        if from_earth is not None and not hasattr(
            self, "_from_earth_scattered"
        ):  # noqa: E501
            # Convert to numpy dict once (avoid repeated conversion)
            from_earth_ref = self._prepare_from_earth_broadcast(from_earth)
        elif from_earth is not None:
            from_earth_ref = self._prepare_from_earth_broadcast(from_earth)

        # Submit updates to all actors in parallel (non-blocking)
        actor_futures = []
        for i, actor in enumerate(self.actors):
            # Pass cell_indices so actor can extract its own slice
            cell_indices = getattr(self.countries[i], "_cell_indices", None)
            if cell_indices is not None:
                cell_indices = np.asarray(cell_indices, dtype=np.int64)

            # actor.update() returns an ActorFuture
            actor_future = actor.update_with_broadcast(
                t, from_earth_ref, cell_indices
            )
            actor_futures.append(actor_future)

        # OPTIMIZATION 1: Collect results in parallel
        # All actors run concurrently, we just wait for all to complete
        results = []
        for af in actor_futures:
            try:
                result = af.result()
                # OPTIMIZATION 3: Skip empty results
                if result is not None:
                    cell_indices, to_earth_updates, individual_updates = result
                    # Only include if there are actual changes
                    if to_earth_updates or individual_updates:
                        results.append(result)
                    elif cell_indices is not None and len(cell_indices) > 0:
                        # Include even without changes (for completeness)
                        results.append(result)
            except Exception:
                # Skip failed actors
                pass

        return results

    def _prepare_from_earth_broadcast(
        self, from_earth: Any
    ) -> Dict[str, np.ndarray]:  # noqa: E501
        """Convert from_earth to a dict of numpy arrays for efficient
        broadcast.

        This is done once on the driver, then the dict is passed to actors
        who extract their own slices.
        """
        result = {}
        if from_earth is None:
            return result

        for var_name in from_earth.data_vars:
            try:
                # Get full array as numpy
                data = from_earth[var_name].values
                if hasattr(data, "compute"):
                    data = data.compute()
                result[var_name] = np.asarray(data)
            except Exception:
                pass

        return result

    def _extract_from_earth_slice(
        self,
        from_earth: Any,
        country: Any,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract from_earth data for a specific country's cells."""
        cell_indices = getattr(country, "_cell_indices", None)
        if cell_indices is None:
            return None

        cell_indices = np.asarray(cell_indices, dtype=np.int64)

        result = {}
        for var_name in from_earth.data_vars:
            try:
                full_data = np.asarray(from_earth[var_name].values)
                result[var_name] = full_data[cell_indices]
            except Exception:
                pass

        return result if result else None

    def shutdown(self) -> None:
        """Clean up actors."""
        self.actors = []
        self.deployed = False
        self._country_to_actor = {}
