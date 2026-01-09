"""Tests for Dask Actor-based parallelization."""

import pytest
import numpy as np

from pycopanlpjml.dask_actors import CountryActor


class MockWorld:
    """Minimal world for testing."""
    def __init__(self):
        self.to_earth = MockDataset({"pft_harvest_frac": np.zeros(3)})
        self.from_earth = MockDataset({"soilc": np.ones(3)})


class MockDataset:
    """Minimal dataset for testing."""
    def __init__(self, data_vars):
        self._data_vars = data_vars
        self.data_vars = list(data_vars.keys())
        for name, values in data_vars.items():
            setattr(self, name, MockDataArray(values))
    
    def __getitem__(self, key):
        return getattr(self, key)


class MockDataArray:
    """Minimal data array for testing."""
    def __init__(self, values):
        self.values = np.asarray(values)


class MockCountry:
    """Minimal country for testing."""
    def __init__(self, cell_indices, world):
        self._cell_indices = np.array(cell_indices)
        self._world = world
        self._direct_cells = set()
        self._direct_individuals = set()
    
    def update(self, t):
        # Simulate changing to_earth
        self._world.to_earth.pft_harvest_frac.values[:] = t * 0.1
    
    def __getstate__(self):
        return {
            "_cell_indices": self._cell_indices,
            "_world": self._world,
            "_direct_cells": self._direct_cells,
            "_direct_individuals": self._direct_individuals,
        }


def test_country_actor_initialization():
    """Test that CountryActor correctly initializes from payload."""
    world = MockWorld()
    country = MockCountry([0, 1, 2], world)
    
    # Create payload
    payload = (
        MockCountry.__module__,
        MockCountry.__qualname__,
        country.__getstate__(),
    )
    
    # Mock deserialize_country
    import pycopanlpjml.serialization as ser
    original_deserialize = ser.deserialize_country
    
    def mock_deserialize(payload):
        mod, qual, state = payload
        obj = MockCountry.__new__(MockCountry)
        obj.__dict__.update(state)
        return obj
    
    ser.deserialize_country = mock_deserialize
    
    try:
        actor = CountryActor(payload)
        
        assert actor.initialized
        assert len(actor.cell_indices) == 3
        assert actor._cell_index_to_local == {0: 0, 1: 1, 2: 2}
    finally:
        ser.deserialize_country = original_deserialize


def test_country_actor_update():
    """Test that CountryActor.update() returns correct deltas."""
    world = MockWorld()
    country = MockCountry([0, 1, 2], world)
    
    # Create payload
    payload = (
        MockCountry.__module__,
        MockCountry.__qualname__,
        country.__getstate__(),
    )
    
    # Mock deserialize_country
    import pycopanlpjml.serialization as ser
    original_deserialize = ser.deserialize_country
    
    def mock_deserialize(payload):
        mod, qual, state = payload
        obj = MockCountry.__new__(MockCountry)
        obj.__dict__.update(state)
        return obj
    
    ser.deserialize_country = mock_deserialize
    
    try:
        actor = CountryActor(payload)
        
        # Run update
        cell_indices, to_earth_updates, individual_updates = actor.update(t=2025)
        
        # Check results
        assert len(cell_indices) == 3
        assert to_earth_updates is not None
        assert "pft_harvest_frac" in to_earth_updates
        # Country.update() sets values to t * 0.1 = 202.5
        np.testing.assert_allclose(
            to_earth_updates["pft_harvest_frac"], 
            np.full(3, 202.5)
        )
    finally:
        ser.deserialize_country = original_deserialize


def test_country_actor_from_earth_update():
    """Test that from_earth data is updated on actor."""
    world = MockWorld()
    country = MockCountry([0, 1, 2], world)
    
    payload = (
        MockCountry.__module__,
        MockCountry.__qualname__,
        country.__getstate__(),
    )
    
    import pycopanlpjml.serialization as ser
    original_deserialize = ser.deserialize_country
    
    def mock_deserialize(payload):
        mod, qual, state = payload
        obj = MockCountry.__new__(MockCountry)
        obj.__dict__.update(state)
        return obj
    
    ser.deserialize_country = mock_deserialize
    
    try:
        actor = CountryActor(payload)
        
        # Provide new from_earth data
        from_earth_slice = {"soilc": np.array([10.0, 20.0, 30.0])}
        
        # Run update with from_earth
        actor.update(t=2025, from_earth_slice=from_earth_slice)
        
        # Check that from_earth was updated
        np.testing.assert_allclose(
            actor.country._world.from_earth.soilc.values,
            [10.0, 20.0, 30.0]
        )
    finally:
        ser.deserialize_country = original_deserialize


def test_country_actor_broadcast_update():
    """Test update_with_broadcast extracts correct slice and returns it."""
    world = MockWorld()
    country = MockCountry([1, 2], world)  # Only cells 1 and 2
    
    payload = (
        MockCountry.__module__,
        MockCountry.__qualname__,
        country.__getstate__(),
    )
    
    import pycopanlpjml.serialization as ser
    original_deserialize = ser.deserialize_country
    
    def mock_deserialize(payload):
        mod, qual, state = payload
        obj = MockCountry.__new__(MockCountry)
        obj.__dict__.update(state)
        return obj
    
    ser.deserialize_country = mock_deserialize
    
    try:
        actor = CountryActor(payload)
        
        # Full broadcast data (all 5 cells)
        from_earth_full = {
            "soilc": np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        }
        cell_indices = np.array([1, 2])  # This country has cells 1 and 2
        
        # Test that slice extraction works
        slice_result = actor._extract_own_slice(from_earth_full, cell_indices)
        np.testing.assert_allclose(
            slice_result["soilc"],
            [200.0, 300.0]  # Values at indices 1 and 2
        )
        
        # Run update with broadcast (returns result tuple)
        result = actor.update_with_broadcast(
            t=2025, 
            from_earth_full=from_earth_full,
            cell_indices=cell_indices
        )
        
        # Check that result is returned correctly
        assert result is not None
        returned_indices, to_earth_updates, individual_updates = result
        assert len(returned_indices) == 2  # Two cells
        
    finally:
        ser.deserialize_country = original_deserialize


def test_country_actor_extract_own_slice():
    """Test _extract_own_slice handles different array shapes."""
    world = MockWorld()
    country = MockCountry([0, 2], world)
    
    payload = (
        MockCountry.__module__,
        MockCountry.__qualname__,
        country.__getstate__(),
    )
    
    import pycopanlpjml.serialization as ser
    original_deserialize = ser.deserialize_country
    
    def mock_deserialize(payload):
        mod, qual, state = payload
        obj = MockCountry.__new__(MockCountry)
        obj.__dict__.update(state)
        return obj
    
    ser.deserialize_country = mock_deserialize
    
    try:
        actor = CountryActor(payload)
        
        # Test 1D array
        full_1d = {"var1": np.array([10, 20, 30, 40, 50])}
        slice_1d = actor._extract_own_slice(full_1d, np.array([0, 2]))
        np.testing.assert_array_equal(slice_1d["var1"], [10, 30])
        
        # Test 2D array
        full_2d = {"var2": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])}
        slice_2d = actor._extract_own_slice(full_2d, np.array([0, 2]))
        np.testing.assert_array_equal(slice_2d["var2"], [[1, 2], [5, 6]])
        
    finally:
        ser.deserialize_country = original_deserialize

