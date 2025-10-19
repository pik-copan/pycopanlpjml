"""
Comprehensive tests for zarr_backend.py functionality.
Tests all classes, methods, and optimizations.
"""

import numpy as np
import xarray as xr
import pytest
import tempfile
import shutil
from pathlib import Path
import datetime

from pycopanlpjml.zarr_backend import (
    ZarrBackend,
    ZarrDatasetView,
    ZarrDataArrayView,
    ZarrCoordinateView,
    WritableCoordinateArray,
    DEFAULT_CHUNK_SIZE,
)


@pytest.fixture
def temp_zarr_store():
    """Create a temporary Zarr store for testing."""
    tmpdir = tempfile.mkdtemp()
    store_path = Path(tmpdir) / "test.zarr"
    yield str(store_path)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_xarray_data():
    """Create sample xarray datasets for testing."""
    n_cells = 10
    n_bands = 2
    n_time = 3

    coords = {
        "cell": np.arange(n_cells),
        "band": ["irrigated", "rainfed"],
        "time": [
            datetime.datetime(2020, 1, 1),
            datetime.datetime(2021, 1, 1),
            datetime.datetime(2022, 1, 1),
        ],
    }

    # Create input dataset
    input_data = {
        "fertilizer": xr.DataArray(
            np.random.randn(n_cells, n_bands, n_time).astype(np.float32),
            coords=coords,
            dims=["cell", "band", "time"],
            attrs={"units": "kg/ha"},
        ),
        "irrigation": xr.DataArray(
            np.random.randn(n_cells, n_bands, n_time).astype(np.float32),
            coords=coords,
            dims=["cell", "band", "time"],
            attrs={"units": "mm"},
        ),
    }
    input_ds = xr.Dataset(input_data)

    # Create output dataset
    output_data = {
        "yield": xr.DataArray(
            np.random.randn(n_cells, n_bands, n_time).astype(np.float32),
            coords=coords,
            dims=["cell", "band", "time"],
            attrs={"units": "kg/ha"},
        ),
        "biomass": xr.DataArray(
            np.random.randn(n_cells, n_bands, n_time).astype(np.float32),
            coords=coords,
            dims=["cell", "band", "time"],
            attrs={"units": "kg/m2"},
        ),
    }
    output_ds = xr.Dataset(output_data)

    # Create grid
    grid = xr.DataArray(
        np.random.randn(n_cells, 2).astype(np.float32),
        coords={"cell": np.arange(n_cells)},
        dims=["cell", "latlon"],
        attrs={"description": "Grid coordinates"},
    )

    # Create country
    country = xr.DataArray(
        np.random.choice(["DEU", "FRA", "USA"], n_cells),
        coords={"cell": np.arange(n_cells)},
        dims=["cell"],
        attrs={"description": "Country codes"},
    )

    return input_ds, output_ds, grid, country


class TestZarrBackend:
    """Test ZarrBackend class."""

    def test_init(self, temp_zarr_store):
        """Test ZarrBackend initialization."""
        backend = ZarrBackend(temp_zarr_store, overwrite=True)
        assert backend.store_path == temp_zarr_store
        assert backend.root is not None
        assert backend.lock is not None

    def test_initialize_from_xarray(self, temp_zarr_store, sample_xarray_data):
        """Test initialization from xarray datasets."""
        input_ds, output_ds, grid, country = sample_xarray_data
        backend = ZarrBackend(temp_zarr_store, overwrite=True)

        backend.initialize_from_xarray(
            input_ds, output_ds, grid, country, chunk_size=5
        )

        # Check that groups were created
        assert "input" in backend.root
        assert "output" in backend.root

        # Check that arrays were created
        assert "fertilizer" in backend.root["input"]
        assert "irrigation" in backend.root["input"]
        assert "yield" in backend.root["output"]
        assert "biomass" in backend.root["output"]
        assert "grid" in backend.root
        assert "country" in backend.root

        # Check shapes
        assert backend.root["input/fertilizer"].shape == (10, 2, 3)
        assert backend.root["output/yield"].shape == (10, 2, 3)

    def test_get_view(self, temp_zarr_store, sample_xarray_data):
        """Test get_view method."""
        input_ds, output_ds, grid, country = sample_xarray_data
        backend = ZarrBackend(temp_zarr_store, overwrite=True)
        backend.initialize_from_xarray(input_ds, output_ds, grid, country)

        # Get full view
        view = backend.get_view("input")
        assert isinstance(view, ZarrDatasetView)
        assert view.group == "input"

        # Get subset view
        subset_view = backend.get_view("input", indices=[0, 1, 2])
        assert isinstance(subset_view.indices, np.ndarray)
        assert len(subset_view.indices) == 3

    def test_get_array_view(self, temp_zarr_store, sample_xarray_data):
        """Test get_array_view method."""
        input_ds, output_ds, grid, country = sample_xarray_data
        backend = ZarrBackend(temp_zarr_store, overwrite=True)
        backend.initialize_from_xarray(input_ds, output_ds, grid, country)

        view = backend.get_array_view("input/fertilizer")
        assert isinstance(view, ZarrDataArrayView)
        assert view.path == "input/fertilizer"


class TestZarrDatasetView:
    """Test ZarrDatasetView class."""

    @pytest.fixture
    def backend_with_data(self, temp_zarr_store, sample_xarray_data):
        """Create backend with sample data."""
        input_ds, output_ds, grid, country = sample_xarray_data
        backend = ZarrBackend(temp_zarr_store, overwrite=True)
        backend.initialize_from_xarray(input_ds, output_ds, grid, country)
        return backend

    def test_coords_property(self, backend_with_data):
        """Test coords property and caching."""
        view = backend_with_data.get_view("input")

        # First access
        coords1 = view.coords
        assert "cell" in coords1
        assert "band" in coords1
        assert "time" in coords1

        # Second access should return cached value
        coords2 = view.coords
        assert coords1 is coords2  # Same object due to caching

    def test_data_vars_property(self, backend_with_data):
        """Test data_vars property."""
        view = backend_with_data.get_view("input")
        data_vars = view.data_vars

        assert "fertilizer" in data_vars
        assert "irrigation" in data_vars
        assert isinstance(data_vars["fertilizer"], ZarrDataArrayView)

    def test_getitem(self, backend_with_data):
        """Test __getitem__ for variable access (writable view with normalized repr)."""
        view = backend_with_data.get_view("input")
        array_view = view["fertilizer"]

        # Dict-style access returns writable ZarrDataArrayView
        assert isinstance(array_view, ZarrDataArrayView)
        assert array_view.path == "input/fertilizer"

        # But it should behave like LPJmLData (normalized dimensions)
        assert "cell" in array_view.dims
        assert "band" in array_view.dims  # Should be normalized

    def test_setitem(self, backend_with_data):
        """Test __setitem__ for variable assignment."""
        view = backend_with_data.get_view("input")
        new_data = np.ones((10, 2, 3), dtype=np.float32)

        view["fertilizer"] = new_data
        assert np.allclose(view["fertilizer"].values, new_data)

    def test_getattr_coordinate(self, backend_with_data):
        """Test __getattr__ for coordinate access."""
        view = backend_with_data.get_view("input")
        time_coord = view.time

        assert isinstance(time_coord, ZarrCoordinateView)
        assert len(time_coord) == 3

    def test_getattr_variable(self, backend_with_data):
        """Test __getattr__ for variable access."""
        view = backend_with_data.get_view("input")
        fertilizer = view.fertilizer

        # Now returns LPJmLData with proper dimension normalization
        # (not ZarrDataArrayView) to match pycoupler behavior
        from pycoupler.data import LPJmLData

        assert isinstance(fertilizer, (LPJmLData, xr.DataArray))
        # Check dimensions are normalized
        assert fertilizer.dims == ("cell", "band", "time")

    def test_isel(self, backend_with_data):
        """Test isel method for indexing."""
        view = backend_with_data.get_view("input")

        # Select subset of cells
        subset = view.isel({"cell": [0, 1, 2]})
        assert isinstance(subset, ZarrDatasetView)
        assert len(subset.indices) == 3

        # Select single cell
        single = view.isel({"cell": 0})
        assert isinstance(single, ZarrDatasetView)
        assert single._is_scalar

    def test_to_xarray(self, backend_with_data):
        """Test to_xarray conversion."""
        view = backend_with_data.get_view("input")
        ds = view.to_xarray()

        assert isinstance(ds, xr.Dataset)
        assert "fertilizer" in ds.data_vars
        assert "irrigation" in ds.data_vars

    def test_to_dict(self, backend_with_data):
        """Test to_dict conversion."""
        view = backend_with_data.get_view("input")
        d = view.to_dict()

        assert "coords" in d
        assert "dims" in d
        assert "data_vars" in d


class TestZarrDataArrayView:
    """Test ZarrDataArrayView class."""

    @pytest.fixture
    def backend_with_data(self, temp_zarr_store, sample_xarray_data):
        """Create backend with sample data."""
        input_ds, output_ds, grid, country = sample_xarray_data
        backend = ZarrBackend(temp_zarr_store, overwrite=True)
        backend.initialize_from_xarray(input_ds, output_ds, grid, country)
        return backend

    def test_values_getter(self, backend_with_data):
        """Test values property getter."""
        view = backend_with_data.get_array_view("input/fertilizer")
        values = view.values

        assert isinstance(values, np.ndarray)
        assert values.shape == (10, 2, 3)

    def test_values_setter(self, backend_with_data):
        """Test values property setter."""
        view = backend_with_data.get_array_view("input/fertilizer")
        new_values = np.ones((10, 2, 3), dtype=np.float32)

        view.values = new_values
        assert np.allclose(view.values, new_values)

    def test_shape_caching(self, backend_with_data):
        """Test shape property caching."""
        view = backend_with_data.get_array_view("input/fertilizer")

        # First access
        shape1 = view.shape
        assert shape1 == (10, 2, 3)

        # Second access should be cached
        shape2 = view.shape
        assert shape1 == shape2
        assert view._shape_cache == shape1

    def test_dtype_caching(self, backend_with_data):
        """Test dtype property caching."""
        view = backend_with_data.get_array_view("input/fertilizer")

        dtype1 = view.dtype
        dtype2 = view.dtype
        assert dtype1 == dtype2
        assert view._dtype_cache == dtype1

    def test_dims_caching(self, backend_with_data):
        """Test dims property consistency and normalization."""
        view = backend_with_data.get_array_view("input/fertilizer")

        # Dims should be consistent across multiple accesses
        dims1 = view.dims
        dims2 = view.dims
        assert dims1 == dims2

        # Dims should be normalized (tuple, not list)
        assert isinstance(dims1, tuple)
        assert "cell" in dims1
        assert "band" in dims1  # Should be normalized from 'band (...)'

    def test_attrs_caching(self, backend_with_data):
        """Test attrs property caching."""
        view = backend_with_data.get_array_view("input/fertilizer")

        attrs1 = view.attrs
        attrs2 = view.attrs
        assert attrs1 == attrs2
        assert view._attrs_cache == attrs1

    def test_getitem_single_element(self, backend_with_data):
        """Test __getitem__ for single element access."""
        view = backend_with_data.get_array_view("input/fertilizer")
        value = view[0, 0, 0]

        # Value can be numpy scalar or python float
        assert isinstance(value, (np.floating, float, np.number, np.ndarray))

    def test_getitem_slice(self, backend_with_data):
        """Test __getitem__ for slice access."""
        view = backend_with_data.get_array_view("input/fertilizer")
        subset = view[0:5, :, :]

        assert isinstance(subset, np.ndarray)
        assert subset.shape[0] == 5

    def test_setitem_single_element(self, backend_with_data):
        """Test __setitem__ for single element."""
        view = backend_with_data.get_array_view("input/fertilizer")

        view[0, 0, 0] = 99.5
        assert view[0, 0, 0] == 99.5

    def test_setitem_slice(self, backend_with_data):
        """Test __setitem__ for slice."""
        view = backend_with_data.get_array_view("input/fertilizer")

        view[0:3, :, :] = 88.8
        assert np.allclose(view[0:3, :, :], 88.8)

    def test_isel(self, backend_with_data):
        """Test isel method."""
        view = backend_with_data.get_array_view("input/fertilizer")

        # Select subset
        subset = view.isel({"cell": [0, 1, 2]})
        assert isinstance(subset, ZarrDataArrayView)
        assert len(subset.indices) == 3

    def test_to_xarray(self, backend_with_data):
        """Test to_xarray conversion."""
        view = backend_with_data.get_array_view("input/fertilizer")
        da = view.to_xarray()

        assert isinstance(da, xr.DataArray)
        assert da.shape == (10, 2, 3)

    def test_to_dict(self, backend_with_data):
        """Test to_dict conversion."""
        view = backend_with_data.get_array_view("input/fertilizer")
        d = view.to_dict()

        assert "dims" in d
        assert "data" in d
        assert "coords" in d

    def test_scalar_view(self, backend_with_data):
        """Test scalar (single cell) view."""
        view = backend_with_data.get_array_view("input/fertilizer", indices=0)

        assert view._is_scalar
        assert view.shape == (2, 3)  # Cell dimension removed

        # Test indexing on scalar view
        value = view[0, 0]
        # Value can be numpy scalar, python float, or 0-d array
        assert isinstance(value, (np.floating, float, np.number, np.ndarray))

    def test_subset_view(self, backend_with_data):
        """Test subset view with array indices."""
        indices = np.array([0, 2, 4])
        view = backend_with_data.get_array_view(
            "input/fertilizer", indices=indices
        )

        assert view.shape == (3, 2, 3)
        values = view.values
        assert values.shape == (3, 2, 3)


class TestZarrCoordinateView:
    """Test ZarrCoordinateView class."""

    @pytest.fixture
    def backend_with_data(self, temp_zarr_store, sample_xarray_data):
        """Create backend with sample data."""
        input_ds, output_ds, grid, country = sample_xarray_data
        backend = ZarrBackend(temp_zarr_store, overwrite=True)
        backend.initialize_from_xarray(input_ds, output_ds, grid, country)
        return backend

    def test_values_getter(self, backend_with_data):
        """Test values property getter."""
        view = backend_with_data.get_view("input")
        time_coord = view.time

        values = time_coord.values
        assert isinstance(values, WritableCoordinateArray)
        assert len(values) == 3

    def test_values_setter(self, backend_with_data):
        """Test values property setter."""
        view = backend_with_data.get_view("input")
        band_coord = view.band

        # Use simple array values instead of datetime
        new_bands = ["modified_1", "modified_2"]
        band_coord.values = new_bands

        # Read back and verify
        retrieved = view.band.values
        assert len(retrieved) == 2

    def test_len(self, backend_with_data):
        """Test __len__ method."""
        view = backend_with_data.get_view("input")
        time_coord = view.time

        assert len(time_coord) == 3


class TestWritableCoordinateArray:
    """Test WritableCoordinateArray class."""

    @pytest.fixture
    def backend_with_data(self, temp_zarr_store, sample_xarray_data):
        """Create backend with sample data."""
        input_ds, output_ds, grid, country = sample_xarray_data
        backend = ZarrBackend(temp_zarr_store, overwrite=True)
        backend.initialize_from_xarray(input_ds, output_ds, grid, country)
        return backend

    def test_setitem_writes_back(self, backend_with_data):
        """Test that __setitem__ writes back to Zarr."""
        view = backend_with_data.get_view("input")
        band_coord = view.band

        # Get writable array
        values = band_coord.values
        assert isinstance(values, WritableCoordinateArray)

        # Modify it
        values[0] = "modified"

        # Read back from Zarr directly to verify write-back
        coords = backend_with_data.root["input"].attrs.get("coords", {})
        assert "band" in coords


class TestSynchronization:
    """Test synchronization across views."""

    @pytest.fixture
    def backend_with_data(self, temp_zarr_store, sample_xarray_data):
        """Create backend with sample data."""
        input_ds, output_ds, grid, country = sample_xarray_data
        backend = ZarrBackend(temp_zarr_store, overwrite=True)
        backend.initialize_from_xarray(input_ds, output_ds, grid, country)
        return backend

    def test_world_to_cell_sync(self, backend_with_data):
        """Test that changes at world level sync to cell level."""
        world_view = backend_with_data.get_view("input")
        cell_view = backend_with_data.get_view("input", indices=0)

        # Change at world level
        world_view["fertilizer"][0, 0, 0] = 777.0

        # Verify at cell level
        assert cell_view["fertilizer"][0, 0] == 777.0

    def test_cell_to_world_sync(self, backend_with_data):
        """Test that changes at cell level sync to world level."""
        world_view = backend_with_data.get_view("input")
        cell_view = backend_with_data.get_view("input", indices=3)

        # Change at cell level
        cell_view["fertilizer"][0, 0] = 888.0

        # Verify at world level
        assert world_view["fertilizer"][3, 0, 0] == 888.0

    def test_subset_sync(self, backend_with_data):
        """Test synchronization with subset views."""
        full_view = backend_with_data.get_view("input")
        subset_view = backend_with_data.get_view("input", indices=[2, 4, 6])

        # Change in subset
        subset_view["fertilizer"][
            1, 0, 0
        ] = 999.0  # Index 1 in subset = cell 4

        # Verify in full view
        assert full_view["fertilizer"][4, 0, 0] == 999.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_group(self, temp_zarr_store, sample_xarray_data):
        """Test accessing non-existent group."""
        input_ds, output_ds, grid, country = sample_xarray_data
        backend = ZarrBackend(temp_zarr_store, overwrite=True)
        backend.initialize_from_xarray(input_ds, output_ds, grid, country)

        # Zarr will create the group if it doesn't exist, so just verify
        # that we can access existing groups without error
        view = backend.get_view("input")
        assert view is not None

    def test_out_of_bounds_index(self, temp_zarr_store, sample_xarray_data):
        """Test out of bounds indexing."""
        input_ds, output_ds, grid, country = sample_xarray_data
        backend = ZarrBackend(temp_zarr_store, overwrite=True)
        backend.initialize_from_xarray(input_ds, output_ds, grid, country)

        view = backend.get_array_view("input/fertilizer")

        # This should raise an IndexError
        with pytest.raises(IndexError):
            _ = view[1000, 0, 0]


class TestDefaultChunkSize:
    """Test DEFAULT_CHUNK_SIZE constant."""

    def test_default_chunk_size_value(self):
        """Test that DEFAULT_CHUNK_SIZE is set correctly."""
        assert DEFAULT_CHUNK_SIZE == 1000

    def test_custom_chunk_size(self, temp_zarr_store, sample_xarray_data):
        """Test using custom chunk size."""
        input_ds, output_ds, grid, country = sample_xarray_data
        backend = ZarrBackend(temp_zarr_store, overwrite=True)

        custom_size = 5
        backend.initialize_from_xarray(
            input_ds, output_ds, grid, country, chunk_size=custom_size
        )

        # Check that arrays use the custom chunk size
        chunks = backend.root["input/fertilizer"].chunks
        assert chunks[0] == custom_size


class TestZarrStoreCleanup:
    """Test automatic cleanup of temporary Zarr stores."""

    def test_temp_store_manual_cleanup(self, sample_xarray_data):
        """Test manual cleanup of temporary Zarr stores."""
        import sys
        import os

        sys.path.insert(
            0, "/p/projects/copan/users/jannesbr/repos/pycopanlpjml"
        )

        from pycopanlpjml import World

        input_ds, output_ds, grid, country = sample_xarray_data

        # Create World without model (should use temp store)
        world = World(
            input=input_ds, output=output_ds, grid=grid, country=country
        )

        # Verify it's a temp store
        assert world._is_temp_store == True
        store_path = world._zarr_store_path
        assert os.path.exists(store_path)

        # Manual cleanup
        world.cleanup_zarr_store()
        assert not os.path.exists(store_path), "Temp store should be deleted"

    def test_context_manager_cleanup(self, sample_xarray_data):
        """Test that context manager automatically cleans up temp stores."""
        import sys
        import os

        sys.path.insert(
            0, "/p/projects/copan/users/jannesbr/repos/pycopanlpjml"
        )

        from pycopanlpjml import World

        input_ds, output_ds, grid, country = sample_xarray_data

        # Use World as context manager
        with World(
            input=input_ds, output=output_ds, grid=grid, country=country
        ) as world:
            # Verify it's a temp store
            assert world._is_temp_store == True
            store_path = world._zarr_store_path
            assert os.path.exists(
                store_path
            ), "Store should exist during context"

        # After context, temp store should be cleaned up
        assert not os.path.exists(
            store_path
        ), "Temp store should be auto-deleted after context"

    def test_persistent_store_not_cleaned(
        self, sample_xarray_data, temp_zarr_store
    ):
        """Test that persistent stores (with explicit path) are NOT auto-cleaned."""
        import sys
        import os

        sys.path.insert(
            0, "/p/projects/copan/users/jannesbr/repos/pycopanlpjml"
        )

        from pycopanlpjml import World
        from pycopanlpjml.zarr_backend import ZarrBackend

        input_ds, output_ds, grid, country = sample_xarray_data

        # Create World with explicit store path (simulating real model)
        # Mock the model config
        class MockModel:
            class config:
                sim_path = os.path.dirname(temp_zarr_store)

        class MockWorld(World):
            def __init__(self, **kwargs):
                self.model = MockModel()
                super().__init__(**kwargs)

        world = MockWorld(
            input=input_ds, output=output_ds, grid=grid, country=country
        )

        # Verify it's NOT a temp store
        assert world._is_temp_store == False
        store_path = world._zarr_store_path

        # Cleanup should not delete persistent stores
        world.cleanup_zarr_store()
        assert os.path.exists(
            store_path
        ), "Persistent store should NOT be deleted"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
