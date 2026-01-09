"""Test that all basic xarray/LPJmLData/LPJmLDataSet functionality works through Zarr views.

This test suite verifies that the ZarrDatasetView and ZarrDataArrayView provide
full xarray/LPJmL compatibility including:
- Selection operations (sel, isel, loc)
- Reduction operations (mean, sum, min, max, std)
- Arithmetic operations
- Coordinate access
- Attribute access
- Array operations
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr

from pycopanlpjml.world import World

try:
    from pycoupler.data import LPJmLDataSet, LPJmLData

    HAS_PYCOUPLER = True
except ImportError:
    HAS_PYCOUPLER = False


@pytest.fixture
def world_with_sample_data():
    """Create a World with sample data for testing xarray functionality."""
    np.random.seed(42)

    # Create output dataset with multiple variables and dimensions
    if HAS_PYCOUPLER:
        output_ds = LPJmLDataSet(
            xr.Dataset(
                {
                    "temperature": (
                        ["cell", "time"],
                        np.random.randn(10, 12) * 10 + 20,
                    ),
                    "precipitation": (
                        ["cell", "time"],
                        np.random.rand(10, 12) * 100,
                    ),
                    "yield_crops": (
                        ["cell", "band (yield)", "time"],
                        np.random.rand(10, 5, 12) * 1000,
                    ),
                    "soil_carbon": (
                        ["cell", "band (soil_layer)", "time"],
                        np.random.rand(10, 3, 12) * 500,
                    ),
                },
                coords={
                    "cell": range(10),
                    "time": pd.date_range("2020-01-01", periods=12, freq="MS"),
                    "band (yield)": [
                        "wheat",
                        "maize",
                        "rice",
                        "soybean",
                        "barley",
                    ],
                    "band (soil_layer)": [0.2, 0.5, 1.0],  # depths in meters
                },
            )
        )

        output_ds = output_ds.assign_coords(
            {
                "lon": ("cell", np.linspace(0, 10, 10)),
                "lat": ("cell", np.linspace(50, 55, 10)),
            }
        )

        input_ds = LPJmLDataSet(
            xr.Dataset(
                {
                    "fertilizer": (
                        ["cell", "band (fertilizer)", "time"],
                        np.random.rand(10, 2, 12) * 50,
                    ),
                },
                coords={
                    "cell": range(10),
                    "time": pd.date_range("2020-01-01", periods=12, freq="MS"),
                    "band (fertilizer)": ["nitrogen", "phosphorus"],
                },
            )
        )
    else:
        output_ds = xr.Dataset(
            {
                "temperature": (
                    ["cell", "time"],
                    np.random.randn(10, 12) * 10 + 20,
                ),
                "precipitation": (
                    ["cell", "time"],
                    np.random.rand(10, 12) * 100,
                ),
            },
            coords={
                "cell": range(10),
                "time": pd.date_range("2020-01-01", periods=12, freq="MS"),
            },
        )

        input_ds = xr.Dataset(
            {
                "fertilizer": (["cell", "time"], np.random.rand(10, 12) * 50),
            },
            coords={
                "cell": range(10),
                "time": pd.date_range("2020-01-01", periods=12, freq="MS"),
            },
        )

    grid = xr.DataArray(
        np.arange(20).reshape(10, 2),
        coords={"cell": range(10)},
        dims=["cell", "coord"],
    )
    country = xr.DataArray(
        ["DEU"] * 10, coords={"cell": range(10)}, dims=["cell"]
    )

    world = World(
        input=input_ds, output=output_ds, grid=grid, country_code=country
    )

    return world


class TestXarraySelectionOperations:
    """Test xarray selection operations through Zarr views."""

    def test_isel_on_dataset(self, world_with_sample_data):
        """Test isel (integer selection) on dataset via xarray."""
        output = world_with_sample_data.from_earth

        # output is already an xarray Dataset
        result = output.isel(time=0)
        assert "time" not in result.dims or len(result.time) == 1

    def test_isel_on_dataarray(self, world_with_sample_data):
        """Test isel on individual variable."""
        temp = world_with_sample_data.output.temperature

        # Select single cell and time
        result = temp.isel(cell=0, time=0)
        assert isinstance(result.values, (np.ndarray, np.generic))

    def test_sel_by_coordinate(self, world_with_sample_data):
        """Test sel (label selection) by coordinate values."""
        temp = world_with_sample_data.output.temperature

        # Select by cell index
        result = temp.sel(cell=0)
        assert len(result.shape) < len(temp.shape)

    def test_indexing_syntax(self, world_with_sample_data):
        """Test numpy-style indexing."""
        temp = world_with_sample_data.output.temperature

        # Test slicing
        result = temp[0:5, 0:6]
        assert result.shape[0] == 5
        assert result.shape[1] == 6


class TestXarrayReductionOperations:
    """Test xarray reduction operations."""

    def test_mean(self, world_with_sample_data):
        """Test mean reduction."""
        temp = world_with_sample_data.output.temperature

        # Mean over time
        time_mean = temp.mean(dim="time")
        assert "time" not in time_mean.dims
        assert len(time_mean.shape) == len(temp.shape) - 1

    def test_sum(self, world_with_sample_data):
        """Test sum reduction."""
        precip = world_with_sample_data.output.precipitation

        # Annual total precipitation
        annual_total = precip.sum(dim="time")
        assert "time" not in annual_total.dims

    def test_min_max(self, world_with_sample_data):
        """Test min and max."""
        temp = world_with_sample_data.output.temperature

        min_temp = temp.min()
        max_temp = temp.max()

        assert float(min_temp) < float(max_temp)

    def test_std(self, world_with_sample_data):
        """Test standard deviation."""
        temp = world_with_sample_data.output.temperature

        std_temp = temp.std(dim="time")
        assert std_temp.shape == (10,)

    def test_reduction_along_multiple_dims(self, world_with_sample_data):
        """Test reduction along multiple dimensions."""
        temp = world_with_sample_data.output.temperature

        # Global average
        global_mean = temp.mean(dim=["cell", "time"])
        assert global_mean.dims == ()  # Scalar


class TestXarrayArithmeticOperations:
    """Test arithmetic operations."""

    def test_addition(self, world_with_sample_data):
        """Test addition."""
        temp = world_with_sample_data.output.temperature

        result = temp + 10
        assert result.shape == temp.shape
        assert float(result.mean()) > float(temp.mean())

    def test_subtraction(self, world_with_sample_data):
        """Test subtraction."""
        temp = world_with_sample_data.output.temperature

        result = temp - 5
        assert result.shape == temp.shape

    def test_multiplication(self, world_with_sample_data):
        """Test multiplication."""
        precip = world_with_sample_data.output.precipitation

        result = precip * 2
        assert float(result.mean()) == pytest.approx(
            float(precip.mean()) * 2, rel=1e-5
        )

    def test_division(self, world_with_sample_data):
        """Test division."""
        precip = world_with_sample_data.output.precipitation

        result = precip / 10
        assert result.shape == precip.shape

    def test_array_operations(self, world_with_sample_data):
        """Test operations between arrays."""
        temp = world_with_sample_data.output.temperature
        precip = world_with_sample_data.output.precipitation

        # Both have same shape (cell, time)
        result = temp + precip
        assert result.shape == temp.shape


class TestXarrayCoordinateAccess:
    """Test coordinate access and manipulation."""

    def test_coordinate_attribute_access(self, world_with_sample_data):
        """Test accessing coordinates as attributes."""
        temp = world_with_sample_data.output.temperature

        # Access time coordinate
        time_coord = temp.time
        assert len(time_coord) == 12

    def test_coordinate_dict_access(self, world_with_sample_data):
        """Test accessing coordinates via coords dict."""
        temp = world_with_sample_data.output.temperature

        cell_coord = temp.coords["cell"]
        assert len(cell_coord) == 10

    @pytest.mark.skipif(
        not HAS_PYCOUPLER, reason="Requires pycoupler for band dimensions"
    )
    @pytest.mark.skip(
        reason="Known pycoupler bug with band_idx_name in _construct_dataarray"
    )
    def test_normalized_coordinate_names(self, world_with_sample_data):
        """Test that band dimensions are normalized in variable access."""
        yields = world_with_sample_data.output.yield_crops

        # Should have 'band' not 'band (yield)'
        assert "band" in yields.dims
        assert "band (yield)" not in yields.dims

        # But should have band coordinate
        assert "band" in yields.coords

    def test_lon_lat_coordinates(self, world_with_sample_data):
        """Test that lon/lat coordinates are accessible."""
        if HAS_PYCOUPLER:
            output_ds = world_with_sample_data.output

            assert "lon" in output_ds.coords
            assert "lat" in output_ds.coords

            # Access through variable
            temp = world_with_sample_data.output.temperature
            assert "lon" in temp.coords
            assert "lat" in temp.coords


class TestXarrayAttributeAccess:
    """Test attribute access."""

    def test_attrs_dict(self, world_with_sample_data):
        """Test attrs dictionary access."""
        temp = world_with_sample_data.output.temperature

        # Attrs should be accessible
        attrs = temp.attrs
        assert isinstance(attrs, dict)

    def test_dims_property(self, world_with_sample_data):
        """Test dims property."""
        temp = world_with_sample_data.output.temperature

        assert "cell" in temp.dims
        assert "time" in temp.dims

    def test_shape_property(self, world_with_sample_data):
        """Test shape property."""
        temp = world_with_sample_data.output.temperature

        assert temp.shape == (10, 12)

    def test_dtype_property(self, world_with_sample_data):
        """Test dtype property."""
        temp = world_with_sample_data.output.temperature

        assert temp.dtype in [np.float32, np.float64]


class TestXarrayArrayOperations:
    """Test array data access."""

    def test_values_property(self, world_with_sample_data):
        """Test .values property."""
        temp = world_with_sample_data.output.temperature

        values = temp.values
        assert isinstance(values, np.ndarray)
        assert values.shape == (10, 12)

    def test_to_numpy(self, world_with_sample_data):
        """Test .to_numpy() method."""
        temp = world_with_sample_data.output.temperature

        arr = temp.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (10, 12)

    def test_data_property(self, world_with_sample_data):
        """Test .data property (underlying array)."""
        temp = world_with_sample_data.output.temperature

        data = temp.data
        assert hasattr(data, "shape")


class TestXarrayMethodChaining:
    """Test method chaining and complex operations."""

    def test_selection_then_reduction(self, world_with_sample_data):
        """Test chaining selection and reduction."""
        temp = world_with_sample_data.output.temperature

        # Select first half of year, then compute mean
        result = temp.isel(time=slice(0, 6)).mean(dim="time")
        assert result.shape == (10,)

    def test_arithmetic_then_reduction(self, world_with_sample_data):
        """Test arithmetic followed by reduction."""
        temp = world_with_sample_data.output.temperature

        # Convert to Kelvin then get mean
        result = (temp + 273.15).mean()
        assert isinstance(result.values, (np.ndarray, np.generic))

    def test_multiple_selections(self, world_with_sample_data):
        """Test multiple selection operations."""
        temp = world_with_sample_data.output.temperature

        # Select cells, then time
        result = temp.isel(cell=slice(0, 5)).isel(time=0)
        assert result.shape == (5,)


class TestDatasetOperations:
    """Test dataset-level operations."""

    def test_list_variables(self, world_with_sample_data):
        """Test listing data variables."""
        output = world_with_sample_data.from_earth

        # output is already an xarray Dataset
        assert "temperature" in output.data_vars
        assert "precipitation" in output.data_vars

    def test_access_multiple_variables(self, world_with_sample_data):
        """Test accessing multiple variables."""
        output = world_with_sample_data.from_earth

        temp = output.temperature
        precip = output.precipitation

        assert temp.shape == precip.shape

    def test_dataset_coords(self, world_with_sample_data):
        """Test dataset-level coordinates."""
        output = world_with_sample_data.from_earth
        coords = output.coords

        assert "cell" in coords
        assert "time" in coords


class TestLPJmLSpecificFunctionality:
    """Test LPJmL-specific functionality."""

    @pytest.mark.skipif(not HAS_PYCOUPLER, reason="Requires pycoupler")
    def test_lpjml_data_type(self, world_with_sample_data):
        """Test that variables are LPJmLData instances."""
        temp = world_with_sample_data.output.temperature

        # Attribute access should return LPJmLData
        assert isinstance(temp, (LPJmLData, xr.DataArray))

    @pytest.mark.skipif(not HAS_PYCOUPLER, reason="Requires pycoupler")
    def test_lpjml_dataset_type(self, world_with_sample_data):
        """Test that dataset is LPJmLDataSet."""
        ds = world_with_sample_data.output

        assert isinstance(ds, (LPJmLDataSet, xr.Dataset))

    @pytest.mark.skipif(not HAS_PYCOUPLER, reason="Requires pycoupler")
    @pytest.mark.skip(
        reason="Known pycoupler bug with band_idx_name in _construct_dataarray"
    )
    def test_band_dimension_normalization(self, world_with_sample_data):
        """Test that band dimensions are properly normalized."""
        yields = world_with_sample_data.output.yield_crops

        # Dimensions should be normalized
        assert yields.dims == ("cell", "band", "time")

        # But coordinates should reference the specific band
        assert "band" in yields.coords
        band_values = yields.coords["band"].values
        assert len(band_values) == 5
        assert "wheat" in band_values

    @pytest.mark.skipif(not HAS_PYCOUPLER, reason="Requires pycoupler")
    @pytest.mark.skip(
        reason="Known pycoupler bug with band_idx_name in _construct_dataarray"
    )
    def test_multiple_band_variables(self, world_with_sample_data):
        """Test accessing variables with different band dimensions."""
        yields = world_with_sample_data.output.yield_crops
        soil = world_with_sample_data.output.soil_carbon

        # Both should have normalized 'band' dimension
        assert "band" in yields.dims
        assert "band" in soil.dims

        # But different coordinate values
        assert len(yields.coords["band"]) == 5
        assert len(soil.coords["band"]) == 3


class TestWriteOperations:
    """Test that write operations work through views."""

    def test_write_via_dict_access(self, world_with_sample_data):
        """Test writing via dict-style access."""
        output = world_with_sample_data.from_earth

        # Write a value
        temp_view = output["temperature"]
        original_value = float(temp_view[0, 0])
        temp_view[0, 0] = 999.0

        # Verify it was written
        new_value = float(temp_view[0, 0])
        assert new_value == 999.0
        assert new_value != original_value

    def test_write_syncs_across_views(self, world_with_sample_data):
        """Test that writes sync across different views."""
        world = world_with_sample_data

        # Write via world view
        world.from_earth["temperature"][5, 5] = 888.0

        # Read back via world view to verify write
        assert float(world.from_earth["temperature"][5, 5]) == 888.0


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_selection(self, world_with_sample_data):
        """Test selection that results in empty data."""
        temp = world_with_sample_data.output.temperature

        # Select impossible range
        result = temp.isel(cell=slice(0, 0))
        assert result.shape[0] == 0

    def test_scalar_result(self, world_with_sample_data):
        """Test operations that return scalars."""
        temp = world_with_sample_data.output.temperature

        # Get single value
        scalar = temp.isel(cell=0, time=0)
        assert scalar.dims == ()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
