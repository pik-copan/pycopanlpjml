"""Tests for correct LPJmLData/LPJmLDataSet representation behavior.

These tests verify that Zarr views behave exactly like pycoupler's
LPJmLDataSet and LPJmLData, including proper dimension normalization.
"""

import pytest
import numpy as np
import xarray as xr
from pycoupler.data import LPJmLDataSet, LPJmLData

from pycopanlpjml.world import World


@pytest.fixture
def world_with_multiple_bands():
    """Create a World instance with multiple band dimensions for testing."""
    n_cells = 5
    n_time = 2

    # Create input dataset
    input_ds = LPJmLDataSet(
        xr.Dataset(
            {
                "with_tillage": (
                    ["cell", "band (with_tillage)", "time"],
                    np.random.rand(n_cells, 1, n_time),
                )
            },
            coords={
                "cell": list(range(n_cells)),
                "band (with_tillage)": ["yes"],
                "time": [
                    np.datetime64("2022-01-01", "ns"),
                    np.datetime64("2022-02-01", "ns"),
                ],
            },
        )
    )

    # Create output dataset with multiple band dimensions
    output_ds = LPJmLDataSet(
        xr.Dataset(
            {
                "hdate": (
                    ["cell", "band (hdate)", "time"],
                    np.random.randint(0, 365, (n_cells, 24, n_time)),
                ),
                "pft_harvestc": (
                    ["cell", "band (pft_harvestc)", "time"],
                    np.random.rand(n_cells, 32, n_time),
                ),
                "soilc_agr_layer": (
                    ["cell", "band (soilc_agr_layer)", "time"],
                    np.random.rand(n_cells, 5, n_time),
                ),
                "cftfrac": (
                    ["cell", "band (cftfrac)", "time"],
                    np.random.rand(n_cells, 32, n_time),
                ),
            },
            coords={
                "cell": list(range(n_cells)),
                "band (hdate)": [f"crop_{i}" for i in range(24)],
                "band (pft_harvestc)": [f"pft_{i}" for i in range(32)],
                "band (soilc_agr_layer)": [
                    200.0,
                    500.0,
                    1000.0,
                    2000.0,
                    3000.0,
                ],
                "band (cftfrac)": [f"cft_{i}" for i in range(32)],
                "time": [
                    np.datetime64("2022-01-01", "ns"),
                    np.datetime64("2022-02-01", "ns"),
                ],
            },
        )
    )

    # Add non-dimension coordinates
    output_ds = output_ds.assign_coords(
        {
            "lon": ("cell", np.random.rand(n_cells) * 10),
            "lat": ("cell", np.random.rand(n_cells) * 10),
        }
    )

    # Create grid
    grid = xr.DataArray(
        np.random.rand(n_cells, 2),
        dims=["cell", "coord"],
        coords={"cell": list(range(n_cells)), "coord": ["lon", "lat"]},
    )

    # Create country codes
    country = xr.DataArray(
        [f"C{i}" for i in range(n_cells)],
        dims=["cell"],
        coords={"cell": list(range(n_cells))},
    )

    world = World(
        input=input_ds, output=output_ds, grid=grid, country_code=country
    )

    return world


class TestLPJmLDataRepresentation:
    """Test that Zarr views behave like pycoupler LPJmLDataSet/LPJmLData."""

    def test_dataset_level_shows_full_dimension_names(
        self, world_with_multiple_bands
    ):
        """
        Test that dataset level (world.output) shows full dimension names.

        Should show: band (hdate), band (pft_harvestc), etc. (not just 'band')
        """
        output = world_with_multiple_bands.output

        # Convert to xarray to check dimensions
        ds = output._get_xarray()

        # Should have full dimension names
        assert "band (hdate)" in ds.dims
        assert "band (pft_harvestc)" in ds.dims
        assert "band (soilc_agr_layer)" in ds.dims
        assert "band (cftfrac)" in ds.dims

        # Should be LPJmLDataSet type
        assert isinstance(ds, LPJmLDataSet)

    def test_dataset_level_has_all_coordinates(
        self, world_with_multiple_bands
    ):
        """
        Test that dataset level includes all dimension coordinates.

        Should NOT show "Dimensions without coordinates".
        """
        output = world_with_multiple_bands.output
        ds = output._get_xarray()

        # Should have all band coordinates
        assert "band (hdate)" in ds.coords
        assert "band (pft_harvestc)" in ds.coords
        assert "band (soilc_agr_layer)" in ds.coords
        assert "band (cftfrac)" in ds.coords

        # Should have dimension coordinates
        assert "cell" in ds.coords
        assert "time" in ds.coords

    def test_variable_access_normalizes_dimensions(
        self, world_with_multiple_bands
    ):
        """
        Test that accessing individual variables normalizes dimensions.

        Should show: band (not band (hdate))
        This matches pycoupler LPJmLDataSet._construct_dataarray behavior.
        """
        output = world_with_multiple_bands.output

        # Access hdate variable
        hdate = output.hdate

        # Dimensions should be normalized to 'band'
        assert hdate.dims == ("cell", "band", "time")

        # Should be LPJmLData type
        assert isinstance(hdate, (LPJmLData, xr.DataArray))

    def test_variable_access_has_correct_coordinates(
        self, world_with_multiple_bands
    ):
        """
        Test that individual variables only have their own coordinates.

        Should NOT have band (pft_harvestc) when accessing hdate.
        """
        output = world_with_multiple_bands.output

        # Access hdate variable
        hdate = output.hdate

        # Should have normalized 'band' coordinate (not 'band (hdate)')
        assert "band" in hdate.coords

        # Should NOT have other band coordinates
        assert "band (pft_harvestc)" not in hdate.coords
        assert "band (soilc_agr_layer)" not in hdate.coords
        assert "band (cftfrac)" not in hdate.coords

        # Should have shared coordinates
        assert "cell" in hdate.coords
        assert "time" in hdate.coords

    def test_all_variables_accessible_without_error(
        self, world_with_multiple_bands
    ):
        """
        Test that all variables can be accessed without coordinate conflicts.

        Previously failed with: "coordinate band (pft_harvestc) has dimensions
        ('band (pft_harvestc)',), but these are not a subset of the DataArray
        dimensions ('cell', 'band (hdate)', 'time')"
        """
        output = world_with_multiple_bands.output

        # All variables should be accessible without error
        variables = ["hdate", "pft_harvestc", "soilc_agr_layer", "cftfrac"]

        for var_name in variables:
            var_data = getattr(output, var_name)

            # Should be LPJmLData type
            assert isinstance(var_data, (LPJmLData, xr.DataArray))

            # Should have normalized dimensions
            assert var_data.dims == ("cell", "band", "time")

            # Should have 'band' coordinate (normalized)
            assert "band" in var_data.coords

    def test_dict_vs_attribute_access(self, world_with_multiple_bands):
        """Test that both dict-style and attribute access provide LPJmLData-like behavior.

        Both access methods now provide:
        - LPJmLData representation with normalized dimensions
        - Write capability that syncs across world/country/cell

        The difference is that dict-style returns a ZarrDataArrayView (writable view)
        while attribute access returns LPJmLData (copy), but both look and behave the same.
        """
        output = world_with_multiple_bands.output

        # Attribute access returns LPJmLData with normalized dimensions
        hdate_attr = output.hdate
        assert isinstance(hdate_attr, (LPJmLData, xr.DataArray))
        assert hdate_attr.dims == ("cell", "band", "time")  # Normalized

        # Dict-style access returns ZarrDataArrayView but behaves like LPJmLData
        from pycopanlpjml.zarr_backend import ZarrDataArrayView

        hdate_dict = output["hdate"]
        assert isinstance(hdate_dict, ZarrDataArrayView)
        # But it should have normalized dimensions too
        assert hdate_dict.dims == ("cell", "band", "time")  # Normalized
        # And normalized coords
        assert "band" in hdate_dict.coords
        assert "band (hdate)" not in hdate_dict.coords  # Should be normalized

    def test_coordinate_values_preserved(self, world_with_multiple_bands):
        """Test that coordinate values are preserved correctly."""
        output = world_with_multiple_bands.output

        hdate = output.hdate

        # The 'band' coordinate should have the values from 'band (hdate)'
        band_values = hdate.coords["band"].values
        assert len(band_values) == 24
        assert band_values[0].startswith("crop_")

    def test_lazy_loading_performance(self, world_with_multiple_bands):
        """Test that lazy loading doesn't convert until needed."""
        import time

        # Getting the view should be very fast (no conversion)
        start = time.time()
        for _ in range(100):
            view = world_with_multiple_bands.output
        view_time = time.time() - start

        # Should be very fast (< 1ms total for 100 accesses)
        assert (
            view_time < 0.01
        ), f"View access too slow: {view_time*1000:.2f}ms for 100 accesses"

        # First variable access triggers conversion (will be slower)
        fresh_view = world_with_multiple_bands.output
        start = time.time()
        _ = fresh_view.hdate
        first_access = time.time() - start

        # Should be reasonable (< 200ms for large dataset)
        assert (
            first_access < 0.2
        ), f"First access too slow: {first_access*1000:.2f}ms"

        # Subsequent accesses use cache (should be very fast)
        start = time.time()
        for _ in range(100):
            _ = fresh_view.hdate
        cached_time = time.time() - start

        # Should be fast (< 10ms total for 100 accesses)
        assert (
            cached_time < 0.01
        ), f"Cached access too slow: {cached_time*1000:.2f}ms for 100 accesses"

    def test_cell_level_dimension_normalization(
        self, world_with_multiple_bands
    ):
        """
        Test that cell-level views also have proper dimension normalization.

        When accessing first_cell.output.hdate, should show:
        - dims: (band, time) not (band (hdate), time)
        - coordinate: band (not band (hdate))
        """
        # Get first cell (note: need to initialize cells first)
        world = world_with_multiple_bands

        # Initialize cells
        from pycopanlpjml.cell import Cell

        cells = [Cell(world=world, cell_index=i) for i in range(5)]

        first_cell = cells[0]

        # Access cell-level output
        cell_hdate = first_cell.output.hdate

        # Dimensions should be normalized and cell dimension dropped
        assert cell_hdate.dims == ("band", "time")

        # Should have normalized 'band' coordinate
        assert "band" in cell_hdate.coords
        assert "band (hdate)" not in cell_hdate.coords

    def test_repr_shows_lpjml_types(self, world_with_multiple_bands):
        """Test that repr shows LPJmLDataSet and LPJmLData types."""
        output = world_with_multiple_bands.output

        # Dataset repr should show LPJmLDataSet
        ds_repr = repr(output)
        assert "LPJmLDataSet" in ds_repr

        # Variable repr should show LPJmLData
        hdate = output.hdate
        hdate_repr = repr(hdate)
        assert "LPJmLData" in hdate_repr or "DataArray" in hdate_repr

    def test_xarray_methods_work(self, world_with_multiple_bands):
        """Test that all xarray methods work through delegation."""
        output = world_with_multiple_bands.output
        hdate = output.hdate

        # Test common xarray methods
        mean_val = hdate.mean()
        assert mean_val is not None

        sum_val = hdate.sum()
        assert sum_val is not None

        # Test selection methods
        subset = hdate.isel(cell=0)
        assert subset.dims == ("band", "time")

    def test_coordinate_attribute_access(self, world_with_multiple_bands):
        """Test that coordinates can be accessed as attributes."""
        output = world_with_multiple_bands.output
        hdate = output.hdate

        # Should be able to access coordinates as attributes
        band_coord = hdate.band
        assert len(band_coord) == 24

        time_coord = hdate.time
        assert len(time_coord) == 2

        cell_coord = hdate.cell
        assert len(cell_coord) == 5

    def test_multiple_variables_with_different_bands(
        self, world_with_multiple_bands
    ):
        """
        Test that multiple variables with different band dimensions work correctly.

        Each variable should have its own normalized 'band' coordinate with the
        correct number of elements.
        """
        output = world_with_multiple_bands.output

        # hdate has 24 bands
        hdate = output.hdate
        assert hdate.dims == ("cell", "band", "time")
        assert len(hdate.coords["band"]) == 24

        # pft_harvestc has 32 bands
        pft_harvestc = output.pft_harvestc
        assert pft_harvestc.dims == ("cell", "band", "time")
        assert len(pft_harvestc.coords["band"]) == 32

        # soilc_agr_layer has 5 bands
        soilc = output.soilc_agr_layer
        assert soilc.dims == ("cell", "band", "time")
        assert len(soilc.coords["band"]) == 5

        # cftfrac has 32 bands
        cftfrac = output.cftfrac
        assert cftfrac.dims == ("cell", "band", "time")
        assert len(cftfrac.coords["band"]) == 32


class TestDimensionNormalizationConsistency:
    """Test that dimension normalization is consistent with pycoupler."""

    def test_dataset_dims_vs_variable_dims(self, world_with_multiple_bands):
        """
        Test the key difference between dataset and variable level.

        Dataset level: full names (band (hdate), band (pft_harvestc))
        Variable level: normalized (band)
        """
        output = world_with_multiple_bands.output

        # Dataset level should show full dimension names
        ds = output._get_xarray()
        assert "band (hdate)" in ds.dims
        assert "band (pft_harvestc)" in ds.dims

        # Variable level should show normalized dimensions
        hdate = output.hdate
        assert hdate.dims == ("cell", "band", "time")
        assert "band (hdate)" not in hdate.dims

    def test_coordinate_renaming_in_variables(self, world_with_multiple_bands):
        """
        Test that coordinates are renamed when accessing variables.

        Dataset: coordinate 'band (hdate)' with dims ('band (hdate)',)
        Variable: coordinate 'band' with dims ('band',)
        """
        output = world_with_multiple_bands.output

        # Dataset level
        ds = output._get_xarray()
        assert "band (hdate)" in ds.coords

        # Variable level - coordinate should be renamed to 'band'
        hdate = output.hdate
        assert "band" in hdate.coords
        assert "band (hdate)" not in hdate.coords
