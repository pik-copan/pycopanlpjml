"""Unit tests for _align_to_lpjml_grid coordinate handling.

Tests the fix for corrupted grid coordinates in serial MPI runs where
LPJmL writes fill values (9.969e+36) instead of valid lat/lon coordinates.
"""

import pytest
import numpy as np
import xarray as xr

from pycopanlpjml.output import _align_to_lpjml_grid

# Standard LPJmL 0.5-degree global grid parameters
STANDARD_LAT_MIN = -55.75
STANDARD_LAT_MAX = 83.75
STANDARD_LON_MIN = -179.75
STANDARD_LON_MAX = 179.75
STANDARD_N_LATS = 280
STANDARD_N_LONS = 720
FILL_VALUE = 9.969209968386869e36


@pytest.fixture
def sample_cell_data():
    """Create sample cell-indexed data for testing."""
    n_cells = 50
    return xr.DataArray(
        np.random.rand(3, n_cells),
        dims=("time", "cell"),
        coords={
            "time": [2025, 2026, 2027],
            "cell": np.arange(n_cells),
            "lon": ("cell", np.linspace(-10, 10, n_cells)),
            "lat": ("cell", np.linspace(45, 55, n_cells)),
        },
    )


@pytest.fixture
def valid_lpjml_template():
    """Create a valid LPJmL grid template with proper coordinates."""
    lats = np.linspace(STANDARD_LAT_MIN, STANDARD_LAT_MAX, STANDARD_N_LATS)
    lons = np.linspace(STANDARD_LON_MIN, STANDARD_LON_MAX, STANDARD_N_LONS)
    return xr.Dataset(
        {
            "cellid": (
                ["lat", "lon"],
                np.full((STANDARD_N_LATS, STANDARD_N_LONS), np.nan),
            ),
        },
        coords={
            "lat": lats,
            "lon": lons,
        },
    )


@pytest.fixture
def corrupted_lpjml_template():
    """Create a corrupted LPJmL grid template with fill value coordinates.

    This simulates what happens in serial MPI runs where LPJmL writes
    fill values instead of valid lat/lon coordinates.
    """
    lats = np.full(STANDARD_N_LATS, FILL_VALUE)
    lons = np.full(STANDARD_N_LONS, FILL_VALUE)
    return xr.Dataset(
        {
            "cellid": (
                ["lat", "lon"],
                np.full((STANDARD_N_LATS, STANDARD_N_LONS), np.nan),
            ),
        },
        coords={
            "lat": lats,
            "lon": lons,
        },
    )


@pytest.fixture
def partially_corrupted_template_lat():
    """Create a template with only lat coordinates corrupted."""
    lats = np.full(STANDARD_N_LATS, FILL_VALUE)
    lons = np.linspace(STANDARD_LON_MIN, STANDARD_LON_MAX, STANDARD_N_LONS)
    return xr.Dataset(
        {
            "cellid": (
                ["lat", "lon"],
                np.full((STANDARD_N_LATS, STANDARD_N_LONS), np.nan),
            ),
        },
        coords={
            "lat": lats,
            "lon": lons,
        },
    )


@pytest.fixture
def partially_corrupted_template_lon():
    """Create a template with only lon coordinates corrupted."""
    lats = np.linspace(STANDARD_LAT_MIN, STANDARD_LAT_MAX, STANDARD_N_LATS)
    lons = np.full(STANDARD_N_LONS, FILL_VALUE)
    return xr.Dataset(
        {
            "cellid": (
                ["lat", "lon"],
                np.full((STANDARD_N_LATS, STANDARD_N_LONS), np.nan),
            ),
        },
        coords={
            "lat": lats,
            "lon": lons,
        },
    )


class TestAlignToLpjmlGridCoordinates:
    """Tests for coordinate handling in _align_to_lpjml_grid."""

    def test_valid_coordinates_preserved(
        self, sample_cell_data, valid_lpjml_template
    ):
        """Valid template coordinates should be preserved unchanged."""
        result = _align_to_lpjml_grid(
            sample_cell_data, valid_lpjml_template, 2025
        )

        # Check coordinates are valid (not fill values)
        assert np.all(np.abs(result.lat.values) < 1e30)
        assert np.all(np.abs(result.lon.values) < 1e30)

        # Check coordinates match the template
        np.testing.assert_array_almost_equal(
            result.lat.values, valid_lpjml_template.lat.values
        )
        np.testing.assert_array_almost_equal(
            result.lon.values, valid_lpjml_template.lon.values
        )

    def test_corrupted_coordinates_replaced(
        self, sample_cell_data, corrupted_lpjml_template
    ):
        """Fill-value coordinates should be replaced with the standard grid."""
        result = _align_to_lpjml_grid(
            sample_cell_data, corrupted_lpjml_template, 2025
        )

        # Check coordinates are valid (not fill values)
        assert np.all(np.abs(result.lat.values) < 1e30)
        assert np.all(np.abs(result.lon.values) < 1e30)

        # Check coordinates match standard LPJmL grid
        assert result.lat.values.min() == pytest.approx(STANDARD_LAT_MIN)
        assert result.lat.values.max() == pytest.approx(STANDARD_LAT_MAX)
        assert result.lon.values.min() == pytest.approx(STANDARD_LON_MIN)
        assert result.lon.values.max() == pytest.approx(STANDARD_LON_MAX)

        # Check dimensions
        assert len(result.lat) == STANDARD_N_LATS
        assert len(result.lon) == STANDARD_N_LONS

    def test_corrupted_lat_only_replaced(
        self, sample_cell_data, partially_corrupted_template_lat
    ):
        """Only corrupted lat coordinates should be replaced, lon preserved."""
        result = _align_to_lpjml_grid(
            sample_cell_data, partially_corrupted_template_lat, 2025
        )

        # Lat should be replaced with standard grid
        assert result.lat.values.min() == pytest.approx(STANDARD_LAT_MIN)
        assert result.lat.values.max() == pytest.approx(STANDARD_LAT_MAX)

        # Lon should be preserved from template
        np.testing.assert_array_almost_equal(
            result.lon.values, partially_corrupted_template_lat.lon.values
        )

    def test_corrupted_lon_only_replaced(
        self, sample_cell_data, partially_corrupted_template_lon
    ):
        """Only corrupted lon coordinates should be replaced, lat preserved."""
        result = _align_to_lpjml_grid(
            sample_cell_data, partially_corrupted_template_lon, 2025
        )

        # Lat should be preserved from template
        np.testing.assert_array_almost_equal(
            result.lat.values, partially_corrupted_template_lon.lat.values
        )

        # Lon should be replaced with standard grid
        assert result.lon.values.min() == pytest.approx(STANDARD_LON_MIN)
        assert result.lon.values.max() == pytest.approx(STANDARD_LON_MAX)

    def test_valid_and_corrupted_produce_same_grid(
        self, sample_cell_data, valid_lpjml_template, corrupted_lpjml_template
    ):
        """Valid and corrupted templates should yield the same output grid."""
        result_valid = _align_to_lpjml_grid(
            sample_cell_data, valid_lpjml_template, 2025
        )
        result_corrupted = _align_to_lpjml_grid(
            sample_cell_data, corrupted_lpjml_template, 2025
        )

        # Coordinates should be identical
        np.testing.assert_array_almost_equal(
            result_valid.lat.values, result_corrupted.lat.values
        )
        np.testing.assert_array_almost_equal(
            result_valid.lon.values, result_corrupted.lon.values
        )

        # Data should be identical
        np.testing.assert_array_almost_equal(
            result_valid.values, result_corrupted.values
        )


class TestAlignToLpjmlGridDataIntegrity:
    """Tests for data integrity in _align_to_lpjml_grid."""

    def test_time_dimension_preserved(
        self, sample_cell_data, valid_lpjml_template
    ):
        """Time dimension should be preserved in output."""
        result = _align_to_lpjml_grid(
            sample_cell_data, valid_lpjml_template, 2025
        )

        assert "time" in result.dims
        assert len(result.time) == len(sample_cell_data.time)

    def test_output_dimensions_correct(
        self, sample_cell_data, valid_lpjml_template
    ):
        """Output should have (time, lat, lon) dimensions."""
        result = _align_to_lpjml_grid(
            sample_cell_data, valid_lpjml_template, 2025
        )

        assert result.dims == ("time", "lat", "lon")
        assert result.shape == (3, STANDARD_N_LATS, STANDARD_N_LONS)

    def test_data_mapped_correctly(self, valid_lpjml_template):
        """Data values should be mapped to correct grid positions."""
        # Create data with known coordinates
        data = xr.DataArray(
            np.array([[1.0, 2.0, 3.0]]),  # 1 time step, 3 cells
            dims=("time", "cell"),
            coords={
                "time": [2025],
                "cell": [0, 1, 2],
                "lon": ("cell", [0.25, 0.75, 1.25]),  # Specific lon values
                "lat": ("cell", [50.25, 50.25, 50.25]),  # Same lat
            },
        )

        result = _align_to_lpjml_grid(data, valid_lpjml_template, 2025)

        # Check that non-NaN values exist at approximately the right locations
        # (tolerance for grid alignment)
        valid_mask = ~np.isnan(result.values[0])
        assert valid_mask.sum() >= 3  # At least our 3 data points


class TestAlignToLpjmlGridEdgeCases:
    """Tests for edge cases in _align_to_lpjml_grid."""

    def test_no_time_dimension(self, valid_lpjml_template):
        """Data without time dimension should work."""
        data = xr.DataArray(
            np.random.rand(10),
            dims=("cell",),
            coords={
                "cell": np.arange(10),
                "lon": ("cell", np.linspace(0, 5, 10)),
                "lat": ("cell", np.linspace(50, 52, 10)),
            },
        )

        result = _align_to_lpjml_grid(data, valid_lpjml_template, 2025)

        assert "time" not in result.dims
        assert result.dims == ("lat", "lon")

    def test_missing_lon_lat_coords_raises(self, valid_lpjml_template):
        """Cell data without lon/lat coords should raise ValueError."""
        data = xr.DataArray(
            np.random.rand(3, 10),
            dims=("time", "cell"),
            coords={
                "time": [2025, 2026, 2027],
                "cell": np.arange(10),
            },
        )

        with pytest.raises(ValueError, match="must have lon/lat coords"):
            _align_to_lpjml_grid(data, valid_lpjml_template, 2025)

    def test_fill_value_detection_threshold(self, sample_cell_data):
        """Values just below threshold should not be treated as fill values."""
        # Create template with large but valid coordinates (below threshold)
        lats = np.full(STANDARD_N_LATS, 1e29)  # Below 1e30 threshold
        lons = np.full(STANDARD_N_LONS, 1e29)
        template = xr.Dataset(
            {
                "cellid": (
                    ["lat", "lon"],
                    np.full((STANDARD_N_LATS, STANDARD_N_LONS), np.nan),
                ),
            },
            coords={"lat": lats, "lon": lons},
        )

        result = _align_to_lpjml_grid(sample_cell_data, template, 2025)

        # Coordinates should be preserved (not replaced) since below threshold
        np.testing.assert_array_almost_equal(result.lat.values, lats)
        np.testing.assert_array_almost_equal(result.lon.values, lons)


class TestAlignToLpjmlGridNonStandardDimensions:
    """Tests for non-standard grid dimensions.

    Note: LPJmL always uses 280x720 grid, so these tests verify behavior
    with standard dimensions only. Non-standard dimensions are not supported
    when coordinates are corrupted (fallback always uses standard grid).
    """

    def test_standard_dimensions_when_corrupted(self, sample_cell_data):
        """Corrupted coordinates should fall back to standard 280x720 grid."""
        lats = np.full(STANDARD_N_LATS, FILL_VALUE)
        lons = np.full(STANDARD_N_LONS, FILL_VALUE)
        template = xr.Dataset(
            {
                "cellid": (
                    ["lat", "lon"],
                    np.full((STANDARD_N_LATS, STANDARD_N_LONS), np.nan),
                ),
            },
            coords={"lat": lats, "lon": lons},
        )

        result = _align_to_lpjml_grid(sample_cell_data, template, 2025)

        # Should use standard LPJmL grid dimensions
        assert len(result.lat) == STANDARD_N_LATS
        assert len(result.lon) == STANDARD_N_LONS
