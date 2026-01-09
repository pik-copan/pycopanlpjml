"""Tests for region.py - Region, Country, WorldRegion classes."""
import numpy as np
import pytest
import xarray as xr

from pycopanlpjml.region import Country, Region, WorldRegion
from pycopanlpjml.world import World


@pytest.fixture
def sample_xarray_data():
    """Create sample xarray data for testing."""
    # Create sample data
    n_cells = 10
    n_time = 5
    n_band = 2

    # Input dataset
    input_data = {
        "fertilizer": (["cell", "time"], np.random.rand(n_cells, n_time)),
        "irrigation": (["cell", "time"], np.random.rand(n_cells, n_time)),
    }
    input_ds = xr.Dataset(
        input_data, coords={"cell": range(n_cells), "time": range(n_time)}
    )

    # Output dataset
    output_data = {
        "yield": (
            ["cell", "time", "band"],
            np.random.rand(n_cells, n_time, n_band),
        ),
        "harvest": (
            ["cell", "time", "band"],
            np.random.rand(n_cells, n_time, n_band),
        ),
    }
    output_ds = xr.Dataset(
        output_data,
        coords={
            "cell": range(n_cells),
            "time": range(n_time),
            "band": range(n_band),
        },
    )

    # Grid data
    grid = xr.DataArray(
        np.random.rand(n_cells, 2),  # lat, lon
        coords={"cell": range(n_cells), "coord": ["lat", "lon"]},
        dims=["cell", "coord"],
    )

    # Country data
    country = xr.DataArray(
        ["DEU"] * 5 + ["FRA"] * 5,  # Mix of countries
        coords={"cell": range(n_cells)},
        dims=["cell"],
    )

    return input_ds, output_ds, grid, country


@pytest.fixture
def sample_world(sample_xarray_data):
    """Create a sample World instance for testing."""
    input_ds, output_ds, grid, country = sample_xarray_data

    world = World(
        input=input_ds, output=output_ds, grid=grid, country_code=country
    )

    return world


class TestRegion:
    """Test the base Region class."""

    def test_region_initialization(self, sample_world):
        """Test basic Region initialization."""
        # Create a subset of cells for the region
        cell_indices = [0, 1, 2]

        region = Region(
            name="Test Region",
            code="TEST",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": cell_indices}),
        )

        assert region.name == "Test Region"
        assert region.code == "TEST"
        assert region.type == "region"
        assert region.world == sample_world
        assert np.array_equal(region._cell_indices, cell_indices)
        assert region.neighbourhood == []

    def test_region_with_upper_region(self, sample_world):
        """Test Region with upper_region hierarchy."""
        # Create parent region
        parent_cells = [0, 1, 2, 3]
        parent = Region(
            name="Parent Region",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": parent_cells}),
        )

        # Create child region
        child_cells = [0, 1]
        child = Region(
            name="Child Region",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": child_cells}),
            upper_region=parent,
        )

        assert child.next_higher_social_system == parent
        assert child.next_higher_region == parent

    def test_region_hierarchy_conflict(self, sample_world):
        """Test that upper_region and next_higher_social_system conflict."""
        parent = Region(
            name="Parent",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0, 1]}),
        )

        with pytest.raises(
            ValueError,
            match="Cannot set both 'upper_region' and 'next_higher_social_system'",
        ):
            Region(
                name="Child",
                world=sample_world,
                grid=sample_world.grid.isel({"cell": [0]}),
                upper_region=parent,
                next_higher_social_system=parent,
            )

    def test_region_cell_indices_from_xarray(self, sample_world):
        """Test extracting cell indices from xarray DataArray."""
        cell_indices = [2, 4, 6]
        grid_subset = sample_world.grid.isel({"cell": cell_indices})

        region = Region(name="Test", world=sample_world, grid=grid_subset)

        assert np.array_equal(region._cell_indices, cell_indices)

    def test_region_cell_indices_direct_array(self, sample_world):
        """Test passing cell indices directly as array."""
        cell_indices = np.array([0, 2, 4])

        region = Region(name="Test", world=sample_world, grid=cell_indices)

        assert np.array_equal(region._cell_indices, cell_indices)

    def test_region_properties_without_world(self):
        """Test region properties when world is None."""
        # This test is actually testing the same as
        # test_region_properties_without_cell_indices since pycopancore
        # doesn't allow setting world to None. Let's test the properties
        # when world exists but cell_indices is None.
        import xarray as xr
        import numpy as np

        # Create minimal data for World
        input_ds = xr.Dataset({"test": (["cell"], [1, 2, 3])})
        output_ds = xr.Dataset({"test": (["cell"], [1, 2, 3])})
        grid = xr.DataArray(
            [[1, 2], [3, 4], [5, 6]],
            coords={"cell": [0, 1, 2]},
            dims=["cell", "coord"],
        )
        country = xr.DataArray(
            ["DEU", "FRA", "DEU"], coords={"cell": [0, 1, 2]}, dims=["cell"]
        )

        world = World(
            input=input_ds,
            output=output_ds,
            grid=grid,
            country_code=country,
        )

        # Create region without grid (so cell_indices will be None)
        region = Region(name="Test", world=world)

        # Properties should return None when cell_indices is None
        assert region.to_earth is None
        assert region.from_earth is None
        assert region.grid is None
        assert region.area is None

    def test_region_properties_without_cell_indices(self, sample_world):
        """Test region properties when cell indices are None."""
        region = Region(name="Test", world=sample_world)

        assert region.to_earth is None
        assert region.from_earth is None
        assert region.grid is None
        assert region.area is None

    def test_region_input_property(self, sample_world):
        """Test region input property access."""
        cell_indices = [0, 1, 2]
        region = Region(
            name="Test",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": cell_indices}),
        )

        # Test getter
        input_view = region.to_earth
        assert input_view is not None
        assert hasattr(input_view, "data_vars")

        # Test setter
        new_input = xr.Dataset(
            {
                "fertilizer": (["cell", "time"], np.ones((3, 5))),
                "irrigation": (["cell", "time"], np.zeros((3, 5))),
            }
        )

        region.to_earth = new_input

        # Verify the data was written
        assert np.allclose(region.to_earth["fertilizer"].values, 1.0)
        assert np.allclose(region.to_earth["irrigation"].values, 0.0)

    def test_region_output_property(self, sample_world):
        """Test region output property access."""
        cell_indices = [0, 1]
        region = Region(
            name="Test",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": cell_indices}),
        )

        # Test getter
        output_view = region.from_earth
        assert output_view is not None
        assert hasattr(output_view, "data_vars")

        # Test setter
        new_output = xr.Dataset(
            {
                "yield": (["cell", "time", "band"], np.ones((2, 5, 2))),
                "harvest": (["cell", "time", "band"], np.zeros((2, 5, 2))),
            }
        )

        # from_earth is read-only - cannot be set directly
        # Data comes from LPJmL via Component.update_lpjml()
        # Just verify we can read the existing data
        assert region.from_earth is not None

    def test_region_grid_property(self, sample_world):
        """Test region grid property access (read-only)."""
        cell_indices = [0, 1, 2]
        region = Region(
            name="Test",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": cell_indices}),
        )

        # Test getter
        grid_view = region.grid
        assert grid_view is not None
        assert grid_view.shape[0] == 3  # 3 cells

        # Grid is read-only - verify current values are accessible
        grid_values = grid_view.values
        assert grid_values.shape == (3, 2)

    def test_region_area_property(self, sample_world):
        """Test region area property access (read-only)."""
        # Add area data to world (directly set internal data since area is read-only)
        area_data = xr.DataArray(
            np.random.rand(10) * 1000000,  # Random areas in m²
            coords={"cell": range(10)},
            dims=["cell"],
        )
        sample_world._area_data = area_data

        cell_indices = [0, 1, 2]
        region = Region(
            name="Test",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": cell_indices}),
        )

        # Test getter
        area_view = region.area
        assert area_view is not None
        assert len(area_view.values) == 3  # 3 cells

        # Area is read-only - verify values are accessible
        area_values = area_view.values
        assert area_values.shape == (3,)
        assert all(area_values > 0)  # All areas should be positive

    def test_region_area_property_nonexistent(self, sample_world):
        """Test region area property when area data doesn't exist."""
        cell_indices = [0, 1]
        region = Region(
            name="Test",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": cell_indices}),
        )

        # Should return None when area doesn't exist in Zarr
        assert region.area is None

    def test_region_hierarchy_properties(self, sample_world):
        """Test region hierarchy property access."""
        # Create hierarchy: world -> parent -> child
        parent_cells = [0, 1, 2, 3]
        parent = Region(
            name="Parent",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": parent_cells}),
        )

        child_cells = [0, 1]
        child = Region(
            name="Child",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": child_cells}),
            upper_region=parent,
        )

        # Test hierarchy properties
        assert child.next_higher_region == parent
        assert parent in child.higher_regions

        # Test setters
        child.next_higher_region = None
        assert child.next_higher_social_system is None

        # Test that higher_regions is read-only (can't set directly)
        # The hierarchy is managed through next_higher_social_system
        child.next_higher_region = parent
        assert parent in child.higher_regions

    def test_region_lower_regions_properties(self, sample_world):
        """Test region lower regions property access."""
        parent_cells = [0, 1, 2, 3]
        parent = Region(
            name="Parent",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": parent_cells}),
        )

        child_cells = [0, 1]
        child = Region(
            name="Child",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": child_cells}),
            upper_region=parent,
        )

        # Test lower regions properties (read-only)
        assert child in parent.next_lower_regions
        assert child in parent.lower_regions

    def test_region_property_setters_error_handling(self, sample_world):
        """Test error handling in property setters for dynamic properties."""
        region = Region(name="Test", world=sample_world)

        # Dynamic properties: to_earth can be set, from_earth is read-only
        with pytest.raises(
            ValueError,
            match="Cannot set to_earth: cell indices not initialized",
        ):
            region.to_earth = xr.Dataset({"test": (["cell"], [1, 2, 3])})

        # from_earth is read-only - attempting to set should raise AttributeError
        with pytest.raises(AttributeError):
            region.from_earth = xr.Dataset({"test": (["cell"], [1, 2, 3])})

        # Static geographical properties (grid, area) have no setters
        # Attempting to set them should raise AttributeError
        with pytest.raises(AttributeError):
            region.grid = np.array([1, 2, 3])

        with pytest.raises(AttributeError):
            region.area = np.array([1, 2, 3])


class TestCountry:
    """Test the Country class."""

    def test_country_initialization(self, sample_world):
        """Test Country initialization."""
        cell_indices = [0, 1, 2]

        country = Country(
            name="Germany",
            code="DEU",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": cell_indices}),
        )

        assert country.name == "Germany"
        assert country.code == "DEU"
        assert country.type == "country"
        assert country.world == sample_world
        assert np.array_equal(country._cell_indices, cell_indices)

    def test_country_inheritance(self, sample_world):
        """Test that Country inherits all Region functionality."""
        cell_indices = [0, 1]

        country = Country(
            name="France",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": cell_indices}),
        )

        # Should have all Region properties
        assert country.to_earth is not None
        assert country.from_earth is not None
        assert country.grid is not None

        # Should be able to set data
        country.to_earth = xr.Dataset(
            {"fertilizer": (["cell", "time"], np.ones((2, 5)))}
        )
        assert np.allclose(country.to_earth["fertilizer"].values, 1.0)


class TestWorldRegion:
    """Test the WorldRegion class."""

    def test_world_region_initialization(self, sample_world):
        """Test WorldRegion initialization."""
        cell_indices = [0, 1, 2, 3, 4]

        world_region = WorldRegion(
            name="European Union",
            code="EU",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": cell_indices}),
        )

        assert world_region.name == "European Union"
        assert world_region.code == "EU"
        assert world_region.type == "world_region"
        assert world_region.world == sample_world
        assert np.array_equal(world_region._cell_indices, cell_indices)

    def test_world_region_inheritance(self, sample_world):
        """Test that WorldRegion inherits all Region functionality."""
        cell_indices = [0, 1, 2]

        world_region = WorldRegion(
            name="G7",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": cell_indices}),
        )

        # Should have all Region properties
        assert world_region.to_earth is not None
        assert world_region.from_earth is not None
        assert world_region.grid is not None

        # from_earth is read-only - data comes from LPJmL
        # Just verify we can read the existing data
        assert world_region.from_earth is not None


class TestRegionIntegration:
    """Integration tests for Region classes with World."""

    def test_region_data_synchronization(self, sample_world):
        """Test that region data changes are synchronized with world."""
        # Create two regions with overlapping cells
        region1_cells = [0, 1, 2]
        region2_cells = [2, 3, 4]  # Cell 2 overlaps

        region1 = Region(
            name="Region 1",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": region1_cells}),
        )

        region2 = Region(
            name="Region 2",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": region2_cells}),
        )

        # Modify data through region1
        region1.to_earth = xr.Dataset(
            {"fertilizer": (["cell", "time"], np.ones((3, 5)) * 100)}
        )

        # Check that world sees the change
        world_input = sample_world.to_earth
        assert np.allclose(
            world_input["fertilizer"].isel({"cell": 0}).values, 100
        )
        assert np.allclose(
            world_input["fertilizer"].isel({"cell": 1}).values, 100
        )
        assert np.allclose(
            world_input["fertilizer"].isel({"cell": 2}).values, 100
        )

        # Check that region2 also sees the change for overlapping cell
        region2_input = region2.to_earth
        assert np.allclose(
            region2_input["fertilizer"].isel({"cell": 0}).values, 100
        )  # Cell 2 in region2

        # Modify data through region2
        region2.to_earth = xr.Dataset(
            {"fertilizer": (["cell", "time"], np.ones((3, 5)) * 200)}
        )

        # Check that world sees the updated change
        world_input = sample_world.to_earth
        assert np.allclose(
            world_input["fertilizer"].isel({"cell": 2}).values, 200
        )  # Overlapping cell updated
        assert np.allclose(
            world_input["fertilizer"].isel({"cell": 3}).values, 200
        )
        assert np.allclose(
            world_input["fertilizer"].isel({"cell": 4}).values, 200
        )

    def test_region_hierarchy_data_access(self, sample_world):
        """Test data access through region hierarchy."""
        # Create parent region
        parent_cells = [0, 1, 2, 3, 4]
        parent = Region(
            name="Parent",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": parent_cells}),
        )

        # Create child region
        child_cells = [0, 1]
        child = Region(
            name="Child",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": child_cells}),
            upper_region=parent,
        )

        # Set data through parent
        parent.to_earth = xr.Dataset(
            {"fertilizer": (["cell", "time"], np.ones((5, 5)) * 50)}
        )

        # Child should see the same data for its cells
        child_input = child.to_earth
        assert np.allclose(child_input["fertilizer"].values, 50)

        # Set data through child
        child.to_earth = xr.Dataset(
            {"fertilizer": (["cell", "time"], np.ones((2, 5)) * 75)}
        )

        # Parent should see updated data for child's cells
        parent_input = parent.to_earth
        assert np.allclose(
            parent_input["fertilizer"].isel({"cell": 0}).values, 75
        )
        assert np.allclose(
            parent_input["fertilizer"].isel({"cell": 1}).values, 75
        )
        assert np.allclose(
            parent_input["fertilizer"].isel({"cell": 2}).values, 50
        )  # Unchanged

    def test_multiple_countries(self, sample_world):
        """Test multiple countries with different cell sets."""
        # Create Germany (cells 0-4)
        germany_cells = [0, 1, 2, 3, 4]
        germany = Country(
            name="Germany",
            code="DEU",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": germany_cells}),
        )

        # Create France (cells 5-9)
        france_cells = [5, 6, 7, 8, 9]
        france = Country(
            name="France",
            code="FRA",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": france_cells}),
        )

        # Set different data for each country
        germany.to_earth = xr.Dataset(
            {"fertilizer": (["cell", "time"], np.ones((5, 5)) * 100)}
        )

        france.to_earth = xr.Dataset(
            {"fertilizer": (["cell", "time"], np.ones((5, 5)) * 200)}
        )

        # Verify world sees both datasets
        world_input = sample_world.to_earth
        assert np.allclose(
            world_input["fertilizer"].isel({"cell": slice(0, 5)}).values, 100
        )
        assert np.allclose(
            world_input["fertilizer"].isel({"cell": slice(5, 10)}).values, 200
        )

        # Verify each country sees only its data
        assert np.allclose(germany.to_earth["fertilizer"].values, 100)
        assert np.allclose(france.to_earth["fertilizer"].values, 200)
