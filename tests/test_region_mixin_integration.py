"""Integration tests for region.py and mixin.py working together."""

import pytest
import numpy as np
import xarray as xr
import sys

# Add the package to path for imports
sys.path.insert(0, "/p/projects/copan/users/jannesbr/repos/pycopanlpjml")

from pycopanlpjml.region import Region, Country, WorldRegion
from pycopanlpjml.world import World
from pycopanlpjml.mixin import AliasMixin


@pytest.fixture
def sample_xarray_data():
    """Create sample xarray data for testing."""
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
        np.random.rand(n_cells, 2),
        coords={"cell": range(n_cells), "coord": ["lat", "lon"]},
        dims=["cell", "coord"],
    )

    # Country data
    country = xr.DataArray(
        ["DEU"] * 5 + ["FRA"] * 5,
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


class TestRegionAliasMixinIntegration:
    """Test integration between Region classes and AliasMixin."""

    def test_region_inherits_alias_mixin(self):
        """Test that Region classes inherit AliasMixin functionality."""
        # Check inheritance
        assert issubclass(Region, AliasMixin)
        assert issubclass(Country, AliasMixin)
        assert issubclass(WorldRegion, AliasMixin)

        # Check that Region has alias map
        assert hasattr(Region, "_alias_map")
        assert "social_system" in Region._alias_map
        assert "social_systems" in Region._alias_map

    def test_region_alias_properties_generated(self, sample_world):
        """Test that Region instances have alias properties."""
        region = Region(
            name="Test Region",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0, 1, 2]}),
        )

        # Should have alias properties
        assert hasattr(region, "region")
        assert hasattr(region, "country")
        assert hasattr(region, "worldregion")
        assert hasattr(region, "regions")
        assert hasattr(region, "countries")

    def test_region_alias_kwargs_work(self, sample_world):
        """Test that Region accepts alias kwargs."""
        # Test with region alias
        region = Region(
            name="Test Region",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0, 1, 2]}),
            region=sample_world,  # Alias for social_system
        )

        assert region.social_system == sample_world
        assert region.region == sample_world

        # Test with country alias
        country = Country(
            name="Test Country",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0, 1]}),
            country=sample_world,  # Alias for social_system
        )

        assert country.social_system == sample_world
        assert country.country == sample_world

    def test_region_alias_properties_sync(self, sample_world):
        """Test that alias properties sync with underlying attributes."""
        region = Region(
            name="Test Region",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0, 1, 2]}),
        )

        # Set via alias
        region.region = sample_world

        # Should sync to underlying attribute
        assert region.social_system == sample_world

        # Set via underlying attribute
        region.social_system = None

        # Should sync to alias
        assert region.region is None

    def test_country_alias_functionality(self, sample_world):
        """Test Country-specific alias functionality."""
        country = Country(
            name="Germany",
            code="DEU",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0, 1, 2]}),
        )

        # Test alias properties work
        assert hasattr(country, "country")
        assert hasattr(country, "countries")

        # Test setting via alias
        country.country = sample_world
        assert country.social_system == sample_world

        # Test getting via alias
        assert country.country == sample_world

    def test_world_region_alias_functionality(self, sample_world):
        """Test WorldRegion-specific alias functionality."""
        world_region = WorldRegion(
            name="European Union",
            code="EU",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0, 1, 2, 3, 4]}),
        )

        # Test alias properties work
        assert hasattr(world_region, "worldregion")
        assert hasattr(world_region, "regions")

        # Test setting via alias
        world_region.worldregion = sample_world
        assert world_region.social_system == sample_world

        # Test getting via alias
        assert world_region.worldregion == sample_world

    def test_region_hierarchy_with_aliases(self, sample_world):
        """Test that mixin aliases work independently of pycopancore hierarchy."""
        # Create parent and child regions
        parent = Region(
            name="Parent Region",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0, 1, 2, 3, 4]}),
        )

        child = Region(
            name="Child Region",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0, 1]}),
            upper_region=parent,  # Sets next_higher_social_system
        )

        # pycopancore hierarchy should be set
        assert child.next_higher_social_system == parent
        assert parent in child.higher_regions

        # Mixin aliases map to social_system (separate from next_higher_social_system)
        # Set social_system directly to test the alias
        child.social_system = parent
        assert child.region == parent  # Alias works
        assert child.country == parent  # Alias works
        assert child.worldregion == parent  # Alias works

    def test_region_entity_aliases_generation(self, sample_world):
        """Test that entity-specific aliases exist for regions."""
        # Note: Detailed filtering tests are in test_mixin.py
        # This just verifies the integration works

        # Create multiple regions
        regions = [
            Region(
                name="R1",
                world=sample_world,
                grid=sample_world.grid.isel({"cell": [0, 1]}),
            ),
            Region(
                name="R2",
                world=sample_world,
                grid=sample_world.grid.isel({"cell": [2, 3]}),
            ),
            Country(
                name="C1",
                world=sample_world,
                grid=sample_world.grid.isel({"cell": [4, 5]}),
            ),
            Country(
                name="C2",
                world=sample_world,
                grid=sample_world.grid.isel({"cell": [6, 7]}),
            ),
        ]

        # Add to a container region
        container = Region(
            name="Container",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0, 1, 2, 3, 4, 5, 6, 7]}),
        )

        container.social_systems = regions
        container._add_entity_aliases("social_systems")

        # Should have filtered aliases created
        assert hasattr(container, "regions")
        assert hasattr(container, "countries")

        # The aliases should return lists (filtering logic tested in test_mixin.py)
        assert isinstance(container.regions, list)
        assert isinstance(container.countries, list)

    def test_region_alias_conflicts_resolution(self, sample_world):
        """Test that alias conflicts are resolved properly."""

        # Create region with pre-existing attribute
        class CustomRegion(Region):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # Pre-existing attribute that conflicts with alias
                self.region = "pre-existing-value"

        region = CustomRegion(
            name="Test",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0, 1]}),
        )

        # Should not overwrite pre-existing attribute
        assert region.region == "pre-existing-value"

        # But should still have other aliases
        assert hasattr(region, "country")
        assert hasattr(region, "worldregion")

    def test_region_alias_property_lambda_capture(self, sample_world):
        """Test that alias property lambdas capture variables correctly."""
        region = Region(
            name="Test Region",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0, 1, 2]}),
        )

        # Test that all aliases point to the same underlying attribute
        test_value = sample_world

        region.region = test_value
        assert region.country == test_value
        assert region.worldregion == test_value

        region.country = None
        assert region.region is None
        assert region.worldregion is None

    def test_region_alias_with_hierarchy(self, sample_world):
        """Test region aliases work with pycopancore hierarchy."""
        # Create hierarchy using upper_region (proper way)
        world_region = WorldRegion(
            name="EU",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0, 1, 2, 3, 4]}),
        )

        country1 = Country(
            name="Germany",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0, 1]}),
            upper_region=world_region,
        )

        country2 = Country(
            name="France",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [2, 3]}),
            upper_region=world_region,
        )

        # Test pycopancore hierarchy works
        assert country1.next_higher_social_system == world_region
        assert country2.next_higher_social_system == world_region
        assert world_region in country1.higher_regions
        assert world_region in country2.higher_regions
        assert country1 in world_region.lower_regions
        assert country2 in world_region.lower_regions

        # Test that mixin aliases can be used to set/get social_system
        country1.social_system = world_region
        assert country1.region == world_region  # Alias works

    def test_region_alias_data_access(self, sample_world):
        """Test that alias properties don't interfere with data access."""
        region = Region(
            name="Test Region",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0, 1, 2]}),
        )

        # Test that data access still works
        assert region.input is not None
        assert region.output is not None
        assert region.grid is not None

        # Test that setting data works
        region.input = xr.Dataset(
            {"fertilizer": (["cell", "time"], np.ones((3, 5)))}
        )

        assert np.allclose(region.input["fertilizer"].values, 1.0)

        # Test that aliases still work
        region.region = sample_world
        assert region.social_system == sample_world

    def test_region_alias_with_multiple_instances(self, sample_world):
        """Test that aliases work correctly with multiple region instances."""
        # Create multiple regions
        region1 = Region(
            name="Region 1",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0]}),
        )

        region2 = Region(
            name="Region 2",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [1]}),
        )

        # Test that each region has the alias properties
        assert hasattr(region1, "region")
        assert hasattr(region2, "region")

        # Test that aliases work independently
        region1.social_system = sample_world
        assert region1.region == sample_world

        # region2 should not be affected (its social_system should not be set)
        # Use the alias to check (it returns None if not set)
        assert region2.region is None

    def test_region_alias_edge_cases(self, sample_world):
        """Test edge cases for region aliases."""
        region = Region(
            name="Test Region",
            world=sample_world,
            grid=sample_world.grid.isel({"cell": [0, 1, 2]}),
        )

        # Test setting None via alias
        region.region = None
        assert region.social_system is None

        # Test setting different types via alias
        region.country = sample_world
        assert region.social_system == sample_world

        region.worldregion = None
        assert region.social_system is None

        # Test that data access still works after alias operations
        assert region.input is not None
        assert region.output is not None
        assert region.grid is not None
