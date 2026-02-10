"""Tests for mixin.py - AliasMixin functionality."""

import sys

from pycopanlpjml.mixin import AliasMixin


class MockEntity:
    """Mock entity class for testing."""

    type = None  # Default type, can be overridden by subclasses

    def __init__(self, name=None, type=None):
        self.name = name
        # Only override type if explicitly provided or if class type is None
        if type is not None:
            self.type = type
        elif self.__class__.type is None:
            self.type = self.__class__.__name__.lower()
        # Otherwise, use the class-level type attribute

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.type}')"  # noqa: E501


class TestEntity(MockEntity):
    """Test entity with specific type."""

    type = "test"


class RegionEntity(MockEntity):
    """Mock region entity."""

    type = "region"


class CountryEntity(MockEntity):
    """Mock country entity."""

    type = "country"


class WorldRegionEntity(MockEntity):
    """Mock world region entity."""

    type = "worldregion"


class TestAliasMixin:
    """Test the AliasMixin class."""

    def test_alias_mixin_initialization(self):
        """Test basic AliasMixin initialization."""

        class TestClass(AliasMixin):
            def __init__(self, **kwargs):
                self.social_system = None
                self.social_systems = []
                super().__init__(**kwargs)

        # Test with no aliases
        obj = TestClass()
        assert obj.social_system is None
        assert obj.social_systems == []

    def test_alias_kwargs_redirection(self):
        """Test that alias kwargs are redirected to real attributes."""

        class TestClass(AliasMixin):
            def __init__(self, **kwargs):
                self.social_system = None
                self.social_systems = []
                super().__init__(**kwargs)

        # Test region alias
        region = RegionEntity("Test Region")
        obj = TestClass(region=region)
        assert obj.social_system == region

        # Test country alias
        country = CountryEntity("Test Country")
        obj = TestClass(country=country)
        assert obj.social_system == country

        # Test worldregion alias
        world_region = WorldRegionEntity("Test World Region")
        obj = TestClass(worldregion=world_region)
        assert obj.social_system == world_region

    def test_alias_kwargs_plural_redirection(self):
        """Test that plural alias kwargs are redirected to real attributes."""

        class TestClass(AliasMixin):
            def __init__(self, **kwargs):
                self.social_system = None
                self.social_systems = []
                super().__init__(**kwargs)

        # Test regions alias
        regions = [RegionEntity("Region 1"), RegionEntity("Region 2")]
        obj = TestClass(regions=regions)
        assert obj.social_systems == regions

        # Test countries alias
        countries = [CountryEntity("Country 1"), CountryEntity("Country 2")]
        obj = TestClass(countries=countries)
        assert obj.social_systems == countries

        # Test worldregions alias (plural)
        world_regions = [WorldRegionEntity("WR 1"), WorldRegionEntity("WR 2")]
        obj = TestClass(worldregions=world_regions)
        assert obj.social_systems == world_regions

    def test_alias_property_generation(self):
        """Test that alias properties are generated dynamically."""

        class TestClass(AliasMixin):
            def __init__(self, **kwargs):
                self.social_system = None
                self.social_systems = []
                super().__init__(**kwargs)

        obj = TestClass()

        # Test singular aliases
        assert hasattr(obj, "region")
        assert hasattr(obj, "country")
        assert hasattr(obj, "worldregion")

        # Test plural aliases
        assert hasattr(obj, "regions")
        assert hasattr(obj, "countries")
        assert hasattr(obj, "worldregion")  # Same as singular for this alias

    def test_alias_property_getter_setter(self):
        """Test that alias properties work as getters and setters."""

        class TestClass(AliasMixin):
            def __init__(self, **kwargs):
                self.social_system = None
                self.social_systems = []
                super().__init__(**kwargs)

        obj = TestClass()

        # Test singular alias getter/setter
        region = RegionEntity("Test Region")
        obj.region = region
        assert obj.social_system == region
        assert obj.region == region

        # Test plural alias getter/setter
        countries = [CountryEntity("C1"), CountryEntity("C2")]
        obj.countries = countries
        assert obj.social_systems == countries
        assert obj.countries == countries

    def test_alias_property_with_existing_attribute(self):
        """Test that instance attributes shadow class-level alias
        properties."""

        class TestClass(AliasMixin):
            def __init__(self, **kwargs):
                self.social_system = None
                self.social_systems = []
                super().__init__(**kwargs)
                # Set instance attribute after mixin initialization
                self.region = "pre-existing"

        obj = TestClass()

        # Instance attribute should shadow the class-level property
        assert obj.region == "pre-existing"
        # The class does have the property (created by
        # _add_all_alias_properties)
        assert hasattr(obj.__class__, "region")
        # But the instance attribute shadows it
        assert hasattr(obj, "region")
        assert obj.region == "pre-existing"

    def test_entity_aliases_singular(self):
        """Test _add_entity_aliases for singular entities."""

        class TestClass(AliasMixin):
            def __init__(self, **kwargs):
                self.social_system = None
                self.social_systems = []
                super().__init__(**kwargs)

        obj = TestClass()

        # Add a region entity
        region = RegionEntity("Test Region")
        obj.social_system = region
        obj._add_entity_aliases("social_system")

        # Should have region alias
        assert hasattr(obj, "region")
        assert obj.region == region

    def test_entity_aliases_plural_list(self):
        """Test _add_entity_aliases for plural list entities."""

        class TestClass(AliasMixin):
            def __init__(self, **kwargs):
                self.social_system = None
                self.social_systems = []
                super().__init__(**kwargs)

        obj = TestClass()

        # Add multiple entities
        regions = [RegionEntity("R1"), RegionEntity("R2")]
        countries = [CountryEntity("C1"), CountryEntity("C2")]
        all_entities = regions + countries

        obj.social_systems = all_entities
        obj._add_entity_aliases("social_systems")

        # Should have regions alias (filtered)
        assert hasattr(obj, "regions")
        assert len(obj.regions) == 2
        assert all(e.type == "region" for e in obj.regions)

        # Should have countries alias (filtered)
        assert hasattr(obj, "countries")
        assert len(obj.countries) == 2
        assert all(e.type == "country" for e in obj.countries)

    def test_entity_aliases_plural_dict(self):
        """Test _add_entity_aliases for plural dict entities."""

        class TestClass(AliasMixin):
            def __init__(self, **kwargs):
                self.social_system = None
                self.social_systems = {}
                super().__init__(**kwargs)

        obj = TestClass()

        # Add entities as dict
        entities_dict = {
            "r1": RegionEntity("R1"),
            "r2": RegionEntity("R2"),
            "c1": CountryEntity("C1"),
            "c2": CountryEntity("C2"),
        }

        obj.social_systems = entities_dict
        obj._add_entity_aliases("social_systems")

        # Should have regions alias (filtered from dict values)
        assert hasattr(obj, "regions")
        assert len(obj.regions) == 2
        assert all(e.type == "region" for e in obj.regions)

        # Should have countries alias (filtered from dict values)
        assert hasattr(obj, "countries")
        assert len(obj.countries) == 2
        assert all(e.type == "country" for e in obj.countries)

    def test_entity_aliases_none_values(self):
        """Test _add_entity_aliases handles None values."""

        class TestClass(AliasMixin):
            def __init__(self, **kwargs):
                self.social_system = None
                self.social_systems = []
                super().__init__(**kwargs)

        obj = TestClass()

        # Add entities with None values
        entities = [RegionEntity("R1"), None, CountryEntity("C1"), None]
        obj.social_systems = entities
        obj._add_entity_aliases("social_systems")

        # Should filter out None values
        assert hasattr(obj, "regions")
        assert len(obj.regions) == 1
        assert obj.regions[0].type == "region"

        assert hasattr(obj, "countries")
        assert len(obj.countries) == 1
        assert obj.countries[0].type == "country"

    def test_entity_aliases_canonical_regions(self):
        """Test that regions alias filters correctly after
        _add_entity_aliases."""

        class TestClass(AliasMixin):
            def __init__(self, **kwargs):
                self.social_system = None
                self.social_systems = []
                super().__init__(**kwargs)

        obj = TestClass()

        # Add mixed entities
        region = RegionEntity("R1")
        country = CountryEntity("C1")
        entities = [region, country]
        obj.social_systems = entities

        # Before calling _add_entity_aliases, regions returns all
        # social_systems
        assert hasattr(obj, "regions")
        assert obj.regions == entities  # Generic alias

        # After calling _add_entity_aliases, regions should filter to only
        # regions
        obj._add_entity_aliases("social_systems")
        assert hasattr(obj, "regions")
        assert obj.regions == [region]  # Filtered to only region entities
        assert hasattr(obj, "countries")
        assert obj.countries == [country]  # Filtered to only country entities

    def test_entity_aliases_no_existing_attribute(self):
        """Test _add_entity_aliases when attribute doesn't exist."""

        class TestClass(AliasMixin):
            def __init__(self, **kwargs):
                self.social_system = None
                self.social_systems = []
                super().__init__(**kwargs)

        obj = TestClass()

        # Should return early if attribute doesn't exist
        obj._add_entity_aliases("nonexistent")
        # No error should be raised

    def test_alias_map_structure(self):
        """Test that _alias_map has expected structure."""
        expected_aliases = {
            "social_system": ["region", "country", "worldregion"],
            "social_systems": ["regions", "countries", "worldregions"],
        }

        assert AliasMixin._alias_map == expected_aliases

    def test_complex_alias_scenario(self):
        """Test complex scenario with multiple aliases and entity types."""

        class TestClass(AliasMixin):
            def __init__(self, **kwargs):
                # Initialize attributes first
                self.social_system = None
                self.social_systems = []
                # Then call super to process aliases
                super().__init__(**kwargs)

        # Create entities
        region = RegionEntity("Main Region")
        countries = [CountryEntity("C1"), CountryEntity("C2")]

        # Initialize with mixed aliases
        obj = TestClass(
            region=region,  # Should go to social_system
            countries=countries,  # Should go to social_systems
        )

        # Verify redirection worked
        assert obj.social_system == region
        assert obj.social_systems == countries

        # Verify properties work
        assert obj.region == region
        assert obj.countries == countries

        # Add entity aliases
        obj._add_entity_aliases("social_system")
        obj._add_entity_aliases("social_systems")

        # Verify entity-specific aliases work
        assert obj.region == region
        assert len(obj.countries) == 2
        assert all(c.type == "country" for c in obj.countries)

    def test_alias_property_lambda_capture(self):
        """Test that lambda functions properly capture variables."""

        class TestClass(AliasMixin):
            def __init__(self, **kwargs):
                self.social_system = None
                self.social_systems = []
                super().__init__(**kwargs)

        obj = TestClass()

        # Test that each alias property captures the right attribute name
        region = RegionEntity("Test")
        obj.social_system = region

        # All aliases should point to the same value
        assert obj.region == region
        assert obj.country == region  # Same social_system
        assert obj.worldregion == region  # Same social_system

        # Change the value
        country = CountryEntity("Test Country")
        obj.social_system = country

        # All aliases should reflect the change
        assert obj.region == country
        assert obj.country == country
        assert obj.worldregion == country

    def test_alias_property_setter_lambda_capture(self):
        """Test that setter lambda functions properly capture variables."""

        class TestClass(AliasMixin):
            def __init__(self, **kwargs):
                self.social_system = None
                self.social_systems = []
                super().__init__(**kwargs)

        obj = TestClass()

        # Test that setters work correctly
        region = RegionEntity("Test Region")
        obj.region = region

        assert obj.social_system == region

        # Test different alias setter
        country = CountryEntity("Test Country")
        obj.country = country

        assert obj.social_system == country
        assert obj.region == country  # Should reflect the change

    def test_entity_aliases_filtering_logic(self):
        """Test the filtering logic in _add_entity_aliases."""

        class TestClass(AliasMixin):
            def __init__(self, **kwargs):
                self.social_system = None
                self.social_systems = []
                super().__init__(**kwargs)

        obj = TestClass()

        # Mix of entity types
        entities = [
            RegionEntity("R1"),
            CountryEntity("C1"),
            RegionEntity("R2"),
            WorldRegionEntity("WR1"),
            CountryEntity("C2"),
            TestEntity("T1"),  # Different type
        ]

        obj.social_systems = entities
        obj._add_entity_aliases("social_systems")

        # Test filtering
        assert len(obj.regions) == 2
        assert all(e.type == "region" for e in obj.regions)

        assert len(obj.countries) == 2
        assert all(e.type == "country" for e in obj.countries)

        # TestEntity should not appear in any filtered list
        assert not any(e.type == "test" for e in obj.regions)
        assert not any(e.type == "test" for e in obj.countries)
