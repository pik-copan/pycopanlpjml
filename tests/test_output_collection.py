"""Unit tests for OutputCollectionMixin output collection functionality."""

import pytest
import numpy as np
import xarray as xr
import tempfile
import os
import shutil

from pycopanlpjml.output import OutputCollectionMixin, OutputDefinitionMixin, Output
from pycopanlpjml.world import World
from pycopanlpjml.model import Model
from pycopancore.data_model.variable import Variable
from pycopancore.data_model.master_data_model.dimensions_and_units import (
    DimensionsAndUnits as DAU,
)


class MockConfig:
    """Mock configuration object for testing."""

    def __init__(self):
        self.sim_path = tempfile.mkdtemp()
        self.sim_name = "test_run"
        self.coupled_config = MockCoupledConfig()


class MockCoupledConfig:
    """Mock coupled configuration."""

    def __init__(self):
        self.output = MockOutputConfig()


class MockOutputConfig:
    """Mock output configuration."""

    def __init__(self):
        self.format = ["csv"]

    def to_dict(self):
        return {
            "world": ["world_var1", "world_var2"],
            "testcountry": ["country_var1"],  # Match class name TestCountry -> testcountry
            "testcell": ["cell_var1"],  # Match class name TestCell -> testcell
            "testfarmer": ["farmer_var1", "farmer_var2"],  # Match class name TestFarmer -> testfarmer
        }


class MockPyCopanLPJMLConfig:
    """Mock pycopanlpjml configuration."""

    def __init__(self):
        self.output = MockOutputSettings()


class MockOutputSettings:
    """Mock output settings."""

    def __init__(self):
        self.enabled = True
        self.use_temp_storage = False
        self.format = ["csv"]


class TestWorld(World, OutputDefinitionMixin):
    """Test World class with output variables."""

    output_variables = Output(
        world_var1=Variable("World Variable 1", "test world var 1"),
        world_var2=Variable("World Variable 2", "test world var 2", unit=DAU.gC_per_m2),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.world_var1 = 100.0
        self.world_var2 = 200.0


class TestCountry(OutputDefinitionMixin):
    """Test Country class with output variables."""

    output_variables = Output(
        country_var1=Variable("Country Variable 1", "test country var"),
    )

    def __init__(self, model, code="TEST", name="Test Country"):
        self._model = model
        self.code = code
        self.name = name
        self.country_var1 = 50.0
        self._cells = set()

    @property
    def model(self):
        return self._model

    @property
    def cells(self):
        return self._cells


class TestCell(OutputDefinitionMixin):
    """Test Cell class with output variables."""

    output_variables = Output(
        cell_var1=Variable("Cell Variable 1", "test cell var"),
    )

    def __init__(self, model, cell_index=0):
        self._model = model
        self.cell_index = cell_index
        self.cell_var1 = 25.0
        self._individuals = set()

    @property
    def model(self):
        return self._model

    @property
    def individuals(self):
        return self._individuals


class TestFarmer(OutputDefinitionMixin):
    """Test Farmer class with output variables."""

    output_variables = Output(
        farmer_var1=Variable("Farmer Variable 1", "test farmer var 1"),
        farmer_var2=Variable("Farmer Variable 2", "test farmer var 2", unit=DAU.gC_per_m2),
    )

    def __init__(self, model, cell, farmer_id=0):
        self._model = model
        self.cell = cell
        self.farmer_id = farmer_id
        self.farmer_var1 = 10.0
        self.farmer_var2 = 20.0
        cell._individuals.add(self)

    @property
    def model(self):
        return self._model


class TestComponent(Model):
    """Test Component with output collection."""

    def __init__(self):
        # Create minimal world
        n_cells = 5
        input_data = xr.Dataset(
            {"test_input": (["cell", "time"], np.zeros((n_cells, 1)))},
            coords={"cell": range(n_cells), "time": [2020]},
        )
        output_data = xr.Dataset(
            {"test_output": (["cell", "time"], np.zeros((n_cells, 1)))},
            coords={"cell": range(n_cells), "time": [2020]},
        )
        grid = xr.DataArray(
            np.random.rand(n_cells, 2),
            coords={"cell": range(n_cells), "coord": ["lat", "lon"]},
        )
        area = xr.DataArray(np.ones(n_cells), coords={"cell": range(n_cells)})

        self.config = MockConfig()
        self.pycopanlpjml_config = MockPyCopanLPJMLConfig()
        self.world = TestWorld(
            model=self,
            input=input_data,
            output=output_data,
            grid=grid,
            area=area,
        )

        # Add countries (need to properly register with world)
        country1 = TestCountry(self, code="C1", name="Country 1")
        country2 = TestCountry(self, code="C2", name="Country 2")
        # Register countries with world (pycopancore pattern)
        if not hasattr(self.world, "_social_systems"):
            self.world._social_systems = set()
        self.world._social_systems.add(country1)
        self.world._social_systems.add(country2)
        # Also set world on countries
        country1.world = self.world
        country2.world = self.world

        # Add cells
        for i in range(n_cells):
            cell = TestCell(self, cell_index=i)
            if i < 3:
                country1._cells.add(cell)
                cell.social_system = country1
            else:
                country2._cells.add(cell)
                cell.social_system = country2
            self.world._cells.add(cell)

        # Add farmers
        for i, cell in enumerate(self.world.cells):
            farmer1 = TestFarmer(self, cell, farmer_id=i * 2)
            farmer2 = TestFarmer(self, cell, farmer_id=i * 2 + 1)


@pytest.fixture
def test_component():
    """Create a test component with sample data."""
    return TestComponent()


@pytest.fixture
def cleanup_temp():
    """Cleanup temporary directories after tests."""
    yield
    # Cleanup handled by component's temp directory


class TestOutputCollectionMixin:
    """Test OutputCollectionMixin functionality."""

    def test_should_collect_outputs(self, test_component):
        """Test _should_collect_outputs returns True when enabled."""
        assert test_component._should_collect_outputs(2020) is True

    def test_should_collect_outputs_disabled(self, test_component):
        """Test _should_collect_outputs returns False when disabled."""
        test_component.pycopanlpjml_config.output.enabled = False
        assert test_component._should_collect_outputs(2020) is False

    def test_collect_world_outputs(self, test_component):
        """Test collection of world-level outputs."""
        ds = test_component._collect_world_outputs(2020)
        assert ds is not None
        assert "world_var1" in ds.data_vars
        assert "world_var2" in ds.data_vars
        assert ds["world_var1"].values[0] == 100.0
        assert ds["world_var2"].values[0] == 200.0
        assert "units" in ds["world_var2"].attrs
        assert ds.sizes["time"] == 1

    def test_collect_country_outputs(self, test_component):
        """Test collection of country-level outputs."""
        ds = test_component._collect_country_outputs(2020)
        assert ds is not None
        assert "country_var1" in ds.data_vars
        assert "country" in ds.coords
        assert "country_name" in ds.coords
        assert ds.sizes["country"] == 2
        assert ds.sizes["time"] == 1
        assert ds["country_var1"].values[0, 0] == 50.0

    def test_collect_cell_outputs(self, test_component):
        """Test collection of cell-level outputs."""
        ds = test_component._collect_cell_outputs(2020)
        assert ds is not None
        assert "cell_var1" in ds.data_vars
        assert ds.sizes["cell"] == 5
        assert ds.sizes["time"] == 1
        assert ds["cell_var1"].values[0, 0] == 25.0

    def test_collect_individual_outputs(self, test_component):
        """Test collection of individual-level outputs."""
        ds = test_component._collect_individual_outputs(2020)
        assert ds is not None
        assert "farmer_var1" in ds.data_vars
        assert "farmer_var2" in ds.data_vars
        assert "individual_id" in ds.coords
        assert "individual_cell" in ds.coords
        assert "individual_class" in ds.coords
        assert ds.sizes["individual_id"] == 10  # 5 cells * 2 farmers
        assert ds.sizes["time"] == 1
        assert ds["farmer_var1"].values[0, 0] == 10.0

    def test_collect_outputs_integration(self, test_component):
        """Test full collect_outputs integration."""
        # Disable Zarr writing for this test (to keep data in memory)
        # Set flag to prevent initialization
        test_component._output_store_initialized = True  # Pretend it's initialized
        # But don't set output_store_path so Zarr write won't happen
        if hasattr(test_component.world, "_output_store_path"):
            test_component.world._output_store_path = None

        # Collect outputs
        test_component.collect_outputs(2020)

        # Check that data was stored (should remain in memory since Zarr write is skipped)
        assert test_component.world._output_data is not None
        ds = test_component.world._output_data

        # Check all entity levels are present
        assert "world_var1" in ds.data_vars
        assert "country_var1" in ds.data_vars
        assert "cell_var1" in ds.data_vars
        assert "farmer_var1" in ds.data_vars

        # Check time coordinate
        assert "time" in ds.coords
        assert ds.coords["time"].values[0] == 2020

    def test_collect_outputs_empty(self, test_component):
        """Test collect_outputs with no output variables defined."""
        # Clear output config
        test_component.config.coupled_config.output.to_dict = lambda: {}
        assert test_component._should_collect_outputs(2020) is False

    def test_collect_outputs_none_world(self):
        """Test collection when world is None."""
        component = TestComponent()
        component.world = None
        assert component._collect_world_outputs(2020) is None
        assert component._collect_country_outputs(2020) is None
        assert component._collect_cell_outputs(2020) is None
        assert component._collect_individual_outputs(2020) is None

    def test_collect_outputs_missing_attributes(self, test_component):
        """Test collection handles missing attributes gracefully."""
        # Remove an attribute
        delattr(test_component.world, "world_var1")
        ds = test_component._collect_world_outputs(2020)
        assert ds is not None
        # Should have NaN for missing attribute
        assert np.isnan(ds["world_var1"].values[0])

    def test_collect_outputs_vectorization(self, test_component):
        """Test that collection uses vectorized operations."""
        # Add more cells and farmers for performance test
        for i in range(5, 100):
            cell = TestCell(test_component, cell_index=i)
            test_component.world._cells.add(cell)
            TestFarmer(test_component, cell, farmer_id=i * 2)
            TestFarmer(test_component, cell, farmer_id=i * 2 + 1)

        import time

        start = time.time()
        ds = test_component._collect_individual_outputs(2020)
        elapsed = time.time() - start

        # Should be fast (< 0.1s for 200 individuals)
        assert elapsed < 0.1
        assert ds is not None
        assert ds.sizes["individual_id"] == 200

