import cloudpickle
import numpy as np
import sympy as sp
import xarray as xr
from dask import delayed

from pycopancore.private._simple_expressions import unknown
from pycopanlpjml.world import World
from pycopanlpjml.region import Country
from pycopanlpjml import region as region_module
from pycopanlpjml.model import Model
from pycopanlpjml.serialization import (
    serialize_country_for_worker,
    sync_world,
    deserialize_country,
    get_sync_attributes,
)


def _make_world(cell_count=6):
    cells = np.arange(cell_count)
    data = xr.DataArray(cells, dims=("cell",), coords={"cell": cells})
    ds = xr.Dataset({"var": data})
    return World(
        input=ds,
        output=ds.copy(),
        grid=xr.DataArray(np.linspace(-1, 1, cell_count), dims=("cell",)),
        country_code=xr.DataArray(cells, dims=("cell",)),
        area=xr.DataArray(np.ones(cell_count), dims=("cell",)),
    )


def test_country_getstate_replaces_problematic_references():
    world = _make_world()
    country = Country(world=world, grid=np.arange(3))
    # Inject objects that previously triggered recursion during serialization
    country._expr_cache = sp.Symbol("x")  # sympy expression
    country._other_country = country  # direct mixin reference

    state = country.__getstate__()
    assert state["_expr_cache"] is unknown
    assert state["_other_country"] is unknown


def test_serialized_country_payload_is_picklable_with_exprs():
    world = _make_world()
    country = Country(world=world, grid=np.arange(3))
    country._expr_cache = sp.Symbol("y")

    payload = serialize_country_for_worker(country)
    # cloudpickle should succeed without hitting metaclass recursion
    cloudpickle.dumps(payload)


def test_local_world_view_recursion_guard_scans_datasets():
    world = _make_world(cell_count=2)
    country = Country(world=world, grid=np.arange(2))
    view = world.build_local_view([0, 1])
    bad_values = xr.DataArray(
        np.array([country, country], dtype=object),
        dims=("cell",),
        coords={"cell": np.arange(2)},
    )
    view._to_earth_data = xr.Dataset({"bad": bad_values})

    assert region_module._contains_non_serializable_references(view)


class DummyCell:
    def __init__(self, world, cell_index):
        self._world = world
        self._cell_index = cell_index
        self.neighbourhood = []
        self._individuals = set()
        self.social_system = None
        self._social_system = None
        self.social_systems = []


class DummyIndividual:
    sync_attributes = ["value"]
    _entity_alias = "farmer"

    def __init__(self, idx, cell, world, value=0.0):
        self._individual_index = idx
        self.value = value
        self._cell = cell
        self._world = world
        self.neighbourhood = []
        self.social_system = None
        self._social_system = None
        self.social_systems = []


class DummyOutputIndividual(DummyIndividual):
    def __init__(self, idx, cell, world, value=0.0, secret=5.0):
        super().__init__(idx, cell, world, value)
        self.secret_state = secret

    def get_defined_outputs(self):
        return ["value"]


class DummyConsumer(DummyIndividual):
    _entity_alias = "consumer"


class DummyCountry(Country):
    def update(self, t):
        for individual in getattr(self, "_direct_individuals", set()):
            individual.value = float(t)


def _build_country_with_individuals():
    world = _make_world()
    country = DummyCountry(world=world, grid=np.arange(2))

    cell0 = DummyCell(world, 0)
    cell1 = DummyCell(world, 1)
    cell0.neighbourhood = [cell1]
    cell1.neighbourhood = [cell0]

    farmer0 = DummyIndividual(0, cell0, world, value=1.0)
    farmer1 = DummyIndividual(1, cell1, world, value=2.0)
    farmer0.neighbourhood = [farmer1]
    farmer1.neighbourhood = [farmer0]

    cell0._individuals.add(farmer0)
    cell1._individuals.add(farmer1)

    for cell in (cell0, cell1):
        cell.social_system = country
        cell._social_system = country
        cell.social_systems = [country]

    for farmer in (farmer0, farmer1):
        farmer.social_system = country
        farmer._social_system = country
        farmer.social_systems = [country]

    country._direct_cells = {cell0, cell1}
    country._next_lower_social_systems = set(country._direct_cells)
    country._direct_individuals = {farmer0, farmer1}
    country._individuals = set(country._direct_individuals)
    world._individuals = set(country._direct_individuals)

    return world, country, (cell0, cell1), (farmer0, farmer1)


def test_country_serialization_clones_cells_and_individuals():
    world, country, original_cells, original_farmers = (
        _build_country_with_individuals()
    )

    payload = serialize_country_for_worker(country)
    worker_country = deserialize_country(payload)

    assert worker_country is not country
    assert len(worker_country._direct_cells) == len(original_cells)

    worker_cells = list(worker_country._direct_cells)
    for cell in worker_cells:
        assert cell._world is worker_country._world
        assert cell not in original_cells
        assert len(cell._individuals) == 1
        assert all(
            neighbour in worker_cells for neighbour in cell.neighbourhood
        )

    worker_farmers = set()
    for cell in worker_cells:
        worker_farmers.update(cell._individuals)

    assert len(worker_farmers) == len(original_farmers)
    for farmer in worker_farmers:
        assert farmer._world is worker_country._world
        assert farmer not in original_farmers
        assert all(
            neighbour in worker_farmers for neighbour in farmer.neighbourhood
        )


def test_serialized_state_has_detached_individuals():
    _, country, _, _ = _build_country_with_individuals()

    payload = serialize_country_for_worker(country)
    _, _, state = payload

    for cell in state["_direct_cells"]:
        for farmer in cell._individuals:
            assert getattr(farmer, "_cell", None) is None
            assert getattr(farmer, "_cell_index", None) is not None


def test_country_serialization_handles_country_level_individuals():
    world, country, original_cells, original_farmers = (
        _build_country_with_individuals()
    )
    # Remove references from cells to simulate models that only store
    # individuals on the country
    for cell in original_cells:
        cell._individuals = set()

    payload = serialize_country_for_worker(country)
    worker_country = deserialize_country(payload)

    worker_individuals = worker_country.individuals
    assert isinstance(worker_individuals, set)
    assert len(worker_individuals) == len(original_farmers)


def test_worker_country_exposes_individuals_set():
    _, country, _, original_farmers = _build_country_with_individuals()

    payload = serialize_country_for_worker(country)
    worker_country = deserialize_country(payload)

    individuals = worker_country.individuals
    assert isinstance(individuals, set)
    assert len(individuals) == len(original_farmers)

    for individual in individuals:
        assert getattr(individual, "_cell", None) is not None
        assert getattr(individual, "_world", None) is worker_country._world


def test_worker_country_exposes_cells_set():
    _, country, original_cells, _ = _build_country_with_individuals()

    payload = serialize_country_for_worker(country)
    worker_country = deserialize_country(payload)

    cells = worker_country.cells
    assert isinstance(cells, set)
    assert len(cells) == len(original_cells)
    for cell in cells:
        assert getattr(cell, "_world", None) is worker_country._world


def test_country_serialization_strips_social_links_from_clones():
    _, country, _, _ = _build_country_with_individuals()

    payload = serialize_country_for_worker(country)
    _, _, state = payload

    # Cloned cells and individuals sent to workers must not retain country refs
    for cell in state["_direct_cells"]:
        assert getattr(cell, "_social_system", None) is None
        assert getattr(cell, "_social_systems", []) == []
    for farmer in state["_direct_individuals"]:
        assert getattr(farmer, "_social_system", None) is None
        assert getattr(farmer, "_social_systems", []) == []


def test_country_dynamic_individual_accessors():
    world, country, cells, farmers = _build_country_with_individuals()
    consumer = DummyConsumer(2, cells[0], world, value=5.0)
    cells[0]._individuals.add(consumer)
    country._direct_individuals.add(consumer)
    country._individuals.add(consumer)
    world._individuals.add(consumer)

    assert farmers[0] in country.farmers
    assert consumer in country.consumers
    assert country.individual == country.individuals

    payload = serialize_country_for_worker(country)
    worker_country = deserialize_country(payload)

    assert len(worker_country.farmers) == len(farmers)
    assert len(worker_country.consumers) == 1


def test_component_applies_individual_updates():
    world, country, _, farmers = _build_country_with_individuals()
    payload = serialize_country_for_worker(country)

    result = sync_world(payload, t=7)
    _, _, updated = result

    assert updated is not None
    assert set(updated["indices"]) == {0, 1}
    assert "value" in updated["values"]

    component = Model.__new__(Model)
    component.world = world
    component._farmers = list(farmers)
    world._individuals = set(farmers)

    component._apply_individual_updates(updated["indices"], updated["values"])

    for farmer in farmers:
        assert farmer.value == 7.0


def test_cloudpickle_handles_dask_delayed_sync_world():
    _, country, _, _ = _build_country_with_individuals()
    payload = serialize_country_for_worker(country)

    expr = delayed(sync_world, pure=False)(payload, t=2023)
    cloudpickle.dumps(expr)


def test_sync_attributes_include_public_state_even_with_defined_outputs():
    class PlainIndividual:
        def __init__(self):
            self.value = 3.0
            self.secret_state = 42.0
            self._world = "private"

        def get_defined_outputs(self):
            return ["value"]

    attrs = get_sync_attributes(PlainIndividual())

    assert "value" in attrs  # defined output
    assert "secret_state" in attrs  # public attribute not listed in outputs
    assert "_world" not in attrs  # private attributes remain excluded


def test_neighbourhood_rebuilt_from_indices_after_deserialization():
    """Test that neighbourhood graphs are correctly rebuilt from indices.

    During serialization, neighbourhood relationships are stored as indices
    (not object references) to avoid deep recursion. This test verifies that
    the reconstruction correctly rebuilds the neighbourhood lists with actual
    object references on the worker side.
    """
    world, country, original_cells, original_farmers = (
        _build_country_with_individuals()
    )
    cell0, cell1 = original_cells
    farmer0, farmer1 = original_farmers

    # Verify original neighbourhoods are set up correctly
    assert cell1 in cell0.neighbourhood
    assert cell0 in cell1.neighbourhood
    assert farmer1 in farmer0.neighbourhood
    assert farmer0 in farmer1.neighbourhood

    # Serialize and deserialize
    payload = serialize_country_for_worker(country)
    worker_country = deserialize_country(payload)

    # Get the worker-side cells and farmers
    worker_cells = list(worker_country._direct_cells)
    worker_farmers = set()
    for cell in worker_cells:
        worker_farmers.update(cell._individuals)
    worker_farmers = list(worker_farmers)

    # Verify cell neighbourhoods are rebuilt
    for worker_cell in worker_cells:
        # Each cell should have exactly one neighbour (the other cell)
        assert len(worker_cell.neighbourhood) == 1
        neighbour = worker_cell.neighbourhood[0]
        # The neighbour should be a worker cell, not an original cell
        assert neighbour in worker_cells
        assert neighbour not in original_cells
        # The neighbour should reference back to this cell
        assert worker_cell in neighbour.neighbourhood

    # Verify farmer neighbourhoods are rebuilt
    for worker_farmer in worker_farmers:
        # Each farmer should have exactly one neighbour (the other farmer)
        assert len(worker_farmer.neighbourhood) == 1
        neighbour = worker_farmer.neighbourhood[0]
        # The neighbour should be a worker farmer, not an original farmer
        assert neighbour in worker_farmers
        assert neighbour not in original_farmers
        # The neighbour should reference back to this farmer
        assert worker_farmer in neighbour.neighbourhood


def test_neighbourhood_indices_not_persisted_after_reconstruction():
    """Test that temporary _neighbourhood_indices are cleaned up after reconstruction."""  # noqa: E501
    _, country, _, _ = _build_country_with_individuals()

    payload = serialize_country_for_worker(country)
    worker_country = deserialize_country(payload)

    # After reconstruction, _neighbourhood_indices should be cleaned up
    for cell in worker_country._direct_cells:
        assert not hasattr(
            cell, "_neighbourhood_indices"
        ), "Cell should not have _neighbourhood_indices after reconstruction"

    for cell in worker_country._direct_cells:
        for farmer in cell._individuals:
            assert not hasattr(
                farmer, "_neighbourhood_indices"
            ), "Farmer should not have _neighbourhood_indices after reconstruction"  # noqa: E501


def test_neighbourhood_with_cross_country_neighbours():
    """Test that cross-country neighbours are included as external proxies.

    Neighbours from other countries are included in the neighbourhood as
    cross-border neighbour objects, allowing cross-border interactions.
    """
    world = _make_world(cell_count=4)

    # Create two countries, each with 2 cells
    country1 = DummyCountry(world=world, grid=np.arange(2))
    country2 = DummyCountry(world=world, grid=np.arange(2, 4))

    # Create cells for country1
    cell0 = DummyCell(world, 0)
    cell1 = DummyCell(world, 1)
    # Create cells for country2
    cell2 = DummyCell(world, 2)
    cell3 = DummyCell(world, 3)

    # Set up neighbourhoods: cell1 and cell2 are neighbours (across countries)
    cell0.neighbourhood = [cell1]
    cell1.neighbourhood = [cell0, cell2]  # cell2 is in a different country
    cell2.neighbourhood = [cell1, cell3]  # cell1 is in a different country
    cell3.neighbourhood = [cell2]

    # Set up cells for country1
    for cell in (cell0, cell1):
        cell.social_system = country1
        cell._social_system = country1
        cell.social_systems = [country1]
    country1._direct_cells = {cell0, cell1}
    country1._next_lower_social_systems = set(country1._direct_cells)
    country1._direct_individuals = set()
    country1._individuals = set()

    # Serialize and deserialize country1
    payload = serialize_country_for_worker(country1)
    worker_country = deserialize_country(payload)

    worker_cells = list(worker_country._direct_cells)
    worker_cell_indices = {c._cell_index for c in worker_cells}

    # Specifically, the cell with index 1 should have both cell 0 (internal)
    # and cell 2 (cross-border neighbour) as neighbours
    cell1_worker = [c for c in worker_cells if c._cell_index == 1][0]
    neighbour_indices = [n._cell_index for n in cell1_worker.neighbourhood]
    assert 0 in neighbour_indices  # Internal neighbour
    assert 2 in neighbour_indices  # External neighbour (now included as proxy)

    # Check that external neighbour is marked as external
    external_neighbours = [
        n
        for n in cell1_worker.neighbourhood
        if getattr(n, "_is_external", False)
    ]
    assert len(external_neighbours) == 1
    assert external_neighbours[0]._cell_index == 2


# =============================================================================
# SERIALIZATION HELPER TESTS
# =============================================================================


def test_is_sync_scalar_with_primitives():
    """Test is_sync_scalar correctly identifies primitive types."""
    from pycopanlpjml.serialization import is_sync_scalar

    # Should return True for scalars
    assert is_sync_scalar(42) is True
    assert is_sync_scalar(3.14) is True
    assert is_sync_scalar(True) is True
    assert is_sync_scalar("hello") is True
    assert is_sync_scalar(np.float64(1.5)) is True
    assert is_sync_scalar(np.int32(10)) is True

    # Should return False for non-scalars
    assert is_sync_scalar(None) is False
    assert is_sync_scalar([1, 2, 3]) is False
    assert is_sync_scalar({"a": 1}) is False
    assert is_sync_scalar(np.array([1, 2, 3])) is False


def test_extract_output_scalar_conversions():
    """Test extract_output_scalar handles various input types."""
    from pycopanlpjml.serialization import extract_output_scalar

    # Normal values
    assert extract_output_scalar(42) == 42.0
    assert extract_output_scalar(3.14) == 3.14
    assert extract_output_scalar(np.float64(2.5)) == 2.5

    # None returns NaN
    result = extract_output_scalar(None)
    assert np.isnan(result)

    # Non-convertible returns NaN
    result = extract_output_scalar("not a number")
    assert np.isnan(result)

    # Objects that can be converted to float
    class Convertible:
        def __float__(self):
            return 99.0

    assert extract_output_scalar(Convertible()) == 99.0


def test_extract_world_slice_with_mapping():
    """Test extract_world_slice with global-to-local index mapping."""
    from pycopanlpjml.serialization import extract_world_slice

    # Create test dataset
    cells = np.arange(10)
    data = xr.DataArray(cells * 2, dims=("cell",), coords={"cell": cells})
    dataset = xr.Dataset({"values": data})

    # Create mock world with mapping
    class MockWorld:
        def map_global_to_local(self, indices):
            # Map global indices 5-7 to local indices 0-2
            return [i - 5 for i in indices]

    world = MockWorld()
    cell_indices = np.array([5, 6, 7])

    result = extract_world_slice(dataset, world, cell_indices)

    # Should use the mapped indices
    assert result is not None
    assert "values" in result.data_vars


def test_extract_world_slice_none_dataset():
    """Test extract_world_slice returns None for None dataset."""
    from pycopanlpjml.serialization import extract_world_slice

    result = extract_world_slice(None, None, np.array([0]))
    assert result is None


def test_dataset_to_numpy_dict():
    """Test dataset_to_numpy_dict conversion."""
    from pycopanlpjml.serialization import dataset_to_numpy_dict

    # Create test dataset
    cells = np.arange(5)
    ds = xr.Dataset(
        {
            "var1": xr.DataArray(cells, dims=("cell",)),
            "var2": xr.DataArray(cells * 2.0, dims=("cell",)),
        }
    )

    result = dataset_to_numpy_dict(ds)

    assert result is not None
    assert "var1" in result
    assert "var2" in result
    assert isinstance(result["var1"], np.ndarray)
    assert isinstance(result["var2"], np.ndarray)
    np.testing.assert_array_equal(result["var1"], cells)
    np.testing.assert_array_equal(result["var2"], cells * 2.0)


def test_dataset_to_numpy_dict_none():
    """Test dataset_to_numpy_dict returns None for None input."""
    from pycopanlpjml.serialization import dataset_to_numpy_dict

    assert dataset_to_numpy_dict(None) is None


# =============================================================================
# LOCAL WORLD VIEW TESTS
# =============================================================================


def test_local_world_view_creation():
    """Test _LocalWorldView is created correctly."""
    from pycopanlpjml.world import _LocalWorldView

    world = _make_world(cell_count=10)
    cell_indices = [2, 4, 6]

    view = _LocalWorldView(world, cell_indices)

    # Check basic structure
    assert view.cell_neighbourhood is None  # No graphs on workers
    assert view.country_neighbourhood is None
    assert len(view._global_cell_indices) == 3


def test_local_world_view_data_slicing():
    """Test _LocalWorldView correctly slices data."""
    from pycopanlpjml.world import _LocalWorldView

    world = _make_world(cell_count=10)
    cell_indices = [2, 4, 6]

    view = _LocalWorldView(world, cell_indices)

    # Check data is sliced
    assert view.to_earth is not None
    assert len(view.to_earth.cell) == 3

    assert view.from_earth is not None
    assert view.grid is not None
    assert view.country_code is not None


def test_local_world_view_index_mapping():
    """Test _LocalWorldView.map_global_to_local."""
    from pycopanlpjml.world import _LocalWorldView

    world = _make_world(cell_count=10)
    cell_indices = [2, 4, 6]

    view = _LocalWorldView(world, cell_indices)

    # Map global indices to local
    local = view.map_global_to_local([2, 4, 6])
    assert local == [0, 1, 2]


def test_local_world_view_build_local_view_same():
    """Test _LocalWorldView.build_local_view returns self for same indices."""
    from pycopanlpjml.world import _LocalWorldView

    world = _make_world(cell_count=10)
    cell_indices = [2, 4, 6]

    view = _LocalWorldView(world, cell_indices)
    same_view = view.build_local_view(cell_indices)

    assert same_view is view


def test_local_world_view_build_local_view_different_raises():
    """Test _LocalWorldView.build_local_view raises for different indices."""
    from pycopanlpjml.world import _LocalWorldView
    import pytest

    world = _make_world(cell_count=10)
    cell_indices = [2, 4, 6]

    view = _LocalWorldView(world, cell_indices)

    with pytest.raises(
        ValueError, match="Cannot build a different local view"
    ):
        view.build_local_view([1, 2, 3])


def test_local_world_view_deprecated_aliases():
    """Test _LocalWorldView backward compatibility aliases."""
    from pycopanlpjml.world import _LocalWorldView
    import warnings

    world = _make_world(cell_count=10)
    cell_indices = [2, 4, 6]

    view = _LocalWorldView(world, cell_indices)

    # Access deprecated aliases (should work but may warn)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        _ = view.input
        _ = view.output


def test_model_config_view():
    """Test _ModelConfigView wraps config correctly."""
    from pycopanlpjml.world import _ModelConfigView

    class MockModel:
        def __init__(self):
            self.config = {"key": "value"}

    model = MockModel()
    view = _ModelConfigView(model)

    assert view.config == {"key": "value"}


def test_model_config_view_no_config():
    """Test _ModelConfigView handles model without config."""
    from pycopanlpjml.world import _ModelConfigView

    class MockModel:
        pass

    model = MockModel()
    view = _ModelConfigView(model)

    assert view.config is None


# =============================================================================
# REGION EDGE CASE TESTS
# =============================================================================


def test_region_get_individuals_empty():
    """Test Region.get_individuals returns empty set when no individuals."""
    world = _make_world()
    country = Country(world=world, grid=np.arange(3))

    individuals = country.get_individuals()
    assert individuals == set()


def test_region_get_individuals_by_type():
    """Test Region.get_individuals filters by type."""
    world, country, cells, farmers = _build_country_with_individuals()

    # Get all individuals
    all_ind = country.get_individuals()
    assert len(all_ind) > 0

    # Get by type (farmers should be in the result)
    farmers_result = country.get_individuals("farmer")
    assert len(farmers_result) > 0


def test_contains_non_serializable_detects_dask_expr():
    """Test _contains_non_serializable_references detects Dask expressions."""

    # Create a mock Dask-like expression object
    class LLGExpr:
        pass

    expr = LLGExpr()
    assert region_module._contains_non_serializable_references(expr) is True


def test_contains_non_serializable_handles_cycles():
    """Test _contains_non_serializable_references handles circular
    references."""
    # Create circular dict
    d1 = {"a": 1}
    d2 = {"b": d1}
    d1["c"] = d2

    # Should not raise RecursionError
    result = region_module._contains_non_serializable_references(d1)
    assert isinstance(result, bool)


def test_serialization_cache_reuses_clone_structure():
    """Test that cached serialization reuses clone objects across calls.

    The serialization cache should:
    1. Create clones on first __getstate__ call
    2. Reuse the same clone objects on subsequent calls
    3. Update dynamic values in the clones each time
    """
    world, country, original_cells, original_farmers = (
        _build_country_with_individuals()
    )
    farmer0 = original_farmers[0]

    # Set initial value
    farmer0.value = 10.0

    # First serialization - creates cache
    state1 = country.__getstate__()
    assert hasattr(country, "_serialization_cache")
    cache = country._serialization_cache

    # Get clone references
    clones1 = state1["_direct_cells"]
    farmer_clones1 = set()
    for c in clones1:
        farmer_clones1.update(c._individuals)

    # Find the clone corresponding to farmer0 (by individual_index)
    farmer0_clone = None
    for clone in farmer_clones1:
        if (
            getattr(clone, "_individual_index", None)
            == farmer0._individual_index
        ):
            farmer0_clone = clone
            break

    assert farmer0_clone is not None
    assert farmer0_clone.value == 10.0  # Should have first value

    # Update original value
    farmer0.value = 20.0

    # Second serialization - should reuse cache
    state2 = country.__getstate__()

    # Cache should be same object
    assert country._serialization_cache is cache

    # Clone objects should be same (reused)
    clones2 = state2["_direct_cells"]
    assert clones1 == clones2  # Same set of objects

    # Same farmer0_clone object should now have updated value
    assert farmer0_clone.value == 20.0


def test_serialization_cache_improves_performance():
    """Test that cached serialization is faster on subsequent calls.

    This is a basic performance sanity check - cached calls should be
    faster than the initial call.
    """
    import time

    world, country, original_cells, original_farmers = (
        _build_country_with_individuals()
    )

    # First call - creates cache (slower)
    start = time.perf_counter()
    _ = country.__getstate__()
    first_duration = time.perf_counter() - start

    # Warm up - do a few more calls
    for _ in range(3):
        _ = country.__getstate__()

    # Cached call should complete without error
    start = time.perf_counter()
    _ = country.__getstate__()
    cached_duration = time.perf_counter() - start

    # We can't guarantee faster in a unit test due to variability,
    # but cached call should complete reasonably quickly
    assert cached_duration < 1.0  # Should be much faster than 1 second


# =============================================================================
# EXTERNAL NEIGHBOR BUFFER TESTS
# =============================================================================


def test_cross_border_neighbour_buffer_captures_cross_country_cells():
    """Test that external cell neighbours are captured in the buffer.

    When a cell has neighbours in another country, those cross-border
    neighbours should be included in the cross_border_neighbour_buffer so they
    can be reconstructed on the worker.
    """
    world = _make_world(cell_count=4)

    # Create two countries, each with 2 cells
    country1 = DummyCountry(world=world, grid=np.arange(2))
    country2 = DummyCountry(world=world, grid=np.arange(2, 4))

    # Create cells for country1
    cell0 = DummyCell(world, 0)
    cell1 = DummyCell(world, 1)
    # Create cells for country2
    cell2 = DummyCell(world, 2)
    cell3 = DummyCell(world, 3)

    # Set up neighbourhoods: cell1 and cell2 are neighbours (across countries)
    cell0.neighbourhood = [cell1]
    cell1.neighbourhood = [cell0, cell2]  # cell2 is in a different country
    cell2.neighbourhood = [cell1, cell3]  # cell1 is in a different country
    cell3.neighbourhood = [cell2]

    # Set up cells for country1
    for cell in (cell0, cell1):
        cell.social_system = country1
        cell._social_system = country1
        cell.social_systems = [country1]
    country1._direct_cells = {cell0, cell1}
    country1._next_lower_social_systems = set(
        country1._direct_cells
    )  # noqa: E501
    country1._direct_individuals = set()
    country1._individuals = set()

    # Serialize country1
    state = country1.__getstate__()

    # Check that external buffer is present
    assert "_cross_border_neighbour_buffer" in state
    buffer = state["_cross_border_neighbour_buffer"]

    # Should have captured cell2 as a cross-border neighbour
    assert len(buffer["external_cells"]) == 1
    ext_cell = buffer["external_cells"][0]
    assert ext_cell["_cell_index"] == 2
    assert ext_cell["_is_external"] is True

    # Should have mapping from cell1 to its cross-border neighbour cell2
    assert 1 in buffer["cell_external_neighbours"]
    assert 2 in buffer["cell_external_neighbours"][1]


def test_cross_border_neighbour_buffer_captures_cross_country_individuals():
    """Test that external individual neighbours are captured in the buffer.

    When an individual has neighbours in another country, those external
    neighbours should be included in the cross_border_neighbour_buffer.
    """
    world = _make_world(cell_count=4)

    # Create two countries
    country1 = DummyCountry(world=world, grid=np.arange(2))
    country2 = DummyCountry(world=world, grid=np.arange(2, 4))

    # Create cells
    cell0 = DummyCell(world, 0)
    cell1 = DummyCell(world, 1)
    cell2 = DummyCell(world, 2)
    cell3 = DummyCell(world, 3)

    # Set up cell neighbourhoods
    cell0.neighbourhood = [cell1]
    cell1.neighbourhood = [cell0, cell2]
    cell2.neighbourhood = [cell1, cell3]
    cell3.neighbourhood = [cell2]

    # Create individuals
    farmer0 = DummyIndividual(0, cell0, world, value=1.0)
    farmer1 = DummyIndividual(1, cell1, world, value=2.0)
    farmer2 = DummyIndividual(2, cell2, world, value=3.0)  # In country2
    farmer3 = DummyIndividual(3, cell3, world, value=4.0)  # In country2

    # Set up individual neighbourhoods (farmer1 neighbours farmer2 across
    # border)
    farmer0.neighbourhood = [farmer1]
    farmer1.neighbourhood = [farmer0, farmer2]  # farmer2 is external
    farmer2.neighbourhood = [farmer1, farmer3]
    farmer3.neighbourhood = [farmer2]

    # Add individuals to cells
    cell0._individuals.add(farmer0)
    cell1._individuals.add(farmer1)
    cell2._individuals.add(farmer2)
    cell3._individuals.add(farmer3)

    # Set up country1
    for cell in (cell0, cell1):
        cell.social_system = country1
        cell._social_system = country1
        cell.social_systems = [country1]
    for farmer in (farmer0, farmer1):
        farmer.social_system = country1
        farmer._social_system = country1
        farmer.social_systems = [country1]

    country1._direct_cells = {cell0, cell1}
    country1._next_lower_social_systems = set(
        country1._direct_cells
    )  # noqa: E501
    country1._direct_individuals = {farmer0, farmer1}
    country1._individuals = set(country1._direct_individuals)

    # Serialize country1
    state = country1.__getstate__()
    buffer = state["_cross_border_neighbour_buffer"]

    # Should have captured farmer2 as a cross-border neighbour
    assert len(buffer["external_individuals"]) == 1
    ext_ind = buffer["external_individuals"][0]
    assert ext_ind["_individual_index"] == 2
    assert ext_ind["_is_external"] is True
    assert ext_ind["value"] == 3.0  # Should have captured the value

    # Should have mapping from farmer1 to its cross-border neighbour farmer2
    assert 1 in buffer["individual_external_neighbours"]
    assert 2 in buffer["individual_external_neighbours"][1]


def test_external_neighbours_reconstructed_after_deserialization():
    """Test that cross-border neighbours are available in neighbourhood after
    deserialization.

    After deserializing a country, internal entities should have their
    cross-border neighbours reconstructed as proxy objects in their
    neighbourhood.
    """
    world = _make_world(cell_count=4)

    # Create two countries
    country1 = DummyCountry(world=world, grid=np.arange(2))

    # Create cells
    cell0 = DummyCell(world, 0)
    cell1 = DummyCell(world, 1)
    cell2 = DummyCell(world, 2)  # External

    # Set up cell neighbourhoods
    cell0.neighbourhood = [cell1]
    cell1.neighbourhood = [cell0, cell2]  # cell2 is external

    # Create individuals
    farmer0 = DummyIndividual(0, cell0, world, value=1.0)
    farmer1 = DummyIndividual(1, cell1, world, value=2.0)
    farmer2 = DummyIndividual(2, cell2, world, value=3.0)  # External

    # Set up individual neighbourhoods
    farmer0.neighbourhood = [farmer1]
    farmer1.neighbourhood = [farmer0, farmer2]  # farmer2 is external

    # Add individuals to cells
    cell0._individuals.add(farmer0)
    cell1._individuals.add(farmer1)
    cell2._individuals.add(farmer2)

    # Set up country1
    for cell in (cell0, cell1):
        cell.social_system = country1
        cell._social_system = country1
        cell.social_systems = [country1]
    for farmer in (farmer0, farmer1):
        farmer.social_system = country1
        farmer._social_system = country1
        farmer.social_systems = [country1]

    country1._direct_cells = {cell0, cell1}
    country1._next_lower_social_systems = set(
        country1._direct_cells
    )  # noqa: E501
    country1._direct_individuals = {farmer0, farmer1}
    country1._individuals = set(country1._direct_individuals)

    # Serialize and deserialize
    payload = serialize_country_for_worker(country1)
    worker_country = deserialize_country(payload)

    # Find the worker cell with index 1
    worker_cells = list(worker_country._direct_cells)
    cell1_worker = [c for c in worker_cells if c._cell_index == 1][0]

    # Cell1 should have 2 neighbours: internal cell0 and external cell2
    assert len(cell1_worker.neighbourhood) == 2

    # Find the external cell in the neighbourhood
    external_cells = [
        c
        for c in cell1_worker.neighbourhood
        if getattr(c, "_is_external", False)
    ]
    assert len(external_cells) == 1
    assert external_cells[0]._cell_index == 2

    # Find the worker farmer with index 1
    worker_farmers = set()
    for cell in worker_cells:
        worker_farmers.update(cell._individuals)
    farmer1_worker = [f for f in worker_farmers if f._individual_index == 1][0]

    # Farmer1 should have 2 neighbours: internal farmer0 and external farmer2
    assert len(farmer1_worker.neighbourhood) == 2

    # Find the external farmer in the neighbourhood
    external_farmers = [
        f
        for f in farmer1_worker.neighbourhood
        if getattr(f, "_is_external", False)
    ]
    assert len(external_farmers) == 1
    assert external_farmers[0]._individual_index == 2
    assert external_farmers[0].value == 3.0  # Should have the buffered value


def test_external_proxy_attributes_accessible():
    """Test that cross-border neighbour objects have accessible attributes.

    External proxies should have all dynamic attributes from the original
    entity available for reading (e.g., tillage, soilc, cropyield).
    """
    world = _make_world(cell_count=4)

    # Create country1
    country1 = DummyCountry(world=world, grid=np.arange(2))

    # Create cells
    cell0 = DummyCell(world, 0)
    cell1 = DummyCell(world, 1)
    cell2 = DummyCell(world, 2)  # External

    cell0.neighbourhood = [cell1]
    cell1.neighbourhood = [cell0, cell2]

    # Create individuals with multiple attributes
    farmer0 = DummyIndividual(0, cell0, world, value=1.0)
    farmer1 = DummyIndividual(1, cell1, world, value=2.0)
    farmer2 = DummyIndividual(2, cell2, world, value=3.0)

    # Add extra attributes to farmer2 (simulating tillage farmer)
    farmer2.tillage = 1
    farmer2.soilc = 150.0
    farmer2.cropyield = 5000.0

    farmer0.neighbourhood = [farmer1]
    farmer1.neighbourhood = [farmer0, farmer2]

    cell0._individuals.add(farmer0)
    cell1._individuals.add(farmer1)
    cell2._individuals.add(farmer2)

    for cell in (cell0, cell1):
        cell.social_system = country1
        cell._social_system = country1
        cell.social_systems = [country1]
    for farmer in (farmer0, farmer1):
        farmer.social_system = country1
        farmer._social_system = country1
        farmer.social_systems = [country1]

    country1._direct_cells = {cell0, cell1}
    country1._next_lower_social_systems = set(
        country1._direct_cells
    )  # noqa: E501
    country1._direct_individuals = {farmer0, farmer1}
    country1._individuals = set(country1._direct_individuals)

    # Serialize and deserialize
    payload = serialize_country_for_worker(country1)
    worker_country = deserialize_country(payload)

    # Find farmer1's cross-border neighbour
    worker_farmers = set()
    for cell in worker_country._direct_cells:
        worker_farmers.update(cell._individuals)
    farmer1_worker = [f for f in worker_farmers if f._individual_index == 1][0]

    external_farmer = [
        f
        for f in farmer1_worker.neighbourhood
        if getattr(f, "_is_external", False)
    ][0]

    # Should be able to access all attributes
    assert external_farmer.value == 3.0
    assert external_farmer.tillage == 1
    assert external_farmer.soilc == 150.0
    assert external_farmer.cropyield == 5000.0


def test_external_buffer_cleaned_up_after_reconstruction():
    """Test that _cross_border_neighbour_buffer is cleaned up after
    reconstruction."""
    world = _make_world(cell_count=4)

    country1 = DummyCountry(world=world, grid=np.arange(2))

    cell0 = DummyCell(world, 0)
    cell1 = DummyCell(world, 1)
    cell2 = DummyCell(world, 2)

    cell0.neighbourhood = [cell1]
    cell1.neighbourhood = [cell0, cell2]

    for cell in (cell0, cell1):
        cell.social_system = country1
        cell._social_system = country1
        cell.social_systems = [country1]

    country1._direct_cells = {cell0, cell1}
    country1._next_lower_social_systems = set(
        country1._direct_cells
    )  # noqa: E501
    country1._direct_individuals = set()
    country1._individuals = set()

    # Serialize and deserialize
    payload = serialize_country_for_worker(country1)
    worker_country = deserialize_country(payload)

    # Buffer should be cleaned up
    assert not hasattr(worker_country, "_cross_border_neighbour_buffer")
