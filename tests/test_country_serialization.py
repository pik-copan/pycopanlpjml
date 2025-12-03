import numpy as np
import xarray as xr
import cloudpickle
import sympy as sp

from pycopancore.private._simple_expressions import unknown
from pycopanlpjml.world import World
from pycopanlpjml.region import Country
from pycopanlpjml.component import _serialize_country_for_worker


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

    payload = _serialize_country_for_worker(country)
    # cloudpickle should succeed without hitting metaclass recursion
    cloudpickle.dumps(payload)

import xarray as xr
import cloudpickle
import sympy as sp

from pycopancore.private._simple_expressions import unknown
from pycopanlpjml.world import World
from pycopanlpjml.region import Country
from pycopanlpjml.component import _serialize_country_for_worker


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

    payload = _serialize_country_for_worker(country)
    # cloudpickle should succeed without hitting metaclass recursion
    cloudpickle.dumps(payload)


