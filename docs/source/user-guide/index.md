# User Guide

Welcome to the copan:LPJmL user guide. This document covers the practical steps
for model authors: how to initialise a coupled model, run it year by year, add
social components, configure outputs, and leverage parallel execution.

## 🚀 Getting started

1. Install the package (and its dependencies) in your environment:

   ```bash
   pip install pycopanlpjml
   ```

2. Create or reuse a `coupled_config` YAML file that describes how LPJmL should
   be launched (grid resolution, climate forcing, management options, etc.) and
   how your social model should be configured.

3. Instantiate the base model component and run a simple loop:

   ```python
   from pycopanlpjml import Model

   model = Model(config_file="config/coupled_config.yml")

   for year in range(2000, 2010):
       model.update(year)
   ```

The bare `Model` gives you a fully coupled LPJmL world (useful for I/O
tests or when you only need LPJmL outputs). It does not add any social logic
until you subclass it. Behind the scenes copan:LPJmL will start the LPJmL
executable through `pycoupler`, create `World`, `Country` and `Cell` entities
that mirror your grid, and expose a Python API for reading/writing LPJmL fields.

## 🧭 Basic workflow

Each call to `Model.update(year)` performs a full "tick":

1. The LPJmL world advances by one year and the new environmental state is
   mapped onto every cell in your social model.
2. The component loops through all configured `Country` instances (either
   sequentially or via Dask if you enabled parallel execution) and runs your
   model's `Country.update`.
3. Any scalar agent attributes marked as outputs or sync fields are collected
   and merged back into the central LPJmL state so the next year starts with
   updated behaviour.

Thanks to the helper `pycopanlpjml.serialization.sync_world`, you don't have to
worry about moving data between the driver process and worker processes; the
component takes care of serialising the correct slices and re-attaching cells
and agents on the other side.

## 🧩 Extending with your own model

Most projects create a subclass that mixes in their domain-specific components,
for example (simplified):

```python
from pycopanlpjml import Model, World, Country, Cell
from inseeds.components.farming import Component as FarmingComponent


class InseedsModel(Model, FarmingComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.world = World(model=self, **self._make_world_kwargs())
        self.init_countries(country_class=Country)
        self.init_cells(cell_class=Cell)
        self.init_farmers(farmer_class=FarmingComponent.Farmer)

    def update(self, year: int) -> None:
        self.update_countries(year)
        self.update_lpjml(year)
```

You only need to provide the domain logic (for example `Country.update`) and
optionally override the `init_*` helpers if you have custom entities. Everything
else—parallel execution, LPJmL coupling and output collection—is already handled
by `Model`.

## 📊 Output System

copan:LPJmL provides a high-performance output system for collecting and writing
model outputs. Outputs are collected incrementally using Zarr storage during
simulation, then converted to final formats after completion.

### Defining outputs

Define output variables on your entity classes using `Output`:

```python
from pycopanlpjml.output import Output
from pycopancore.data_model.variable import Variable
from pycopancore.data_model.unit import DimensionedAbsoluteUnit as DAU

class MyCell(Cell):
    output_variables = Output(
        soilc=Variable("soil carbon", unit=DAU.gC_per_m2),
        yield_val=Variable("crop yield", unit=DAU.t_per_ha),
    )
```

### Configuring output formats

In your `coupled_config`, specify which formats to write:

```yaml
output:
  format: ["netcdf", "parquet"]  # or just ["csv"]
  cell:
    - soilc
    - yield_val
  country:
    - total_yield
  individual:
    - farmer_income
```

### Accessing outputs during or after simulation

Collected outputs are available lazily on **World**, **Region** (Country), and
**Cell** via ``output_array`` and ``output_table``. No extra work during
simulation; computed only when accessed.

- ``output_array``: xarray Dataset (raw format) for the last collected year.
- ``output_table``: long-format DataFrame (year, cell, entity, variable, value, unit).

On Region and Cell, both are filtered to that entity's cells.

```python
# After model.update(year):
world.output_array    # xarray Dataset for last year
world.output_table   # DataFrame for last year
country.output_table  # DataFrame filtered to this country's cells
cell.output_table    # DataFrame filtered to this cell
```

### Writing outputs to files

Outputs are collected automatically during simulation. After completion, call
`finalize_output_streams()` or use `run_simulation()` which handles this
automatically:

```python
from pycopanlpjml.run import run_simulation

run_simulation(
    config_file="config.json",
    model_factory=lambda cf: InseedsModel(config_file=cf),
)
```

## ⚡ Parallel Execution

copan:LPJmL supports parallel execution of country-level updates using Dask:

### Configuration

```yaml
parallelization:
  mode: "dask"      # or "serial" for single-threaded
  max_workers: 64   # optional, auto-detected if not set
```

### Automatic runtime management

Use `run_simulation()` for automatic parallel runtime management:

```python
from pycopanlpjml.run import run_simulation

run_simulation(
    config_file="config.json",
    model_factory=build_my_model,
    description="Production simulation",
)
```

This handles:
- Parallel runtime startup and shutdown
- Single-instance locking
- Driver and worker profiling
- Output writing
- Graceful error handling

### Manual control

For more control, use the parallelization utilities directly:

```python
from pycopanlpjml.parallel import (
    start_local_dask_cluster,
    configure_model_for_dask,
)

with start_local_dask_cluster(n_workers=8) as runtime:
    configure_model_for_dask(model, runtime.client)
    for year in years:
        model.update(year)
```

## 🧠 Advanced topics

- **Custom synchronisation:** implement `get_sync_attributes` on your agent
  classes (or set a `sync_attributes` list) to explicitly control which
  variables are sent back from workers. Anything returned here will be included
  in the synchronisation payload.
- **Data handling:** use the fields exposed on `model.world` plus
  `cell.to_earth` / `cell.from_earth` to feed LPJmL with management decisions
  (e.g. irrigation, fertiliser) or to read the resulting crop yields, soil
  carbon and climate variables.
- **Entity aliasing:** use `AliasMixin` to provide semantic aliases for
  pycopancore relationships (e.g., `cell.country` instead of
  `cell.social_system`).
- **Profiling:** enable driver/worker profiling via configuration to identify
  performance bottlenecks:

  ```yaml
  profiling:
    driver: true
    workers: false
  ```

For more exhaustive API details—including every public class, helper and their
parameters—check the [API reference](../api/index.rst).