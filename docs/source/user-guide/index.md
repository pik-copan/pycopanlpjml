# User Guide

pycopanlpjml is the right framework for you if you want to put people — farmers, regions,
policies — on the same grid as a process-based land biosphere model, and let
them talk to each other every year.

copan:LPJmL (`pycopanlpjml`) is the Python side of that coupling. It extends
[copan:CORE](https://github.com/pik-copan/pycopancore) with [LPJmL](https://github.com/PIK-LPJmL/LPJmL)
as the Earth-system interface ([Breier et al., 2026](https://doi.org/10.5194/gmd-19-6829-2026)).
LPJmL I/O goes through [pycoupler](https://pycoupler.readthedocs.io/).

You write a social-ecological model as a `Model` subclass. The library starts
LPJmL, maps the grid onto entities you already know from copan:CORE, and each
year exchanges management (`to_earth`) and biosphere state (`from_earth`).

A complete application is [InSEEDS](https://github.com/pik-copan/inseeds).
The papers behind the design are
[Donges et al. (2020)](https://doi.org/10.5194/esd-11-395-2020) (copan:CORE)
and [Breier et al. (2026)](https://doi.org/10.5194/gmd-19-6829-2026) (copan:LPJmL).

## 🌐🌍 World–Earth entities (copan:CORE)

copan:CORE treats a World–Earth model as **entities** (“things that are”)
involved in **processes** (“things that happen”) that change **attributes**
(“how things are”) — a cell’s harvest, a country’s institutions, a farmer’s
practice ([Donges et al., 2020](https://doi.org/10.5194/esd-11-395-2020)).

The entity types you will use:

| Entity | Meaning | In copan:LPJmL |
|---|---|---|
| **World** | The whole simulation space | Owns the LPJmL arrays |
| **Cell** | A spatial unit | One LPJmL grid cell (live view) |
| **Social system** | A human-reproduced structure (country, city, …) | `Country` / `Region` |
| **Individual** | One agent, not a “representative consumer” | Farmers and other agents on a cell |
| **Group** | A loose social structure, not tied to a place | Optional (norms, networks) |

Processes sit in three overlapping taxa: **ENV** (biogeophysical), **MET**
(socio-metabolic: harvest, fertiliser, trade), **CUL** (socio-cultural:
learning, norms, policy). copan:LPJmL plugs LPJmL in as the ENV component
and couples it once per year. You implement MET and CUL on countries,
cells and individuals.

```{mermaid}
flowchart LR
    Model --> Coupler["LPJmLCoupler"]
    Model --> World
    World --> Country
    Country --> Cell
    Cell --> Individual
    World -->|"owns arrays"| Arrays["to_earth / from_earth"]
    Cell -->|"scalar isel view"| Arrays
    Country -->|"copy on access"| Arrays
```

**How data is shared.** World is the source of truth. A cell stores a
scalar `isel` into those arrays — write `cell.to_earth` and world (and
therefore LPJmL) sees it. A country cannot share that view: its cells are a
gapped list. `country.output` is a **copy of the current world slice**,
recomputed when you ask for it. Read countries when you need an aggregate;
write through cells.

Aliases: `input` = `to_earth` (you → LPJmL), `output` = `from_earth`
(LPJmL → you).

## 🚀 Getting started

1. Compile [LPJmL](https://github.com/PIK-LPJmL/LPJmL) and set its
   [working environment](https://github.com/PIK-LPJmL/LPJmL/blob/master/INSTALL.md)
   if you are not on the PIK HPC.

2. Install the Python package (this pulls in `pycoupler` and `pycopancore`):

   ```bash
   pip install pycopanlpjml
   ```

3. Subclass `Model`, build the entity tree, and implement `update(t)`.
   `Model` opens the coupler. It does **not** create World / Country / Cell
   for you, and it has no `update()` of its own.

```python
import pycopanlpjml as lpjml


class MyModel(lpjml.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.world = lpjml.World(
            input=self.lpjml.read_input(copy=False),
            output=self.lpjml.read_historic_output(),
            grid=self.lpjml.grid,
            country_code=self.lpjml.country,
            area=self.lpjml.terr_area,
        )
        self.init_countries(country_class=lpjml.Country)
        self.init_cells(cell_class=lpjml.Cell)

    def update(self, t):
        self.update_countries(t)  # social step (countries)
        self.update_lpjml(t)      # exchange with LPJmL
        self.collect_outputs(t)   # optional


model = MyModel(config_file="config.json")
for year in model.lpjml.get_sim_years():
    model.update(year)
```

Countries are optional. Skip `init_countries` and write the social step
yourself — loop cells or farmers in `update(t)`, then call `update_lpjml`.
There is no `update_world`: World is the data holder, not a second
iterator. `update_countries` exists because a country list is something
the library can walk for you.

`init_countries` makes one `Country` per ISO code on the grid.
`init_cells` attaches each cell to its country (or to world if you
skipped countries), stores the live views, and builds neighbourhood
graphs.

Pass your own `country_class` / `cell_class` (and later farmer classes)
when you have domain logic. `Model.__init__` takes `config_file` (or an
existing `lpjml=` coupler), plus `lpjml_host` / `lpjml_port`
(default `localhost:2224`).

## 🧭 Yearly coupling

Each `update(t)` is one annual increment:

1. **Social step** — with countries, `update_countries(t)` calls
   `country.update(t)` in sequence (a country usually loops its farmers
   or cells). Without countries, put that loop in your own `update(t)`.
2. **Earth step** — `update_lpjml(t)` sends `world.to_earth`, writes the
   new `from_earth` **in place**, and steps the time coordinates. After
   `config.lastyear` the coupler closes.
3. **Outputs** — call `collect_outputs(t)` yourself if you want tables or
   NetCDF. The base `Model` does not.

Writes LPJmL must see go on `cell.to_earth` or `world.to_earth`. Edits on
`country.input` are discarded unless you assign the object back through the
setter.

If you **replace** `world.input` / `world.output` (new object, not
`values[:]`), call `refresh_cell_views()`. In-place updates from
`update_lpjml` do not need that.

## 🧩 Data on entities

```python
cell.to_earth["with_tillage"].values[...] = 0
yield_c = cell.from_earth["pft_harvestc"]
country_slice = country.output  # current world values, this country only
```

`country.cells` is the cell list. `cell.country` is the `Country` object
(or a string code if you skipped countries). Neighbourhoods
(`cell.neighbourhood`, `country.neighbourhood`) come from the graphs built
in `init_cells`.

`World.statistic` and `Country.statistic` are optional key/value caches
for your own aggregates. The library does not fill them.

## 📊 Outputs

Declare variables on the entity class. Only those names are collected.

```python
from pycopanlpjml.output import Output
from pycopancore.data_model.variable import Variable


class MyCell(lpjml.Cell):
    output_variables = Output(
        soilc=Variable("soil carbon", "topsoil SOC"),
        cropyield=Variable("crop yield", "area-weighted harvest"),
    )
```

In `coupled_config`:

```yaml
output:
  format: ["netcdf", "parquet", "csv"]
  cell:
    - soilc
    - cropyield
  country:
    - total_yield
  individual:
    - farmer_income
```

`format: []` turns collection off. Variable lists usually live in the
application config (e.g. InSEEDS), not in the library default
`pycopanlpjml/config.yaml`.

`collect_outputs(t)` appends a Zarr buffer. After the last year,
`finalize_output_streams()` or `run_simulation()` writes the requested
formats.

```python
world.output_array    # xarray Dataset (last collected year)
world.output_table    # long DataFrame
country.output_table  # this country's cells
cell.output_table     # this cell
```

## ▶️ Running a simulation

Use `run_simulation` when you want paths, optional profiling, and output
writing around your yearly loop:

```python
from pycopanlpjml.run import run_simulation

run_simulation(
    config_file="config.json",
    model_factory=MyModel(config_file=config_file),
    description="Coupled run",
)
```

## ⚙️ Configuration

Library defaults (`pycopanlpjml/config.yaml`) become
`pycopanlpjml_config` / parts of `coupled_config`:

```yaml
profiling: false
output:
  format: ["netcdf", "parquet", "csv"]
```

`profiling: true` writes a PyInstrument HTML file next to the run outputs
(`profiling_{timestamp}.html`; needs `pyinstrument`).

The LPJmL JSON from pycoupler still owns the grid, years and coupler
settings. `model.config` is that LPJmL config.

## 📚 Where next

| If you want… | Go to |
|---|---|
| Signatures and class members | [API reference](../api/index.rst) |
| A full coupled application | [InSEEDS](https://github.com/pik-copan/inseeds) |
| The framework paper | [Breier et al., 2026](https://doi.org/10.5194/gmd-19-6829-2026) |
| Entity types and process taxa | [Donges et al., 2020](https://doi.org/10.5194/esd-11-395-2020) |
