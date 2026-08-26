# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [2.0.0] - 2026-08-26

Breaking release versus 1.x (`Component` + `World` + `Cell`). Models now
use a World → Country/Region → Cell hierarchy, a dedicated output system,
and `run_simulation`.

### Added

- `Model` as the coupler base class: `init_countries`, `init_cells`,
  `update_countries`, `update_lpjml`, `refresh_cell_views`,
  `collect_outputs`.
- `Country`, `Region`, and `WorldRegion` (copan:CORE social systems).
- Cell and country neighbourhood graphs from the LPJmL grid.
- Output system: `Output`, `OutputDefinitionMixin`,
  `OutputCollectionMixin`, incremental Zarr, writers for NetCDF, Parquet,
  and CSV.
- `run_simulation` and `RunContext` (paths, optional PyInstrument
  profiling, finalize outputs after the last year).
- User guide (copan:CORE entities, yearly coupling, country vs no-country
  `update`), API reference, and paper citations
  ([Donges et al., 2020](https://doi.org/10.5194/esd-11-395-2020),
  [Breier et al., 2026](https://doi.org/10.5194/gmd-19-6829-2026)).

### Changed

- Subclass `Model` instead of mixing in `Component`. The coupler still
  opens in `__init__`; the subclass creates `World` / entities and
  implements `update(t)`. `update_lpjml` is unchanged in role.
- With countries, call `update_countries(t)` from `update(t)`. Without
  countries, write the social step in `update(t)` yourself (there is no
  `update_world`).
- `World(...)` takes `country_code=` (ISO array). The 1.x `country=`
  argument is gone.
- World still owns the LPJmL arrays; cells are scalar `isel` views
  (shared memory), as in 1.x. Countries re-isel on access (copy: gapped
  cell lists cannot share a numpy view). Writes that LPJmL must see go
  through `cell` or `world`.
- Default coupler port is `2224` (was `2042`).
- `to_earth` / `from_earth` remain aliases of `input` / `output`.

### Deprecated

### Removed

- `pycopanlpjml.Component` — use `Model`.

### Fixed

### Security

## [1.1.9] - 2026-07

Last 1.x release on `main`: `Component`, `World`, and `Cell` with
world↔cell numpy views. No country entity, output collection, or
`run_simulation`.

## [1.1.0] - 2025-01

### Added

- Documentation and integration examples.

## [1.0.7] - 2025-01

### Fixed

- Stability fixes.

## [1.0.0] - 2025-01

### Added

- Initial release (`Component`, `World`, `Cell`, LPJmL coupler).
