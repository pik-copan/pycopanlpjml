=============
API reference
=============

The copan:LPJmL World-Earth Modeling (WEM) framework.


copan:LPJmL Model Component
===========================

The copan:LPJmL model component integrates the LPJmL land surface model
with copan:CORE, providing parallel country-level updates and output collection.

.. autosummary::
   :toctree: generated
   :caption: copan:LPJmL Model Component

   pycopanlpjml.ModelComponent


Entities
========

Entities representing the spatial and social hierarchy: World (global simulation
space), Cell (grid cell), Region/Country (social-territorial units).

.. autosummary::
   :toctree: generated
   :caption: Entities

   pycopanlpjml.World
   pycopanlpjml.Cell
   pycopanlpjml.Region
   pycopanlpjml.Country
   pycopancore.Individual
   pycopancore.Group


Output System
=============

Efficient batch collection and writing of model outputs to NetCDF, Parquet, and
CSV formats. Uses incremental Zarr storage during simulation for memory efficiency.

.. autosummary::
   :toctree: generated
   :caption: Output System

   pycopanlpjml.output.Output
   pycopanlpjml.output.OutputDefinitionMixin
   pycopanlpjml.output.OutputCollectionMixin
   pycopanlpjml.output.write_outputs_netcdf
   pycopanlpjml.output.write_outputs_parquet
   pycopanlpjml.output.write_outputs_csv
   pycopanlpjml.output.write_outputs_tables
   pycopanlpjml.output.collect_variable_metadata_from_model


Parallelization
===============

Parallel execution utilities for distributing country-level updates across
Dask workers. Automatically detects and configures parallel environments.

.. autosummary::
   :toctree: generated
   :caption: Parallelization

   pycopanlpjml.parallelization.ParallelConfig
   pycopanlpjml.parallelization.ParallelExecutor
   pycopanlpjml.parallelization.LocalDaskRuntime
   pycopanlpjml.parallelization.detect_parallel_environment
   pycopanlpjml.parallelization.get_executor
   pycopanlpjml.parallelization.start_local_dask_cluster
   pycopanlpjml.parallelization.configure_model_for_dask


Serialization
=============

Utilities for serializing and deserializing country and agent state between
the main process and parallel workers.

.. autosummary::
   :toctree: generated
   :caption: Serialization

   pycopanlpjml.serialization.serialize_country_for_worker
   pycopanlpjml.serialization.deserialize_country
   pycopanlpjml.serialization.sync_world
   pycopanlpjml.serialization.get_sync_attributes


Run Orchestration
=================

High-level functions for orchestrating complete coupled simulations including
parallel runtime management, profiling, and output writing.

.. autosummary::
   :toctree: generated
   :caption: Run Orchestration

   pycopanlpjml.run.run_simulation
   pycopanlpjml.run.build_run_context
   pycopanlpjml.run.RunContext
   pycopanlpjml.run.ProfilingOptions
   pycopanlpjml.run.load_profiling_options
   pycopanlpjml.run.detect_worker_target
   pycopanlpjml.run.ensure_single_instance


Mixins
======

Utility mixins for entity aliasing and other cross-cutting concerns.

.. autosummary::
   :toctree: generated
   :caption: Mixins

   pycopanlpjml.mixin.AliasMixin
   pycopanlpjml.mixin.pluralize


Data Handling
=============

Data array and metadata formats for handling LPJmL data (from pycoupler).

.. autosummary::
   :toctree: generated
   :caption: Data

   pycoupler.LPJmLData
   pycoupler.LPJmLDataSet
   pycoupler.read_data
   pycoupler.LPJmLMetaData
   pycoupler.read_meta
   pycoupler.read_header


Configuration
=============

Configure LPJmL simulations to run standalone or integrated with copan:CORE.

.. autosummary::
   :toctree: generated
   :caption: Configurations

   pycoupler.LpjmlConfig
   pycoupler.CoupledConfig
   pycoupler.read_config


Run Simulations
===============

Run LPJmL standalone and integrated copan:LPJmL simulations locally or on
HPC clusters (from pycoupler).

.. autosummary::
   :toctree: generated
   :caption: Simulations

   pycoupler.start_lpjml
   pycoupler.submit_lpjml
   pycoupler.check_lpjml






