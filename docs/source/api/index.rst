=============
API reference
=============

Public classes and functions of ``pycopanlpjml``, plus the pycoupler types
used to configure and exchange LPJmL data.


Model
=====

``Model`` opens the LPJmL coupler. Subclasses create ``World`` / ``Country`` /
``Cell`` and implement ``update(t)``.

.. autosummary::
   :toctree: generated
   :caption: Model

   pycopanlpjml.Model


Entities
========

World holds the LPJmL arrays. Cells store scalar views. Countries re-isel a
copy of the current world slice on each access.

.. autosummary::
   :toctree: generated
   :caption: Entities

   pycopanlpjml.World
   pycopanlpjml.Cell
   pycopanlpjml.Region
   pycopanlpjml.Country
   pycopanlpjml.WorldRegion
   pycopancore.Individual
   pycopancore.Group


Output
======

Declare variables with ``Output`` on entity classes. ``Model`` inherits
``OutputCollectionMixin`` (``collect_outputs``, ``finalize_output_streams``).

.. autosummary::
   :toctree: generated
   :caption: Output

   pycopanlpjml.output.Output
   pycopanlpjml.output.OutputDefinitionMixin
   pycopanlpjml.output.OutputCollectionMixin
   pycopanlpjml.output.write_outputs_netcdf
   pycopanlpjml.output.write_outputs_tables
   pycopanlpjml.output.write_outputs_parquet
   pycopanlpjml.output.write_outputs_csv
   pycopanlpjml.output.collect_variable_metadata_from_model
   pycopanlpjml.output.read_output_table_from_zarr


Run
===

Process wrapper around a model's yearly ``update``.

.. autosummary::
   :toctree: generated
   :caption: Run

   pycopanlpjml.run.run_simulation
   pycopanlpjml.run.build_run_context
   pycopanlpjml.run.RunContext
   pycopanlpjml.run.read_profiling
   pycopanlpjml.run.driver_profiler_session


Data handling (pycoupler)
=========================

Data array and metadata formats, plus reading functions, for LPJmL input and
output (``to_earth`` / ``from_earth`` on World and cells).

.. autosummary::
   :toctree: generated
   :caption: Data

   pycoupler.LPJmLData
   pycoupler.LPJmLDataSet
   pycoupler.read_data
   pycoupler.LPJmLMetaData
   pycoupler.read_meta
   pycoupler.read_header


Configure simulations (pycoupler)
=================================

Configure LPJmL to run standalone or coupled with a copan:CORE model.
``CoupledConfig`` is the pycopanlpjml / pycoupler block (ports, outputs,
``lpjml_settings``).

.. autosummary::
   :toctree: generated
   :caption: Configurations

   pycoupler.LpjmlConfig
   pycoupler.CoupledConfig
   pycoupler.read_config


Run simulations (pycoupler)
===========================

Start or submit LPJmL itself (locally or on an HPC cluster), and check a
config before a run. The social–Earth yearly loop is
``pycopanlpjml.run.run_simulation`` above.

.. autosummary::
   :toctree: generated
   :caption: Simulations

   pycoupler.start_lpjml
   pycoupler.submit_lpjml
   pycoupler.check_lpjml
