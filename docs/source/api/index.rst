=============
API reference
=============

The copan:LPJmL World-Earth Modeling (WEM) framework.


copan:LPJmL Component
=====================

The copan:LPJmL model component offers the integration of the LPJmL model
instance with copan:CORE at the model level.


.. autosummary::
   :toctree: generated
   :caption: copan:LPJmL Component

   pycopanlpjml.Component


Entities
========

World and Cell entities to integrate the LPJmL simulation space (World) and
the smallest spatial unit (Cell) with the copan:CORE entities.

.. autosummary::
   :toctree: generated
   :caption: Entities

   pycopanlpjml.World
   pycopanlpjml.Cell
   pycopancore.Individual
   pycopancore.Group


Data handling
=============

Data array and meta data formats as well as reading functions to handle LPJmL
data.

.. autosummary::
   :toctree: generated
   :caption: Data

   pycoupler.LPJmLData
   pycoupler.LPJmLDataSet
   pycoupler.read_data
   pycoupler.LPJmLMetaData
   pycoupler.read_meta
   pycoupler.read_header


Configure simulations
====================

Configure LPJmL simulations to run standalone or integrated with copan:CORE.

.. autosummary::
   :toctree: generated
   :caption: Configurations

   pycoupler.LpjmlConfig
   pycoupler.CoupledConfig
   pycoupler.read_config



Run simulations
===============

Run LPJmL standalone and integrated copan:LPJmL simulations locally or on
HPC clusters.

.. autosummary::
   :toctree: generated
   :caption: Simulations

   pycoupler.run_lpjml
   pycoupler.submit_lpjml
   pycoupler.check_lpjml






