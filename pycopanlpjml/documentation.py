"""Interface for lpjml coupler on CORE-side."""

# This file is part of pycopancore.
#
# Copyright (C) 2022 by COPAN team at Potsdam Institute for Climate
# Impact Research
#
# URL: <http://www.pik-potsdam.de/copan/software>

from pycoupler.data import LPJmLData, LPJmLDataSet
from pycoupler.coupler import LPJmLCoupler
from networkx import Graph


from pycopancore.data_model.variable import Variable


class Component:
    """Interface for Model mixin."""

    # metadata:
    name = "LPJmL based model component"
    description = "model component implementing the bidirectional coupling of\
                    copan:CORE to lpjml"
    requires = []
    """list of other model components required for this model component to
    make sense"""
