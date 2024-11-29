import sys
import numpy as np
import networkx as nx
import pycopancore.model_components.base.implementation as base

from . import Cell


class World(base.World):
    """World entity type mixin class for LPJmL component. A world
    instance holds data attributes as pycoupler.LPJmLData or
    pycoupler.LPJmLDataSet that are received and send via the lpjml
    instance of the pycoupler.LPJmLCoupler class.
    """

    def __init__(
        self, input=None, output=None, grid=None, country=None, area=None, **kwargs
    ):
        """Initialize an instance of World.

        :param input: Coupled LPJmL model inputs
        :param output: pycoupler.LPJmLDataSet

        :param output: Coupled LPJmL model outputs
        :param output: pycoupler.LPJmLDataSet

        :param grid: Grid of the LPJmL model
        :param grid: pycoupler.LPJmLData

        :param country: Countries of each cell as country code
        :param country: pycoupler.LPJmLData

        :param area: Area of each cell in m^2
        :param area: pycoupler.LPJmLData

        :param kwargs: Additional keyword arguments.
        :type data_dict: dict

        :return: An instance of the LPJmL World.
        :rtype: World
        """
        super().__init__(**kwargs)

        # hold the input data for LPJmL
        if input is not None:
            self.input = input

        # hold the output data from LPJmL
        if output is not None:
            self.output = output

        # hold the grid information for each cell (lon, lat) from LPJmL
        if grid is not None:
            self.grid = grid
            # initialize the neighbourhood as networkx graph
            self.neighbourhood = nx.Graph()

        # hold the country information (country code str) from LPJmL
        if country is not None:
            self.country = country

        # hold the area in m2 from LPJmL
        if area is not None:
            self.area = area
