import sys
import numpy as np
import networkx as nx
import pycopancore.model_components.base.implementation as base

from . import Cell


class World(base.World):
    """World entity type mixin class for LPJmL component. A world
    instance holds data attributes that are received and send via the lpjml
    instance of the pycoupler.LPJmLCoupler class.

    :param input: coupled LPJmL model input
    :param output: pycoupler.LPJmLDataSet

    :param output: coupled LPJmL model output
    :param output: pycoupler.LPJmLData

    :param grid: grid of the LPJmL model
    :param grid: pycoupler.LPJmLData

    :param country: country of the cell as country code
    :param country: str

    :param area: area of the cell in m^2
    :param area: float

    :param kwargs: Additional keyword arguments.
    :type data_dict: dict

    :return: An instance of the LPJmL World.
    :rtype: World
    """

    def __init__(
        self, input=None, output=None, grid=None, country=None, area=None, **kwargs
    ):
        """Initialize an instance of World."""
        super().__init__(**kwargs)

        if input is not None:
            self.input = input

        if output is not None:
            self.output = output

        if grid is not None:
            self.grid = grid
            self.neighbourhood = nx.Graph()

        if country is not None:
            self.country = country

        if area is not None:
            self.area = area
