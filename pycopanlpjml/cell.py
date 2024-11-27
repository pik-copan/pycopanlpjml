"""Cell entity for LPJmL coupling component."""

# This file is part of pycopancore.
#
# Copyright (C) 2022 by COPAN team at Potsdam Institute for Climate
# Impact Research
#
# URL: <http://www.pik-potsdam.de/copan/software>

import networkx as nx
import pycopancore.model_components.base.implementation as base


class Cell(base.Cell):
    """Cell entity type mixin class for LPJmL component. A cell
    instance holds numpy views of attributes to the corresponding cell in the
    LPJmL world instance.

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

    :return: An instance of the LPJmL Cell.
    :rtype: Cell
    """

    def __init__(
        self, input=None, output=None, grid=None, country=None, area=None, **kwargs
    ):
        """Initialize an instance of Cell."""
        super().__init__(**kwargs)

        self.input = input
        self.output = output
        self.neighbourhood = list()
        self.grid = grid
        if country is not None:
            self.country = country
        if area is not None:
            self.area = area
