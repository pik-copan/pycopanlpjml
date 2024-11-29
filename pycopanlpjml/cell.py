import networkx as nx
import pycopancore.model_components.base.implementation as base


class Cell(base.Cell):
    """Cell entity type mixin class for LPJmL component. A cell
    instance holds numpy views of attributes to the corresponding cell in the
    LPJmL world instance.

    """

    def __init__(
        self, input=None, output=None, grid=None, country=None, area=None, **kwargs
    ):
        """Initialize an instance of Cell.

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
        super().__init__(**kwargs)

        # hold the input data for LPJmL on cell level
        if input is not None:
            self.input = input

        # hold the output data from LPJmL on cell level
        if output is not None:
            self.output = output

        # hold the grid information for each cell (lon, lat) from LPJmL on
        #   cell level
        if grid is not None:
            self.grid = grid
            # initialize the neighbourhood of the cell
            self.neighbourhood = list()

        # hold the country information (country code str) from LPJmL on
        #   cell level
        if country is not None:
            self.country = country

        # hold the area in m2 from LPJmL on cell level
        if area is not None:
            self.area = area
