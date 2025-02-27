"""World entity type mixin class for copan:LPJmL component."""

import numpy as np
import networkx as nx
import pycopancore.model_components.base.implementation as base


class World(base.World):
    """An LPJmL-integrating world entity.

    World entity type (mixin) class for copan:LPJmL component. A world
    instance holds data attributes as pycoupler.LPJmLData or
    pycoupler.LPJmLDataSet that are received and send via the lpjml
    instance of the pycoupler.LPJmLCoupler class.

    Parameters
    ----------
    input : pycoupler.LPJmLDataSet
        Coupled LPJmL model inputs.
    output : pycoupler.LPJmLDataSet
        Coupled LPJmL model outputs.
    grid : pycoupler.LPJmLData
        Grid of the LPJmL model.
    country : pycoupler.LPJmLData
        Countries of each cell as country code.
    area : pycoupler.LPJmLData
        Area of each cell in square meters.
    kwargs : dict, optional
        Additional keyword arguments.

    Returns
    -------
    World
        An instance of the LPJmL World.


    Examples
    --------
    In this example, we will demonstate an exammplaric initialization of a
    `pycopanlpjml.Cell` instance independent of the `pycopanlpjml.Component`
    that automatizes the initializtion of all cells belonging to a world.

    A prerequisite is the start of an LPJmL simulation in coupled mode
    described in ...
    To connect to the LPJmL simulation we use the `pycoupler.LPJmLCoupler`
    class.

    >>> from pycoupler.coupler import LPJmLCoupler
    >>> from pycopanlpjml import World


    The configuration file is a json file that holds the configuration for
    the integrated copan:LPJmL model simulation.

    >>> config_file = "path/to/config_file.json"
    >>> lpjml = LPJmLCoupler(
    ...     config_file=config_file,
    ...     host="localhost",
    ...     port=2042,
    ... )

    Initialize LPJmL world, all data is read/send from and to the LPJmL model
    >>> world = World(
    ...     input=lpjml.read_input(copy=False),
    ...     output=lpjml.read_historic_output(),
    ...     grid=lpjml.grid,
    ...     country=lpjml.country,
    ... )

    """

    def __init__(
        self,
        input=None,
        output=None,
        grid=None,
        country=None,
        area=None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        # hold the input data for LPJmL
        if input is not None:
            self.input = input
            if self.model and self.model.lpjml:
                self.input.time.values[0] = np.datetime64(
                    f"{self.model.lpjml.sim_year}-12-31"
                )

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
