import pycopancore.model_components.base.implementation as base
from .mixin import AliasMixin

class Cell(base.Cell, AliasMixin):
    """An LPJmL-integrating cell entity.

    Cell entity type (mixin) class for copan:LPJmL component. It inherits the
    copan:CORE cell entity and structure and integrates LPJmL input and output
    data (attributes) as well as grid, country and area information as
    `pycoupler.LPJmLData` and `pycoupler.LPJmLDataSet` instances.
    A cell instance should hold (numpy) views of attributes to the
    corresponding cell in the copan:LPJmL world instance.

    Parameters
    ----------
    input : pycoupler.LPJmLDataSet
        Coupled LPJmL model input.
    output : pycoupler.LPJmLData
        Coupled LPJmL model output.
    grid : pycoupler.LPJmLData
        Grid of the LPJmL model.
    country : str
        Country of the cell as a country code.
    area : float
        Area of the cell in square meters.
    kwargs : dict, optional
        Additional keyword arguments.

    Returns
    -------
    Cell
        An instance of the copan:LPJmL Cell.

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

    Initialize a (first) cell instance
    >>> cell_id = 0
    >>> cell = Cell(
    ...     world=world,
    ...     input=world.input.isel(cell=cell_id),
    ...     output=world.output.isel(cell=cell_id),
    ...     grid=world.grid.isel(cell=cell_id),
    ...     country=world.country.isel(cell=cell_id),
    ...     area=world.area.isel(cell=cell_id)
    ... )

    >>> cell
    """

    _entity_alias = "cell"

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
