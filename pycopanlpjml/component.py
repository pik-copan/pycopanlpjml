"""Model mixin class to build copan:LPJmL models."""

import sys
import numpy as np
import xarray as xr
from pycoupler.coupler import LPJmLCoupler
from pycoupler.utils import get_countries
import networkx as nx


class Component:
    """An LPJmL-integrating mixin model component to build copan:LPJmL models.

    The model component initializes the LPJmL coupler and establishes a
    connection to the LPJmL model. It provides a method to initialize cell
    instances and an update method to exchange input and output data with LPJmL
    while updating the output in world and implicitly in the corresponding
    cells through (numpy) views.
    It acts as a mixin to enable integrated copan:LPJmL modeling and to be
    combined with further model components in a model class.

    Parameters
    ----------
    config_file : str
        File path to the integrated model configuration file.
    lpjml : LPJmLCoupler
        LPJmL coupler instance.
    lpjml_couplerversion : int
        LPJmL coupler version.
    lpjml_host : str
        Hostname of the LPJmL coupler.
    lpjml_port : int
        Port of the LPJmL coupler.
    kwargs : dict, optional
        Additional keyword arguments.

    Returns
    -------
    Component
        An instance of the LPJmL component.

    Examples
    --------

    >>> class StopFertilizationModel(lpjml.Component):

    ... name = "Model to simulate a stop global artificial fertilization."

    ... def __init__(self, stop_year, **kwargs):
    ...     super().__init__(**kwargs)

    ...     # initialize LPJmL world
    ...     self.world = lpjml.World(
    ...         input=self.lpjml.read_input(copy=False),
    ...         output=self.lpjml.read_historic_output(),
    ...         grid=self.lpjml.grid,
    ...         country=self.lpjml.country,
    ...     )

    ...     # initialize cells
    ...     self.init_cells(cell_class=lpjml.Cell)

    ... def stop_fertilization(self, t, stop_year):
    ...     if t == stop_year:
    ...         self.world.input.fertilization.values[:] = 0

    ... def update(self, t):
    ...     self.stop_fertilization(t)
    ...     self.update_lpjml(t)


    >>> model = StopFertilizationModel(
    ...     config_file="path/to/config_file.json",
    ...     stop_year=2025
    ... )

    >>> for year in model.lpjml.get_sim_years():
    ...     model.update(year)

    """

    def __init__(
        self,
        config_file=None,
        lpjml=None,
        lpjml_couplerversion=3,
        lpjml_host="localhost",
        lpjml_port=2042,
        **kwargs,
    ):

        super().__init__(**kwargs)

        if config_file is not None:
            # establish coupler connection to LPJmL
            self.lpjml = LPJmLCoupler(
                config_file=config_file,
                version=lpjml_couplerversion,
                host=lpjml_host,
                port=lpjml_port,
            )
        elif lpjml is not None:
            self.lpjml = lpjml
        else:
            raise ValueError("Either config_file or lpjml must be provided")

        self._countries_as_names()
        self.config = self.lpjml.config

    def _countries_as_names(self):
        """Convert country codes to names"""
        if (
            self.lpjml.config.coupled_config.lpjml_settings.country_code_to_name  # noqa
        ):  # noqa
            self.lpjml.code_to_name(
                self.lpjml.config.coupled_config.lpjml_settings.iso_country_code  # noqa
            )

    def init_countries(self, country_class, world_views=None, **kwargs):
        """Initialize country instances for each corresponding country."""
        countries = []
        breakpoint()
        country_names = _get_country_names()

        unique_countries = np.unique(self.world.country.values)
        for country in unique_countries:

            country_indices = np.where(self.world.country.values == country)[0]
            countries.append(country_class(
                name=country_names[country]['name'],
                code=country_names[country]['code'],
                world=self.world,
                input=self.world.input.isel(cell=country_indices, drop=False),
                output=self.world.output.isel(cell=country_indices, drop=False),  # noqa
                grid=self.world.grid.isel(cell=country_indices, drop=False),
                area=self.world.area.isel(cell=country_indices, drop=False),
                **(
                    {
                        view: getattr(self.world, view).isel(
                            cell=country_indices, drop=False
                        )
                        for view in world_views
                        if hasattr(self.world, view)
                    }
                    if world_views
                    else {}
                ),
                **kwargs,
            ))

    def init_cells(self, cell_class, world_views=None, **kwargs):
        """Initialize cell instances for each corresponding cell via numpy
            views.

        Parameters
        ----------
        cell_class : Cell
            Cell class to be instantiated for each cell.
        world_views : list, optional
            List of world attributes which are are of type xarray.Dataarray,
            xarray.DataSet, pycoupler.LPJmLData or pycoupler.LPJmLDataSet
            to generate cell views from, to access corresponding cell entity
            data.
        kwargs : dict, optional
            Additional keyword arguments for cell instances.

        """
        # https://docs.xarray.dev/en/stable/user-guide/indexing.html#copies-vs-views

        # Get neighbourhood of surrounding cells as matrix
        #   (cell, neighbour cells)
        neighbour_matrix = self.lpjml.grid.get_neighbourhood(id=False)

        # Create cell instances
        country_by_code = {
            country.code: country for country in self.world.countries
        }
        cells = [
            cell_class(
                world=self.world,
                input=self.world.input.isel(cell=icell, drop=False),
                output=self.world.output.isel(cell=icell, drop=False),
                grid=self.world.grid.isel(cell=icell, drop=False),
                area=(
                    self.world.area.isel(cell=icell, drop=False)
                    if hasattr(self.world, "area")
                    else None
                ),
                country=country_by_code.get(
                    self.world.country.isel(cell=icell, drop=False).item(), None
                ),
                **(
                    {
                        view: getattr(self.world, view).isel(cell=icell, drop=False)
                        for view in world_views
                        if hasattr(self.world, view)
                    }
                    if world_views
                    else {}
                ),
                **kwargs,
            )
            for icell in self.lpjml.get_cells(id=False)
        ]

        self._assign_cell_neighbourhood(cells, neighbour_matrix)
        self._assign_country_neighbourhood()

    def update_lpjml(self, t):
        """Exchange input and output data with LPJmL. Update output in world.
        Update corresponding time stamps in input and output attributes.

        Parameters
        ----------
        t : int
            Current time step (year) to exchange data with LPJmL.

        """

        # update input time values
        self.world.input.time.values[0] = np.datetime64(f"{t+1}-12-31")

        if not hasattr(sys, "_called_from_test"):
            # send input data to lpjml
            self.lpjml.send_input(self.world.input, t)

            # read output data from lpjml
            for name, output in self.lpjml.read_output(t).items():
                self.world.output[name].values[:] = (
                    xr.concat([self.world.output[name], output[:]], dim="time")
                    .drop_isel(time=0)
                    .values[:]
                )

            # update output time values
            self.world.output.time.values[:] = np.array(
                [
                    np.datetime64(f"{year}-12-31")
                    for year in range(
                        t + 1 - len(self.world.output.time), t + 1
                    )  # noqa
                ]
            )

            if t == self.lpjml.config.lastyear:
                self.lpjml.close()
        else:
            # only update output time values for testing
            self.world.output.time.values[:] = np.array(
                [
                    np.datetime64(f"{year}-12-31")
                    for year in range(
                        t + 1 - len(self.world.output.time), t + 1
                    )  # noqa
                ]
            )

    def _assign_country_neighbourhood(self):
        # Create a new graph for country neighbourhoods
        self.world.country_neighbourhood = nx.Graph()
        self.world.country_neighbourhood.add_nodes_from(self.world.countries)

        # Build edges between neighbouring countries based on cell neighbours
        for cell1, cell2 in self.world.cell_neighbourhood.edges:
            country1 = getattr(cell1, 'country', None)
            country2 = getattr(cell2, 'country', None)
            if (
                country1 and country2 and country1 is not country2
            ):
                self.world.country_neighbourhood.add_edge(country1, country2)

        # Assign neighbourhood attribute for each country
        for country in self.world.countries:
            country.neighbourhood = set(
                self.world.country_neighbourhood.neighbors(country)
            )

    def _assign_cell_neighbourhood(self, cells, neighbour_matrix):
        # Create the cell neighbourhood graph
        self.world.cell_neighbourhood = nx.Graph()
        self.world.cell_neighbourhood.add_nodes_from(cells)

        # Add edges between neighbouring cells
        for icell, cell in enumerate(cells):
            for neighbour in neighbour_matrix.isel(cell=icell).values:
                if neighbour >= 0:
                    self.world.cell_neighbourhood.add_edge(
                        cell, cells[neighbour]
                    )

        # Assign neighbourhood attribute for each cell (as a set)
        for cell in cells:
            cell.neighbourhood = set(self.world.cell_neighbourhood.neighbors(
                cell
            ))

        # Assign subgraph for each country
        for country in self.world.countries:
            country.cell_neighbourhood = self.world.cell_neighbourhood.subgraph(  # noqa
                country.cells
            ).copy()


def _get_country_names():
    return {
        value['code']: {
            'name': value['name'],
            'code': value['code']
        } for key, value in get_countries().items()
    }
