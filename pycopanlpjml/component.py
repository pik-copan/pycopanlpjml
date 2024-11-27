"""Model mixin class for the LPJmL coupling component."""

# This file is part of pycopancore.
#
# Copyright (C) 2022 by COPAN team at Potsdam Institute for Climate
# Impact Research
#
# URL: <http://www.pik-potsdam.de/copan/software>
# Contact: core@pik-potsdam.de
# License: BSD 2-clause license
import os
import sys
import numpy as np
from pycoupler.coupler import LPJmLCoupler

from . import documentation as doc

from . import Cell


class Component(doc.Component):
    """Mixin class for the LPJmL coupling component.

    :param config_file: File path to the configuration file.
    :type data_dict: str

    :param lpjml: LPJmL coupler instance.
    :type data_dict: LPJmLCoupler

    :param lpjml_couplerversion: LPJmL coupler version.
    :type data_dict: int

    :param lpjml_host: Hostname of the LPJmL coupler.
    :type data_dict: str

    :param lpjml_port: Port of the LPJmL coupler.
    :type data_dict: int

    :param kwargs: Additional keyword arguments.
    :type data_dict: dict

    :return: An instance of the LPJmL component.
    :rtype: Component
    """

    def __init__(
        self,
        config_file=None,
        lpjml=None,
        lpjml_couplerversion=3,
        lpjml_host="localhost",
        lpjml_port=2224,
        **kwargs,
    ):
        """Initialize an instance of World."""
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
            ValueError("Either config_file or lpjml must be provided")

        self.countries_as_names()
        self.config = self.lpjml.config

    def countries_as_names(self):
        """Convert country codes to names"""
        if self.lpjml.config.coupled_config.lpjml_settings.country_code_to_name:  # noqa
            self.lpjml.code_to_name(
                self.lpjml.config.coupled_config.lpjml_settings.iso_country_code  # noqa
            )

    def init_cells(self, **kwargs):
        """Init cell instances for each corresponding cell via numpy views

        :param kwargs: Additional keyword arguments for cell instances.
        :type kwargs: dict
        """
        # https://docs.xarray.dev/en/stable/user-guide/indexing.html#copies-vs-views

        # Get neighbourhood of surrounding cells as matrix
        #   (cell, neighbour cells)
        neighbour_matrix = self.lpjml.grid.get_neighbourhood(id=False)

        # Create cell instances
        cells = [
            Cell(
                world=self.world,
                input=self.world.input.isel(cell=icell),
                output=self.world.output.isel(cell=icell),
                grid=self.world.grid.isel(cell=icell),
                country=(
                    self.world.country.isel(cell=icell)
                    if hasattr(self.world, "country")
                    else None
                ),  # noqa
                area=(
                    self.world.terr_area.isel(cell=icell)
                    if hasattr(self.world, "terr_area")
                    else None
                ),  # noqa
                **kwargs,
            )
            for icell in self.lpjml.get_cells(id=False)
        ]
        # Build neighbourhood graph nodes from cells
        self.world.neighbourhood.add_nodes_from(cells)

        # Create neighbourhood graph edges from neighbour matrix
        for icell in self.lpjml.get_cells(id=False):
            for neighbour in neighbour_matrix.isel(cell=icell).values:
                if neighbour >= 0:  # Ignore negative values (-1 or NaN)
                    self.world.neighbourhood.add_edge(cells[icell], cells[neighbour])

        # Add neighbourhood subgraph for each cell
        for icell in self.lpjml.get_cells(id=False):
            cells[icell].neighbourhood = self.world.neighbourhood.neighbors(
                cells[icell]
            )

    def update_lpjml(self, t):
        """Exchange input and output data with LPJmL

        :param t: Current time step (year) to exchange data with LPJmL
        :type t: int
        """
        self.world.input.time.values[0] = np.datetime64(f"{t}-12-31")
        # send input data to lpjml
        if not hasattr(sys, "_called_from_test"):
            self.lpjml.send_input(self.world.input, t)
        # read output data from lpjml
        self.world.output.time.values[0] = np.datetime64(f"{t}-12-31")

        if not hasattr(sys, "_called_from_test"):
            for name, output in self.lpjml.read_output(t).items():
                self.world.output[name][:] = output[:]

            if t == self.lpjml.config.lastyear:
                self.lpjml.close()