"""Test the LPJmLCoupler class."""

import os
import datetime
import numpy as np
import pytest
from unittest.mock import patch
from pycoupler.config import read_config
from pycoupler.coupler import LPJmLCoupler

import pycopanlpjml as lpjml
from .conftest import get_test_path


class Model(lpjml.Component):
    """Test class representing the model."""

    name = "Test LPJmL coupled model component"

    def __init__(self, **kwargs):
        """Initialize an instance of World."""
        super().__init__(**kwargs)

        # initialize LPJmL world
        self.world = lpjml.World(
            input=self.lpjml.read_input(copy=False),
            output=self.lpjml.read_historic_output(),
            grid=self.lpjml.grid,
            country=self.lpjml.country,
        )

        # initialize cells
        self.init_cells(cell_class=lpjml.Cell)

    def update(self, t):
        self.update_lpjml(t)


@patch.dict(
    os.environ, {"TEST_PATH": get_test_path(), "TEST_LINE_COUNTER": "0"}
)  # noqa
def test_lpjml_component(test_path):
    """Test the LPJmLCoupler class."""

    lpjml_config = f"{test_path}/data/config_coupled_test.json"
    model = Model(config_file=lpjml_config)

    # test LPJmL coupling attributes
    assert model.lpjml.version == 3
    assert model.lpjml.sim_year == 2023
    assert all(
        [year for year in model.lpjml.get_sim_years()] == np.arange(2023, 2051)
    )  # noqa
    assert model.lpjml.ncell == 2

    # test LPJmL data attributes
    expected_input_dict = {
        "coords": {
            "lat": {
                "dims": ("cell",),
                "attrs": {
                    "units": "degrees_north",
                    "long_name": "Latitude",
                    "standard_name": "latitude",
                    "axis": "Y",
                },
                "data": [51.25, 51.75],
            },
            "lon": {
                "dims": ("cell",),
                "attrs": {
                    "units": "degrees_east",
                    "long_name": "Longitude",
                    "standard_name": "longitude",
                    "axis": "X",
                },
                "data": [6.75, 6.75],
            },
            "band (with_tillage)": {
                "dims": ("band",),
                "attrs": {},
                "data": ["1"],
            },
            "time": {
                "dims": ("time",),
                "attrs": {},
                "data": [datetime.datetime(2010, 12, 31, 0, 0)],
            },
            "cell": {"dims": ("cell",), "attrs": {}, "data": [27410, 27411]},
        },
        "attrs": {},
        "dims": {"cell": 2, "band": 1, "time": 1},
        "data_vars": {
            "with_tillage": {
                "dims": ("cell", "band", "time"),
                "attrs": {"missing_value": -999999, "_FillValue": -999999},
                "data": [[[-999999]], [[-999999]]],
            }
        },
    }
    assert expected_input_dict == model.world.input.to_dict()

    expected_output_dict = {
        "coords": {
            "cell": {"dims": ("cell",), "attrs": {}, "data": [27410, 27411]},
            "lon": {"dims": ("cell",), "attrs": {}, "data": [7.75, 7.75]},
            "lat": {"dims": ("cell",), "attrs": {}, "data": [51.25, 51.75]},
            "band (hdate)": {
                "dims": ("band",),
                "attrs": {},
                "data": [
                    "rainfed temperate cereals",
                    "rainfed rice",
                    "rainfed maize",
                    "rainfed tropical cereals",
                    "rainfed pulses",
                    "rainfed temperate roots",
                    "rainfed tropical roots",
                    "rainfed oil crops sunflower",
                    "rainfed oil crops soybean",
                    "rainfed oil crops groundnut",
                    "rainfed oil crops rapeseed",
                    "rainfed sugarcane",
                    "irrigated temperate cereals",
                    "irrigated rice",
                    "irrigated maize",
                    "irrigated tropical cereals",
                    "irrigated pulses",
                    "irrigated temperate roots",
                    "irrigated tropical roots",
                    "irrigated oil crops sunflower",
                    "irrigated oil crops soybean",
                    "irrigated oil crops groundnut",
                    "irrigated oil crops rapeseed",
                    "irrigated sugarcane",
                ],
            },
            "time": {
                "dims": ("time",),
                "attrs": {},
                "data": [datetime.datetime(2022, 12, 31, 0, 0)],
            },
            "band (pft_harvestc)": {
                "dims": ("band",),
                "attrs": {},
                "data": [
                    "rainfed temperate cereals",
                    "rainfed rice",
                    "rainfed maize",
                    "rainfed tropical cereals",
                    "rainfed pulses",
                    "rainfed temperate roots",
                    "rainfed tropical roots",
                    "rainfed oil crops sunflower",
                    "rainfed oil crops soybean",
                    "rainfed oil crops groundnut",
                    "rainfed oil crops rapeseed",
                    "rainfed sugarcane",
                    "rainfed others",
                    "rainfed grassland",
                    "rainfed biomass grass",
                    "rainfed biomass tree",
                    "irrigated temperate cereals",
                    "irrigated rice",
                    "irrigated maize",
                    "irrigated tropical cereals",
                    "irrigated pulses",
                    "irrigated temperate roots",
                    "irrigated tropical roots",
                    "irrigated oil crops sunflower",
                    "irrigated oil crops soybean",
                    "irrigated oil crops groundnut",
                    "irrigated oil crops rapeseed",
                    "irrigated sugarcane",
                    "irrigated others",
                    "irrigated grassland",
                    "irrigated biomass grass",
                    "irrigated biomass tree",
                ],
            },
            "band (soilc_agr_layer)": {
                "dims": ("band",),
                "attrs": {},
                "data": [200.0, 500.0, 1000.0, 2000.0, 3000.0],
            },
            "band (cftfrac)": {
                "dims": ("band",),
                "attrs": {},
                "data": [
                    "rainfed temperate cereals",
                    "rainfed rice",
                    "rainfed maize",
                    "rainfed tropical cereals",
                    "rainfed pulses",
                    "rainfed temperate roots",
                    "rainfed tropical roots",
                    "rainfed oil crops sunflower",
                    "rainfed oil crops soybean",
                    "rainfed oil crops groundnut",
                    "rainfed oil crops rapeseed",
                    "rainfed sugarcane",
                    "rainfed others",
                    "rainfed grassland",
                    "rainfed biomass grass",
                    "rainfed biomass tree",
                    "irrigated temperate cereals",
                    "irrigated rice",
                    "irrigated maize",
                    "irrigated tropical cereals",
                    "irrigated pulses",
                    "irrigated temperate roots",
                    "irrigated tropical roots",
                    "irrigated oil crops sunflower",
                    "irrigated oil crops soybean",
                    "irrigated oil crops groundnut",
                    "irrigated oil crops rapeseed",
                    "irrigated sugarcane",
                    "irrigated others",
                    "irrigated grassland",
                    "irrigated biomass grass",
                    "irrigated biomass tree",
                ],
            },
        },
        "attrs": {
            "source": "LPJmL C Version 5.8.1",
            "history": "/p/projects/open/Jannes/copan_core/lpjml/LPJmL_internal/bin/lpjml /p/projects/open/Jannes/copan_core/lpjml/config_coupled_test.json",  # noqa
            "cellsize": 0.5,
        },
        "dims": {
            "cell": 2,
            "band (hdate)": 24,
            "time": 1,
            "band (pft_harvestc)": 32,
            "band (soilc_agr_layer)": 5,
            "band (cftfrac)": 32,
        },
        "data_vars": {
            "hdate": {
                "dims": ("cell", "band", "time"),
                "attrs": {
                    "standard_name": "hdate",
                    "long_name": "harvesting date",
                    "units": "",
                    "source": "LPJmL C Version 5.8.1",
                    "history": "/p/projects/open/Jannes/copan_core/lpjml/LPJmL_internal/bin/lpjml /p/projects/open/Jannes/copan_core/lpjml/config_coupled_test.json",  # noqa
                    "cellsize": 0.5,
                },
                "data": [
                    [
                        [217],
                        [0],
                        [26],
                        [0],
                        [271],
                        [13],
                        [0],
                        [17],
                        [17],
                        [0],
                        [217],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                    ],
                    [
                        [203],
                        [0],
                        [300],
                        [0],
                        [255],
                        [9],
                        [0],
                        [12],
                        [12],
                        [0],
                        [221],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                    ],
                ],
            },
            "pft_harvestc": {
                "dims": ("cell", "band", "time"),
                "attrs": {
                    "standard_name": "pft_harvestc",
                    "long_name": "harvested carbon excluding residuals",
                    "units": "gC/m2/yr",
                    "source": "LPJmL C Version 5.8.1",
                    "history": "/p/projects/open/Jannes/copan_core/lpjml/LPJmL_internal/bin/lpjml /p/projects/open/Jannes/copan_core/lpjml/config_coupled_test.json",  # noqa
                    "cellsize": 0.5,
                },
                "data": [
                    [
                        [238.27952575683594],
                        [0.0],
                        [100.90016174316406],
                        [0.0],
                        [276.4978332519531],
                        [333.13970947265625],
                        [0.0],
                        [54.27776336669922],
                        [18.48753547668457],
                        [0.0],
                        [154.17788696289062],
                        [0.0],
                        [228.88429260253906],
                        [101.40410614013672],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                    ],
                    [
                        [291.4123840332031],
                        [0.0],
                        [355.4585266113281],
                        [0.0],
                        [266.79632568359375],
                        [394.00311279296875],
                        [0.0],
                        [65.5930404663086],
                        [56.49080276489258],
                        [0.0],
                        [164.47923278808594],
                        [0.0],
                        [267.38336181640625],
                        [176.49566650390625],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                    ],
                ],
            },
            "soilc_agr_layer": {
                "dims": ("cell", "band", "time"),
                "attrs": {
                    "standard_name": "soilc_agr_layer",
                    "long_name": "total soil carbon density agricultural stands in layer",  # noqa
                    "units": "gC/m2",
                    "source": "LPJmL C Version 5.8.1",
                    "history": "/p/projects/open/Jannes/copan_core/lpjml/LPJmL_internal/bin/lpjml /p/projects/open/Jannes/copan_core/lpjml/config_coupled_test.json",  # noqa
                    "cellsize": 0.5,
                },
                "data": [
                    [
                        [3214.2353515625],
                        [1453.852294921875],
                        [1227.374755859375],
                        [1433.9185791015625],
                        [1304.71337890625],
                    ],
                    [
                        [3904.43359375],
                        [1778.8968505859375],
                        [1463.5445556640625],
                        [1711.4876708984375],
                        [1575.8492431640625],
                    ],
                ],
            },
            "cftfrac": {
                "dims": ("cell", "band", "time"),
                "attrs": {
                    "standard_name": "cftfrac",
                    "long_name": "CFT fraction",
                    "units": "",
                    "source": "LPJmL C Version 5.8.1",
                    "history": "/p/projects/open/Jannes/copan_core/lpjml/LPJmL_internal/bin/lpjml /p/projects/open/Jannes/copan_core/lpjml/config_coupled_test.json",  # noqa
                    "cellsize": 0.5,
                },
                "data": [
                    [
                        [0.08206391334533691],
                        [0.0],
                        [0.018817877396941185],
                        [0.0],
                        [0.0021041869185864925],
                        [0.0053836628794670105],
                        [0.0],
                        [4.38910246884916e-05],
                        [0.0002015677746385336],
                        [0.0],
                        [0.016676178202033043],
                        [0.0],
                        [0.03566932678222656],
                        [0.037834376096725464],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                    ],
                    [
                        [0.19205573201179504],
                        [0.0],
                        [0.10328730195760727],
                        [0.0],
                        [0.006376297678798437],
                        [0.008108393289148808],
                        [0.0],
                        [4.185701618553139e-05],
                        [0.000625348649919033],
                        [0.0],
                        [0.022904807701706886],
                        [0.0],
                        [0.10959978401660919],
                        [0.12501367926597595],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                    ],
                ],
            },
        },
    }
    assert expected_output_dict == model.world.output.to_dict()

    expected_grid_dict = {
        "dims": ("cell", "band"),
        "attrs": {
            "standard_name": "grid",
            "long_name": "coordinates",
            "units": "degree",
            "source": "LPJmL C Version 5.8.1",
            "history": "/p/projects/open/Jannes/copan_core/lpjml/LPJmL_internal/bin/lpjml /p/projects/open/Jannes/copan_core/lpjml/config_coupled_test.json",  # noqa
            "cellsize": 0.5,
        },
        "data": [[7.75, 51.25], [7.75, 51.75]],
        "coords": {
            "cell": {"dims": ("cell",), "attrs": {}, "data": [27410, 27411]},
            "band": {"dims": ("band",), "attrs": {}, "data": ["lon", "lat"]},
            "lon": {"dims": ("cell",), "attrs": {}, "data": [7.75, 7.75]},
            "lat": {"dims": ("cell",), "attrs": {}, "data": [51.25, 51.75]},
        },
        "name": "grid",
    }
    assert expected_grid_dict == model.world.grid.to_dict()

    expected_country_dict = {
        "dims": ("cell", "band"),
        "attrs": {
            "standard_name": "country",
            "long_name": "country iso alpha-3 code",
            "units": "",
            "source": "LPJmL C Version 5.8.1",
            "history": "/p/projects/open/Jannes/copan_core/lpjml/LPJmL_internal/bin/lpjml /p/projects/open/Jannes/copan_core/lpjml/config_coupled_test.json",  # noqa
            "cellsize": 0.5,
        },
        "data": [["DEU"], ["DEU"]],
        "coords": {
            "cell": {"dims": ("cell",), "attrs": {}, "data": [27410, 27411]},
            "band": {"dims": ("band",), "attrs": {}, "data": [0]},
            "lon": {"dims": ("cell",), "attrs": {}, "data": [7.75, 7.75]},
            "lat": {"dims": ("cell",), "attrs": {}, "data": [51.25, 51.75]},
        },
        "name": "country",
    }
    assert expected_country_dict == model.world.country.to_dict()


@patch.dict(
    os.environ, {"TEST_PATH": get_test_path(), "TEST_LINE_COUNTER": "0"}
)  # noqa
def test_run_model(test_path):
    """Test the LPJmLCoupler class."""

    lpjml_config = f"{test_path}/data/config_coupled_test.json"
    model = Model(config_file=lpjml_config)

    for year in model.lpjml.get_sim_years():
        model.update(year)

    last_year = (
        model.world.output.time.values[0]
        .astype("datetime64[Y]")
        .astype(int)
        .item()  # noqa
        + 1970
    )

    # last year set to 2030 in test data set
    assert last_year == 2050
