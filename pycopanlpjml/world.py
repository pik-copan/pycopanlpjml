"""World entity type mixin class for copan:LPJmL component."""

import numpy as np
import networkx as nx
import pycopancore.model_components.base.implementation as base
from .mixin import AliasMixin
from .zarr_backend import ZarrBackend, DEFAULT_CHUNK_SIZE
import zarr


class World(base.World, AliasMixin):
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
    country_code : pycoupler.LPJmLData
        Country code of each cell (ISO 3-letter codes).
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
    ...     country_code=lpjml.country,  # ISO 3-letter country codes per cell
    ... )

    """

    def __init__(
        self,
        input=None,
        output=None,
        grid=None,
        country_code=None,
        area=None,
        chunk_size=DEFAULT_CHUNK_SIZE,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.chunk_size = chunk_size
        self.cell_neighbourhood = nx.Graph()
        self.country_neighbourhood = nx.Graph()

        # Initialize Zarr backend as single source of truth
        # Determine store path
        if hasattr(self, "model") and hasattr(self.model, "config"):
            store_path = f"{self.model.config.sim_path}/world_data.zarr"
            self._is_temp_store = False
        else:
            # Fallback path if model not yet initialized
            # Use a unique temp location per World instance to avoid test interference
            import tempfile
            import os
            import uuid

            unique_id = str(uuid.uuid4())[:8]
            store_path = os.path.join(
                tempfile.gettempdir(), f"world_data_{unique_id}.zarr"
            )
            self._is_temp_store = True  # Mark for cleanup

        self._zarr_backend = ZarrBackend(store_path, overwrite=True)
        self._zarr_store_path = store_path

        # Register cleanup for temporary stores (backup if context manager not used)
        if self._is_temp_store:
            import atexit

            atexit.register(self.cleanup_zarr_store)

        # Initialize backend with xarray data
        if all(
            [
                input is not None,
                output is not None,
                grid is not None,
                country_code is not None,
            ]
        ):
            self._zarr_backend.initialize_from_xarray(
                input_ds=input,
                output_ds=output,
                grid=grid,
                country=country_code,
                area=area,
                chunk_size=self.chunk_size,
            )

            # Update time if model is available
            if hasattr(self, "model") and hasattr(self.model, "lpjml"):
                input_view = self._zarr_backend.get_view("input")
                time_values = input_view.coords.get("time")
                if time_values is not None:
                    # Set initial time
                    self._zarr_backend.root["input"].attrs["initial_time"] = (
                        str(
                            np.datetime64(f"{self.model.lpjml.sim_year}-12-31")
                        )
                    )

    @property
    def input(self):
        """Get input dataset view.

        Returns a ZarrDatasetView that provides xarray-like interface
        to the input data. Changes made through this view are immediately
        synchronized with all Country and Cell instances.

        For full xarray/LPJmLDataSet compatibility, use .to_xarray()
        """
        return self._zarr_backend.get_view("input")

    @input.setter
    def input(self, value):
        """Set input dataset values.

        Note: This writes to the underlying Zarr store, so changes
        are immediately visible to all views.
        """
        # If value is xarray Dataset, update all variables
        if hasattr(value, "data_vars"):
            for var_name, var_data in value.data_vars.items():
                self._zarr_backend.root["input"][var_name][:] = var_data.values
        else:
            raise ValueError(
                "Input must be an xarray Dataset or ZarrDatasetView"
            )

    @property
    def output(self):
        """Get output dataset view."""
        return self._zarr_backend.get_view("output")

    @output.setter
    def output(self, value):
        """Set output dataset values."""
        if hasattr(value, "data_vars"):
            for var_name, var_data in value.data_vars.items():
                self._zarr_backend.root["output"][var_name][
                    :
                ] = var_data.values
        else:
            raise ValueError(
                "Output must be an xarray Dataset or ZarrDatasetView"
            )

    @property
    def grid(self):
        """Get grid data array view."""
        return self._zarr_backend.get_array_view("grid")

    @grid.setter
    def grid(self, value):
        """Set grid values."""
        if hasattr(value, "values"):
            self._zarr_backend.root["grid"][:] = value.values
        else:
            self._zarr_backend.root["grid"][:] = value

    @property
    def country_code(self):
        """Get country code data array view.

        Returns a ZarrDataArrayView for the country code of each cell (ISO 3-letter code).

        Note: Use `world.countries` to access the Country entity instances.
        """
        if "country" in self._zarr_backend.root:
            return self._zarr_backend.get_array_view("country")
        return None

    @country_code.setter
    def country_code(self, value):
        """Set country code values."""
        if hasattr(value, "values"):
            self._zarr_backend.root["country"][:] = value.values
        else:
            self._zarr_backend.root["country"][:] = value

    @property
    def area(self):
        """Get area data array view."""
        if "area" in self._zarr_backend.root:
            return self._zarr_backend.get_array_view("area")
        return None

    @area.setter
    def area(self, value):
        """Set area values."""
        if hasattr(value, "values"):
            self._zarr_backend.root["area"][:] = value.values
        else:
            self._zarr_backend.root["area"][:] = value

    def cleanup_zarr_store(self):
        """
        Clean up the Zarr store, especially if it's a temporary store.

        This method should be called when the World instance is no longer needed,
        especially in testing or when using temporary stores.
        """
        if hasattr(self, "_is_temp_store") and self._is_temp_store:
            import shutil
            import os

            if hasattr(self, "_zarr_store_path") and os.path.exists(
                self._zarr_store_path
            ):
                try:
                    shutil.rmtree(self._zarr_store_path)
                except Exception as e:
                    # Don't fail if cleanup fails (e.g., permissions)
                    import warnings

                    warnings.warn(
                        f"Failed to clean up temporary Zarr store {self._zarr_store_path}: {e}"
                    )

    def __enter__(self):
        """Context manager entry - allows using World with 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically cleans up temporary stores."""
        self.cleanup_zarr_store()
        return False  # Don't suppress exceptions
