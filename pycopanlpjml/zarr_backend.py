"""Thin Zarr I/O helper for world-level state.

This module used to provide a complex view layer (ZarrDatasetView,
ZarrDataArrayView, etc.) so that World/Region/Country/Cell could work
directly on a shared Zarr store. The entity layer has been refactored to
operate purely on in-memory LPJmLData/xarray objects instead.

The remaining functionality here is intentionally minimal and focused on
bulk I/O only: initializing a Zarr store from xarray/LPJmLData{Set}
objects and (optionally) loading them back.
"""

from typing import Optional, Tuple

import xarray as xr

# Default chunk size for Zarr arrays (number of cells per chunk).
# Kept for backwards compatibility in places that still pass chunk_size.
DEFAULT_CHUNK_SIZE = 1000


class ZarrBackend:
    """Thin helper to write/read world-level state to/from a Zarr store.

    Parameters
    ----------
    store_path : str
        Path to the Zarr store on disk.
    overwrite : bool, default=False
        If True, any existing store at this path will be overwritten by
        :meth:`initialize_from_xarray`.
    """

    def __init__(self, store_path: str, overwrite: bool = False):
        self.store_path = store_path
        self.overwrite = overwrite

    # ------------------------------------------------------------------
    # Bulk initialization from xarray / LPJmLData{Set}
    # ------------------------------------------------------------------
    def initialize_from_xarray(
        self,
        input_ds: xr.Dataset,
        output_ds: xr.Dataset,
        grid: xr.DataArray,
        country: xr.DataArray,
        area: Optional[xr.DataArray] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Initialize a Zarr store from xarray/LPJmLData objects.

        This writes:

        - ``input_ds``  → group ``\"input\"``
        - ``output_ds`` → group ``\"output\"``
        - ``grid``      → variable ``\"grid\"`` in group ``\"meta\"``
        - ``country``   → variable ``\"country\"`` in group ``\"meta\"``
        - ``area``      → variable ``\"area\"`` in group ``\"meta\"`` (if provided)

        Notes
        -----
        The exact on-disk layout is not meant to be consumed directly by
        entities; it is a persistence format only. All model code should
        work on in-memory xarray / LPJmLData objects.
        """
        mode = "w" if self.overwrite else "a"

        # Chunk along cell dimension if present; otherwise let xarray decide.
        cell_dim = "cell" if "cell" in input_ds.dims else None
        if cell_dim:
            input_ds = input_ds.chunk({cell_dim: min(chunk_size, input_ds.dims[cell_dim])})
        input_ds.to_zarr(self.store_path, group="input", mode=mode, consolidated=False)

        cell_dim_out = "cell" if "cell" in output_ds.dims else None
        if cell_dim_out:
            output_ds = output_ds.chunk(
                {cell_dim_out: min(chunk_size, output_ds.dims[cell_dim_out])}
            )
        output_ds.to_zarr(self.store_path, group="output", mode="a", consolidated=False)

        meta = xr.Dataset()
        meta["grid"] = grid
        meta["country"] = country
        if area is not None:
            meta["area"] = area

        if "cell" in meta.dims:
            meta = meta.chunk({"cell": min(chunk_size, meta.dims["cell"])})

        meta.to_zarr(self.store_path, group="meta", mode="a", consolidated=False)

    # ------------------------------------------------------------------
    # Optional helpers to load world state back from Zarr
    # ------------------------------------------------------------------
    def load_world_state(self) -> Tuple[xr.Dataset, xr.Dataset, xr.DataArray, xr.DataArray, Optional[xr.DataArray]]:
        """Load world-level state from this Zarr store.

        Returns
        -------
        (input_ds, output_ds, grid_da, country_da, area_da)
            xarray.Dataset/DataArray objects corresponding to the groups
            written by :meth:`initialize_from_xarray`.
        """
        input_ds = xr.open_zarr(self.store_path, group="input", consolidated=False)
        output_ds = xr.open_zarr(self.store_path, group="output", consolidated=False)
        meta = xr.open_zarr(self.store_path, group="meta", consolidated=False)

        grid_da = meta["grid"]
        country_da = meta["country"]
        area_da = meta["area"] if "area" in meta.variables else None

        return input_ds, output_ds, grid_da, country_da, area_da


