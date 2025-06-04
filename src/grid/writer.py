import xarray as xr
import rioxarray

from pathlib import Path
import logging
from typing import Callable, Dict, List, Optional, Union, Any

from .utils import generate_preallocated_zarr_store, ensure_zarr_store_aligns

class GridWriter:
    def __init__(self, root):
        self.root = Path(root)

    def to_netcdf(self, ds: xr.Dataset, filename: str, **kwargs):
        """
        Write an xarray Dataset to a NetCDF file.

        Args:
            ds: The dataset to write.
            filename: The name of the output file.
            **kwargs: Additional arguments passed to ds.to_netcdf.
        """
        output_path = self.root / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ds.to_netcdf(output_path, **kwargs)
        logging.info(f"Data written to {output_path}")

    def to_geotiff(self, da: xr.DataArray | xr.Dataset, filename: str, **kwargs):
        """
        Write an xarray DataArray to a GeoTIFF file.

        Args:
            da: The data array to write. Must have spatial coordinates and CRS.
            filename: The name of the output file.
            **kwargs: Additional arguments passed to da.rio.to_raster.
        """

        if not hasattr(da, 'rio'):
            raise ValueError("DataArray must have spatial information (rioxarray extension) to save to geotiff.")

        if isinstance(da, xr.DataArray):
            da = da.to_dataset(name='')

        for nam, da in da.data_vars.items():
            output_path = Path(self.root, filename.replace('.tif', f'_{nam}.tif'))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            da.rio.to_raster(output_path, **kwargs)

        logging.info(f"Data written to {output_path}")

    def to_zarr(
        data,
        filename,
        append_dims=None,
        drop_attrs=False,
        **kwargs,
    ):
        """
        Inserts data into a Zarr store, either by appending to an existing store or creating a new one. To
        insert into an existing store, append_dims must be specified.
        Inserting of new variables to an existing store is currently not supported and will raise an error.
        If the zarr store already exists, the function will check if the new data is compatible with the
        existing store and raise an error if not.

        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The data to be inserted into the Zarr store.
        filename : str or Path
            The path to the Zarr store.
        append_dims : list of str, optional
            List of dimensions along which to append data in an existing zarr store.
        drop_attrs: boolean
            If true, all attributes of the dataset will be dropped before writing to zarr.
        **kwargs
            passed on to _ensure_preallocated_store

        Raises
        ------
        ValueError
            If the Zarr store does not exist and preallocate_attrs is not provided, or if preallocate_attrs is invalid.
        NotImplementedError
            If attempting to add new variables to an existing Zarr store.
        """

        if isinstance(data, xr.DataArray):
            if data.name is None:
                data.name = 'var'
            data = data.to_dataset()

        if isinstance(filename, str):
            filename = Path(filename)

        if isinstance(append_dims, str):
            append_dims = [append_dims]

        if filename.exists() and append_dims is not None:
            data_align = ensure_zarr_store_aligns(filename, data, append_dims = append_dims, **kwargs)

        if drop_attrs:
            data_align = data_align.drop_attrs()

        data_align.drop_vars('spatial_ref', errors = 'ignore').to_zarr(filename, mode = "a", region = "auto")