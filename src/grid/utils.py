"""Utility functions for managing Zarr stores for geospatial data.

This module provides functions for creating, validating, and working with
Zarr stores that contain geospatial data using xarray and rioxarray.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any, List

import numpy as np
import xarray as xr
import rioxarray

logger = logging.getLogger(__name__)


##TODO: Check if dtypes between old and new data corresponds
def ensure_zarr_store_aligns(
    path: Union[str, Path], 
    data: Union[xr.Dataset, xr.DataArray], 
    append_dims: Optional[List[str]] = None
) -> None:
    """
    Ensures that an existing Zarr store is correctly preallocated and aligns with new data for appending.
    
    Parameters
    ----------
    path : Path or str
        The path to the Zarr store.
    data : xr.Dataset or xr.DataArray
        The new data to be appended.
    append_dims : List[str], optional
        List of dimensions along which data will be appended. Default is None (empty list).

    Raises
    ------
    ValueError
        If the existing dataset dimensions or chunks do not match the expected dimensions or chunks,
        or if not all values of a coordinate in append_dims are present in the existing dataset.
    NotImplementedError
        If attempting to add new variables to an existing Zarr store or if data type is unsupported.
    ValueError
        If the zarr store does not exist.
        
    Returns
    -------
    xr.Dataset
        The validated dataset ready for appending
    """

    if not isinstance(data, (xr.DataArray, xr.Dataset)):
        raise NotImplementedError(f"Unsupported data type {type(data)} for Zarr store operations.")

    if isinstance(data, xr.DataArray):
        data = data.to_dataset(name="variable")
        
    if append_dims is None:
        append_dims = []

    if not Path(path).exists():
        raise ValueError(f"Zarr store at {path} does not exist.")
        
    ds_existing = xr.open_zarr(path, decode_coords = 'all')
    
    if ds_existing.rio.crs is None:
        logger.warning("CRS for the existing zarr store is not set.")
    elif ds_existing.rio.crs != data.rio.crs:
        raise ValueError(
            f"Coordinate systems of existing store and data are not equal! "
            f"Got {ds_existing.rio.crs} vs {data.rio.crs}"
        )

    for vname in list(data.keys()):
        if vname not in ds_existing:
            raise NotImplementedError(
                f"Found existing zarr store but variable '{vname}' is not present. "
                f"Adding new variables is not supported."
            )
            
        arr = ds_existing[vname]
        existing_dims = {i: j for i, j in arr.sizes.items() if i not in append_dims}
        new_dims = {i: j for i, j in data[vname].sizes.items() if i not in append_dims}

        # Check if dimension names and shapes match
        if existing_dims != new_dims:
            raise ValueError(
                f"Existing dataset dimensions and shapes {existing_dims} "
                f"do not match expected {new_dims}."
            )

        # Check if dimension values are equal
        dims_unequal = {i: not np.array_equal(data[i].values, arr[i].values) for i in existing_dims.keys()}
        if any(dims_unequal.values()):
            raise ValueError(
                f"Cannot insert in zarr store if dimensions are not equal. "
                f"Unequal values found for dimensions {dims_unequal}."
            )

        # Check if chunking is equal
        if arr.chunks and data[vname].chunks and arr.chunks != data[vname].chunks:
            raise ValueError(
                f"Existing dataset chunks {arr.chunks} do not match expected {data[vname].chunks}."
            )

        # Check if all values of append_dim are present in existing dataset
        for i in append_dims:
            if not data[i].isin(arr[i]).all():
                raise ValueError(
                    f"Not all values of append_dim {i} are present in existing dataset. "
                    f"Writing to a zarr region slice requires that no dimensions or metadata are changed by the write."
                )

        if arr.dtype != data[vname].dtype:
            raise ValueError(f"Dtypes between old and new variable '{vname}' do not match. Got {arr.dtype} and {data[vname].dtype}.")

    logger.debug("Existing variable in zarr store is correctly aligned with new data.")


def assert_spatial_info(da: xr.DataArray) -> bool:
    """
    Check if a DataArray has proper spatial information.
    
    Args:
        da: xarray.DataArray to check
        
    Returns:
        bool: True if spatial info is complete
        
    Raises:
        ValueError: If spatial dimensions are not set or CRS is missing
    """
    try:
        # Check if spatial dimensions are set        # Just accessing these properties will raise an exception if not set        da.rio.x_dim
        da.rio.y_dim
    except rioxarray.exceptions.MissingSpatialDimensionError as exc:
        raise ValueError(
            'MissingSpatialDimensionError. Use "da.rio.set_spatial_dims(x_dim="x", y_dim="y")" to set the dimensions.'
        ) from exc

    # Check if CRS is set
    if da.rio.crs is None:
        raise ValueError('CRS not set. Use "da.rio.write_crs()" to set the CRS.')

    return True