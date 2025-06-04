import xarray as xr
import rioxarray
import numpy as np


def generate_preallocated_zarr_store(
    filename,
    shape,
    coords,
    chunks: tuple = None,
    crs: int = None,
    encoding=None,
    vars: dict[str, type] = None,
):

    if vars is None:
        vars = {"variable": float}

    # Create dummy dataset to preallocate a zarr store with necessary metadata but no data
    dummy = xr.DataArray(np.empty(shape), coords=coords)

    if chunks is not None:
        dummy = dummy.chunk(chunks)
        
    dummy = dummy.expand_dims({'var': list(vars.keys())}).to_dataset(dim='var')

    if crs is not None:
        dummy = dummy.rio.write_crs(crs)

    for v, dtype in vars.items():
        dummy[v] = dummy[v].astype(dtype)

    dummy.to_zarr(filename, mode="w", compute=False, encoding=encoding)


def ensure_zarr_store_aligns():
    """
    Ensures that an existing Zarr store is correctly preallocated and aligns with ds_new for appending data.
    If align is true, ds_new will be reprojected to match the coordinate values of the existing zarr store.

    Parameters
    ----------
    zarr_path : Path
        The path to the Zarr store.
    ds_new : xr.Dataset
        The new data to be appended.
    append_dims : list of str
        List of dimensions along which data will be appended.
    align: boolean
        Boolean indicating if ds_new should be reprojected to match coordinate values in existing zarr store
    crs_existing: pyproj.CRS
        If the existing zarr store has no crs, this crs will be set.
    **kwargs
        Passed on to alignment.align_arrays

    Raises
    ------
    ValueError
        If the existing dataset dimensions or chunks do not match the expected dimensions or chunks,
        or if not all values of a coordinate in append_dims are present in the existing dataset.
    NotImplementedError
        If attempting to add new variables to an existing Zarr store.
    ValueError
        If the zarr store does not exist.
    """
    if zarr_path.exists():
        ds_existing = xr.open_zarr(zarr_path)
        if ds_existing.rio.crs is None and crs_existing is not None:
            logger.debug(f'Setting manual crs on zarr store: {crs_existing}')
            ds_existing = ds_existing.rio.write_crs(crs_existing)

        ##Check if crs is equal
        if ds_existing.rio.crs != ds_new.rio.crs:
            raise ValueError(f"Coordinate systems of ds_existing and ds_new are not equal! Got {ds_existing.rio.crs} vs {ds_new.rio.crs}")

        for vname in list(ds_new.keys()):

            if vname in ds_existing:
                arr = ds_existing[vname]
                existing_dims = {i:j for i,j in arr.sizes.items() if i not in append_dims}
                new_dims = {i:j for i,j in ds_new[vname].sizes.items() if i not in append_dims}

                ##Check if dimension names and shapes match
                if existing_dims != new_dims:
                    raise ValueError(f"Existing dataset dimensions and shapes {existing_dims} do not match expected {new_dims}.")

                ##Check if dimension values are equal
                dims_unequal = {i: not ds_new[i].equals(arr[i]) for i in existing_dims.keys()}
                if any(dims_unequal.values()):
                    if not align:
                        raise ValueError(f"Cannot insert in zarr store if dimensions are not equal. Unequal values found for dimensions {dims_unequal}.")
                    else:
                        logger.debug(f"Unequal values found for dimensions {dims_unequal}. Reprojecting...")
                        ds_new = align_arrays(ds_new, base = arr, **kwargs)[0]

                ##Check if chunking is equal
                if arr.chunks and ds_new[vname].chunks and  arr.chunks != ds_new[vname].chunks:
                    raise ValueError(f"Existing dataset chunks {arr.chunks} do not match expected {ds_new[vname].chunks}.")

                ##Check if all values of append_dim are present in existing dataset
                for i in append_dims:
                    if not ds_new[i].isin(arr[i]).all():
                        raise ValueError(f"Not all values of coordinate {i} are present in existing dataset. Writing to a zarr region slice requires that no dimensions or metadata are changed by the write.")

                logger.debug("Existing variable in zarr store is correctly aligned with new data.")
                return ds_new

            else:
                raise NotImplementedError(f"Found existing zarr store but variable '{vname}' is not present. Adding new variables is not supported.")       
    else:
        raise ValueError(f"zarr store at {zarr_path} does not exist. Create one before using _ensure_preallocated_store.")

def assert_spatial_info(da):
    """
    Check if a DataArray has proper spatial information.
    
    Args:
        da: xarray.DataArray to check
        
    Returns:
        bool: True if spatial info is complete
    """
    try:
        # This will raise an exception if spatial dims aren't set
        x_dim = da.rio.x_dim
        y_dim = da.rio.y_dim
                    
    except rioxarray.exceptions.MissingSpatialDimensionError:
        raise ValueError('MissingSpatialDimensionError. Use "da.rio.set_spatial_dims(x_dim="x", y_dim="y")" to set the dimensions.')

    # Check if CRS is set
    if da.rio.crs is None:
        raise ValueError('CRS not set. Use "da.rio.write_crs()" to set the CRS.')

    return True