import numpy as np
import xarray as xr
import rioxarray

from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any, List

from .utils import ensure_zarr_store_aligns, assert_spatial_info

class GridWriter:
    def __init__(self, root):
        self.root = Path(root)

    def to_netcdf(self, data: xr.Dataset, filename: str, **kwargs):
        """
        Write an xarray Dataset to a NetCDF file.

        Args:
            ds: The dataset to write.
            filename: The name of the output file.
            **kwargs: Additional arguments passed to ds.to_netcdf.
        """
        if not isinstance(data, (xr.DataArray, xr.Dataset)):
            raise NotImplementedError(f"Unsupported data type {type(data)} for Netcdf conversion.")

        if isinstance(data, xr.DataArray):
            data = data.to_dataset(name="variable")

        output_path = self.root / filename
        data.to_netcdf(output_path, **kwargs)

    def to_geotiff(self, data: xr.DataArray | xr.Dataset, filename: str | list[str], **kwargs):
        """
        Write an xarray DataArray to a GeoTIFF file.

        Args:
            da: The data array to write. Must have spatial coordinates and CRS.
            filename: The name of the output file.
            **kwargs: Additional arguments passed to da.rio.to_raster.
        """

        if not isinstance(data, (xr.DataArray, xr.Dataset)):
            raise NotImplementedError(f"Unsupported data type {type(data)} for GeoTIFF conversion.")

        if not hasattr(data, 'rio'):
            raise ValueError("DataArray must have spatial information (rioxarray extension) to save to geotiff.")

        if isinstance(data, xr.DataArray):
            assert_spatial_info(data)
            data.rio.to_raster(self.root / filename, **kwargs)
        elif isinstance(data, xr.Dataset):
            if isinstance(filename, str):
                filename = [filename]
            if len(data.data_vars) != len(filename):
                raise ValueError(f"Number of filenames must match the number of data variables in the Dataset. Got {len(filename)} filenames for {len(data.data_vars)} data variables.")  
            for i, (nam, da) in enumerate(data.data_vars.items()):
                output_path = self.root / filename[i]
                assert_spatial_info(da)
                da.rio.to_raster(output_path, **kwargs)

    def to_zarr(
        self,
        data: Union[xr.Dataset, xr.DataArray],
        filename: Union[str, Path],
        append_dims: Optional[List[str]] = None,
        drop_attrs: bool =False,
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
        append_dims : list of str or str, optional
            List of dimensions along which to append data in an existing zarr store.
        drop_attrs: boolean, default False
            If true, all attributes of the dataset will be dropped before writing to zarr.
        **kwargs
            Additional arguments passed to xarray's to_zarr method.

        Raises
        ------
        ValueError
            If the Zarr store does not exist and append_dims is provided, or if the data is incompatible with the existing store.
        NotImplementedError
            If attempting to add new variables to an existing Zarr store or if data type is unsupported.
        """
        if not isinstance(data, (xr.DataArray, xr.Dataset)):
            raise NotImplementedError(f"Unsupported data type {type(data)} for zarr conversion.")

        assert_spatial_info(data)

        # Convert DataArray to Dataset if needed
        if isinstance(data, xr.DataArray):
            if data.name is None:
                data.name = 'var'
            data = data.to_dataset()

        # Ensure filename is a Path object
        if isinstance(filename, str):
            filename = Path(filename)
        filename = self.root / filename

        # Convert string append_dims to list
        if isinstance(append_dims, str):
            append_dims = [append_dims]

        # Make a copy of the data to avoid modifying the original
        data_to_write = data.copy()

        # Handle existing Zarr store
        if filename.exists():
            if append_dims is not None:
                # Validate and align with existing store
                ensure_zarr_store_aligns(filename, data_to_write, append_dims=append_dims)
                mode = "a"
                region = "auto"
                drop_vars = True
            else:
                raise ValueError(f"Found existing zarr store at {filename}. Append_dims must be specified to align with an existing Zarr store.")
                # Overwrite existing store if append_dims is None
                # mode = "w"
                # region = None
                # drop_vars = False
        else:
            # Create new store
            if append_dims is not None:
                raise ValueError(f"Cannot append to non-existent Zarr store at {filename}. Store must exist when append_dims is provided.")
            mode = "w"
            region = None
            drop_vars = False

        # Drop attributes if requested
        if drop_attrs:
            data_to_write = data_to_write.drop_attrs()

        # Remove spatial_ref variable if present to avoid conflicts
        if drop_vars:
            data_to_write = data_to_write.drop_vars(['spatial_ref', 'transverse_mercator'], errors='ignore')

        # Write to Zarr store
        data_to_write.to_zarr(filename, mode=mode, region=region, **kwargs)

    def generate_preallocated_zarr_store(
        self,
        filename: Union[str, Path],
        shape: Tuple[int, ...],
        coords: Dict,
        crs: int,
        chunks: Optional[Tuple[int, ...]] = None,
        encoding: Optional[Dict] = None,
        variables: Optional[Dict[str, type]] = None,
    ) -> None:
        """
        Generate a pre-allocated zarr store with specified dimensions and metadata.

        Args:
            filename: Path to create zarr store
            shape: Tuple defining the array dimensions
            coords: Dictionary of coordinates for each dimension
            crs: Coordinate Reference System code (EPSG)
            chunks: Optional tuple specifying chunk sizes
            encoding: Optional dictionary for zarr encoding settings
            variables: Optional dictionary mapping variable names to their data types

        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If zarr store creation fails
        """

        if (self.root / filename).exists():
            raise ValueError(f"Found existing zarr store at {self.root / filename}.")

        if variables is None:
            variables = {"variable": float}

        if not isinstance(variables, dict):
            raise ValueError("The 'variables' parameter must be a dictionary with the following structure: VariableName: VariableType.")

        try:
            # Create dummy dataset to preallocate a zarr store with necessary metadata but no data
            dummy = xr.DataArray(np.empty(shape), coords=coords)

            if chunks is not None:
                if len(chunks) != len(shape):
                    raise ValueError(f"Chunks {chunks} must match shape dimensions {shape}")
                dummy = dummy.chunk(chunks)
                
            dummy = dummy.expand_dims({'var': list(variables.keys())}).to_dataset(dim='var')

            dummy = dummy.rio.write_crs(crs)

            for v, dtype in variables.items():
                dummy[v] = dummy[v].astype(dtype)

            dummy.to_zarr(self.root / filename, mode="w", compute=False, encoding=encoding)

        except Exception as e:
            raise RuntimeError(f"Failed to create zarr store: {str(e)}") from e