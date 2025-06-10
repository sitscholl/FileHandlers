
import pytest
import xarray as xr
import zarr
import numpy as np
import rioxarray # noqa, imported for accessor and exceptions

from pathlib import Path
from zarr.codecs import BloscCodec
import os

from filehandler.grid.writer import GridWriter

# Fixture for GridWriter instance
@pytest.fixture
def writer(tmp_path: Path) -> GridWriter:
    """Provides a GridWriter instance with a temporary root directory."""
    return GridWriter(tmp_path)

# Helper function to create a sample DataArray
def create_sample_dataarray(
    name: str = "test_var",
    add_spatial_info: bool = True,
    num_bands: int = 1,
    height: int = 10,
    width: int = 10,
) -> xr.DataArray:
    """Creates a sample xarray.DataArray for testing."""
    if num_bands > 1:
        data_shape = (num_bands, height, width)
        coords = {
            "band": np.arange(1, num_bands + 1),
            "y": np.linspace(50, 40, height),  # Latitude-like, decreasing
            "x": np.linspace(-120, -110, width), # Longitude-like
        }
        dims = ("band", "y", "x")
    else:
        # For single band, rioxarray prefers (y, x) dims for 2D data to correctly infer band count as 1
        data_shape = (height, width)
        coords = {
            "y": np.linspace(50, 40, height),
            "x": np.linspace(-120, -110, width),
        }
        dims = ("y", "x")

    data = np.random.rand(*data_shape).astype(np.float32)
    
    da = xr.DataArray(
        data,
        coords=coords,
        dims=dims,
        name=name,
    )
    
    if add_spatial_info:
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
        da = da.rio.write_crs("epsg:4326", inplace=True)
        # rioxarray can often infer transform from 1D coordinates if regularly spaced
    return da

def test_to_geotiff_dataarray_single_band(writer: GridWriter, tmp_path: Path):
    """Tests writing a single-band DataArray to GeoTIFF."""
    da = create_sample_dataarray(name="test_single_band", num_bands=1, height=10, width=10)
    filename = "test_dataarray_single.tif"
    
    writer.to_geotiff(da, filename)
    
    expected_file = tmp_path / filename
    assert expected_file.exists(), f"File {expected_file} was not created."
    assert expected_file.is_file()

    # Verify some basic properties of the created GeoTIFF
    with rioxarray.open_rasterio(expected_file) as rds:
        assert rds.rio.count == 1 # Single band
        assert rds.rio.crs.to_epsg() == 4326
        assert (rds.rio.height, rds.rio.width) == (10, 10)

def test_to_geotiff_dataarray_multi_band(writer: GridWriter, tmp_path: Path):
    """Tests writing a multi-band DataArray to GeoTIFF."""
    da = create_sample_dataarray(name="test_multi_band", num_bands=3, height=10, width=10)
    filename = "test_dataarray_multi.tif"
    
    writer.to_geotiff(da, filename)
    
    expected_file = tmp_path / filename
    assert expected_file.exists(), f"File {expected_file} was not created."
    assert expected_file.is_file()

    with rioxarray.open_rasterio(expected_file) as rds:
        assert rds.rio.count == 3 # Multi-band
        assert rds.rio.crs.to_epsg() == 4326
        assert (rds.rio.height, rds.rio.width) == (10, 10)

def test_to_geotiff_dataset(writer: GridWriter, tmp_path: Path):
    """Tests writing an xarray.Dataset to multiple GeoTIFF files."""
    var1_name = "temperature"
    var2_name = "precipitation"
    
    da1 = create_sample_dataarray(name=var1_name, num_bands=1, height=5, width=5)
    # Ensure da2 has different band structure for a more robust test
    da2_data = np.random.rand(2, 6, 6).astype(np.float32) # 2 bands, 6x6
    da2 = xr.DataArray(
        da2_data,
        coords={"band": [1,2], "y": np.linspace(30,25,6), "x": np.linspace(-100,-95,6)},
        dims=("band", "y", "x"),
        name=var2_name
    )
    da2 = da2.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True).rio.write_crs("epsg:4326", inplace=True)

    ds = xr.Dataset({var1_name: da1, var2_name: da2})
    
    base_filenames = ["test_dataset1.tif", "test_dataset2.tif"] 
    writer.to_geotiff(ds, base_filenames)
    
    # Check for var1
    expected_file1_name = base_filenames[0]
    expected_file1 = tmp_path / expected_file1_name
    assert expected_file1.exists(), f"File {expected_file1} was not created."
    assert expected_file1.is_file()
    with rioxarray.open_rasterio(expected_file1) as rds:
        assert rds.rio.count == da1.rio.count # Should be 1
        assert rds.rio.crs.to_epsg() == 4326

    # Check for var2
    expected_file2_name = base_filenames[1]
    expected_file2 = tmp_path / expected_file2_name
    assert expected_file2.exists(), f"File {expected_file2} was not created."
    assert expected_file2.is_file()
    with rioxarray.open_rasterio(expected_file2) as rds:
        assert rds.rio.count == da2.rio.count # Should be 2
        assert rds.rio.crs.to_epsg() == 4326

def test_to_geotiff_dataarray_no_spatial_info(writer: GridWriter, tmp_path: Path, caplog):
    """Tests writing a DataArray with missing spatial information."""
    da = create_sample_dataarray(name="no_spatial_var", add_spatial_info=False)
    filename = "no_spatial.tif"
    
    # Check if an error is raised when trying to write a DataArray without spatial info
    with pytest.raises(Exception) as excinfo:
        writer.to_geotiff(da, filename)
    
    # Verify that the error is related to missing spatial information
    error_message = str(excinfo.value)
    assert any(msg in error_message for msg in [
        "MissingSpatialDimensionError", 
        "MissingCRS", 
        "No X dimension", 
        "No Y dimension", 
        "CRS not set", 
        "must be an xarray.DataArray with spatial reference information"
    ]), f"Unexpected error message: {error_message}"
    
    # Verify that no file was created
    expected_file = tmp_path / filename
    assert not expected_file.exists(), f"File {expected_file} should not have been created due to missing spatial info."

def test_to_geotiff_unsupported_type(writer: GridWriter, tmp_path: Path):
    """Tests passing an unsupported data type to to_geotiff."""
    unsupported_data = [1, 2, 3] # A list, not a DataArray or Dataset
    filename = "unsupported.tif"

    # Based on the provided summary, GridWriter.to_geotiff does not have an explicit 'else' 
    # to catch and log unsupported types. It will simply not enter the isinstance checks.
    # Thus, no file should be created, and no "Failed to write" error from the writer's 
    # try-except blocks for raster writing should occur.
    with pytest.raises(Exception) as excinfo:
        writer.to_geotiff(unsupported_data, filename) # type: ignore
    
    expected_file = tmp_path / filename
    assert not expected_file.exists(), f"File {expected_file} should not have been created for an unsupported type."
    assert "Unsupported data type" in str(excinfo.value), "Expected an error message indicating unsupported data type."


# Tests for to_zarr method
def test_to_zarr_dataarray_new_store(writer: GridWriter, tmp_path: Path):
    """Tests writing a DataArray to a new Zarr store."""
    da = create_sample_dataarray(name="test_zarr_var", num_bands=1, height=5, width=5)
    filename = "test_dataarray.zarr"
    
    writer.to_zarr(da, filename)
    
    expected_path = tmp_path / filename
    assert expected_path.exists(), f"Zarr store {expected_path} was not created."
    assert expected_path.is_dir(), "Zarr store should be a directory."
    
    # Verify the contents of the Zarr store
    with xr.open_zarr(expected_path, decode_coords = 'all') as ds:
        assert "test_zarr_var" in ds, "Variable name not preserved in Zarr store."
        assert ds.rio.crs.to_epsg() == 4326, "CRS not preserved in Zarr store."
        assert ds["test_zarr_var"].shape == (5, 5), "Data shape not preserved in Zarr store."

def test_to_zarr_dataset_new_store(writer: GridWriter, tmp_path: Path):
    """Tests writing a Dataset to a new Zarr store."""
    var1_name = "temperature"
    var2_name = "precipitation"
    
    da1 = create_sample_dataarray(name=var1_name, num_bands=1, height=5, width=5)
    da2 = create_sample_dataarray(name=var2_name, num_bands=1, height=5, width=5)
    
    ds = xr.Dataset({var1_name: da1, var2_name: da2})
    filename = "test_dataset.zarr"
    
    writer.to_zarr(ds, filename)
    
    expected_path = tmp_path / filename
    assert expected_path.exists(), f"Zarr store {expected_path} was not created."
    assert expected_path.is_dir(), "Zarr store should be a directory."
    
    # Verify the contents of the Zarr store
    with xr.open_zarr(expected_path, decode_coords = 'all') as ds_read:
        assert var1_name in ds_read, f"Variable {var1_name} not found in Zarr store."
        assert var2_name in ds_read, f"Variable {var2_name} not found in Zarr store."
        assert ds_read.rio.crs.to_epsg() == 4326, "CRS not preserved in Zarr store."

def test_generate_preallocated_zarr_store(writer: GridWriter, tmp_path: Path):
    """Tests generating a preallocated Zarr store."""
    shape = (10, 5, 5)  # Example shape: (time, height, width)
    coords = {
        "time": np.arange(shape[0]),
        "x": np.linspace(-100, -95, shape[2]),
        "y": np.linspace(30, 25, shape[1]),
    }
    chunks = {'time': -1, 'x': 2, 'y': 2}
    var_dict = {'var1': float, 'var2': int}

    filename = "preallocated_store.zarr"
    writer.generate_preallocated_zarr_store(
        filename,
        shape = shape,
        coords = coords,
        crs = 4326,
        chunks = chunks,
        encoding = None,
        variables = var_dict
    )

    zarr_path = tmp_path / filename

    assert zarr_path.exists(), f"Zarr store {zarr_path} was not created."
    assert zarr_path.is_dir(), f"Expected {zarr_path} to be a directory."

    with xr.open_zarr(zarr_path, decode_coords = 'all') as ds_read:
        assert ds_read.rio.crs.to_epsg() == 4326, "CRS not preserved in Zarr store."

        assert ds_read.chunks == {'time': (10,), 'x': (2, 2, 1), 'y': (2, 2, 1)}, "Chunks not preserved in Zarr store."
        assert all([i in list(var_dict.keys()) for i in list(ds_read.data_vars)]), "Data variables not preserved in Zarr store."

        assert ds_read['var1'].shape == shape, "Dimensions of var1 not preserved in Zarr store."
        assert ds_read['var2'].shape == shape, "Dimensions of var2 not preserved in Zarr store."

        assert ds_read['var1'].dtype == float, "Data type of 'var1' not preserved in Zarr store."
        assert ds_read['var2'].dtype == int, "Data type of 'var2' not preserved in Zarr store."

def test_generate_preallocated_zarr_store_chunks_wrong_keys(writer: GridWriter, tmp_path: Path):
    """Tests generating a preallocated Zarr store."""
    shape = (10, 5, 5)  # Example shape: (time, height, width)
    coords = {
        "time": np.arange(shape[0]),
        "x": np.linspace(-100, -95, shape[2]),
        "y": np.linspace(30, 25, shape[1]),
    }
    chunks = {'time': -1, 'lon': 2, 'lat': 2}
    var_dict = {'var1': float, 'var2': int}

    filename = "preallocated_store.zarr"
    with pytest.raises(Exception) as excinfo:
        writer.generate_preallocated_zarr_store(
            filename,
            shape = shape,
            coords = coords,
            crs = 4326,
            chunks = chunks,
            encoding = None,
            variables = var_dict
        )

def test_generate_preallocated_zarr_store_dtype_encoding(writer: GridWriter, tmp_path: Path):
    """Tests generating a preallocated Zarr store with dtype encoding."""
    shape = (10, 5, 5)  # Example shape: (time, height, width)
    coords = {
        "time": np.arange(shape[0]),
        "x": np.linspace(-100, -95, shape[2]),
        "y": np.linspace(30, 25, shape[1]),
    }
    var_dict = {'var1': float, 'var2': int}
    
    # Test explicit dtype encoding
    encoding = {
        'var1': {'dtype': 'float32'},  # Downcast from float64 to float32
        'var2': {'dtype': 'int16'}     # Downcast from int64 to int16
    }

    filename = "dtype_encoded_store.zarr"
    writer.generate_preallocated_zarr_store(
        filename,
        shape = shape,
        coords = coords,
        crs = 4326,
        encoding = encoding,
        variables = var_dict
    )

    zarr_path = tmp_path / filename

    assert zarr_path.exists(), f"Zarr store {zarr_path} was not created."
    
    with xr.open_zarr(zarr_path, decode_coords = 'all') as ds_read:
        # Check that the dtypes were properly encoded
        assert ds_read['var1'].dtype == np.dtype('float32'), "Dtype encoding for 'var1' not applied correctly."
        assert ds_read['var2'].dtype == np.dtype('int16'), "Dtype encoding for 'var2' not applied correctly."
        
        # Verify other properties are still preserved
        assert ds_read.rio.crs.to_epsg() == 4326, "CRS not preserved in Zarr store with dtype encoding."
        assert all([i in list(var_dict.keys()) for i in list(ds_read.data_vars)]), "Data variables not preserved in Zarr store with dtype encoding."

def test_generate_preallocated_zarr_store_scale_factor(writer: GridWriter, tmp_path: Path):
    """Tests generating a preallocated Zarr store with scale_factor encoding."""
    shape = (10, 5, 5)  # Example shape: (time, height, width)
    coords = {
        "time": np.arange(shape[0]),
        "x": np.linspace(-100, -95, shape[2]),
        "y": np.linspace(30, 25, shape[1]),
    }
    var_dict = {'var1': float}
    
    # Test scale_factor and add_offset encoding
    encoding = {
        'var1': {
            'dtype': 'int16',
            'scale_factor': 0.1,
            'add_offset': 0.0,
            '_FillValue': -9999
        }
    }

    filename = "scale_factor_store.zarr"
    writer.generate_preallocated_zarr_store(
        filename,
        shape = shape,
        coords = coords,
        crs = 4326,
        encoding = encoding,
        variables = var_dict
    )

    zarr_path = tmp_path / filename

    assert zarr_path.exists(), f"Zarr store {zarr_path} was not created."
    
    # Open the zarr store and check encoding attributes
    with xr.open_zarr(zarr_path, decode_coords = 'all') as ds_read:
        # Check that the encoding was properly applied
        var1_encoding = ds_read['var1'].encoding
        assert 'scale_factor' in var1_encoding, "scale_factor not found in encoding"
        assert var1_encoding['scale_factor'] == 0.1, "scale_factor value not preserved"
        assert 'add_offset' in var1_encoding, "add_offset not found in encoding"
        assert var1_encoding['add_offset'] == 0.0, "add_offset value not preserved"
        assert '_FillValue' in var1_encoding, "_FillValue not found in encoding"
        assert var1_encoding['_FillValue'] == -9999, "_FillValue not preserved"
        
        # Check that the dtype was properly encoded
        assert ds_read['var1'].dtype == np.dtype('float64'), "Data is decoded to float with scale_factor applied"

    # Open the zarr store without decoding and check dtype
    with xr.open_zarr(zarr_path, decode_cf = False) as ds_read:
        assert ds_read['var1'].dtype == np.dtype('int16'), "Data is decoded to float with scale_factor applied"


def test_generate_preallocated_zarr_store_compression(writer: GridWriter, tmp_path: Path):
    """Tests generating a preallocated Zarr store with compression encoding."""
    shape = (10, 5, 5)  # Example shape: (time, height, width)
    coords = {
        "time": np.arange(shape[0]),
        "x": np.linspace(-100, -95, shape[2]),
        "y": np.linspace(30, 25, shape[1]),
    }
    var_dict = {'var1': float, 'var2': int}
    
    # Test compression settings
    encoding = {
        'var1': {
            'compressor': BloscCodec(cname="zstd", clevel=3, shuffle="shuffle"),  # zlib compression
            'dtype': 'float32'
        },
        'var2': {
            'compressor': BloscCodec(cname="zstd", clevel=9, shuffle="shuffle"),  # higher compression level
            'dtype': 'int32'
        }
    }

    filename = "compression_store.zarr"
    writer.generate_preallocated_zarr_store(
        filename,
        shape = shape,
        coords = coords,
        crs = 4326,
        encoding = encoding,
        variables = var_dict
    )

    zarr_path = tmp_path / filename

    assert zarr_path.exists(), f"Zarr store {zarr_path} was not created."
    
    # Open the zarr store and check encoding
    with xr.open_zarr(zarr_path, decode_coords = 'all') as ds_read:
        # Check that the dtypes were properly encoded
        assert ds_read['var1'].dtype == np.dtype('float32'), "Dtype encoding for 'var1' not applied correctly."
        assert ds_read['var2'].dtype == np.dtype('int32'), "Dtype encoding for 'var2' not applied correctly."
        
        # Verify other properties are still preserved
        assert ds_read.rio.crs.to_epsg() == 4326, "CRS not preserved in Zarr store with compression encoding."


def test_generate_preallocated_zarr_store_chunking_with_encoding(writer: GridWriter, tmp_path: Path):
    """Tests generating a preallocated Zarr store with both chunking and encoding."""
    shape = (10, 5, 5)  # Example shape: (time, height, width)
    coords = {
        "time": np.arange(shape[0]),
        "x": np.linspace(-100, -95, shape[2]),
        "y": np.linspace(30, 25, shape[1]),
    }
    chunks = {'time': 2, 'x': 2, 'y': 2}  # Explicit chunks for all dimensions
    var_dict = {'var1': float, 'var2': int}
    
    # Combine chunking with encoding
    encoding = {
        'var1': {
            'dtype': 'float32',
            'scale_factor': 0.01,
            '_FillValue': -9999.0
        },
        'var2': {
            'dtype': 'int16',
            'compressor': BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
        }
    }

    filename = "chunked_encoded_store.zarr"
    writer.generate_preallocated_zarr_store(
        filename,
        shape = shape,
        coords = coords,
        crs = 4326,
        chunks = chunks,
        encoding = encoding,
        variables = var_dict
    )

    zarr_path = tmp_path / filename

    assert zarr_path.exists(), f"Zarr store {zarr_path} was not created."
    
    with xr.open_zarr(zarr_path, decode_coords = 'all') as ds_read:
        # Check that chunks were properly applied
        assert ds_read.chunks == {'time': (2, 2, 2, 2, 2), 'x': (2, 2, 1), 'y': (2, 2, 1)}, "Chunks not preserved correctly."
        
        # Check that encoding was properly applied
        assert ds_read['var1'].dtype == np.dtype('float64'), "Dtype encoding for 'var1' not applied correctly."
        assert ds_read['var2'].dtype == np.dtype('int16'), "Dtype encoding for 'var2' not applied correctly."
        
        # Check scale_factor encoding
        var1_encoding = ds_read['var1'].encoding
        assert 'scale_factor' in var1_encoding, "scale_factor not found in encoding"
        assert var1_encoding['scale_factor'] == 0.01, "scale_factor value not preserved"
        assert '_FillValue' in var1_encoding, "_FillValue not found in encoding"
        
        # Verify CRS is still preserved
        assert ds_read.rio.crs.to_epsg() == 4326, "CRS not preserved in Zarr store with chunking and encoding."

    #check dtypes without decoding
    with xr.open_zarr(zarr_path, decode_cf = False) as ds_read:
        assert ds_read['var1'].dtype == np.dtype('float32'), "Dtype encoding for 'var1' not saved correctly."
        assert ds_read['var2'].dtype == np.dtype('int16'), "Dtype encoding for 'var2' not saved correctly."


def test_to_zarr_append_new_variable_raise(writer: GridWriter, tmp_path: Path):
    """Tests appending to an existing Zarr store along a dimension."""
    # Create initial dataset with time dimension
    time_coords = [np.datetime64("2023-01-01"), np.datetime64("2023-01-02")]
    new_time_coords = [np.datetime64("2023-01-03"), np.datetime64("2023-01-04")]
    all_time_coords = time_coords + new_time_coords
    bands, height, width = 1, 4, 4

    da1 = create_sample_dataarray(name="new_var", num_bands=bands, height=height, width=width)
    da1 = da1.expand_dims({"time": time_coords})
    
    filename = "append_test.zarr"
    writer.generate_preallocated_zarr_store(
        filename,
        shape = (len(all_time_coords), height, width),
        coords = {'time': all_time_coords, 'x': da1.x, 'y': da1.y},
        crs = da1.rio.crs.to_epsg(),
        chunks = {'time': -1, 'x': 2, 'y': 2},
        variables = {'existing_var': float}
    )
    
    with pytest.raises(Exception) as excinfo:
        writer.to_zarr(da1, filename, append_dims=["time"])
    
def test_to_zarr_append_along_dimension(writer: GridWriter, tmp_path: Path):
    """Tests appending to an existing Zarr store along a dimension."""
    # Create initial dataset with time dimension
    time_coords = [np.datetime64("2023-01-01"), np.datetime64("2023-01-02")]
    new_time_coords = [np.datetime64("2023-01-03"), np.datetime64("2023-01-04")]
    all_time_coords = time_coords + new_time_coords
    bands, height, width = 1, 4, 4
    chunk_structure = {'time': -1, 'x': 2, 'y': 2}

    da1 = create_sample_dataarray(name="data", num_bands=bands, height=height, width=width)
    da1 = da1.expand_dims({"time": time_coords})
    da1 = da1.chunk(chunk_structure)
    
    filename = "append_test.zarr"
    writer.generate_preallocated_zarr_store(
        filename,
        shape = (len(all_time_coords), height, width),
        coords = {'time': all_time_coords, 'x': da1.x, 'y': da1.y},
        crs = da1.rio.crs.to_epsg(),
        chunks = chunk_structure,
        variables = {"data": float}
    )
    
    # Write initial data
    writer.to_zarr(da1, filename, append_dims=["time"])
    
    # Create new data with additional time steps
    da2 = create_sample_dataarray(name="data", num_bands=bands, height=height, width=width)
    da2 = da2.expand_dims({"time": new_time_coords})
    da2 = da2.chunk(chunk_structure)
    
    # Append to existing store along time dimension
    writer.to_zarr(da2, filename, append_dims=["time"])
    
    # Verify the appended data
    zarr_path = tmp_path / filename
    with xr.open_zarr(zarr_path) as ds:
        assert "data" in ds, "Variable not found in Zarr store."
        assert ds["data"].shape[0] == 4, f"Expected 4 time steps after append, got {ds['data'].shape[0]}."
        assert all(t in ds.time.dt.date.values for t in all_time_coords), "Not all time coordinates present after append."
        assert ds.chunks['time'][0] == len(all_time_coords)

def test_to_zarr_overwrite_along_dimension(writer: GridWriter, tmp_path: Path):
    """Tests overwriting coordinate in an existing Zarr store along a dimension."""
    # Create initial dataset with time dimension
    time_coords = [np.datetime64("2023-01-01"), np.datetime64("2023-01-02")]
    new_time_coords = [np.datetime64("2023-01-02"), np.datetime64("2023-01-03")]
    all_time_coords = [np.datetime64("2023-01-01"), np.datetime64("2023-01-02"), np.datetime64("2023-01-03")]
    bands, height, width = 1, 4, 4
    chunk_structure = {'time': -1, 'x': 2, 'y': 2}

    da1 = create_sample_dataarray(name="data", num_bands=bands, height=height, width=width)
    da1 = da1.expand_dims({"time": time_coords})
    da1 = da1.chunk(chunk_structure)
    
    filename = "append_test.zarr"
    writer.generate_preallocated_zarr_store(
        filename,
        shape = (len(all_time_coords), height, width),
        coords = {'time': all_time_coords, 'x': da1.x, 'y': da1.y},
        crs = da1.rio.crs.to_epsg(),
        chunks = {'time': -1, 'x': 2, 'y': 2},
        variables = {"data": float}
    )
    
    # Write initial data
    writer.to_zarr(da1, filename, append_dims=["time"])
    
    # Create new data with additional time steps
    da2 = create_sample_dataarray(name="data", num_bands=bands, height=height, width=width)
    da2 = da2.expand_dims({"time": new_time_coords})
    da2 = da2.chunk(chunk_structure)
    
    # Append to existing store along time dimension
    writer.to_zarr(da2, filename, append_dims=["time"])
    
    # Verify the appended data
    zarr_path = tmp_path / filename
    with xr.open_zarr(zarr_path) as ds:
        assert "data" in ds, "Variable not found in Zarr store."
        assert ds["data"].shape[0] == 3, f"Expected 3 time steps after append, got {ds['data'].shape[0]}."
        assert all(t in ds.time.dt.date.values for t in all_time_coords), "Not all time coordinates present after append."
        assert ds.chunks['time'][0] == len(all_time_coords)

def test_to_zarr_different_dtypes(writer: GridWriter, tmp_path: Path):
    """Make sure error is raised when dtypes in existing zarr store differ from new dtypes."""
    # Create initial dataset with time dimension
    time_coords = [np.datetime64("2023-01-01"), np.datetime64("2023-01-02")]
    bands, height, width = 1, 4, 4

    da1 = create_sample_dataarray(name="data", num_bands=bands, height=height, width=width)
    da1 = da1.expand_dims({"time": time_coords})
    da1 = da1.astype(int)
    
    filename = "append_test.zarr"
    writer.generate_preallocated_zarr_store(
        filename,
        shape = (len(time_coords), height, width),
        coords = {'time': time_coords, 'x': da1.x, 'y': da1.y},
        crs = da1.rio.crs.to_epsg(),
        chunks = {'time': -1, 'x': 2, 'y': 2},
        variables = {"data": float}
    )
    
    # Write initial data
    with pytest.raises(Exception) as excinfo:
        writer.to_zarr(da1, filename, append_dims=["time"])


def test_to_zarr_drop_attrs(writer: GridWriter, tmp_path: Path):
    """Tests writing to Zarr with drop_attrs=True."""
    da = create_sample_dataarray(name="test_attrs", num_bands=1, height=3, width=3)
    
    # Add some custom attributes
    da.attrs["custom_attr"] = "test_value"
    da.attrs["units"] = "meters"
    
    filename = "attrs_test.zarr"
    
    # Write with drop_attrs=True
    writer.to_zarr(da, filename, drop_attrs=True)
    
    # Verify attributes were dropped
    with xr.open_zarr(tmp_path / filename) as ds:
        assert "test_attrs" in ds, "Variable not found in Zarr store."
        assert "custom_attr" not in ds["test_attrs"].attrs, "Custom attribute should have been dropped."
        assert "units" not in ds["test_attrs"].attrs, "Units attribute should have been dropped."

def test_to_zarr_unsupported_type(writer: GridWriter, tmp_path: Path):
    """Tests passing an unsupported data type to to_zarr."""
    unsupported_data = [1, 2, 3]  # A list, not a DataArray or Dataset
    filename = "unsupported.zarr"
    
    with pytest.raises(NotImplementedError) as excinfo:
        writer.to_zarr(unsupported_data, filename)  # type: ignore
    
    expected_path = tmp_path / filename
    assert not expected_path.exists(), f"Zarr store {expected_path} should not have been created for an unsupported type."
    assert "Unsupported data type" in str(excinfo.value), "Expected an error message indicating unsupported data type."

def test_to_zarr_append_to_nonexistent_store(writer: GridWriter, tmp_path: Path):
    """Tests appending to a non-existent Zarr store."""
    da = create_sample_dataarray(name="test_var", num_bands=1, height=3, width=3)
    filename = "nonexistent.zarr"
    
    with pytest.raises(ValueError) as excinfo:
        writer.to_zarr(da, filename, append_dims=["y"])
    
    assert "Cannot append to non-existent Zarr store" in str(excinfo.value), "Expected error about non-existent store."
    