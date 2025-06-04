
import pytest
import xarray as xr
import numpy as np
import rioxarray # noqa, imported for accessor and exceptions
from pathlib import Path
import logging # For caplog

from src.grid.writer import GridWriter

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
    