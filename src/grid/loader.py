import xarray as xr
import rioxarray

from pathlib import Path
import logging
from typing import Callable, Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class GridLoader:

    # Common names for x and y coordinates
    X_COORD_NAMES = {'x', 'lon', 'longitude', 'easting', 'x_coord', 'xc', 'x_dim', 'x_pos'}
    Y_COORD_NAMES = {'y', 'lat', 'latitude', 'northing', 'y_coord', 'yc', 'y_dim', 'y_pos'}
    TIME_COORD_NAMES = {
                        'time', 't', 'datetime', 'date', 'timestamp',
                        'times', 'timestep', 'time_step', 'timepoint',
                        'date_time', 'valid_time', 'forecast_time',
                        'reference_time', 'reftime', 'analysis_time',
                        'init_time', 'verification_time', 'obs_time',
                        'temporal', 'temporal_dim', 'time_dim',
                        'day', 'month', 'year', 'hour', 'minute', 'second'
                    }

    def __init__(self, root):
        self.root = Path(root)
        self.preprocess_steps = []
        self.default_attrs = {}

    def list_files(self, pattern: str = '*'):
        return list(self.root.glob(pattern))

    def add_preprocess_step(self, step: Callable[[xr.Dataset], xr.Dataset], name: Optional[str] = None):
        """Add a preprocessing step function to be applied in order during loading."""
        if name:
            step.__name__ = name
        self.preprocess_steps.append(step)
        logger.debug('Added preprocessing step to loading pipeline')
        return self  # Enable method chaining

    def set_default_attrs(self, attrs: Dict[str, Any]):
        """Set default attributes to be added to all loaded datasets."""
        self.default_attrs.update(attrs)
        return self

    def set_crs(self, epsg: int):
        """Set the coordinate system of all loaded datasets."""
        def _set_crs(ds):
            return ds.rio.write_crs(epsg)
        return self.add_preprocess_step(_set_crs, "set_crs")

    def rename_variables(self, mapping: Dict[str, str]):
        """Add a preprocessing step to rename variables according to mapping."""
        def _rename(ds):
            return ds.rename({k: v for k, v in mapping.items() if k in ds.data_vars})
        return self.add_preprocess_step(_rename, "rename_variables")

    def rename_coordinates(self, x_name: str = 'x', y_name: str = 'y', time_name: str | None = None):
        """
        Add a preprocessing step to rename x and y coordinates to user-defined names.

        Args:
            x_name: Target name for x-like coordinates
            y_name: Target name for y-like coordinates

        Returns:
            self for method chaining
        """
        def _rename_coords(ds):
            # Find x and y coordinates in the dataset
            x_coord = None
            y_coord = None
            time_coord = None

            # Check for common x coordinate names
            for coord in ds.coords:
                if coord.lower() in self.X_COORD_NAMES:
                    x_coord = coord
                elif coord.lower() in self.Y_COORD_NAMES:
                    y_coord = coord
                elif coord.lower() in self.TIME_COORD_NAMES:
                    time_coord = coord

            # Rename coordinates if found
            rename_dict = {}
            if x_coord and x_coord != x_name:
                rename_dict[x_coord] = x_name
            elif not x_coord:
                logger.warning(f"No x-like coordinate found in dataset. Looking for: {self.X_COORD_NAMES}")

            if y_coord and y_coord != y_name:
                rename_dict[y_coord] = y_name
            elif not y_coord:
                logger.warning(f"No y-like coordinate found in dataset. Looking for: {self.Y_COORD_NAMES}")

            if time_coord and time_name and time_coord != time_name:
                rename_dict[time_coord] = time_name
            elif not time_coord and time_name:
                logger.warning(f"No time-like coordinate found in dataset. Looking for: {self.TIME_COORD_NAMES}")

            if rename_dict:
                return ds.rename(rename_dict)
            return ds

        return self.add_preprocess_step(_rename_coords, "rename_coordinates")

    def drop_variables(self, variables: List[str]):
        """Add a preprocessing step to drop specified variables if they exist."""
        def _drop(ds):
            return ds.drop([v for v in variables if v in ds.data_vars or v in ds.coords])
        return self.add_preprocess_step(_drop, "drop_variables")

    def _preprocess(self, ds):
        """Apply all preprocessing steps in order."""
        # Apply default attributes
        if self.default_attrs:
            ds.attrs.update(self.default_attrs)

        # Apply all preprocessing steps
        logger.debug('Preprocessing dataset...')
        for step in self.preprocess_steps:
            ds = step(ds)

        return ds

    def load(self, pattern: str = '*',
             dask: bool = False, preprocess_override: Optional[Callable] = None, **kwargs):
        """
        Load .nc files with flexible preprocessing options.

        Args:
            pattern: Glob pattern to match files
            epsg: Optional EPSG code to set CRS
            dask: Whether to use dask for lazy loading
            preprocess_override: Optional function to override the default preprocessing pipeline
            **kwargs: Additional arguments passed to xr.merge or xr.open_mfdataset
        """
        files = self.list_files(pattern)

        if isinstance(files, str):
            files = [files]

        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")

        preprocess_func = preprocess_override if preprocess_override else self._preprocess

        def _load_nc(file):
            ds = xr.open_dataset(file)
            ds = preprocess_func(ds)
            return ds

        logger.info(f"Loading {len(files)} files...")
        if not dask:
            # Load and preprocess each file
            datasets = [_load_nc(f) for f in files]
            # Merge the datasets
            ds = xr.merge(datasets, **kwargs)
            # Preserve CRS information from the first dataset if available
            if datasets and hasattr(datasets[0], 'rio') and hasattr(datasets[0].rio, 'crs') and datasets[0].rio.crs:
                ds = ds.rio.write_crs(datasets[0].rio.crs)
        else:
            ds = xr.open_mfdataset(files, preprocess=preprocess_func, **kwargs)

        return ds

class STLoader(GridLoader):

    def __init__(self, root):
        super().__init__(root)

        self.set_default_attrs({"source": "Crespi (2021): https://doi.org/10.5194/essd-13-2801-2021"}) \
            .rename_coordinates(x_name="lon", y_name="lat", time_name = 'time') \
            .rename_variables({"tmean": "temperature", "prec": "precipitation"}) \
            .drop_variables(["transverse_mercator"]) \
            .set_crs(32632)
