from zarr.codecs import BloscCodec
import numpy as np
import xarray as xr

from datetime import datetime, timedelta
from pathlib import Path

from filehandler.grid.writer import GridWriter

zarr_path = Path("examples/data/large_store.zarr")
writer = GridWriter(zarr_path.parent)
filename = zarr_path.name
compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
all_dates = np.arange(datetime(1980,1,1,12), datetime(2024,12,31,12), timedelta(days=1))
store_shape = (len(all_dates), 800, 800)
chunk_structure = {'time': -1, 'lon': 80, 'lat': 80}
encoding = {i: {
        'dtype': 'int32',
        'scale_factor': 0.01,
        '_FillValue': -9999,
        'compressors': compressor
    } for i in ['temperature', 'precipitation']
}

writer.generate_preallocated_zarr_store(
    filename = filename,
    shape = store_shape,
    coords = {'time': all_dates, "lon": np.arange(800), "lat": np.arange(800)},
    chunks = None,
    crs = 32632,
    encoding = encoding,
    variables = {'temperature': np.dtype('int32'), "precipitation": np.dtype('int32')}
)

store = xr.open_zarr(zarr_path)
print(store['temperature'].shape)