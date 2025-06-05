import logging
from datetime import datetime, timedelta

import numpy as np
import rioxarray
import xarray as xr
from zarr.codecs import BloscCodec

from src.grid.loader import STLoader
from src.grid.writer import GridWriter

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    loader = STLoader(r"C:\OneDrive\Scientific Network South Tyrol\Obwegs Lisa - 7. Climate_data")

    # Load data with configured preprocessing
    ds = loader.load("TEMPERATURE/**/*201901.nc", dask = False)
    print(ds)
    print(ds.chunks)
    print(ds.rio.crs)

    # Save
    writer = GridWriter('examples/data')
    filename = 'test.zarr'
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
    all_dates = np.arange(datetime(2019,1,1,12), datetime(2019,12,31,12), timedelta(days=1))
    store_shape = (len(all_dates), len(ds.lon), len(ds.lat))
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
        coords = {'time': all_dates, "lon": ds.lon, "lat": ds.lat},
        chunks = chunk_structure,
        crs = ds.rio.crs.to_epsg(),
        encoding = encoding,
        variables = {'temperature': np.dtype('int32'), "precipitation": np.dtype('int32')}
    )

    ds_re = ds.chunk(chunk_structure)
    writer.to_zarr(ds_re, filename, append_dims=["time"])

    store = xr.open_zarr(f"examples/data/{filename}")
    print(store)
    print(store.chunks)
    print(store.rio.crs)
    


if __name__ == "__main__":
    main()
