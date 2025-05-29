import logging

from src.grid.loader import STLoader, GridLoader
from src.grid.utils import generate_preallocated_zarr_store

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    # loader = GridLoader(r"D:\Scientific Network South Tyrol\Obwegs Lisa - Climate_data_RAW") \
    #     .set_default_attrs({"source": "Crespi (2019)"}) \
    #     .rename_coordinates(x_name="lon", y_name="lat", time_name = 'datetime') \
    #     .rename_variables({"tmean": "temperature", "prec": "precipitation"}) \
    #     .drop_variables(["transverse_mercator"]) \
    #     .set_crs(32632)

    loader = STLoader(r"D:\Scientific Network South Tyrol\Obwegs Lisa - Climate_data_RAW")

    # Load data with configured preprocessing
    ds = loader.load("TEMPERATURE/**/*01.nc", dask = True, chunks = 'auto')
    print(ds)
    print(ds.chunks)
    print(ds.rio.crs)

    # Save
    # generate_preallocated_zarr_store(
    #     filename = 'out.zarr',
    #     shape = ds.shape,
    #     coords = ds.coords,
    #     chunks = {'datetime': -1, 'lon': 'auto', 'lat': 'auto'},
    #     crs = 32632,
    #     encoding=None,
    #     vars = ds.dtypes,
    # )


if __name__ == "__main__":
    main()
