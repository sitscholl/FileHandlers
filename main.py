import logging

from src.grid.loader import GridLoader

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    loader = GridLoader(r"D:\Scientific Network South Tyrol\Obwegs Lisa - Climate_data_RAW") \
        .set_default_attrs({"source": "climate_model_v1"}) \
        .rename_coordinates(x_name="x", y_name="y", time_name = 'datetime') \
        .rename_variables({"tmean": "temperature", "prec": "precipitation"}) \
        .add_preprocess_step(lambda ds: ds.assign_attrs(spatial_ref=ds.transverse_mercator.attrs)) \
        .drop_variables(["transverse_mercator"])

    # Load data with configured preprocessing
    ds = loader.load("TEMPERATURE/**/*01.nc", epsg=32632, dask = True, chunks = 'auto')
    print(ds)
    print(ds.chunks)


if __name__ == "__main__":
    main()
