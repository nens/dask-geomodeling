import dask.config

defaults = {
    "root": "/",
    "raster_limit": 12 * (1024 ** 2),  # ca. 100 MB of float64
    "geometry_limit": 10000,
}

dask.config.update_defaults({"geomodeling": defaults})
