import dask.config
import os

defaults = {
    "root": os.getcwd(),
    "strict-file-paths": False,
    "raster-limit": 12 * (1024 ** 2),  # ca. 100 MB of float64
    "raster-limit-timesteps": 65536,  # the same as GDAL band limit
    "geometry-limit": 10000,
}

dask.config.update_defaults({"geomodeling": defaults})
