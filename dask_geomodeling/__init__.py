from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

gdal.SetConfigOption("GDAL_MEM_ENABLE_OPEN", "YES")

from . import config  # NOQA
from .core import Block, construct  # NOQA
from . import raster  # NOQA
from . import geometry  # NOQA

# Version is managed in pyproject.toml and read from package metadata at runtime
try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("dask-geomodeling")
except PackageNotFoundError:
    # Package not installed, use a placeholder
    __version__ = "unknown"
