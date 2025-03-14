from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

gdal.SetConfigOption("GDAL_MEM_ENABLE_OPEN", "YES")

from . import config  # NOQA
from .core import Block, construct  # NOQA
from . import raster  # NOQA
from . import geometry  # NOQA
