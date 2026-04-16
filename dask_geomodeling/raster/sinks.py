import glob
import os
import logging

import numpy as np
from osgeo import gdal, gdal_array
from dask.base import tokenize
from dask_geomodeling import utils

from .base import BaseSingle, RasterBlock
from .parallelize import RasterTiler

__all__ = ["RasterFileSink", "to_file"]

logger = logging.getLogger(__name__)


class RasterFileSink(BaseSingle):
    """Write raster data to GeoTIFF files in a specified directory.

    Use RasterFileSink.merge_files to merge tiles into a VRT file.

    Args:
      source (RasterBlock): The raster block the data is coming from.
      url (str): The target directory to put the files in. If relative, it is
        taken relative to the geomodeling.root setting.

    Relevant settings can be adapted as follows:
      >>> from dask import config
      >>> config.set({"geomodeling.root": '/my/output/data/path'})
    """

    def __init__(self, source, url):
        if not isinstance(source, RasterBlock):
            raise TypeError("'{}' object is not allowed".format(type(source)))
        safe_url = utils.safe_file_url(url)
        super().__init__(source, safe_url)

    @property
    def url(self):
        return self.args[1]

    def get_sources_and_requests(self, **request):
        if request["mode"] != "vals":
            return [(self.store, request), ({}, None)]

        process_kwargs = {
            "url": self.url,
            "hash": tokenize(request)[:7],
            "bbox": request["bbox"],
            "projection": request["projection"],
        }
        return [(self.store, request), (process_kwargs, None)]

    @staticmethod
    def process(data, process_kwargs):
        if not process_kwargs:
            # non-vals mode: forward data as-is
            return data

        if data is None or "values" not in data:
            return None

        values = data["values"]
        no_data_value = data["no_data_value"]

        if values.ndim != 3 or values.shape[0] != 1:
            raise ValueError(
                "Expected a single-band raster (shape (1, H, W)), got shape {}".format(
                    values.shape
                )
            )

        band_data = values[0]

        # Skip saving if all values are nodata
        if no_data_value is not None and np.all(band_data == no_data_value):
            return None

        height, width = band_data.shape

        # Map numpy dtype to GDAL type
        gdal_type = gdal_array.NumericTypeCodeToGDALTypeCode(band_data.dtype)
        if gdal_type is None:
            raise ValueError("Unsupported dtype '{}' for GDAL".format(band_data.dtype))

        path = utils.safe_abspath(process_kwargs["url"])
        os.makedirs(path, exist_ok=True)

        filename = process_kwargs["hash"] + ".tif"
        filepath = os.path.join(path, filename)

        # Derive geo_transform from bbox
        bbox = process_kwargs["bbox"]
        x1, y1, x2, y2 = bbox
        geo_transform = (
            x1,
            (x2 - x1) / width,
            0,
            y2,
            0,
            -(y2 - y1) / height,
        )

        projection = process_kwargs["projection"]

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            filepath, width, height, 1, gdal_type, ["COMPRESS=DEFLATE", "TILED=YES"]
        )
        dataset.SetGeoTransform(geo_transform)

        sr = utils.get_sr(projection)
        dataset.SetSpatialRef(sr)

        band = dataset.GetRasterBand(1)
        if no_data_value is not None:
            band.SetNoDataValue(float(no_data_value))
        band.WriteArray(band_data)

        return None

    @staticmethod
    def merge_files(path, target):
        """Merge GeoTIFF files (the output of this Block) into a VRT file.

        Args:
          path (str): The source directory containing .tif files.
          target (str): The target .vrt file path.
        """
        path = utils.safe_abspath(path)
        target = utils.safe_abspath(target)

        if os.path.exists(target):
            raise IOError("Target '{}' already exists".format(target))

        source_paths = sorted(glob.glob(os.path.join(path, "*.tif")))
        if len(source_paths) == 0:
            raise IOError("No source .tif files found in '{}'".format(path))

        gdal.BuildVRT(target, source_paths)


def to_file(source, url, tile_size, **request):
    """Utility function to export data from a RasterBlock to a file on disk.

    The raster is exported as tiled GeoTIFFs which are merged into a VRT
    at ``url``.

    Args:
      source (RasterBlock): the block the data is coming from
      url (str): The target VRT file path.
      tile_size (int or list): The tile size in pixels. The export is split
        into tiles of this size which are merged into a VRT.
      bbox (tuple): bounding box ``(x1, y1, x2, y2)``
      projection (str): The projection as a WKT string or EPSG code.
      width (int): The output width in pixels.
      height (int): The output height in pixels.
      start (datetime): start date as UTC datetime
      stop (datetime): stop date as UTC datetime
      **request: see RasterBlock request specification

    Relevant settings can be adapted as follows:
      >>> from dask import config
      >>> config.set({"geomodeling.root": '/my/output/data/path'})
    """
    request["mode"] = "vals"
    if "projection" not in request:
        if source.projection is None:
            raise ValueError(
                "Cannot determine the projection from the source raster. "
                "Please provide a 'projection' argument."
            )
        request["projection"] = source.projection
    if "bbox" not in request:
        if source.geometry is None:
            raise ValueError(
                "Cannot determine the extent from the source raster. "
                "Please provide a 'bbox' argument."
            )
        x1, x2, y1, y2 = source.geometry.GetEnvelope()
        request["bbox"] = x1, y1, x2, y2
    if "width" not in request or "height" not in request:
        if source.geo_transform is None:
            raise ValueError(
                "Cannot determine the pixel size from the source raster. "
                "Please provide 'width' and 'height' arguments."
            )
        geo_transform = source.geo_transform
        x1, y1, x2, y2 = request["bbox"]
        request["width"] = int(round((x2 - x1) / abs(float(geo_transform[1]))))
        request["height"] = int(round((y2 - y1) / abs(float(geo_transform[5]))))

    path = utils.safe_abspath(url)

    if os.path.isdir(path):
        path = os.path.join(path, "output.vrt")
    tiles_dir = os.path.join(os.path.split(path)[0], "tiles")

    sink = RasterFileSink(source, tiles_dir)
    tiler = RasterTiler(sink, tile_size)
    tiler.get_data(**request)

    RasterFileSink.merge_files(tiles_dir, path)
