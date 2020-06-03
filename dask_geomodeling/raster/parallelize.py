"""
Module containing blocks that parallelize raster blocks
"""
from math import floor, ceil
import numpy as np

from dask_geomodeling import utils
from .base import BaseSingle


__all__ = ["RasterTiler"]


class RasterTiler(BaseSingle):
    """Parallelize operations on a RasterBlock by tiling the request.

    Args:
      source (GeometryBlock): The source RasterBlock
      size (float or list): The maximum size of a tile in units of the
        projection. To specify different tile sizes for horizontal and vertical
        directions, provide the two as a list [lon, lat].
      projection (str): The projection as EPSG or WKT string in which to
        compute tiles (e.g. ``"EPSG:28992"``)
      topleft (list): The (lon, lat) coordinates of the topleft corner of any
        tile. This defines the tile grid. Default [0, 0].

    Note that the tile size is adjusted automatically so that there is an
    integer amount of cells inside each tile.

    Returns:
      RasterBlock
    """

    def __init__(self, source, size, projection, topleft=None):
        if not hasattr(size, "__iter__"):
            size = [size, size]
        elif len(size) != 2:
            raise ValueError(
                "The 'size' parameter should be a scalar or a list of length 2."
            )
        size = [float(x) for x in size]
        if size[0] <= 0 or size[1] <= 0:
            raise TypeError("Tile size should be greater than 0")
        if not isinstance(projection, str):
            raise TypeError(
                "'{}' object is not allowed".format(type(projection))
            )
        if topleft is None:
            topleft = [0.0, 0.0]
        elif len(topleft) != 2:
            raise ValueError(
                "The 'topleft' parameter should be a list of length 2."
            )
        super().__init__(
            source,
            [float(x) for x in size],
            projection,
            [float(x) for x in topleft],
        )

    @property
    def size(self):
        return self.args[1]

    @property
    def projection(self):
        return self.args[2]

    @property
    def topleft(self):
        return self.args[3]

    def get_sources_and_requests(self, **request):
        if request["mode"] != "vals":
            return [(self.source, request)]

        sr = utils.get_sr(request["projection"])
        if not sr.IsSame(utils.get_sr(self.projection)):
            raise RuntimeError("RasterTiler does not support reprojection")

        # get the requested cell size
        x1, y1, x2, y2 = request["bbox"]
        cell_width = (x2 - x1) / request["width"]
        cell_height = (y2 - y1) / request["height"]
        if cell_width <= 0 or cell_height <= 0:
            return [(self.source, request)]  # pass point requests through

        # adjust tile size so that it matches the cell size
        h = max(round(self.size[0] / cell_width), 1) * cell_width
        w = max(round(self.size[1] / cell_width), 1) * cell_width

        # generate tile IDs (given by tile_height, tile_width, topleft)
        tile_x, tile_y = self.topleft
        x_ids = floor((x1 - tile_x) / w), ceil((x2 - tile_x) / w)
        y_ids = floor((y1 - tile_y) / h), ceil((y2 - tile_y) / h)

        # get the tile corners as a single grid
        for x_id, y_id in np.ndindex(y_ids[1] - y_ids[0], x_ids[1] - x_ids[0]):
            _x1 = x_id * w + tile_x
            _y1 = x_id * w + tile_x
            _x2 = _x1 + w
            _y2 = _y1 + h
            # resize the bbox with an integer amount of cells if necessary
            if _x1 < x1:
                _x1 += floor((x1 - _x1) / cell_width) * cell_width
            if _y1 < y1:
                _y1 += floor((y1 - _y1) / cell_height) * cell_height
            if _x2 > x2:
                _x2 -= floor((_x2 - x2) / cell_width) * cell_width
            if _y2 > y2:
                _y2 -= floor((_y2 - y2) / cell_height) * cell_height
            _request = {
                **request,
                "bbox": (_x1, _x2, _y1, _y2),
                "width": int((_x2 - _x1) * cell_width),
                "height": int((_y2 - _y1) * cell_height),
            }
            yield (self.source, _request)

    @staticmethod
    def process(*all_data):
        if len(all_data) == 0:
            return
        else:
            return all_data[0]  # for non-tiled / meta / time requests
