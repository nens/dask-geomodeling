"""
Module containing blocks that parallelize raster blocks
"""
from math import floor, ceil
from itertools import product
import numpy as np

from dask import config
from dask_geomodeling import utils
from .base import BaseSingle


__all__ = ["RasterTiler"]


class RasterTiler(BaseSingle):
    """Parallelize operations on a RasterBlock by tiling the request.

    Note that the RasterTiler may adjust the request bbox, width and height
    so that the raster cells precisely fit in a tile.

    Args:
      source (GeometryBlock): The source RasterBlock
      size (float or list): The maximum size of a tile in units of the
        projection. To specify different tile sizes for horizontal and vertical
        directions, provide the two as a list [x,y] or [lon, lat].
      projection (str): The projection as EPSG or WKT string in which to
        compute tiles (e.g. ``"EPSG:28992"``). This request projection should
        equal the tiling projection.
      corner (list): The [x, y] or [lon, lat] coordinates of a tile corner.
        This defines the tile grid (together with `size`). Default [0, 0].

    Returns:
      RasterBlock
    """

    def __init__(self, source, size, projection, corner=None):
        if hasattr(size, "__iter__"):
            if len(size) != 2:
                raise ValueError("'size' should be a scalar or a list of length 2.")
            size = [float(x) for x in size]
        else:
            size = [float(size), float(size)]
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError("'size' should be greater than 0")
        if not isinstance(projection, str):
            raise TypeError("'{}' object is not allowed".format(type(projection)))
        try:
            utils.get_sr(projection)
        except RuntimeError:
            raise ValueError("Could not parse projection {}".format(projection))
        if corner is None:
            corner = [0.0, 0.0]
        elif len(corner) != 2:
            raise ValueError("The 'corner' parameter should None or lenght-2 list.")
        else:
            corner = [float(x) for x in corner]
        super().__init__(source, size, projection, corner)

    @property
    def size(self):
        return self.args[1]

    @property
    def projection(self):
        return self.args[2]

    @property
    def corner(self):
        return self.args[3]

    def get_sources_and_requests(self, **request):
        if request["mode"] != "vals":
            return [(None, None), (self.store, request)]

        sr = utils.get_sr(request["projection"])
        if not sr.IsSame(utils.get_sr(self.projection)):
            raise RuntimeError("RasterTiler does not support reprojection")

        # get the requested cell size
        x1, y1, x2, y2 = request["bbox"]
        cell_width = (x2 - x1) / request["width"]
        cell_height = (y2 - y1) / request["height"]
        if cell_width <= 0 or cell_height <= 0:
            # pass point requests through
            return [(None, None), (self.store, request)]

        # get tile grid
        tile_h, tile_w = self.size
        tile_x, tile_y = self.corner

        # adjust the cell size so that it fits an integer times in a tile
        cell_height = tile_h / max(round(tile_h / cell_height), 1)
        cell_width = tile_w / max(round(tile_w / cell_width), 1)

        # compute the tile edge coordinates (N + 1 edges for N tiles)
        edges_x = np.arange(
            floor((x1 - tile_x) / tile_w) * tile_w + tile_x,
            ceil((x2 - tile_x) / tile_w) * tile_w + tile_x + tile_w,
            tile_w,
        )
        edges_y = np.arange(
            floor((y1 - tile_y) / tile_h) * tile_h + tile_y,
            ceil((y2 - tile_y) / tile_h) * tile_h + tile_y + tile_h,
            tile_h,
        )

        # shrink the outmost edges with an integer amount of cells if necessary
        if edges_x[0] < x1:
            edges_x[0] += floor((x1 - edges_x[0]) / cell_width) * cell_width
        if edges_y[0] < y1:
            edges_y[0] += floor((y1 - edges_y[0]) / cell_height) * cell_height
        if edges_x[-1] > x2:
            edges_x[-1] -= floor((edges_x[-1] - x2) / cell_width) * cell_width
        if edges_y[-1] > y2:
            edges_y[-1] -= floor((edges_y[-1] - y2) / cell_height) * cell_height

        # yield process_kwargs to be able piece back together the tiles later
        result = [
            (
                {
                    "dtype": self.dtype,
                    "fillvalue": self.fillvalue,
                    "tile_ij": (
                        ((edges_x[:-1] - edges_x[0]) / cell_width).astype(int),
                        ((edges_y[-1] - edges_y[:-1]) / cell_height).astype(int),
                    ),
                    "shape_yx": (
                        int((edges_y[-1] - edges_y[0]) / cell_height),
                        int((edges_x[-1] - edges_x[0]) / cell_width),
                    ),
                },
                None,
            )
        ]

        # yield the tile requests
        for i, j in product(range(len(edges_x) - 1), range(len(edges_y) - 1)):
            _x1 = edges_x[i]
            _y1 = edges_y[j]
            _x2 = edges_x[i + 1]
            _y2 = edges_y[j + 1]
            _request = {
                **request,
                "bbox": (_x1, _y1, _x2, _y2),
                "width": int((_x2 - _x1) / cell_width),
                "height": int((_y2 - _y1) / cell_height),
            }
            result.append((self.store, _request))

        return result

    @staticmethod
    def process(process_kwargs, *all_data):
        if len(all_data) == 0:
            return
        elif process_kwargs is None:
            return all_data[0]  # for non-tiled / meta / time requests

        # go through all_data and get the shape
        shape_yx = process_kwargs["shape_yx"]
        for data in all_data:
            if data is not None:
                shape = (data["values"].shape[0],) + shape_yx
                break
        else:
            return  # return None if all data is None

        # check the size of the total raster to prevent going out of memory
        max_pixels = config.get("geomodeling.raster-limit")
        required_pixels = int(np.prod(shape))
        if required_pixels > max_pixels:
            raise RuntimeError(
                "The required raster size after combing the tiles exceeded "
                "the maximum ({} > {})".format(required_pixels, max_pixels)
            )

        # create the output array
        values = np.full(shape, process_kwargs["fillvalue"], process_kwargs["dtype"])

        # piece together the output array
        coords_x, coords_y = process_kwargs["tile_ij"]
        for (x, y), data in zip(product(coords_x, coords_y), all_data):
            if data is None:
                continue
            vals = data["values"]
            values[:, y - vals.shape[1] : y, x : x + vals.shape[2]] = vals
        return {"values": values, "no_data_value": process_kwargs["fillvalue"]}
