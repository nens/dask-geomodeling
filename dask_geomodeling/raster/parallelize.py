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

    The tiles are computed starting at the topleft corner of the requested box.

    Args:
      source (GeometryBlock): The source RasterBlock
      size (int or list): The maximum size of a tile in pixels (cells)

    Returns:
      RasterBlock
    """

    def __init__(self, source, size):
        if hasattr(size, "__iter__"):
            if len(size) != 2:
                raise ValueError("'size' should be a scalar or a list of length 2.")
            size = [int(x) for x in size]
        else:
            size = [int(size), int(size)]
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError("'size' should be greater than 0")
        super().__init__(source, size)

    @property
    def size(self):
        return self.args[1]

    def get_sources_and_requests(self, **request):
        if request["mode"] != "vals":
            return [(None, None), (self.store, request)]

        # get the requested cell size
        x1, y1, x2, y2 = request["bbox"]
        cellsize_x = (x2 - x1) / request["width"]
        cellsize_y = (y2 - y1) / request["height"]
        if cellsize_x == 0 and cellsize_y == 0:
            # pass point requests through
            return [(None, None), (self.store, request)]

        # get tile size and compute tile edge coordinates
        tilesize_x = cellsize_x * self.size[0]
        tilesize_y = cellsize_y * self.size[1]
        x = np.arange(x1, x2, tilesize_x)
        y = np.arange(y1, y2, tilesize_y)

        # handle the 'leftover' tiles
        if x[-1] != x2:
            x = np.append(x, x2)
        if y[-1] != y2:
            y = np.append(y, y2)

        count_x, count_y = len(x) - 1, len(y) - 1

        # yield process_kwargs to be able piece back together the tiles later
        result = [
            (
                {
                    "dtype": self.dtype,
                    "fillvalue": self.fillvalue,
                    "shape_yx": (request["height"], request["width"]),
                    "count_xy": (count_x, count_y),
                    "size_xy": self.size,
                },
                None,
            )
        ]
        for i, j in product(range(count_x), range(count_y)):
            _request = {
                **request,
                "bbox": (x[i], y[j], x[i + 1], y[j + 1]),
                "width": int((x[i + 1] - x[i]) / cellsize_x),
                "height": int((y[j + 1] - y[j]) / cellsize_y),
            }
            result.append((self.store, _request))

        return result

    @staticmethod
    def process(process_kwargs, *all_data):
        if len(all_data) == 0:
            return
        elif process_kwargs is None:
            return all_data[0]  # for non-tiled / meta / time requests

        # go through all_data and get the temporal shape
        shape_yx = process_kwargs["shape_yx"]
        for data in all_data:
            if data is not None:
                shape = (data["values"].shape[0],) + shape_yx
                break
        else:
            return  # return None if all data is None

        # create the output array
        values = np.full(shape, process_kwargs["fillvalue"], process_kwargs["dtype"])

        # The tile requests where generated from topleft to bottomright. Due to
        # y-axis swapping, the returned boxes are from bottomleft to topright.
        count_x, count_y = process_kwargs["count_xy"]
        size_x, size_y = process_kwargs["size_xy"]
        for (i, j), data in zip(product(range(count_x), range(count_y)), all_data):
            if data is None:
                continue
            vals = data["values"]
            x = i * size_x
            y = j * size_y
            values[:, -(y + vals.shape[1]) : -y or None, x : x + vals.shape[2]] = vals
        return {"values": values, "no_data_value": process_kwargs["fillvalue"]}
