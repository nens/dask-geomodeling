"""
Module containing blocks that parallelize raster blocks
"""
from itertools import product
import numpy as np

from .base import BaseSingle


__all__ = ["RasterTiler"]


class RasterTiler(BaseSingle):
    """Parallelize operations on a RasterBlock by tiling the request.

    Args:
      source (GeometryBlock): The source RasterBlock
      tile_size (int or list): The maximum size of a tile in pixels (cells)
         To specify different tile sizes for horizontal and vertical
         directions, provide the two as a list [width, height].

    Returns:
      RasterBlock
    """

    def __init__(self, source, tile_size):
        if hasattr(tile_size, "__iter__"):
            if len(tile_size) != 2:
                raise ValueError(
                    "'tile_size' should be a scalar or a list of length 2."
                )
            tile_size = [int(x) for x in tile_size]
        else:
            tile_size = [int(tile_size), int(tile_size)]
        if tile_size[0] <= 0 or tile_size[1] <= 0:
            raise ValueError("'tile_size' should be greater than 0")
        super().__init__(source, tile_size)

    @property
    def tile_size(self):
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
        tilesize_x = cellsize_x * self.tile_size[0]
        tilesize_y = cellsize_y * self.tile_size[1]
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
                    "tilesize_xy": self.tile_size,
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

        # The tile order that was generated in the get_sources_and_request
        # starts at low x, low y and ends at high x, high y. As we are working
        # here with array indices [i, j] we need to take into account that the
        # vertical axis swaps direction: high y maps to low i.
        count_x, count_y = process_kwargs["count_xy"]
        tilesize_x, tilesize_y = process_kwargs["tilesize_xy"]
        for index, data in zip(product(range(count_x), range(count_y)), all_data):
            if data is None:
                continue
            vals = data["values"]
            j = index[0] * tilesize_x
            i = index[1] * tilesize_y
            values[:, -(i + vals.shape[1]) : -i or None, j : j + vals.shape[2]] = vals
        return {"values": values, "no_data_value": process_kwargs["fillvalue"]}
