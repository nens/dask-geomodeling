"""
Module containing blocks that parallelize non-geometry fields
"""
from itertools import product
from math import ceil
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from dask_geomodeling import utils

from .base import BaseSingle


__all__ = ["GeometryTiler"]


class GeometryTiler(BaseSingle):
    """Parallelize operations on a GeometryBlock by tiling the request.

    Args:
      source (GeometryBlock): The source GeometryBlock
      size (float): The maximum size of a tile in units of the projection
      projection (str): The projection as EPSG or WKT string in which to
        compute tiles (e.g. ``"EPSG:28992"``)

    Returns:
      GeometryBlock that only supports ``"centroid"`` and ``"extent"`` request
      modes.
    """

    def __init__(self, source, size, projection):
        if not isinstance(projection, str):
            raise TypeError("'{}' object is not allowed".format(type(projection)))
        super().__init__(source, float(size), projection)

    @property
    def size(self):
        return self.args[1]

    @property
    def projection(self):
        return self.args[2]

    def get_sources_and_requests(self, **request):
        mode = request["mode"]
        if mode == "extent":
            return [(self.source, request)]
        if mode != "centroid":
            raise NotImplementedError("Cannot process '{}' mode".format(mode))

        # tile the requested geometry in boxes that have a maximum size
        req_geometry = request["geometry"]
        tile_srs = self.projection
        request_srs = request["projection"]

        # transform the requested geometry into the tile geometry
        geometry = utils.shapely_transform(req_geometry, request_srs, tile_srs)

        x1, y1, x2, y2 = geometry.bounds
        ncols = ceil((x2 - x1) / self.size)
        nrows = ceil((y2 - y1) / self.size)

        if ncols <= 1 and nrows <= 1:
            return [(self.source, request)]  # shortcut

        # resize the tiles so that all tiles have similar dimensions
        size_x = (x2 - x1) / ncols
        size_y = (y2 - y1) / nrows
        tiles = [
            box(
                x1 + i * size_x,
                y1 + j * size_y,
                x1 + (i + 1) * size_x,
                y1 + (j + 1) * size_y,
            )
            for i, j in product(range(ncols), range(nrows))
        ]

        # intersect the tiles with the requested geometry
        series = gpd.GeoSeries(tiles).intersection(geometry)

        # drop empty tiles
        series = series[~series.is_empty]

        source = self.source
        request["projection"] = tile_srs
        return [(source, {**request, "geometry": tile}) for tile in series]

    @staticmethod
    def process(*all_data):
        if len(all_data) == 0:
            return {"features": gpd.GeoDataFrame([]), "projection": None}
        if len(all_data) == 1:
            return all_data[0]  # for non-tiled or extent requests
        features_lst = [
            data["features"]
            for data in all_data
            if data is not None and len(data.get("features")) != 0
        ]
        if len(features_lst) == 0:
            features = gpd.GeoDataFrame([])
        elif len(features_lst) == 1:
            features = features_lst[0]
        else:
            features = pd.concat(features_lst)

        projection = all_data[0]["projection"]
        return {"features": features, "projection": projection}
