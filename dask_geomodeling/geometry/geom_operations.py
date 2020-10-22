"""
Module containing operations that return series from geometry fields
"""

import pandas as pd

from dask_geomodeling.utils import shapely_transform

from .base import SeriesBlock, GeometryBlock


__all__ = ["Area"]


class Area(SeriesBlock):
    """
    Calculate the area of features in a GeometryBlock.

    Provide a GeometryBlock and a projection. Returns the area of each
    individual geometry in the input block, in that projection.

    Args:
      source (GeometryBlock): Source GeometryBlock which contains the features.
      projection (str): Projection in which to compute the area
        (i.e. ``"epsg:28992"``).

    Returns:
      SeriesBlock with only the computed area
    """

    def __init__(self, source, projection):
        if not isinstance(source, GeometryBlock):
            raise TypeError("'{}' object is not allowed".format(type(source)))
        if not isinstance(projection, str):
            raise TypeError("Argument 'projection' must be a str.")
        super().__init__(source, projection)

    @property
    def source(self):
        return self.args[0]

    @property
    def projection(self):
        return self.args[1]

    @staticmethod
    def process(data, projection):
        if "features" not in data or len(data["features"]) == 0:
            return pd.Series([], dtype=float)

        return (
            data["features"]
            .geometry.apply(shapely_transform, args=(data["projection"], projection))
            .area
        )
