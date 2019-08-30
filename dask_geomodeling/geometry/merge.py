"""
Module containing merge operation that act on geometry blocks
"""
import pandas as pd
from shapely.geometry import box

from .base import GeometryBlock

__all__ = ["MergeGeometryBlocks"]


class MergeGeometryBlocks(GeometryBlock):
    """Merge two GeometryBlocks into one by index

    :param left: left geometry data to merge
    :param right: right geometry data to merge
    :param how: type of merge to be performed. One of ``‘left’, ‘right’,
      ‘outer’, ‘inner'``. Default ``‘inner’``.
    :param suffixes: suffix to apply to overlapping column names in the left
      and right side, respectively. Default ``('', '_right')``.

    :type left: GeometryBlock
    :type right: GeometryBlock
    :type how: string
    :type suffixes: tuple

    See also merge:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
    """

    allow_how_joins = ("left", "right", "outer", "inner")

    def __init__(self, left, right, how="inner", suffixes=("", "_right")):
        if not isinstance(left, GeometryBlock):
            raise TypeError("'{}' object is not allowed".format(type(left)))
        if not isinstance(right, GeometryBlock):
            raise TypeError("'{}' object is not allowed".format(type(right)))
        if how not in self.allow_how_joins:
            raise KeyError(
                "'{}' is not part of the list of operations: "
                "{}".format(how, self.allow_how_joins)
            )
        if (
            not isinstance(suffixes[0], str)
            or not isinstance(suffixes[1], str)
            or len(suffixes) != 2
        ):
            raise TypeError("'{}' object is not " "allowed".format(type(suffixes)))
        super().__init__(left, right, how, suffixes)

    @property
    def left(self):
        return self.args[0]

    @property
    def right(self):
        return self.args[1]

    @property
    def how(self):
        return self.args[2]

    @property
    def suffixes(self):
        return self.args[3]

    @property
    def columns(self):
        left = self.left.columns
        right = self.right.columns
        result = left ^ right  # column in left or right, not both
        overlap = left & right
        for col in overlap:
            result |= {col + self.suffixes[0], col + self.suffixes[1]}
        return result

    def get_sources_and_requests(self, **request):
        process_kwargs = {
            "how": self.how,
            "suffixes": self.suffixes,
            "mode": request["mode"],
        }
        return [(self.left, request), (self.right, request), (process_kwargs, None)]

    @staticmethod
    def process(left, right, kwargs):
        mode = kwargs["mode"]
        how = kwargs["how"]
        projection = left["projection"]

        if mode == "intersects" or mode == "centroid":
            merged = pd.merge(
                left["features"],
                right["features"],
                how=kwargs.get("how"),
                suffixes=kwargs.get("suffixes"),
                left_index=True,  # we merge by index, left and right.
                right_index=True,
            )
            return {"features": merged, "projection": projection}
        elif mode == "extent":
            if how == "left":
                return left

            elif how == "right":
                return right

            elif how == "inner":
                values = None
                if left["extent"] and right["extent"]:
                    left_shape = box(*left["extent"])
                    right_shape = box(*right["extent"])
                    extent = left_shape.intersection(right_shape)
                    if not extent.is_empty:
                        values = extent.bounds
                return {"extent": values, "projection": projection}

            elif how == "outer":
                values = None
                if left["extent"] and right["extent"]:
                    left_shape = box(*left["extent"])
                    right_shape = box(*right["extent"])
                    values = left_shape.union(right_shape).bounds
                elif left["extent"]:
                    values = left["extent"]
                elif right["extent"]:
                    values = right["extent"]
                return {"extent": values, "projection": projection}
