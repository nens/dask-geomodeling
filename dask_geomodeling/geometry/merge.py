"""
Module containing merge operation that act on geometry blocks
"""
import pandas as pd
from shapely.geometry import box

from .base import GeometryBlock

__all__ = ["MergeGeometryBlocks"]


class MergeGeometryBlocks(GeometryBlock):
    """
    Merge two GeometryBlocks into one by index

    Provide two GeometryBlocks with the same original source to make sure they
    can be matched on index. The additional SeriesBlocks that have been added
    to the GeometryBlock will be combined to one GeometryBlock that contains
    all the information.

    Args:
      left (GeometryBlock): The left GeometryBlock to be combined.
      right (GeometryBlock): The right GeometryBlock to be combined.
      how (str, optional): The parameter that describes how the merge should
        be performed. There are four options:

        1. ``"left"``: The resulting GeometryBlock will have all the features
           that are present in the left GeometryBlock, no matter the features
           in the right GeometryBlock.
        2. ``"right"``: The resulting GeometryBlock will have all the features
           that are present in the right GeometryBlock, no matter the features
           in the left GeometryBlock.
        3. ``"inner"`` (default): The outcome will contain all the features
           that are present in both input GeometryBlocks. Features that are
           absent in one of the GeometryBlocks will be absent in the result.
        4. ``outer``: The result will contain all the features which are
           present in one of the input GeometryBlocks.

      suffixes (tuple, optional): Text to be added to the column
        names to distinguish whether they originate from the left or right
        GeometryBlock. Default: ``("", "_right")``.

    Returns:
      GeometryBlock that contains a combination of features and columns of the
      two input GeometryBlocks.
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
