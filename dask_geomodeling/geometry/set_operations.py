"""
Module containing geometry block set operations
"""
import geopandas as gpd
from shapely.geometry import box

from .base import GeometryBlock, BaseSingle

__all__ = ["Difference", "Intersection"]


class Difference(BaseSingle):
    """
    Calculate the geometric difference of two GeometryBlocks.

    All geometries in the source GeometryBlock will be adapted by geometries
    with the same index from the second GeometryBlock. The difference operation
    removes any overlap between the geometries from the first geometry.

    Args:
      source (GeometryBlock): First geometry source.
      other (GeometryBlock): Second geometry source.

    Returns:
      A GeometryBlock with altered geometries. Properties are preserved.
    """

    def __init__(self, source, other):
        if not isinstance(other, GeometryBlock):
            raise TypeError("'{}' object is not allowed".format(type(other)))
        super().__init__(source, other)  # type of source is checked in super()

    @property
    def other(self):
        return self.args[1]

    def get_sources_and_requests(self, **request):
        if request["mode"] == "extent":
            # the resulting extent might be smaller than the source's extent,
            # but we would have to execute the Difference in order to know.
            return [(self.source, request)]

        # we need to get the extent of the geometries in source, in order to
        # fetch all needed geometries in other
        extent_request = request.copy()
        extent_request["mode"] = "extent"
        extent = self.source.get_data(**extent_request)["extent"]

        if extent is None:  # shortcut when there are no geometries present
            projection = request["projection"]
            return [({"empty": True, "projection": projection}, None)]

        other_request = request.copy()
        other_request["geometry"] = box(*extent)
        return [(self.source, request), (self.other, other_request)]

    @staticmethod
    def process(source_data, other_data=None):
        if other_data is None:
            if source_data.get("empty"):  # catches the no-geometry shortcut
                return {
                    "features": gpd.GeoDataFrame([]),
                    "projection": source_data["projection"],
                }
            else:
                return source_data

        a = source_data["features"]
        b = other_data["features"]
        if len(a) == 0 or len(b) == 0:
            return source_data  # do nothing

        a_series = a["geometry"]
        b_series = b["geometry"].reindex(a_series.index)
        result_series = a_series.difference(b_series)

        # re-insert geometries that were missing in b (we want A - None = A)
        missing_in_b = b_series.isna()
        result_series[missing_in_b] = a_series[missing_in_b]
        result = a.set_geometry(result_series)
        return {"features": result, "projection": source_data["projection"]}


class Intersection(BaseSingle):
    """Calculate the intersection of a GeometryBlock with the request geometry.

    Normally, geometries returned by a GeometryBlock may be partially outside
    of the requested geometry. This block ensures that the geometries are
    strictly inside the requested geometry by taking the intersection of
    each geometry with the request geometry.

    Args:
      source (GeometryBlock): Input geometry source.

    Returns:
      A GeometryBlock with altered geometries. Properties are preserved.
    """

    def __init__(self, source, other=None):
        if isinstance(other, GeometryBlock):
            raise NotImplementedError(
                "Cannot compare geometries with another geometry datasource"
            )
        elif other is not None:
            raise NotImplementedError(
                "Cannot compare geometries with a constant geometry"
            )
        super().__init__(source, other)

    @property
    def other(self):
        return self.args[1]

    def get_sources_and_requests(self, **request):
        return [(self.source, request), (request["geometry"], None)]

    @staticmethod
    def process(data, geometry):
        # the features will always be in the projection of geometry so we can
        # safely not think about projections here.
        # This cover 'intersects' and 'centroid' mode.
        if "features" in data:
            features = data["features"]
            features = features.set_geometry(features.geometry.intersection(geometry))
            return {"features": features, "projection": data["projection"]}
        elif "extent" in data:
            # intersect the bboxes
            bbox1 = data["extent"]
            bbox2 = geometry.bounds
            bbox = (
                max(bbox1[0], bbox2[0]),
                max(bbox1[1], bbox2[1]),
                min(bbox1[2], bbox2[2]),
                min(bbox1[3], bbox2[3]),
            )
            return {"extent": bbox, "projection": data["projection"]}
