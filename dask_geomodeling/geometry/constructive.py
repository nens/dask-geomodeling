"""
Module containing geometry block constructive operations
"""
import numbers

from dask_geomodeling.utils import Extent, shapely_transform, transform_extent

from .base import BaseSingle

__all__ = ["Buffer", "Simplify"]


class Buffer(BaseSingle):
    """
    Buffer ('expand') geometries with a given value.

    A GeometryBlock and a buffer distance are provided. Each feature in the
    GeometryBlock is buffered with the distance provided, resulting in updated
    geometries.

    Args:
      source (GeometryBlock): The source GeometryBlock whose geometry will be
        updated.
      distance (float): The distance used to buffer all features. The distance
        is measured in the unit of the given projection (e.g. m, °).
      projection (str): The projection used in the operation provided in the
        format: ``"EPSG:28992"``.
      resolution (integer, optional): The resolution of the buffer provided as
        the number of points used to represent a quarter of a circle. The
        default value is ``16``.

    Returns:
      GeometryBlock with buffered geometries.

    """

    def __init__(self, source, distance, projection, resolution=16):
        if not isinstance(distance, numbers.Real):
            raise TypeError("Argument 'distance' must be a float or int.")
        if not isinstance(projection, str):
            raise TypeError("Argument 'projection' must be a str.")
        if not isinstance(resolution, int):
            raise TypeError("Argument 'resolution' must be an int.")
        super().__init__(source, distance, projection, resolution)

    @property
    def distance(self):
        return self.args[1]

    @property
    def projection(self):
        return self.args[2]

    @property
    def resolution(self):
        return self.args[3]

    def get_sources_and_requests(self, **request):
        process_kwargs = {
            "distance": self.distance,
            "buf_srs": self.projection,
            "resolution": self.resolution,
        }
        return [(self.source, request), (process_kwargs, None)]

    @staticmethod
    def process(data, kwargs):
        if "features" in data:
            if len(data["features"]) == 0:
                return data
            req_srs = data["projection"]
            buf_srs = kwargs["buf_srs"]
            features = data["features"].set_geometry(
                data["features"]
                .geometry.apply(shapely_transform, args=(req_srs, buf_srs))
                .buffer(distance=kwargs["distance"], resolution=kwargs["resolution"])
                .apply(shapely_transform, args=(buf_srs, req_srs))
            )
            return {"features": features, "projection": req_srs}
        elif "extent" in data:
            if not data["extent"]:
                return data
            req_srs = data["projection"]
            buf_srs = kwargs["buf_srs"]
            distance = kwargs["distance"]
            extent = transform_extent(data["extent"], req_srs, buf_srs)
            extent = Extent(extent, buf_srs).buffered(distance).bbox
            extent = transform_extent(extent, buf_srs, req_srs)
            return {"extent": extent, "projection": req_srs}
        else:
            raise NotImplementedError("Dunno this mode!")


class Simplify(BaseSingle):
    """
    Simplify geometries, mainly to make them computationally more efficient.

    Provide a GeometryBlock and a tolerance value to simplify the geometries.
    As a result all features in the GeometryBlock are simplified.

    Args:
      source (GeometryBlock): Source of the geometries to be simplified.
      tolerance (float): The tolerance used in the simplification. If no
        tolerance is given the ``"min_size"`` request parameter is used.
      preserve_topology (boolean, optional): Determines whether the topology
        should be preserved in the operation. Defaults to ``True``.

    Returns:
      GeometryBlock which was provided as input with a simplified geometry.

    """

    def __init__(self, source, tolerance=None, preserve_topology=True):
        if tolerance is not None:
            tolerance = float(tolerance)
        super().__init__(source, tolerance, bool(preserve_topology))

    @property
    def tolerance(self):
        return self.args[1]

    @property
    def preserve_topology(self):
        return self.args[2]

    def get_sources_and_requests(self, **request):
        process_kwargs = {
            "tolerance": self.tolerance or request.get("min_size") or 0.0,
            "preserve_topology": self.preserve_topology,
        }
        return [(self.source, request), (process_kwargs, None)]

    @staticmethod
    def process(data, kwargs):
        if "features" not in data:
            # basically assumes that the extent will not change during simplify
            return data
        features = data["features"].set_geometry(
            data["features"].geometry.simplify(**kwargs)
        )  # for 'intersects' and 'centroid' mode.
        return {"features": features, "projection": data["projection"]}
