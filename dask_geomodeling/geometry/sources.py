"""
Module containing geometry sources.
"""
import fiona
import geopandas as gpd
import warnings

from dask import config
from dask_geomodeling import utils
from .base import GeometryBlock

from shapely.errors import WKTReadingError
from shapely.wkt import loads as load_wkt


# this import is a copy from geopandas.io.files

__all__ = ["GeometryFileSource", "GeometryWKTSource"]


class GeometryFileSource(GeometryBlock):
    """A geometry source that opens a geometry file from disk.

    The input of this blocks is by default limited by the global
    geomodeling.geometry-limit setting.

    Args:
      url (str): Path (URL) to the file. If relative, it is taken relative to
        the geomodeling.root setting.
      layer (str, optional): The layer name in the source to select. If None,
        (default) the first layer is used.
      id_field (str, optional): The field name to use as feature index.
        Default ``"id"``.

    Relevant settings can be adapted as follows:
      >>> from dask import config
      >>> config.set({"geomodeling.root": '/my/data/path'})
      >>> config.set({"geomodeling.geometry-limit": 100000})
    """

    def __init__(self, url, layer=None, id_field="id"):
        safe_url = utils.safe_file_url(url)
        super().__init__(safe_url, layer, id_field)

    @property
    def url(self):
        return self.args[0]

    @property
    def layer(self):
        return self.args[1]

    @property
    def id_field(self):
        return self.args[2]

    @property
    def path(self):
        return utils.safe_abspath(self.url)

    @property
    def columns(self):
        # this is exactly how geopandas determines the columns
        with utils.fiona_env(), fiona.open(self.path) as reader:
            properties = reader.meta["schema"]["properties"]
            return set(properties.keys()) | {"geometry"}

    def get_sources_and_requests(self, **request):
        # check the filters: this block does not support lookups
        if request.get("filters") is None:
            request["filters"] = dict()
        if request["filters"]:
            for field, value in request["filters"].items():
                if "__" in field:
                    raise ValueError("Filter '{}' is not supported".format(field))
        mode = request.get("mode", "intersects").lower()
        if mode not in ("extent", "intersects", "centroid"):
            raise ValueError("Unknown mode '{}'".format(mode))
        request["mode"] = mode
        # just pass on the args and request here
        request["layer"] = self.layer
        request["id_field"] = self.id_field
        return [(self.url, None), (request, None)]

    @staticmethod
    def process(url, request):
        path = utils.safe_abspath(url)

        # convert the requested projection to a geopandas CRS
        crs = utils.get_crs(request["projection"])

        # convert the requested shapely geometry object to a GeoSeries
        filt_geom = gpd.GeoSeries([request["geometry"]], crs=crs)

        # acquire the data, filtering on the filt_geom bbox
        f = gpd.GeoDataFrame.from_file(path, bbox=filt_geom, layer=request["layer"])
        if len(f) == 0:
            # return directly if there is no data
            if request.get("mode") == "extent":
                return {"projection": request["projection"], "extent": None}
            else:  # this takes modes 'centroid' and 'intersects'
                return {
                    "projection": request["projection"],
                    "features": gpd.GeoDataFrame([]),
                }

        f.set_index(request["id_field"], inplace=True)

        # apply the non-geometry field filters first
        mask = None
        for field, value in request["filters"].items():
            if field not in f.columns:
                continue
            _mask = f[field] == value
            if mask is None:
                mask = _mask
            else:
                mask &= _mask
        if mask is not None:
            f = f[mask]

        # convert the data to the requested crs
        utils.geodataframe_transform(f, utils.crs_to_srs(f.crs), request["projection"])

        # compute the bounds of each geometry and filter on min_size
        min_size = request.get("min_size")
        if min_size:
            bounds = f["geometry"].bounds
            widths = bounds["maxx"] - bounds["minx"]
            heights = bounds["maxy"] - bounds["miny"]
            f = f[(widths > min_size) | (heights > min_size)]

        # only return geometries that truly intersect the requested geometry
        if request["mode"] == "centroid":
            with warnings.catch_warnings():  # geopandas warns if in WGS84
                warnings.simplefilter("ignore")
                f = f[f["geometry"].centroid.within(filt_geom.iloc[0])]
        else:
            f = f[f["geometry"].intersects(filt_geom.iloc[0])]

        if request.get("mode") == "extent":
            return {
                "projection": request["projection"],
                "extent": tuple(f.total_bounds),
            }
        else:  # this takes modes 'centroid' and 'intersects'
            # truncate the number of geometries if necessary
            if request.get("limit") and len(f) > request["limit"]:
                f = f.iloc[: request["limit"]]
            elif request.get("limit") is None:
                global_limit = config.get("geomodeling.geometry-limit")
                if len(f) > global_limit:
                    raise RuntimeError(
                        "The amount of returned geometries exceeded "
                        "the maximum of {} geometries.".format(global_limit)
                    )

            return {"projection": request["projection"], "features": f}


class GeometryWKTSource(GeometryBlock):
    """Converts a single geometry to a geometry source

    Args:
      wkt (string): the WKT representation of a geometry
      projection (string): the projection of the geometry

    Returns:
      GeometryBlock
    """

    def __init__(self, wkt, projection):
        if not isinstance(wkt, str):
            raise TypeError("'{}' object is not allowed".format(type(wkt)))
        if not isinstance(projection, str):
            raise TypeError("'{}' object is not allowed".format(type(projection)))
        try:
            load_wkt(wkt)
        except WKTReadingError:
            raise ValueError("The provided geometry is not a valid WKT")
        try:
            utils.get_sr(projection)
        except TypeError:
            raise ValueError("The provided projection is not a valid WKT")
        super().__init__(wkt, projection)

    @property
    def wkt(self):
        return self.args[0]

    @property
    def projection(self):
        return self.args[1]

    @property
    def columns(self):
        return {"geometry"}

    def get_sources_and_requests(self, **request):
        data = {"wkt": self.wkt, "projection": self.projection}
        return [(data, None), (request, None)]

    @staticmethod
    def process(data, request):

        mode = request["mode"]
        if mode not in ("extent", "intersects", "centroid"):
            raise ValueError("Unknown mode '{}'".format(mode))

        # load the geometry and transform it into the requested projection
        geometry = load_wkt(data["wkt"])
        if data["projection"] != request["projection"]:
            geometry = utils.shapely_transform(
                geometry, data["projection"], request["projection"]
            )

        f = gpd.GeoDataFrame(
            geometry=[geometry], crs=utils.get_crs(request["projection"])
        )

        # compute the bounds of each geometry and filter on min_size
        min_size = request.get("min_size")
        if min_size:
            minx, miny, maxx, maxy = geometry.bounds
            if (maxy - miny) < min_size or (maxx - minx) < min_size:
                return {
                    "projection": request["projection"],
                    "features": gpd.GeoDataFrame([]),
                }

        if mode == "intersects":
            if not geometry.intersects(request["geometry"]):
                return {
                    "projection": request["projection"],
                    "features": gpd.GeoDataFrame([]),
                }
            return {"features": f, "projection": request["projection"]}

        elif mode == "centroid":
            with warnings.catch_warnings():  # geopandas warns if in WGS84
                warnings.simplefilter("ignore")
                centroid = geometry.centroid
            if not centroid.intersects(request["geometry"]):
                return {
                    "projection": request["projection"],
                    "features": gpd.GeoDataFrame([]),
                }
            return {"features": f, "projection": request["projection"]}

        elif mode == "extent":
            if not geometry.intersects(request["geometry"]):
                return {"projection": request["projection"], "extent": None}
            return {
                "extent": tuple(geometry.bounds),
                "projection": request["projection"],
            }
