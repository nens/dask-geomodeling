import numpy as np
import math
from scipy import ndimage
import shutil
import tempfile
from osgeo import osr, gdal

import geopandas as gpd
from shapely.geometry import Polygon

from dask import config
from dask_geomodeling.config import defaults
from dask_geomodeling.geometry import GeometryBlock
from dask_geomodeling.raster import RasterBlock
from dask_geomodeling.utils import (
    get_dtype_max,
    Extent,
    get_crs,
    get_sr,
    get_epsg_or_wkt,
    shapely_transform,
    Dataset,
)


class MockRaster(RasterBlock):
    """An in-memory raster source for testing purposes

    The dtype is fixed to uint8, and the fillvalue to 255. If the value param
    is an array, the requested bbox is interpreted as indices into this array.

    :param origin: datetime of the first frame
    :param timedelta: timedelta between frames
    :param bands: the number of frames
    :param value: either a constant value or an array of values
    """

    def __init__(
        self, origin=None, timedelta=None, bands=None, value=1, projection="EPSG:3857"
    ):
        self.origin = origin
        self._timedelta = timedelta
        self.bands = bands
        self.value = value
        super(MockRaster, self).__init__(origin, timedelta, bands, value, projection)

    @property
    def dtype(self):
        try:
            return self.value.dtype
        except AttributeError:
            return np.dtype(np.uint8)

    @property
    def fillvalue(self):
        return get_dtype_max(self.dtype)

    def get_sources_and_requests(self, **request):
        return [(self.args, None), (request, None)]

    @staticmethod
    def process(args, request):
        origin, timedelta, bands, value, src_projection = args
        if origin is None or timedelta is None or bands is None:
            return
        td_seconds = timedelta.total_seconds()
        lo = origin
        start = request.get("start", None)
        stop = request.get("stop", None)

        if start is None:
            # take the latest
            bands_lo = bands - 1
            bands_hi = bands
        elif stop is None:
            # take the nearest to start
            start_band = (request["start"] - lo).total_seconds() / td_seconds
            bands_lo = min(max(int(round(start_band)), 0), bands - 1)
            bands_hi = bands_lo + 1
        else:
            bands_lo = (request["start"] - lo).total_seconds() / td_seconds
            bands_hi = (request["stop"] - lo).total_seconds() / td_seconds
            bands_lo = max(int(math.ceil(bands_lo)), 0)
            bands_hi = min(int(math.floor(bands_hi)) + 1, bands)

        depth = bands_hi - bands_lo

        if depth <= 0:
            return

        if request["mode"] == "time":
            return {"time": [origin + i * timedelta for i in range(bands_lo, bands_hi)]}
        if request["mode"] == "meta":
            return {
                "meta": [
                    "Testmeta for band {}".format(i) for i in range(bands_lo, bands_hi)
                ]
            }
        if request["mode"] != "vals":
            raise ValueError('Invalid mode "{}"'.format(request["mode"]))

        height = request.get("height", 1)
        width = request.get("width", 1)
        shape = (depth, height, width)

        # simple mode: return a filled value with type uint8
        if not hasattr(value, "shape"):
            fillvalue = 255
            result = np.full(shape, value, dtype=np.uint8)
            return {"values": result, "no_data_value": fillvalue}

        # there is an actual data array
        fillvalue = get_dtype_max(value.dtype)
        bbox = request.get("bbox", (0, 0, width, height))
        projection = request.get("projection", "EPSG:3857")
        if projection != src_projection:
            extent = Extent(bbox, get_sr(projection))
            bbox = extent.transformed(get_sr(src_projection)).bbox
        x1, y1, x2, y2 = [int(round(x)) for x in bbox]

        if x1 == x2 or y1 == y2:  # point request
            if x1 < 0 or x1 >= value.shape[1] or y1 < 0 or y1 >= value.shape[0]:
                result = np.array([[255]], dtype=np.uint8)
            else:
                result = value[y1 : y1 + 1, x1 : x1 + 1]
        else:
            _x1 = max(x1, 0)
            _y1 = max(y1, 0)
            _x2 = min(x2, value.shape[1])
            _y2 = min(y2, value.shape[0])
            result = value[_y1:_y2, _x1:_x2]
            result = np.pad(
                result,
                ((_y1 - y1, y2 - _y2), (_x1 - x1, x2 - _x2)),
                mode=str("constant"),
                constant_values=fillvalue,
            )
            if result.shape != (height, width):
                zoom = (height / result.shape[0], width / result.shape[1])
                mask = ndimage.zoom((result == fillvalue).astype(float), zoom) > 0.5
                result[result == fillvalue] = 0
                result = ndimage.zoom(result, zoom)
                result[mask] = fillvalue
        result = np.repeat(result[np.newaxis], depth, axis=0)

        # fill nan values
        result[~np.isfinite(result)] = fillvalue
        return {"values": result, "no_data_value": fillvalue}

    @property
    def period(self):
        if self.origin is None or self.bands is None or self.timedelta is None:
            return None
        return self.origin, self.origin + (self.bands - 1) * self.timedelta

    @property
    def timedelta(self):
        return self._timedelta

    @property
    def extent(self):
        if self.value is None:
            return None
        if np.isscalar(self.value):
            return 0, 0, 1, 1
        else:
            height, width = self.value.shape
            return 0, 0, width, height

    @property
    def projection(self):
        return self.args[4]

    @property
    def geo_transform(self):
        x1, y1, x2, y2 = self.extent
        return x1, 1, 0, y2, 0, -1

    @property
    def geometry(self):
        if self.extent is None:
            return
        return Extent(self.extent, get_sr(self.projection)).as_geometry()


class MockGeometry(GeometryBlock):
    """An in-memory geometry source for testing purposes

    All polygons will be returned always, irrespective of the requested bbox.
    MockGeometry does not reproject. The returned projection equals the
    requested projection.

    :param polygons: the polygon geometries
    :param properties: properties associated with the given polygons

    :type polygons: list of lists of 2-tuples of numbers
    :type properties: list of dicts
    """

    def __init__(self, polygons, properties=None, projection="EPSG:3857"):
        super(MockGeometry, self).__init__(polygons, properties, projection)

    @property
    def polygons(self):
        return self.args[0]

    @property
    def properties(self):
        return self.args[1]

    @property
    def projection(self):
        return self.args[2]

    @property
    def columns(self):
        result = {"geometry"}  # 'geometry' is hardcoded in MockGeometry
        if self.properties:
            result |= set(self.properties[0].keys())
        result.discard("id")  # 'id' is reserved for the index in MockGeometry
        return result

    def get_sources_and_requests(self, **request):
        return [
            (self.polygons, None),
            (self.properties, None),
            (self.projection, None),
            (request, None),
        ]

    @staticmethod
    def process(polygons, properties, projection, request):
        if request.get("limit") is not None:
            polygons = polygons[: request["limit"]]
            if properties is not None:
                properties = properties[: request["limit"]]
        mode = request.get("mode", "intersects")

        geoseries = gpd.GeoSeries(
            [Polygon(x) for x in polygons], crs=get_crs(projection)
        )

        if get_epsg_or_wkt(projection) != get_epsg_or_wkt(request["projection"]):
            geoseries = geoseries.apply(
                shapely_transform, args=(projection, request["projection"])
            )

        if mode == "extent":
            if len(geoseries) > 0:
                extent = tuple(geoseries.total_bounds)
            else:
                extent = None
            return {"extent": extent, "projection": request["projection"]}

        if len(geoseries) == 0:
            return {
                "features": gpd.GeoDataFrame([]),
                "projection": request["projection"],
            }

        if properties is not None:
            df = gpd.GeoDataFrame.from_records(properties)
            df.set_geometry(geoseries, inplace=True)
            if "id" in df.columns:
                df.set_index("id", inplace=True, drop=True)
        else:
            df = gpd.GeoDataFrame(geometry=geoseries)
            df.index.name = "id"

        if mode == "centroid":
            df = df[df["geometry"].centroid.within(request["geometry"])]
        elif mode == "intersects":
            df = df[df["geometry"].intersects(request["geometry"])]

        return {"features": df, "projection": request["projection"]}


def setup_temp_root(**kwargs):
    """ Setup a temporary file root for testing purposes. """
    path = tempfile.mkdtemp(**kwargs)
    config.set({"geomodeling.root": path})
    return path


def teardown_temp_root(path):
    """ Delete the temporary file root. """
    shutil.rmtree(path)
    config.set({"geomodeling.root": defaults["root"]})


def create_tif(
    path,
    bands=1,
    no_data_value=255,
    base_level=7,
    dtype="i2",
    projection="EPSG:28992",
    geo_transform=None,
    shape=(16, 16),
):
    """ Create a test source dataset at path. """

    kwargs = {
        "no_data_value": no_data_value,
        "geo_transform": geo_transform or (-16, 2, 0, 16, 0, -2),
        "projection": osr.GetUserInputAsWKT(str(projection)),
    }

    data = np.full(shape, base_level, dtype=dtype)

    array = data[np.newaxis][bands * [0]]
    with Dataset(array, **kwargs) as dataset:
        gdal.GetDriverByName(str("gtiff")).CreateCopy(path, dataset)
