import re
import pytz
import os
import warnings
from functools import lru_cache
from itertools import repeat

from math import floor, log10

import numpy as np
import pandas as pd
from scipy import ndimage
from dask import config

from osgeo import gdal, ogr, osr, gdal_array
from shapely.geometry import box, Point
from shapely import wkb as shapely_wkb

import fiona

POLYGON = "POLYGON (({0} {1},{2} {1},{2} {3},{0} {3},{0} {1}))"


try:
    from fiona import Env as fiona_env  # NOQA
except ImportError:
    from fiona import drivers as fiona_env  # NOQA


def get_index(values, no_data_value):
    """ Return an index to access for data values in values. """
    equal = np.isclose if values.dtype.kind == "f" else np.equal
    return np.logical_not(equal(values, no_data_value))


def get_dtype_max(dtype):
    """
    Return the maximum value for a dtype as a python scalar value.

    :param dtype: numpy dtype
    """
    d = np.dtype(dtype)
    if d.kind == "f":
        return np.finfo(d).max.item()
    return np.iinfo(d).max


def get_dtype_min(dtype):
    """
    Return the minimum value for a dtype as a python scalar value.

    :param dtype: numpy dtype
    """
    d = np.dtype(dtype)
    if d.kind == "f":
        return np.finfo(d).min.item()
    return np.iinfo(d).min


def get_uint_dtype(n):
    """Get the smallest uint dtype that accomodates 'n' values"""
    for dtype in ("u1", "u2", "u4", "u8"):
        if n - 1 <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise ValueError("Too many values for uint dtype ({})".format(n))


def get_rounded_repr(obj, significant=4, fmt="{} (rounded)"):
    """
    http://stackoverflow.com/questions/3410976/
    how-to-round-a-number-to-significant-figures-in-python

    :param significant: number of significant digits
    :param fmt: template for rounded repr
    """
    digits = (
        -int(floor(log10(abs(n)))) + (significant - 1) if n else None for n in obj
    )
    rounded = obj.__class__(round(n, d) if n else n for n, d in zip(obj, digits))
    if obj == rounded:
        return repr(obj)
    return fmt.format(repr(rounded))


class Extent(object):
    """ Spatially aware extent. """

    def __init__(self, bbox, sr):
        self.bbox = bbox
        self.sr = sr

    def __repr__(self):
        return "<{}: {} / {}>".format(
            self.__class__.__name__,
            get_projection(self.sr),
            get_rounded_repr(self.bbox),
        )

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @classmethod
    def from_geometry(cls, geometry):
        """ Return Extent instance. """
        x1, x2, y1, y2 = geometry.GetEnvelope()
        bbox = x1, y1, x2, y2
        sr = geometry.GetSpatialReference()
        return cls(bbox=bbox, sr=sr)

    def as_geometry(self):
        """ Return ogr Geometry instance. """
        x1, y1, x2, y2 = self.bbox

        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint_2D(x1, y1)
        ring.AddPoint_2D(x2, y1)
        ring.AddPoint_2D(x2, y2)
        ring.AddPoint_2D(x1, y2)
        ring.AddPoint_2D(x1, y1)
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        polygon.AssignSpatialReference(self.sr)
        return polygon

    def buffered(self, size):
        """ Return Extent instance. """
        x1, y1, x2, y2 = self.bbox
        buffered_bbox = x1 - size, y1 - size, x2 + size, y2 + size
        return self.__class__(bbox=buffered_bbox, sr=self.sr)

    def transformed(self, sr):
        geometry = self.as_geometry()
        geometry.TransformTo(sr)
        return Extent.from_geometry(geometry)


class GeoTransform(tuple):
    """
    Wrapper with handy methods for the GeoTransform tuple
    as used by the GDAL library.

    In the dask-geomodeling as well as in the GDAL library the geo_transform
    defines the transformation from array pixel indices to projected
    coordinates. A pair of projected coordinates (x, y) is calculated
    from a pair of array indices (i, j) as follows:

    p, a, b, q, c, d = self  # the geo_transform tuple

    x = p + a * j + b * i
    y = q + c * j + d * i

    or in a kind of vector notation:

    [x, y] = [p, q] + [[a, b], [c, d]] * [j, i]
    """

    @classmethod
    def from_bbox(cls, bbox, height, width):
        x1, y1, x2, y2 = bbox
        p, q = x1, y2
        b, c = 0, 0
        a = (x2 - x1) / width
        d = (y1 - y2) / height
        return cls((p, a, b, q, c, d))

    def __init__(self, tpl):
        if len(tpl) != 6:
            raise ValueError("GeoTransform expected an iterable of length 6")
        if tpl[2] != 0.0 or tpl[4] != 0.0:
            raise ValueError("Tilted geo_transforms are not supported")
        if tpl[1] == 0.0 or tpl[5] == 0.0:
            raise ValueError("Pixel size should not be zero")

    def __repr__(self):
        return get_rounded_repr(tuple(self))

    @property
    def cell_area(self):
        p, a, b, q, c, d = self
        return abs(a * d - b * c)

    @property
    def origin(self):
        """Return the (x, y) coordinate of pixel with indices (0, 0)."""
        return self[0], self[3]

    @property
    def origin_normalized(self):
        """Return the (x, y) coordinate of the pixel closest to x = 0, y = 0.
        """
        return self[0] % self[1], self[3] % self[5]

    def get_inverse(self):
        """Return 2 x 2 matrix for the inverse of self"""
        _, a, b, _, c, d = self
        D = 1 / (a * d - b * c)
        return d * D, -b * D, -c * D, a * D

    def scale(self, x, y):
        """
        return scaled geo transform.

        :param x: Multiplication factor for the pixel width.
        :param y: Multiplication factor for the pixel height.

        Adjust the second, third, fifth and sixth elements of the geo
        transform so that the extent of the respective image is multiplied
        by scale.
        """
        p, a, b, q, c, d = self
        return self.__class__([p, a * x, b * x, q, c * y, d * y])

    def shift(self, origin):
        """
        Return shifted geo transform.

        :param origin: Integer pixel coordinates.

        Adjust the first and fourth element of the geo transform to match
        a subarray that starts at origin.
        """
        p, a, b, q, c, d = self
        i, j = origin
        return self.__class__([p + a * j + b * i, a, b, q + c * j + d * i, c, d])

    def get_indices(self, points):
        """
        Return pixel indices as a tuple of linear numpy arrays
        """
        # inverse transformation
        p, _, _, q, _, _ = self
        e, f, g, h = self.get_inverse()

        # calculate
        x, y = np.asarray(points).transpose()
        return (
            np.floor(g * (x - p) + h * (y - q)).astype(np.int64),
            np.floor(e * (x - p) + f * (y - q)).astype(np.int64),
        )

    def get_points(self, indices):
        """
        Return point coordinates as N x 2 numpy array.
        """
        p, a, b, q, c, d = self
        i, j = indices.transpose()
        points = np.empty(indices.shape)
        points[:, 0] = p + a * j + b * i
        points[:, 1] = q + c * j + d * i
        return points

    def get_bbox(self, offset, shape):
        """Return the bounding box coordinates defined by an offset and shape
        of an array"""
        p, a, b, q, c, d = self
        i, j = offset
        m, n = shape
        west = p + a * j + b * i
        north = q + c * j + d * i
        east = west + a * n + b * m
        south = north + c * n + d * m
        return west, south, east, north

    def get_indices_for_bbox(self, bbox):
        """Return (i1, i2), (j1, j2) array coordinates for given bbox."""
        x1, y1, x2, y2 = bbox

        # inverse transformation
        p, _, _, q, _, _ = self
        e, f, g, h = self.get_inverse()

        # apply to envelope corners
        x_index_1 = int(floor(e * (x1 - p) + f * (y2 - q)))
        y_index_1 = int(floor(g * (x1 - p) + h * (y2 - q)))
        x_index_2 = int(floor(e * (x2 - p) + f * (y1 - q)))
        y_index_2 = int(floor(g * (x2 - p) + h * (y1 - q)))

        # sort for flipped geo-transfoms
        x_pair = tuple(sorted((x_index_1, x_index_2)))
        y_pair = tuple(sorted((y_index_1, y_index_2)))
        return y_pair, x_pair

    def get_array_ranges(self, bbox, shape):
        """Compute how much to slice and pad an array with given shape to
        obtain `bbox`.

        :param bbox: tuple of (x1, y1, x2, y2)
        :param shape: tuple of 2 int

        :returns:
          - two [start, stop) ranges for the to array dimensions
          - two [before, after] paddings, or None if no padding is required
        """
        (i1, i2), (j1, j2) = self.get_indices_for_bbox(bbox)

        # increase the end index by one if necessary to deal with points
        if i1 == i2:
            i2 += 1
        if j1 == j2:
            j2 += 1

        # clip the indices to the array bounds for getting the data
        _i1, _i2 = np.clip([i1, i2], 0, shape[1])
        _j1, _j2 = np.clip([j1, j2], 0, shape[2])
        ranges = (_i1, _i2), (_j1, _j2)

        # pad the data to the shape given by the index
        padding_i = (i2 - i1, 0) if _i1 == _i2 else (_i1 - i1, i2 - _i2)
        padding_j = (j2 - j1, 0) if _j1 == _j2 else (_j1 - j1, j2 - _j2)
        padding = padding_i, padding_j
        if np.all(np.array(padding) <= 0):
            padding = None
        return ranges, padding

    def aligns_with(self, other):
        """Compares with other geotransform, checks if rasters are aligned."""
        if not isinstance(other, GeoTransform):
            other = GeoTransform(other)
        # compare resolutions
        if abs(self[1]) != abs(other[1]) or abs(self[5]) != abs(other[5]):
            return False
        return self.origin_normalized == other.origin_normalized


@lru_cache(32)  # least-recently-used cache of size 32
def get_sr(user_input):
    """ Return osr.SpatialReference for user input. """
    return osr.SpatialReference(osr.GetUserInputAsWKT(str(user_input)))


def get_crs(user_input):
    """
    Return fiona CRS dictionary for user input.
    """
    wkt = osr.GetUserInputAsWKT(str(user_input))
    sr = osr.SpatialReference(wkt)
    key = str("GEOGCS") if sr.IsGeographic() else str("PROJCS")
    name = sr.GetAuthorityName(key)
    if name == "EPSG":
        # we can specify CRS in EPSG code which is more compatible with output
        # file types
        return fiona.crs.from_epsg(int(sr.GetAuthorityCode(key)))
    else:
        # we have to go through Proj4
        return fiona.crs.from_string(sr.ExportToProj4())


def crs_to_srs(crs):
    """
    Recover our own WKT definition of projections from a fiona CRS
    """
    proj4_str = fiona.crs.to_string(crs)
    return get_epsg_or_wkt(proj4_str)


def wkb_transform(wkb, src_sr, dst_sr):
    """
    Return a shapely geometry transformed from src_sr to dst_sr.

    :param wkb: wkb bytes
    :param src_sr: source osr SpatialReference
    :param dst_sr: destination osr SpatialReference
    """
    result = ogr.CreateGeometryFromWkb(wkb, src_sr)
    result.TransformTo(dst_sr)
    return result.ExportToWkb()


def shapely_transform(geometry, src_srs, dst_srs):
    """
    Return a shapely geometry transformed from src_srs to dst_srs.

    :param geometry: shapely geometry
    :param src_srs: source projection string
    :param dst_srs: destination projection string

    Note that we do not use geopandas for the transformation, because this is
    much slower than OGR.
    """
    return shapely_wkb.loads(
        wkb_transform(geometry.wkb, src_sr=get_sr(src_srs), dst_sr=get_sr(dst_srs))
    )


def geoseries_transform(df, src_srs, dst_srs):
    """
    Transform a GeoSeries to a different SRS. Returns a copy.

    :param df: GeoSeries to transform
    :param src_srs: source projection string
    :param dst_srs: destination projection string

    Note that we do not use .to_crs() for the transformation, because this is
    much slower than OGR. Also, we ignore the .crs property (but we do set it)
    """
    result = df.geometry.apply(shapely_transform, args=(src_srs, dst_srs))
    result.crs = get_crs(dst_srs)
    return result


def geodataframe_transform(df, src_srs, dst_srs):
    """
    Transform the geometry column of a GeoDataFrame to a different SRS.

    :param df: GeoDataFrame to transform (will be changed inplace)
    :param src_srs: source projection string
    :param dst_srs: destination projection string

    Note that we do not use .to_crs() for the transformation, because this is
    much slower than OGR. Also, we ignore the .crs property (but we do set it)
    """
    geoseries = geoseries_transform(df.geometry, src_srs, dst_srs)
    df.crs = geoseries.crs
    df.set_geometry(geoseries)
    return df


def transform_min_size(min_size, geometry, src_srs, dst_srs):
    """
    Return min_size converted from source crs to a target crs.

    :param min_size: float indicating the minimum size in the source crs
    :param geometry: shapely geometry of the location in the source crs
    :param src_srs: source projection string
    :param dst_srs: destination projection string

    Note that in order to guarantee the minimum size, the reverse operation may
    result in a larger value than the original minimum size.
    """
    source = geometry.centroid.buffer(min_size / 2)
    target = shapely_transform(source, src_srs=src_srs, dst_srs=dst_srs)
    x1, y1, x2, y2 = target.bounds
    return max(x2 - x1, y2 - y1)


def transform_extent(extent, src_srs, dst_srs):
    """
    Return extent tuple transformed from src_srs to dst_srs.

    :param extent: xmin, ymin, xmax, ymax
    :param src_srs: source projection string
    :param dst_srs: destination projection string
    """
    source = box(*extent)
    target = shapely_transform(source, src_srs=src_srs, dst_srs=dst_srs)
    return target.bounds


EPSG3857 = get_sr("EPSG:3857")
EPSG4326 = get_sr("EPSG:4326")


def get_projection(sr):
    """ Return simple userinput string for spatial reference, if any. """
    key = str("GEOGCS") if sr.IsGeographic() else str("PROJCS")
    return "{name}:{code}".format(
        name=sr.GetAuthorityName(key), code=sr.GetAuthorityCode(key)
    )


def get_epsg_or_wkt(text):
    """
    Return EPSG:<code> where possible, and WKT otherwise.

    :param text: Textual representation of a spatial reference system.
    """
    wkt = osr.GetUserInputAsWKT(str(text))
    sr = osr.SpatialReference(wkt)
    key = str("GEOGCS") if sr.IsGeographic() else str("PROJCS")
    name = sr.GetAuthorityName(key)
    if name is None:
        return wkt
    code = sr.GetAuthorityCode(key)
    return "{name}:{code}".format(name=name, code=code)


def get_footprint(size):
    """
    Return numpy array of booleans representing a circular footprint.

    :param size: diameter of the circle, coerced to uneven values.
    """
    s = size // 2 * 2 + 1
    o = (s - 1) // 2
    r = s / 2

    x, y = np.indices((s, s)) - o
    return (x ** 2 + y ** 2) < (r ** 2)


def create_dataset(array, geo_transform=None, projection=None, no_data_value=None):
    """
    Create and return a gdal dataset.

    :param array: A numpy array.
    :param geo_transform: 6-tuple of floats
    :param projection: wkt projection string
    :param no_data_value: integer or float

    This is the fastest way to get a gdal dataset from a numpy array, but
    keep a reference to the array around, or a segfault will occur. Also,
    don't forget to call FlushCache() on the dataset after any operation
    that affects the array.
    """
    # prepare dataset name pointing to array
    datapointer = array.ctypes.data
    bands, lines, pixels = array.shape
    datatypecode = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype.type)
    datatype = gdal.GetDataTypeName(datatypecode)
    bandoffset, lineoffset, pixeloffset = array.strides
    # if projection is wrong there will be a segfault in RasterizeLayer
    projection = osr.GetUserInputAsWKT(str(projection))

    dataset_name_template = (
        "MEM:::"
        "DATAPOINTER={datapointer},"
        "PIXELS={pixels},"
        "LINES={lines},"
        "BANDS={bands},"
        "DATATYPE={datatype},"
        "PIXELOFFSET={pixeloffset},"
        "LINEOFFSET={lineoffset},"
        "BANDOFFSET={bandoffset}"
    )
    dataset_name = dataset_name_template.format(
        datapointer=datapointer,
        pixels=pixels,
        lines=lines,
        bands=bands,
        datatype=datatype,
        pixeloffset=pixeloffset,
        lineoffset=lineoffset,
        bandoffset=bandoffset,
    )

    # access the array memory as gdal dataset
    dataset = gdal.Open(dataset_name, gdal.GA_Update)

    # set additional properties from kwargs
    if geo_transform is not None:
        dataset.SetGeoTransform(geo_transform)
    if projection is not None:
        dataset.SetProjection(projection)
    if no_data_value is not None:
        for i in range(len(array)):
            dataset.GetRasterBand(i + 1).SetNoDataValue(no_data_value)

    return dataset


class Dataset(object):
    """
    Usage:
        >>> with Dataset(array) as dataset:
        ...     # do gdal things.
    """

    def __init__(self, array, **kwargs):
        self.array = array
        self.dataset = create_dataset(array, **kwargs)

    def __enter__(self):
        return self.dataset

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.dataset.FlushCache()


def _finalize_rasterize_result(array, no_data_value):
    if array.dtype == np.uint8:  # cast to bool
        array = array.astype(np.bool)
        no_data_value = None
    return {"values": array, "no_data_value": no_data_value}


def rasterize_geoseries(geoseries, bbox, projection, height, width, values=None):
    """Transform a geoseries to a raster, optionally.

    :param geoseries: GeoSeries or None
    :param bbox: tuple of floats, (x1, y1, x2, y2)
    :param projection: wkt projection string
    :param height: int
    :param width: int
    :param values: Series or None, object containing values (int, float, bool)
      that will be burned into the raster. If None, the result will be a
      boolean array indicating geometries.

    :returns: dictionary containing
      - values: a 3D numpy array (time, y, x) of int32 / float64 / bool type
      - no_data_value

    It is assumed that all geometries intersect the requested bbox. If
    geoseries is None or empty, an array full of nodata will be returned.
    """
    # determine the dtype based on `values`
    if values is None or values.dtype == np.bool:
        dtype = np.uint8  # we cast to bool later as GDAL does not know bools
        no_data_value = 0  # False
        ogr_dtype = None
        if values is not None and geoseries is not None:
            geoseries = geoseries[values]  # values is a boolean mask
            values = None  # discard values
    elif str(values.dtype) == "category":
        # transform pandas Categorical dtype to normal dtype
        values = pd.Series(np.asarray(values), index=values.index)

    if values is not None:
        if np.issubdtype(values.dtype, np.floating):
            dtype = np.float64  # OGR only knows float64
            no_data_value = get_dtype_max(dtype)
            ogr_dtype = ogr.OFTReal
            if geoseries is not None:
                # filter out the inf and NaN values
                mask = np.isfinite(values)
                geoseries = geoseries[mask]
                values = values[mask]
        elif np.issubdtype(values.dtype, np.integer):
            dtype = np.int32  # OGR knows int32 and int64, but GDAL only int32
            no_data_value = get_dtype_max(dtype)
            ogr_dtype = ogr.OFTInteger
        else:
            raise TypeError(
                "Unsupported values dtype to rasterize: '{}'".format(values.dtype)
            )

    # initialize the array
    array = np.full((1, height, width), no_data_value, dtype=dtype)

    # if there are no features, return directly
    if geoseries is None or len(geoseries) == 0:
        return _finalize_rasterize_result(array, no_data_value)

    # drop empty geometries
    mask = ~geoseries.isnull()
    geoseries = geoseries[mask]
    values = values[mask] if values is not None else None

    # be strict about the bbox, it may lead to segfaults else
    x1, y1, x2, y2 = bbox
    if not ((x2 == x1 and y2 == y1) or (x1 < x2 and y1 < y2)):
        raise ValueError("Invalid bbox ({})".format(bbox))

    # if the request is a point, we find the intersecting polygon and
    # "rasterize" it
    if x2 == x1 and y2 == y1:
        mask = geoseries.intersects(Point(x1, y1))
        if not mask.any():
            pass
        elif values is not None:
            array[:] = values[mask].iloc[-1]  # take the last one
        else:
            array[:] = True
        return _finalize_rasterize_result(array, no_data_value)

    # create an output datasource in memory
    driver = ogr.GetDriverByName(str("Memory"))
    burn_attr = str("BURN_IT")
    sr = get_sr(projection)

    # prepare in-memory ogr layer
    ds_ogr = driver.CreateDataSource(str(""))
    layer = ds_ogr.CreateLayer(str(""), sr)
    layer_definition = layer.GetLayerDefn()
    if ogr_dtype is not None:
        field_definition = ogr.FieldDefn(burn_attr, ogr_dtype)
        layer.CreateField(field_definition)

    iterable = (
        zip(geoseries, values) if values is not None else zip(geoseries, repeat(True))
    )
    for geometry, value in iterable:
        feature = ogr.Feature(layer_definition)
        feature.SetGeometry(ogr.CreateGeometryFromWkb(geometry.wkb))
        if ogr_dtype is not None:
            feature[burn_attr] = value
        layer.CreateFeature(feature)

    geo_transform = GeoTransform.from_bbox(bbox, height, width)
    dataset_kwargs = {
        "no_data_value": no_data_value,
        "projection": sr.ExportToWkt(),
        "geo_transform": geo_transform,
    }

    # ATTRIBUTE=BURN_ATTR burns the BURN_ATTR value of each feature
    if dtype == np.uint8:  # this is our boolean dtype
        options = []
    else:
        options = [str("ATTRIBUTE=") + burn_attr]

    with Dataset(array, **dataset_kwargs) as dataset:
        gdal.RasterizeLayer(dataset, (1,), layer, options=options)

    return _finalize_rasterize_result(array, no_data_value)


def safe_abspath(url, start=None):
    """Executes safe_file_url but only returns the path and not the protocol.
    """
    url = safe_file_url(url)
    _, path = url.split("://")
    return path


def safe_file_url(url, start=None):
    """Formats an URL so that it meets the following safety conditions:

    - the URL starts with file:// (else: raises NotImplementedError)
    - the path is absolute (relative paths are taken relative to
      geomodeling.root)
    - if geomodeling.strict_paths: the path has to be contained inside
      `start` (else: raises IOError)

    For backwards compatibility, geomodeling.root can be overriden using the
    'start' argument.
    """
    try:
        protocol, path = url.split("://")
    except ValueError:
        protocol = "file"
        path = url
    else:
        if protocol != "file":
            raise NotImplementedError('Unknown protocol: "{}"'.format(protocol))
    if start is not None:
        warnings.warn(
            "Using the start argument in safe_file_url is deprecated. Use the "
            "'geomodeling.root' in the dask config",
            DeprecationWarning,
        )
    else:
        start = config.get("geomodeling.root")

    if not os.path.isabs(path):
        if start is None:
            raise IOError(
                "Relative path '{}' provided but start was not given.".format(path)
            )
        abspath = os.path.abspath(os.path.join(start, path))
    else:
        abspath = os.path.abspath(path)
    strict = config.get("geomodeling.strict-file-paths")
    if strict and not abspath.startswith(start):
        raise IOError("'{}' is not contained in '{}'".format(path, start))
    return "://".join([protocol, abspath])


PERCENTILE_REGEX = re.compile(r"^p([\d.]+)$")  # regex to match e.g. 'p50'


def parse_percentile_statistic(statistic):
    """Parses p<float> to a float, or None if it does not match."""
    # interpret percentile statistic
    percentile_match = PERCENTILE_REGEX.findall(statistic)
    if percentile_match:
        percentile = float(percentile_match[0])
        if not 0 <= percentile <= 100:
            raise ValueError("Percentiles must be in the range [0, 100]")
        return percentile


def dtype_for_statistic(dtype, statistic):
    """Return the result dtype of given statistic function"""
    # the dtype depends on the statistic
    if statistic in ("min", "max"):
        return dtype
    elif statistic == "sum":
        # same logic as Add block
        if np.issubdtype(dtype, np.integer) or dtype == np.bool:
            # use at least int32
            return np.result_type(dtype, np.int32)
        elif np.issubdtype(dtype, np.floating):
            # use at least float32
            return np.result_type(dtype, np.float32)
        else:
            return dtype
    elif statistic == "count":
        return np.int32
    else:
        # same logic as Divide block
        return np.result_type(np.float32, dtype)


def snap_start_stop(start, stop, time_first, time_delta, length):
    """This function interprets 'start' and 'stop' request parameters and
    returns the actual 'start' 'stop' with corresponding index range.

    There are 3 variants:
     - start and stop are None: the last frame is returned
     - only stop is None: the frame closest to start is returned
     - both start and stop are not None: the frames in the given interval are
       returned. The interval is closed on both sides.

    :param start: the requested 'start'
    :param stop: the requested 'stop'
    :param time_first: the timestamp of the first frame
    :param time_delta: the timedelta between consecutive frames
    :param length: the number of frames

    :type start: naive datetime
    :type stop: naive datetime
    :type time_first: naive datetime
    :type time_delta: timedelta or NoneType
    :type length: int
    """
    if length == 0:
        return (None,) * 4

    if length == 1:
        time_delta = None
        period = (time_first, time_first)
    elif length > 1 and time_delta is None:
        raise ValueError("Length > 1 requires a timedelta")
    else:
        period = (time_first, time_first + (length - 1) * time_delta)

    if start is None:
        # take the latest
        start = stop = period[-1]
        first_i = last_i = length - 1
    elif stop is None:
        # snap 'start' to the closest
        if start <= period[0]:
            start = stop = period[0]
            first_i = last_i = 0
        elif start >= period[1]:
            start = stop = period[1]
            first_i = last_i = length - 1
        elif length == 1:
            start = stop = period[0]
            first_i = last_i = 0
        else:
            # snap 'start' to the nearest frame
            first_i = last_i = int(round((start - period[0]) / time_delta))
            start = stop = period[0] + time_delta * first_i
    else:
        if start > period[1] or stop < period[0]:
            start = stop = first_i = last_i = None
        elif length == 1:
            start = stop = period[0]
            first_i = last_i = 0
        else:
            # ceil 'start'
            first_i = int(np.ceil((start - period[0]) / time_delta))
            first_i = max(first_i, 0)
            # floor 'stop'
            last_i = int(np.floor((stop - period[0]) / time_delta))
            last_i = min(last_i, length - 1)
            start = period[0] + time_delta * first_i
            stop = period[0] + time_delta * last_i
    return start, stop, first_i, last_i


def zoom_raster(data, no_data_value, height, width):
    """Zooms a data array to specified height and width

    Deals with no_data by setting these to 0, zooming, and putting back nodata.
    Edges around nodata will be biased towards 0."""
    if data.shape[1:] == (height, width):
        return data
    factor = 1, height / data.shape[1], width / data.shape[2]

    # first zoom the nodata mask
    src_mask = data == no_data_value
    dst_mask = ndimage.zoom(src_mask.astype(np.float), factor) > 0.5

    # set nodata to 0 and zoom the data
    data = data.copy()
    data[src_mask] = 0
    result = ndimage.zoom(data, factor)

    # set nodata to nodata again using the zoomed mask
    result[dst_mask] = no_data_value
    return result


def dt_to_ms(dt):
    """Converts a datetime to a POSIX timestamp in milliseconds"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.UTC)
    return int(dt.timestamp() * 1000)
