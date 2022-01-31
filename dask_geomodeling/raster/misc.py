"""
Module containing miscellaneous raster blocks.
"""
from osgeo import ogr
import numpy as np
from geopandas import GeoSeries

from shapely.geometry import box
from shapely.geometry import Point
from shapely.errors import WKTReadingError
from shapely.wkt import loads as load_wkt

from dask import config
from dask_geomodeling.geometry import GeometryBlock
from dask_geomodeling import utils

from .base import RasterBlock, BaseSingle


__all__ = [
    "Clip",
    "Classify",
    "Reclassify",
    "Mask",
    "MaskBelow",
    "Step",
    "Rasterize",
    "RasterizeWKT",
]


class Clip(BaseSingle):
    """
    Clip one raster to the extent of another raster.

    Takes two raster inputs, one raster ('store') whose values are returned in
    the output and one raster ('source') that is used as the extent. Cells of
    the 'store' raster are replaced with 'no data' if there is no data in the
    'source' raster.

    If the 'source' raster is a boolean raster, False will result in 'no data'.

    Note that the input rasters are required to have the same time resolution.

    Args:
      store (RasterBlock): Raster whose values are clipped
      source (RasterBlock): Raster that is used as the clipping mask

    Returns:
      RasterBlock with clipped values.
    """

    def __init__(self, store, source):
        if not isinstance(source, RasterBlock):
            raise TypeError("'{}' object is not allowed".format(type(store)))
        # timedeltas are required to be equal
        if store.timedelta != source.timedelta:
            raise ValueError(
                "Time resolution of the clipping mask does not match that of "
                "the values raster. Consider using Snap."
            )
        super(Clip, self).__init__(store, source)

    @property
    def source(self):
        return self.args[1]

    def get_sources_and_requests(self, **request):
        start = request.get("start", None)
        stop = request.get("stop", None)

        if start is not None and stop is not None:
            # limit request to self.period so that resulting data is aligned
            period = self.period
            if period is not None:
                request["start"] = max(start, period[0])
                request["stop"] = min(stop, period[1])

        return ((source, request) for source in self.args)

    @staticmethod
    def process(data, source_data):
        """ Mask store_data where source_data has no data """
        if data is None:
            return None

        if "values" not in data:
            return data

        # check if values contain data
        if np.all(data["values"] == data["no_data_value"]):
            return data

        # make the boolean mask
        if source_data is None:
            return None

        if source_data["values"].dtype == np.dtype("bool"):
            mask = ~source_data["values"]
        else:
            mask = source_data["values"] == source_data["no_data_value"]

        # adjust values
        values = data["values"].copy()
        values[mask] = data["no_data_value"]
        return {"values": values, "no_data_value": data["no_data_value"]}

    @property
    def extent(self):
        """Intersection of bounding boxes of 'store' and 'source'. """
        result, mask = [s.extent for s in self.args]
        if result is None or mask is None:
            return

        # return the overlapping box
        x1 = max(result[0], mask[0])
        y1 = max(result[1], mask[1])
        x2 = min(result[2], mask[2])
        y2 = min(result[3], mask[3])
        if x2 <= x1 or y2 <= y1:
            return None  # no overlap
        else:
            return x1, y1, x2, y2

    @property
    def geometry(self):
        """Intersection of geometries of 'store' and 'source'. """
        result, mask = [x.geometry for x in self.args]
        if result is None or mask is None:
            return
        sr = result.GetSpatialReference()
        if not mask.GetSpatialReference().IsSame(sr):
            mask = mask.Clone()
            mask.TransformTo(sr)
        result = result.Intersection(mask)
        if result.GetArea() == 0.0:
            return
        return result

    @property
    def period(self):
        """ Return period datetime tuple. """
        periods = [x.period for x in self.args]
        if any(period is None for period in periods):
            return None  # None precedes

        # multiple periods: return the overlapping period
        start = max([p[0] for p in periods])
        stop = min([p[1] for p in periods])
        if stop < start:
            return None  # no overlap
        else:
            return start, stop


class Mask(BaseSingle):
    """
    Replace values in a raster with a single constant value. 'no data' values
    are preserved.

    Args:
      store (RasterBlock): The raster whose values are to be converted.
      value (number): The constant value to be given to 'data' values.

    Returns:
      RasterBlock containing a single value
    """

    def __init__(self, store, value):
        if not isinstance(value, (float, int)):
            raise TypeError("'{}' object is not allowed".format(type(value)))
        super(Mask, self).__init__(store, value)

    @property
    def value(self):
        return self.args[1]

    @property
    def fillvalue(self):
        return 1 if self.value == 0 else 0

    @property
    def dtype(self):
        return self._dtype_from_value(self.value)

    @staticmethod
    def _dtype_from_value(value):
        if isinstance(value, float):
            return np.dtype("float32")
        elif value >= 0:
            return utils.get_uint_dtype(value)
        else:
            return utils.get_int_dtype(value)

    @staticmethod
    def process(data, value):
        if data is None or "values" not in data:
            return data

        index = utils.get_index(
            values=data["values"], no_data_value=data["no_data_value"]
        )

        fillvalue = 1 if value == 0 else 0
        dtype = Mask._dtype_from_value(value)

        values = np.full_like(data["values"], fillvalue, dtype=dtype)
        values[index] = value
        return {"values": values, "no_data_value": fillvalue}


class MaskBelow(BaseSingle):
    """
    Converts raster cells below the supplied value to 'no data'.

    Raster cells with values greater than or equal to the supplied value are
    returned unchanged.

    Args:
      store (RasterBlock): The raster whose values are to be masked.
      value (number): The constant value below which values are masked.

    Returns:
      RasterBlock with cells below the input value converted to 'no data'.
    """

    def __init__(self, store, value):
        if not isinstance(value, (float, int)):
            raise TypeError("'{}' object is not allowed".format(type(value)))
        super(MaskBelow, self).__init__(store, value)

    @staticmethod
    def process(data, value):
        if data is None or "values" not in data:
            return data
        values, no_data_value = data["values"].copy(), data["no_data_value"]
        values[values < value] = no_data_value
        return {"values": values, "no_data_value": no_data_value}


class Step(BaseSingle):
    """
    Apply a step function to a raster.

    This operation classifies the elements of a raster into three categories:
    less than, equal to, and greater than a value.

    The step function is defined as follows, with x being the value of a raster
    cell:

    - 'left' if *x < value*

    - 'at' if *x == value*

    - 'right' if *x > value*

    Args:
      store (RasterBlock): The input raster
      left (number): Value given to cells lower than the input value,
        defaults to 0
      right (number): Value given to cells higher than the input value,
        defaults to 1
      value (number): The constant value which raster cells are compared to,
        defaults to 0
      at (number): Value given to cells equal to the input value, defaults to
        the average of left and right

    Returns:
      RasterBlock containing three values; left, right and at.

    """

    def __init__(self, store, left=0, right=1, value=0, at=None):
        at = (left + right) / 2 if at is None else at
        for x in left, right, value, at:
            if not isinstance(x, (float, int)):
                raise TypeError("'{}' object is not allowed".format(type(x)))
        super(Step, self).__init__(store, left, right, value, at)

    @property
    def left(self):
        return self.args[1]

    @property
    def right(self):
        return self.args[2]

    @property
    def value(self):
        return self.args[3]

    @property
    def at(self):
        return self.args[4]

    @staticmethod
    def process(data, left, right, location, at):
        if data is None or "values" not in data:
            return data

        values, no_data_value = data["values"].copy(), data["no_data_value"]

        # determine boolean index arrays
        mask = values == no_data_value
        left_index = values < location
        at_index = values == location
        right_index = values > location
        # perform mapping
        values[left_index] = left
        values[at_index] = at
        values[right_index] = right
        # put no data values back
        values[mask] = no_data_value

        return {"values": values, "no_data_value": no_data_value}


class Classify(BaseSingle):
    """
    Classify raster data into binned categories

    Takes a RasterBlock and classifies its values based on bins. The bins are
    supplied as a list of increasing bin edges.

    For each raster cell this operation returns the index of the bin to which
    the raster cell belongs. The lowest possible output cell value is 0, which
    means that the input value was lower than the lowest bin edge. The highest
    possible output value is equal to the number of supplied bin edges.

    Args:
      store (RasterBlock): The raster whose cell values are to be classified
      bins (list): An increasing list of bin edges
      right (boolean): Whether the intervals include the right or the left bin
        edge, defaults to False.

    Returns:
      RasterBlock with classified values

    """

    def __init__(self, store, bins, right=False):
        if not isinstance(store, RasterBlock):
            raise TypeError("'{}' object is not allowed".format(type(store)))
        if not hasattr(bins, "__iter__"):
            raise TypeError("'{}' object is not allowed".format(type(bins)))
        bins_arr = np.asarray(bins)
        if bins_arr.ndim != 1:
            raise TypeError("'bins' should be one-dimensional")
        if not np.issubdtype(bins_arr.dtype, np.number):
            raise TypeError("'bins' should be numeric")
        bins_diff = np.diff(bins)
        if not np.all(bins_diff > 0) or np.all(bins_diff < 0):
            raise TypeError("'bins' should be monotonic")
        super(Classify, self).__init__(store, bins_arr.tolist(), right)

    @property
    def bins(self):
        return self.args[1]

    @property
    def right(self):
        return self.args[2]

    @property
    def dtype(self):
        # with 254 bin edges, we have 255 bins, and we need 256 possible values
        # to include no_data
        return utils.get_uint_dtype(len(self.bins) + 2)

    @property
    def fillvalue(self):
        return utils.get_dtype_max(self.dtype)

    @staticmethod
    def process(data, bins, right):
        if data is None or "values" not in data:
            return data

        values = data["values"]
        dtype = utils.get_uint_dtype(len(bins) + 2)
        fillvalue = utils.get_dtype_max(dtype)

        result_values = np.digitize(values, bins, right).astype(dtype)
        result_values[values == data["no_data_value"]] = fillvalue

        return {"values": result_values, "no_data_value": fillvalue}


class Reclassify(BaseSingle):
    """
    Reclassify a raster of integer values.

    This operation can be used to reclassify a classified raster into desired
    values. Reclassification is done by supplying a list of [from, to] pairs.

    Args:
      store (RasterBlock): The raster whose cell values are to be reclassified
      bins (list): A list of [from, to] pairs defining the reclassification.
        The from values can be of bool or int datatype; the to values can be of
        int or float datatype
      select (boolean): Whether to set all non-reclassified cells to 'no data',
        defaults to False.

    Returns:
      RasterBlock with reclassified values
    """

    def __init__(self, store, data, select=False):
        dtype = store.dtype
        if dtype != bool and not np.issubdtype(dtype, np.integer):
            raise TypeError("The store must be of boolean or integer datatype")

        # validate "data"
        if not hasattr(data, "__iter__"):
            raise TypeError("'{}' object is not allowed".format(type(data)))
        try:
            source, target = self._data_as_ndarray(data)
        except ValueError:
            raise ValueError("Please supply a list of [from, to] values")
        # "from" can have bool or int dtype, "to" can also be float
        if source.dtype != bool and not np.issubdtype(source.dtype, np.integer):
            raise TypeError(
                "Cannot reclassify from value with type '{}'".format(source.dtype)
            )
        if len(np.unique(source)) != len(source):
            raise ValueError("There are duplicates in the reclassify values")
        if not np.issubdtype(target.dtype, np.number):
            raise TypeError(
                "Cannot reclassify to value with type '{}'".format(target.dtype)
            )
        # put 'data' into a list with consistent dtypes
        data = [list(x) for x in zip(source.tolist(), target.tolist())]

        if select is not True and select is not False:
            raise TypeError("'{}' object is not allowed".format(type(select)))
        super().__init__(store, data, select)

    @staticmethod
    def _data_as_ndarray(data):
        source, target = zip(*data)
        return np.asarray(source), np.asarray(target)

    @property
    def data(self):
        return self.args[1]

    @property
    def select(self):
        return self.args[2]

    @property
    def dtype(self):
        _, target = self._data_as_ndarray(self.data)
        return target.dtype

    @property
    def fillvalue(self):
        return utils.get_dtype_max(self.dtype)

    def get_sources_and_requests(self, **request):
        process_kwargs = {
            "dtype": self.dtype.str,
            "fillvalue": self.fillvalue,
            "data": self.data,
            "select": self.select,
        }
        return [(self.store, request), (process_kwargs, None)]

    @staticmethod
    def process(store_data, process_kwargs):
        if store_data is None or "values" not in store_data:
            return store_data

        no_data_value = store_data["no_data_value"]
        values = store_data["values"]
        source, target = Reclassify._data_as_ndarray(process_kwargs["data"])
        dtype = np.dtype(process_kwargs["dtype"])
        fillvalue = process_kwargs["fillvalue"]

        # add the nodata value to the source array and map it to the target
        # nodata
        if no_data_value is not None and no_data_value not in source:
            source = np.append(source, no_data_value)
            target = np.append(target, fillvalue)

        # sort the source and target values
        inds = np.argsort(source)
        source = source[inds]
        target = target[inds]

        # create the result array
        if process_kwargs["select"]:  # select = True: initialize with nodata
            result = np.full(values.shape, fillvalue, dtype=dtype)
        else:  # select = True: initialize with existing data
            result = values.astype(dtype)  # makes a copy

        # find all values in the source data that are to be mapped
        mask = np.in1d(values.ravel(), source)
        mask.shape = values.shape
        # place the target values (this also maps nodata values)
        result[mask] = target[np.searchsorted(source, values[mask])]
        return {"values": result, "no_data_value": fillvalue}


class Rasterize(RasterBlock):
    """
    Converts geometry source to raster

    This operation is used to transform GeometryBlocks into RasterBlocks. Here
    geometries (from for example a shapefile) are converted to a raster, using
    the values from one of the columns.

    Note that to rasterize floating point values, it is necessary to pass
    ``dtype="float"``.

    Args:
      source (GeometryBlock): The geometry source to be rasterized
      column_name (string): The name of the column whose values will be
        returned in the raster. If column_name is not provided, a boolean
        raster will be generated indicating where there are geometries.
      dtype (string): A numpy datatype specification to return the array.
        Defaults to 'int32' if column_name is provided, or to 'bool' otherwise.

    Returns:
      RasterBlock with values from 'column_name' or a boolean raster.

    See also:
      https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html

    The global geometry-limit setting can be adapted as follows:
      >>> from dask import config
      >>> config.set({"geomodeling.geometry-limit": 100000})
    """

    def __init__(self, source, column_name=None, dtype=None, limit=None):
        if not isinstance(source, GeometryBlock):
            raise TypeError("'{}' object is not allowed".format(type(source)))
        if column_name is not None and not isinstance(column_name, str):
            raise TypeError("'{}' object is not allowed".format(type(column_name)))
        if dtype is None:  # set default values
            dtype = "bool" if column_name is None else "int32"
        else:  # parse to numpy dtype and back to string
            dtype = str(np.dtype(dtype))
        if limit and not isinstance(limit, int):
            raise TypeError("'{}' object is not allowed".format(type(limit)))
        if limit and limit < 1:
            raise ValueError("Limit should be greater than 1")
        super(Rasterize, self).__init__(source, column_name, dtype, limit)

    @property
    def source(self):
        return self.args[0]

    @property
    def column_name(self):
        return self.args[1]

    @property
    def limit(self):
        return self.args[3]

    @property
    def dtype(self):
        return np.dtype(self.args[2])

    @property
    def fillvalue(self):
        return None if self.dtype == bool else utils.get_dtype_max(self.dtype)

    @property
    def period(self):
        return (self.DEFAULT_ORIGIN,) * 2

    @property
    def extent(self):
        return None

    @property
    def timedelta(self):
        return None

    @property
    def geometry(self):
        return None

    @property
    def projection(self):
        return None

    @property
    def geo_transform(self):
        return None

    def get_sources_and_requests(self, **request):
        # first handle the 'time' and 'meta' requests
        mode = request["mode"]
        if mode == "time":
            return [(self.period[-1], None), ({"mode": "time"}, None)]
        elif mode == "meta":
            return [(None, None), ({"mode": "meta"}, None)]
        elif mode != "vals":
            raise ValueError("Unknown mode '{}'".format(mode))

        # build the request to be sent to the geometry source
        x1, y1, x2, y2 = request["bbox"]
        width, height = request["width"], request["height"]

        # be strict about the bbox, it may lead to segfaults else
        if x2 == x1 and y2 == y1:  # point
            min_size = None
        elif x1 < x2 and y1 < y2:
            min_size = min((x2 - x1) / width, (y2 - y1) / height)
        else:
            raise ValueError("Invalid bbox ({})".format(request["bbox"]))

        limit = self.limit
        if self.limit is None:
            limit = config.get("geomodeling.geometry-limit")

        geom_request = {
            "mode": "intersects",
            "geometry": box(*request["bbox"]),
            "projection": request["projection"],
            "min_size": min_size,
            "limit": limit,
            "start": request.get("start"),
            "stop": request.get("stop"),
        }
        # keep some variables for use in process()
        process_kwargs = {
            "mode": "vals",
            "column_name": self.column_name,
            "dtype": self.dtype,
            "no_data_value": self.fillvalue,
            "width": width,
            "height": height,
            "bbox": request["bbox"],
        }
        return [(self.source, geom_request), (process_kwargs, None)]

    @staticmethod
    def process(data, process_kwargs):
        # first handle the time and meta requests
        mode = process_kwargs["mode"]
        if mode == "time":
            return {"time": [data]}
        elif mode == "meta":
            return {"meta": [None]}

        column_name = process_kwargs["column_name"]
        height = process_kwargs["height"]
        width = process_kwargs["width"]
        no_data_value = process_kwargs["no_data_value"]
        dtype = process_kwargs["dtype"]
        f = data["features"]

        # get the value column to rasterize
        if column_name is None:
            values = None
        else:
            try:
                values = f[column_name]
            except KeyError:
                if f.index.name == column_name:
                    values = f.index.to_series()
                else:
                    values = False

        if len(f) == 0 or values is False:  # there is no data to rasterize
            values = np.full((1, height, width), no_data_value, dtype=dtype)
            return {"values": values, "no_data_value": no_data_value}

        result = utils.rasterize_geoseries(
            geoseries=f["geometry"] if "geometry" in f else None,
            values=values,
            bbox=process_kwargs["bbox"],
            projection=data["projection"],
            height=height,
            width=width,
        )

        values = result["values"]

        # cast to the expected dtype if necessary
        cast_values = values.astype(process_kwargs["dtype"])

        # replace the nodata value if necessary
        if result["no_data_value"] != no_data_value:
            cast_values[values == result["no_data_value"]] = no_data_value

        return {"values": cast_values, "no_data_value": no_data_value}


class RasterizeWKT(RasterBlock):
    """Converts a single geometry to a raster mask

    Args:
      wkt (string): the WKT representation of a geometry
      projection (string): the projection of the geometry

    Returns:
      RasterBlock with True for cells that are inside the geometry.
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
    def dtype(self):
        return np.dtype("bool")

    @property
    def fillvalue(self):
        return None

    @property
    def period(self):
        return (self.DEFAULT_ORIGIN,) * 2

    @property
    def extent(self):
        return tuple(
            utils.shapely_transform(
                load_wkt(self.wkt), self.projection, "EPSG:4326"
            ).bounds
        )

    @property
    def timedelta(self):
        return None

    @property
    def geometry(self):
        return ogr.CreateGeometryFromWkt(self.wkt, utils.get_sr(self.projection))

    @property
    def geo_transform(self):
        return None

    def get_sources_and_requests(self, **request):
        # first handle the 'time' and 'meta' requests
        mode = request["mode"]
        if mode == "time":
            data = self.period[-1]
        elif mode == "meta":
            data = None
        elif mode == "vals":
            data = {"wkt": self.wkt, "projection": self.projection}
        else:
            raise ValueError("Unknown mode '{}'".format(mode))
        return [(data, None), (request, None)]

    @staticmethod
    def process(data, request):
        mode = request["mode"]
        if mode == "time":
            return {"time": [data]}
        elif mode == "meta":
            return {"meta": [None]}
        # load the geometry and transform it into the requested projection
        geometry = load_wkt(data["wkt"])
        if data["projection"] != request["projection"]:
            geometry = utils.shapely_transform(
                geometry, data["projection"], request["projection"]
            )

        # take a shortcut when the geometry does not intersect the bbox
        x1, y1, x2, y2 = request["bbox"]
        if (x1 == x2) and (y1 == y2):
            # Don't do box(x1, y1, x2, y2), this gives an invalid geometry.
            bbox_geom = Point(x1, y1)
        else:
            bbox_geom = box(x1, y1, x2, y2)
        if not geometry.intersects(bbox_geom):
            return {
                "values": np.full(
                    (1, request["height"], request["width"]), False, dtype=bool
                ),
                "no_data_value": None,
            }

        return utils.rasterize_geoseries(
            geoseries=GeoSeries([geometry]) if not geometry.is_empty else None,
            bbox=request["bbox"],
            projection=request["projection"],
            height=request["height"],
            width=request["width"],
        )
