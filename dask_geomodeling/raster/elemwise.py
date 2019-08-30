"""
Module containing elementwise raster blocks.
"""
from functools import wraps

import numpy as np

from dask_geomodeling.utils import get_dtype_max, get_index

from .base import RasterBlock, BaseSingle

__all__ = [
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "Power",
    "FillNoData",
    "Equal",
    "NotEqual",
    "Greater",
    "GreaterEqual",
    "Less",
    "LessEqual",
    "Invert",
    "And",
    "Or",
    "Xor",
    "IsData",
    "IsNoData",
]


class BaseElementwise(RasterBlock):
    """ Base block for elementwise operations on rasters.

    This block only accepts stores that have aligned frames. The``extent``
    and ``period`` attributes are limited to the intersection of the
    ancestor stores. If the ancestor stores do not overlap (spatially or
    temporally), the elementwise block will be empty.
    """

    def __init__(self, *args):
        super(BaseElementwise, self).__init__(*args)
        # check the timedelta so that an error is raised if incompatible
        if len(self._sources) > 1:
            self.timedelta  # NOQA

    @property
    def _sources(self):
        # go through the arguments and list the blocks
        return [arg for arg in self.args if isinstance(arg, RasterBlock)]

    def get_sources_and_requests(self, **request):
        start = request.get("start", None)
        stop = request.get("stop", None)

        if start is not None and stop is not None:
            # limit request to self.period so that resulting data is aligned
            period = self.period
            request["start"] = max(start, period[0])
            request["stop"] = min(stop, period[1])

        process_kwargs = {"dtype": self.dtype.name, "fillvalue": self.fillvalue}
        sources_and_requests = [(source, request) for source in self.args]

        return [(process_kwargs, None)] + sources_and_requests

    @property
    def timedelta(self):
        """ The period between timesteps in case of equidistant time. """
        if len(self._sources) == 1:
            return self._sources[0].timedelta

        timedeltas = [s.timedelta for s in self._sources]
        if any(timedelta is None for timedelta in timedeltas):
            return None  # None precedes

        # multiple timedeltas: assert that they are equal
        if not timedeltas[1:] == timedeltas[:-1]:
            raise ValueError(
                "Time resolutions of input rasters are not "
                "equal ({}).".format(timedeltas)
            )
        else:
            return timedeltas[0]

    @property
    def period(self):
        """ Return period datetime tuple. """
        if len(self._sources) == 1:
            return self._sources[0].period

        periods = [s.period for s in self._sources]
        if any(period is None for period in periods):
            return None  # None precedes

        # multiple periods: return the overlapping period
        start = max([p[0] for p in periods])
        stop = min([p[1] for p in periods])
        if stop < start:
            return None  # no overlap
        else:
            return start, stop

    @property
    def extent(self):
        """ Boundingbox of contents in WGS84 projection. """
        if len(self._sources) == 1:
            return self._sources[0].extent

        extents = [s.extent for s in self._sources]
        if any(extent is None for extent in extents):
            return None  # None precedes

        # multiple extents: return the overlapping box
        x1 = max([e[0] for e in extents])
        y1 = max([e[1] for e in extents])
        x2 = min([e[2] for e in extents])
        y2 = min([e[3] for e in extents])
        if x2 <= x1 or y2 <= y1:
            return None  # no overlap
        else:
            return x1, y1, x2, y2

    @property
    def dtype(self):
        dtype = np.result_type(*self.args)
        if np.issubdtype(dtype, np.integer) or dtype == np.bool:
            # use at least int32
            return np.result_type(dtype, np.int32)
        elif np.issubdtype(dtype, np.floating):
            # use at least float32
            return np.result_type(dtype, np.float32)
        else:
            return dtype

    @property
    def fillvalue(self):
        dtype = self.dtype
        if dtype == np.bool:
            return
        else:
            return get_dtype_max(dtype)

    @property
    def geometry(self):
        """Intersection of geometries in the projection of the first store
        geometry. """
        geometries = [x.geometry for x in self.args]
        if any(x is None for x in geometries):
            return
        if len(geometries) == 1:
            return geometries[0]
        result = geometries[0]
        sr = result.GetSpatialReference()
        for geometry in geometries[1:]:
            if not geometry.GetSpatialReference().IsSame(sr):
                geometry = geometry.Clone()
                geometry.TransformTo(sr)
            result = result.Intersection(geometry)
        return result

    @property
    def projection(self):
        """Projection of the data if they match, else None"""
        projection = self.args[0].projection
        if projection is None:
            return
        for arg in self.args[1:]:
            if projection != arg.projection:
                return
        return projection

    @property
    def geo_transform(self):
        return  # TODO: Propagate geo_transform if the sources match


class BaseMath(BaseElementwise):
    """A block that applies basic math to two store-like objects."""

    def __init__(self, a, b):
        for x in (a, b):
            if not isinstance(x, (RasterBlock, np.ndarray, float, int)):
                raise TypeError("'{}' object is not allowed".format(type(x)))
        super(BaseMath, self).__init__(a, b)


class BaseComparison(BaseMath):
    """A block that applies basic comparisons to two store-like objects."""

    @property
    def dtype(self):
        return np.dtype("bool")


class BaseLogic(BaseElementwise):
    """A block that applies basic logical operations to two store-like objects.
    """

    def __init__(self, a, b):
        for x in (a, b):
            if isinstance(x, (RasterBlock, np.ndarray)):
                dtype = x.dtype
                if dtype != np.dtype("bool"):
                    raise TypeError("inputs must have boolean dtypes")
            elif not isinstance(x, bool):
                raise TypeError("'{}' object is not allowed".format(type(x)))
        super(BaseLogic, self).__init__(a, b)

    @property
    def dtype(self):
        return np.dtype("bool")

    @property
    def fillvalue(self):
        return None


def wrap_math_process_func(func):
    """This enables normal math functions to operate only on the data values.
    Nodata is propagated. In case of comparison operators, nodata becomes
    False, except for NotEqual, where it becomes True.

    'meta' and 'time' fields are propagated from the first source. """

    @wraps(func)  # propagates the name and docstring
    def math_process_func(process_kwargs, *args):
        compute_args = []  # the args for the math operation
        # perform the nodata masking manually, as numpy maskedarrays are slow
        nodata_mask = None

        dtype = process_kwargs["dtype"]
        fillvalue = process_kwargs["fillvalue"]

        for data in args:
            if data is None:
                return None
            if not isinstance(data, dict):
                compute_args.append(data)
            elif "time" in data or "meta" in data:
                # return the time / meta right away. assumes there are no
                # mixed requests and that time is aligned
                return data
            elif "values" in data:
                compute_args.append(data["values"])
                # update the nodata mask
                if data["values"].dtype == np.dtype("bool"):
                    continue  # boolean data does not contain nodata values.
                if "no_data_value" not in data:
                    continue
                _nodata_mask = data["values"] == data["no_data_value"]
                if nodata_mask is None:
                    nodata_mask = _nodata_mask
                else:
                    nodata_mask |= _nodata_mask

        if dtype == np.dtype("bool"):
            no_data_value = None
            if func is np.not_equal:
                fillvalue = True
            else:
                fillvalue = False
            func_kwargs = {}
        else:
            func_kwargs = {"dtype": dtype}
            no_data_value = fillvalue

        with np.errstate(all="ignore"):  # suppresses warnings
            result_values = func(*compute_args, **func_kwargs)

        if nodata_mask is not None:
            result_values[nodata_mask] = fillvalue
        return {"no_data_value": no_data_value, "values": result_values}

    return math_process_func


class Add(BaseMath):
    """
    Add a value.

    :param a: Addition parameter a
    :param b: Addition parameter b

    :type a: RasterBlock, scalar
    :type b: RasterBlock, scalar

    At least one of the parameters should be a RasterBlock. If the
    params are both RasterBlocks, they should share exactly the
    same time structure. The Snap block can be used to accomplish this.
    """

    process = staticmethod(wrap_math_process_func(np.add))


class Subtract(BaseMath):
    """
    Subtract a constant value from a store or vice versa.

    :param a: Subtraction parameter a
    :param b: Subtraction parameter b

    :type a: RasterBlock, scalar
    :type b: RasterBlock, scalar

    At least one of the parameters should be a RasterBlock. If the
    params are both RasterBlocks, they should share exactly the
    same time structure. The Snap block can be used to accomplish this.
    """

    process = staticmethod(wrap_math_process_func(np.subtract))


class Multiply(BaseMath):
    """
    Multiply by a value.

    :param a: Multiplication parameter a
    :param b: Multiplication parameter b

    :type a: RasterBlock, scalar
    :type b: RasterBlock, scalar

    At least one of the parameters should be a RasterBlock. If the
    params are both RasterBlocks, they should share exactly the
    same time structure. The Snap block can be used to accomplish this.
    """

    process = staticmethod(wrap_math_process_func(np.multiply))


class Divide(BaseMath):
    """
    Divide Store by a constant value or vice versa.

    :param a: Division parameter a
    :param b: Division parameter b

    :type a: RasterBlock, scalar
    :type b: RasterBlock, scalar

    At least one of the parameters should be a RasterBlock. If the
    params are both RasterBlocks, they should share exactly the
    same time structure. The Snap block can be used to accomplish this.
    """

    process = staticmethod(wrap_math_process_func(np.divide))

    @property
    def dtype(self):
        # use at least float32
        return np.result_type(np.float32, *self.args)


class Power(BaseMath):
    """
    Raise each number in a to the power b.

    :param a: base
    :param b: exponent

    :type a: RasterBlock, scalar
    :type b: RasterBlock, scalar

    At least one of the parameters should be a RasterBlock. If the
    params are both RasterBlocks, they should share exactly the
    same time structure. The Snap block can be used to accomplish this.
    """

    process = staticmethod(wrap_math_process_func(np.power))

    def __init__(self, a, b):
        # cast negative integer exponents to float, because if a is also of
        # integer type, power will raise a ValueError
        if isinstance(b, int) and b < 0:
            b = float(b)
        super(Power, self).__init__(a, b)


class Equal(BaseComparison):
    """
    Compares the values of two stores and returns True if they are equal.

    Note that "no data" is not equal to "no data".

    :param a: Comparison parameter a
    :param b: Comparison parameter b

    :type a: RasterBlock, scalar
    :type b: RasterBlock, scalar

    At least one of the parameters should be a RasterBlock. If the
    params are both RasterBlocks, they should share exactly the
    same time structure. The Snap block can be used to accomplish this.
    """

    process = staticmethod(wrap_math_process_func(np.equal))


class NotEqual(BaseComparison):
    """
    Compares the values of two stores and returns False if they are equal.

    Note that "no data" is not equal to "no data".

    :param a: Comparison parameter a
    :param b: Comparison parameter b

    :type a: RasterBlock, scalar
    :type b: RasterBlock, scalar

    At least one of the parameters should be a RasterBlock. If the
    params are both RasterBlocks, they should share exactly the
    same time structure. The Snap block can be used to accomplish this.
    """

    process = staticmethod(wrap_math_process_func(np.not_equal))


class Greater(BaseComparison):
    """
    Returns True if a is greater than b.

    Note that "no data" will always return False

    :param a: Comparison parameter a
    :param b: Comparison parameter b

    :type a: RasterBlock, scalar
    :type b: RasterBlock, scalar

    At least one of the parameters should be a RasterBlock. If the
    params are both RasterBlocks, they should share exactly the
    same time structure. The Snap block can be used to accomplish this.
    """

    process = staticmethod(wrap_math_process_func(np.greater))


class GreaterEqual(BaseComparison):
    """
    Returns True if a is greater than or equal to b.

    Note that "no data" will always return False

    :param a: Comparison parameter a
    :param b: Comparison parameter b

    :type a: RasterBlock, scalar
    :type b: RasterBlock, scalar

    At least one of the parameters should be a RasterBlock. If the
    params are both RasterBlocks, they should share exactly the
    same time structure. The Snap block can be used to accomplish this.
    """

    process = staticmethod(wrap_math_process_func(np.greater_equal))


class Less(BaseComparison):
    """
    Returns True if a is less than b.

    Note that "no data" will always return False

    :param a: Comparison parameter a
    :param b: Comparison parameter b

    :type a: RasterBlock, scalar
    :type b: RasterBlock, scalar

    At least one of the parameters should be a RasterBlock. If the
    params are both RasterBlocks, they should share exactly the
    same time structure. The Snap block can be used to accomplish this.
    """

    process = staticmethod(wrap_math_process_func(np.less))


class LessEqual(BaseComparison):
    """
    Returns True if a is less than or equal to b.

    Note that "no data" will always return False

    :param a: Comparison parameter a
    :param b: Comparison parameter b

    :type a: RasterBlock, scalar
    :type b: RasterBlock, scalar

    At least one of the parameters should be a RasterBlock. If the
    params are both RasterBlocks, they should share exactly the
    same time structure. The Snap block can be used to accomplish this.
    """

    process = staticmethod(wrap_math_process_func(np.less_equal))


class Invert(BaseSingle):
    """
    Swaps False and True ("not x" or "~x").

    :param x: raster data to invert
    :type x: RasterBlock
    """

    def __init__(self, x):
        super(Invert, self).__init__(x)
        if x.dtype != np.dtype("bool"):
            raise TypeError("input block must have boolean dtype")

    @staticmethod
    def process(data):
        if "values" in data:
            return {"values": ~data["values"], "no_data_value": None}
        else:
            return data

    @property
    def dtype(self):
        return np.dtype("bool")


class IsData(BaseSingle):
    """Returns True where raster has data.

    :param store:
    :type store: RasterBlock
    """

    def __init__(self, store):
        if store.dtype == np.dtype("bool"):
            raise TypeError("input block must not have boolean dtype")
        super(IsData, self).__init__(store)

    @staticmethod
    def process(data):
        if data is None or "values" not in data:
            return data
        values = data["values"]
        no_data_value = data["no_data_value"]
        return {"values": values != no_data_value, "no_data_value": None}

    @property
    def dtype(self):
        return np.dtype("bool")

    @property
    def fillvalue(self):
        return None


class IsNoData(IsData):
    """Returns True where raster has no data.

    :param store:
    :type store: RasterBlock
    """

    @staticmethod
    def process(data):
        if data is None or "values" not in data:
            return data
        values = data["values"]
        no_data_value = data["no_data_value"]
        return {"values": values == no_data_value, "no_data_value": None}


class And(BaseLogic):
    """
    Returns True where a and b are True.

    :param a: Comparison parameter a
    :param b: Comparison parameter b

    :type a: RasterBlock, boolean
    :type b: RasterBlock, boolean

    All input parameters should have a boolean dtype and at least one of the
    parameters should be a RasterBlock. If both are RasterBlocks, they should
    share exactly the same time structure. The Snap block can be used to
    accomplish this.
    """

    process = staticmethod(wrap_math_process_func(np.logical_and))


class Or(BaseLogic):
    """
    Returns True where a or b are True.

    :param a: Comparison parameter a
    :param b: Comparison parameter b

    :type a: RasterBlock, boolean
    :type b: RasterBlock, boolean

    All input parameters should have a boolean dtype and at least one of the
    parameters should be a RasterBlock. If both are RasterBlocks, they should
    share exactly the same time structure. The Snap block can be used to
    accomplish this.
    """

    process = staticmethod(wrap_math_process_func(np.logical_or))


class Xor(BaseLogic):
    """
    Returns True where either a or b is True (exclusive-or)

    :param a: Comparison parameter a
    :param b: Comparison parameter b

    :type a: RasterBlock, boolean
    :type b: RasterBlock, boolean

    All input parameters should have a boolean dtype and at least one of the
    parameters should be a RasterBlock. If both are RasterBlocks, they should
    share exactly the same time structure. The Snap block can be used to
    accomplish this.
    """

    process = staticmethod(wrap_math_process_func(np.logical_xor))


class FillNoData(BaseElementwise):
    """
    Combines multiple rasters, filling in nodata values.

    :param args: list of raster sources to be combined.
    :type args: list of RasterBlock

    Values at equal timesteps in the contributing rasters are pasted left to
    right. Therefore values from rasters that are more 'to the left' are
    shadowed by values from rasters more 'to the right'. However, 'no data'
    values are transparent and do not shadow underlying data values.
    """

    def __init__(self, *args):
        # TODO Expand this block so that it takes ndarrays and scalars.
        for arg in args:
            if not isinstance(arg, RasterBlock):
                raise TypeError("'{}' object is not allowed".format(type(arg)))
        super(FillNoData, self).__init__(*args)

    @staticmethod
    def process(*args):
        """Combine data, filling in nodata values."""
        data_list = []
        no_data_values = []
        for data in args:
            if data is None:
                continue
            elif "time" in data or "meta" in data:
                # return the time / meta right away. assumes there are no
                # mixed requests and that time is aligned
                return data
            elif "values" in data and "no_data_value" in data:
                data_list.append(data["values"])
                no_data_values.append(data["no_data_value"])

        dtype = np.result_type(*data_list)
        fillvalue = get_dtype_max(dtype)

        # initialize values array
        values = np.full(data_list[0].shape, fillvalue, dtype=dtype)

        # populate values array
        for data, no_data_value in zip(data_list, no_data_values):
            # index is where the source has data
            index = get_index(data, no_data_value)
            values[index] = data[index]

        return {"values": values, "no_data_value": fillvalue}
