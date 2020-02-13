"""
Module containing elementwise raster blocks.
"""
from functools import wraps

import numpy as np

from dask_geomodeling.utils import get_dtype_max, get_index, GeoTransform

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
    "ArgMax",
    "ArgMin",
    "CountData",
    "Max",
    "Mean",
    "Median",
    "Min",
    "Product",
    "Std",
    "Sum",
    "Variance",
    "Round",
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
            if period is not None:
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
        geometries = [x.geometry for x in self._sources]
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
        if result.GetArea() == 0.0:
            return
        return result

    @property
    def projection(self):
        """Projection of the data if they match, else None"""
        projection = self._sources[0].projection
        if projection is None:
            return
        for arg in self._sources[1:]:
            if projection != arg.projection:
                return
        return projection

    @property
    def geo_transform(self):
        geo_transform = self._sources[0].geo_transform
        if geo_transform is None:
            return
        geo_transform = GeoTransform(geo_transform)
        for arg in self._sources[1:]:
            other = arg.geo_transform
            if other is None or not geo_transform.aligns_with(other):
                return
        return geo_transform


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
            else:
                raise TypeError("Cannot apply math function to value {}".format(data))

        if dtype == np.dtype("bool"):
            no_data_value = None
            if func is np.not_equal:
                fillvalue = True
            else:
                fillvalue = False
            func_kwargs = {}
        elif dtype == np.dtype("float32"):
            func_kwargs = {}
            no_data_value = fillvalue        
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
    Add two rasters together or add a constant value to a raster.

    Either one or both of the inputs should be a RasterBlock. In case of
    two raster inputs the temporal properties of the rasters should be equal,
    however spatial properties can be different.

    Args:
      a (RasterBlock, number): Addition term a
      b (RasterBlock, number): Addition term b

    Returns:
      RasterBlock containing the result of the addition.
	"""

    process = staticmethod(wrap_math_process_func(np.add))


class Subtract(BaseMath):
    """
    Subtract two rasters or subtract a constant value from a raster

    Either one or both of the inputs should be a RasterBlock. In case of
    two raster inputs the temporal properties of the rasters should be equal,
    however spatial properties can be different.

    Args:
      a (RasterBlock, number): Term be subtracted from
      b (RasterBlock, number): Term to be subtracted

    Returns:
      RasterBlock containing the result of the function subtract.
    """

    process = staticmethod(wrap_math_process_func(np.subtract))


class Multiply(BaseMath):
    """
    Multiply two rasters or multiply a raster by a constant value.

    Either one or both of the inputs should be a RasterBlock. In case of
    two raster inputs the temporal properties of the rasters should be equal,
    however spatial properties can be different.

    Args:
      a (RasterBlock, number): Multiplication factor a
      b (RasterBlock, number): Multiplication factor b
     
    Returns:
      RasterBlock containing the result of the multiplication.
    """

    process = staticmethod(wrap_math_process_func(np.multiply))


class Divide(BaseMath):
    """
    Divide two rasters or divide a raster by a constant value.

    Either one or both of the inputs should be a RasterBlock. In case of
    two raster inputs the temporal properties of the rasters should be equal,
    however spatial properties can be different.

    Args:
      a (RasterBlock, number): Numerator
      b (RasterBlock, number): Denominator
     
    Returns:
      RasterBlock containing the result of the division.
    """

    process = staticmethod(wrap_math_process_func(np.divide))

    @property
    def dtype(self):
        # use at least float32
        return np.result_type(np.float32, *self.args)


class Round(BaseMath):
    """
    Round a raster to the given number of decimals.

    Either one or both of the inputs should be a RasterBlock. In case of
    two raster inputs the temporal properties of the rasters should be equal,
    however spatial properties can be different.

    Args:
      RasterBlock: Base raster with decimal values
      decimals: Whole number of decimals
      
    Returns:
      RasterBlock containing the result of the rounded function. 
    """

    process = staticmethod(wrap_math_process_func(np.round))

    def __init__(self, a, decimals=0):
        if not isinstance(a, RasterBlock):
            raise TypeError("'{}' object is not allowed.".format(type(a)))
        if not isinstance(decimals, int):
            raise TypeError("'{}' object is not allowed.".format(type(decimals)))
        super(Round, self).__init__(a, decimals)


class Power(BaseMath):
    """
    Exponential function with either a raster and a number or two rasters.

    Either one or both of the inputs should be a RasterBlock. In case of
    two raster inputs the temporal properties of the rasters should be equal,
    however spatial properties can be different.

    Args:
      a (RasterBlock, number): Base
      b (RasterBlock, number): Exponent
      
    Returns:
      RasterBlock containing the result of the exponential function. 
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
    Compares the values of two rasters and returns True for equal elements.

    This operation can be used to compare two rasters or to compare a raster
    with a static value. Note that 'no data' is not equal to 'no data':
    False is returned if any of the two terms is 'no data'.

    Either one or both of the inputs should be a RasterBlock. In case of
    two raster inputs the temporal properties of the rasters should be equal,
    however spatial properties can be different.

    Args:
      a (RasterBlock, number): Comparison term a
      b (RasterBlock, number): Comparison term b
      
    Returns:
      RasterBlock containing boolean values
    """

    process = staticmethod(wrap_math_process_func(np.equal))


class NotEqual(BaseComparison):
    """
    Compares the values of two rasters and returns False for equal elements.

    This operation can be used to compare two rasters or to compare a raster
    with a static value. Note that 'no data' is not equal to 'no data':
    True is returned if any of the two terms is 'no data'.

    Either one or both of the inputs should be a RasterBlock. In case of
    two raster inputs the temporal properties of the rasters should be equal,
    however spatial properties can be different.

    Args:
      a (RasterBlock, number): Comparison term a
      b (RasterBlock, number): Comparison term b

    Returns:
      RasterBlock containing boolean values
    """

    process = staticmethod(wrap_math_process_func(np.not_equal))


class Greater(BaseComparison):
    """
    Compares the values of two rasters and returns True if an element in the
    first term is greater.

    This operation can be used to compare two rasters or to compare a raster
    with a static value. Note that False is returned if any of the two terms is
    'no data'.

    Either one or both of the inputs should be a RasterBlock. In case of
    two raster inputs the temporal properties of the rasters should be equal,
    however spatial properties can be different.

    Args:
      a (RasterBlock, number): Comparison term a
      b (RasterBlock, number): Comparison term b

    Returns:
      RasterBlock containing boolean values
    """

    process = staticmethod(wrap_math_process_func(np.greater))


class GreaterEqual(BaseComparison):
    """"
    Compares the values of two rasters and returns True if an element in the
    first term is greater or equal.

    This operation can be used to compare two rasters or to compare a raster
    with a static value. Note that False is returned if any of the two terms is
    'no data'.

    Either one or both of the inputs should be a RasterBlock. In case of
    two raster inputs the temporal properties of the rasters should be equal,
    however spatial properties can be different.
    
    Args:
      a (RasterBlock, number): Comparison term a
      b (RasterBlock, number): Comparison term b

    Returns:
      RasterBlock containing boolean values
    """

    process = staticmethod(wrap_math_process_func(np.greater_equal))


class Less(BaseComparison):
    """
    Compares the values of two rasters and returns True if an element in the
    first term is less.

    This operation can be used to compare two rasters or to compare a raster
    with a static value. Note that False is returned if any of the two terms is
    'no data'.

    Either one or both of the inputs should be a RasterBlock. In case of
    two raster inputs the temporal properties of the rasters should be equal,
    however spatial properties can be different.
    
    Args:
      a (RasterBlock, number): Comparison term a
      b (RasterBlock, number): Comparison term b

    Returns:
      RasterBlock containing boolean values
    """

    process = staticmethod(wrap_math_process_func(np.less))


class LessEqual(BaseComparison):
    """
    Compares the values of two rasters and returns True if an element in the
    first term is less or equal.

    This operation can be used to compare two rasters or to compare a raster
    with a static value. Note that False is returned if any of the two terms is
    'no data'.

    Either one or both of the inputs should be a RasterBlock. In case of
    two raster inputs the temporal properties of the rasters should be equal,
    however spatial properties can be different.
    
    Args:
      a (RasterBlock, number): Comparison term a
      b (RasterBlock, number): Comparison term b

    Returns:
      RasterBlock containing boolean values
    """

    process = staticmethod(wrap_math_process_func(np.less_equal))


class Invert(BaseSingle):
    """
    Logically invert a raster (swap True and False).
    
    Takes a single input raster containing boolean values and outputs a boolean
    raster with the same spatial and temportal properties.

    Args:
      x (RasterBlock): Boolean raster with values to invert

    Returns:
      RasterBlock with boolean values opposite to the input raster. 
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
    """
    Returns True where raster has data.
    
    Takes a single input raster and outputs a boolean raster with the same
    spatial and temporal properties.

    Args:
      store (RasterBlock): Input raster

    Returns:
      RasterBlock with boolean values. 
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
    """
    Returns True where raster has no data.

    Takes a single input raster and outputs a boolean raster with the same
    spatial and temporal properties.

    Args:
      store (RasterBlock): Input raster

    Returns:
      RasterBlock with boolean values. 
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
    Returns True where both inputs are True.

    Either one or both of the inputs should be a boolean RasterBlock. In case
    of two raster inputs the temporal properties of the rasters should be
    equal, however spatial properties can be different.
    
    Args:
      a (RasterBlock, boolean): Logical term a
      b (RasterBlock, boolean): Logical term b

    Returns:
      RasterBlock containing boolean values
    """

    process = staticmethod(wrap_math_process_func(np.logical_and))


class Or(BaseLogic):
    """
    Returns True where any of inputs is True.

    Either one or both of the inputs should be a boolean RasterBlock. In case
    of two raster inputs the temporal properties of the rasters should be
    equal, however spatial properties can be different.

    Args:
      a (RasterBlock, boolean): Logical term a
      b (RasterBlock, boolean): Logical term b

    Returns:
      RasterBlock containing boolean values
    """

    process = staticmethod(wrap_math_process_func(np.logical_or))


class Xor(BaseLogic):
    """
    Exclusive or: returns True where exactly one of the inputs is True.

    Where both inputs are True, False is returned.

    Either one or both of the inputs should be a boolean RasterBlock. In case
    of two raster inputs the temporal properties of the rasters should be
    equal, however spatial properties can be different.

    Args:
      a (RasterBlock, boolean): Logical term a
      b (RasterBlock, boolean): Logical term b

    Returns:
      RasterBlock containing boolean values
    """

    process = staticmethod(wrap_math_process_func(np.logical_xor))


class BaseElementwiseMultiple(BaseElementwise):
    """Base class for elementwise combinations of N rasters"""
    def __init__(self, *args):
        # TODO Expand this block so that it takes ndarrays and scalars.
        for arg in args:
            if not isinstance(arg, RasterBlock):
                raise TypeError("'{}' object is not allowed".format(type(arg)))
        super().__init__(*args)

    @staticmethod
    def preprocess(*args):
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

        dtype = np.float64

        # populate values array
        multivalues = np.full(
            (len(data_list), ) + data_list[0].shape, np.nan, dtype=dtype
        )
        for i, (data, no_data_value) in enumerate(zip(data_list, no_data_values)):
            # index is where the source has data
            index = get_index(data, no_data_value)
            multivalues[i, index] = data[index]
        return multivalues


class FilNoData(BaseElementwiseMultiple):
    @staticmethod
    def process(*args):
        """Combine data, filling in nodata values."""
        multivalues = BaseElementwiseMultiple.preprocess(*args)
        # convert here
        return values_fillvalue


class ArgMax(BaseElementwiseMultiple):
    """Compare multiple rasters to return the element-wise indices of the maximum values (ignores partial NoData)."""
    @staticmethod
    def process(kwargs, *args):
        """Combine data, filling in nodata values."""
        multivalues = BaseElementwiseMultiple.preprocess(*args)
        values = np.nanargmax(multivalues, axis=0, dtype=kwargs["dtype"])
        values[np.isnan(values)] = kwargs["fillvalue"]
        return {"values": values, "no_data_value": kwargs["fillvalue"]}

    @property
    def dtype(self):
        return np.uint8

    @property
    def fillvalue(self):
        return 255


class ArgMin(FillNoData):
    """Compare multiple rasters to return the element-wise indices of the minimum values (ignores partial NoData)."""
    @staticmethod
    def process(self, *args):
        values_fillvalue, multivalues = FillNoData.preprocess(*args)
        fillvalue = 255
        values = np.nanargmin(multivalues, axis=0)
        values[np.isnan(values)] = fillvalue 
        return {"values": values, "no_data_value": fillvalue}


class CountData(FillNoData):
    """Compare multiple rasters to return the element-wise count of data occurrence (ignores partial NoData)."""
    @staticmethod
    def process(self, *args):
        values_fillvalue, multivalues = FillNoData.preprocess(*args)
        fillvalue = 255
        values = np.count_nonzero(~np.isnan(multivalues), axis=0)
        values[np.isnan(values)] = fillvalue 
        return {"values": values, "no_data_value": fillvalue}


class Max(BaseElementwiseMultiple):
    """Compares the values of multiple rasters and returns element-wise max (ignorese partial NoData)."""
    @staticmethod
    def process(kwargs, *args):
        multivalues = BaseElementwiseMultiple.preprocess(*args)
        values = np.nanmax(multivalues, axis=0, keepdims=True)
        no_data = np.isnan(values)
        values.astype(kwargs["dtype"])
        values[no_data] = kwargs["fillvalue"]
        return {"values": values, "no_data_value": kwargs["fillvalue"]}


class Mean(FillNoData):
    """Compares the values of multiple rasters and returns element-wise mean (ignores partial NoData)."""
    @staticmethod
    def process(self, *args):
        values_fillvalue, multivalues = FillNoData.preprocess(*args)
        fillvalue = values_fillvalue["no_data_value"]
        values = np.nanmean(multivalues, axis=0, keepdims=True)
        values[np.isnan(values)] = fillvalue 
        return {"values": values, "no_data_value": fillvalue}


class Median(FillNoData):
    """Compares the values of multiple rasters and returns element-wise median (ignores partial NoData)."""
    @staticmethod
    def process(self, *args):
        values_fillvalue, multivalues = FillNoData.preprocess(*args)
        fillvalue = values_fillvalue["no_data_value"]
        values = np.nanmedian(multivalues, axis=0, keepdims=True)
        values[np.isnan(values)] = fillvalue 
        return {"values": values, "no_data_value": fillvalue}


class Min(FillNoData):
    """Compares the values of multiple rasters and returns element-wise min (ignores partial NoData)."""
    @staticmethod
    def process(self, *args):
        values_fillvalue, multivalues = FillNoData.preprocess(*args)
        fillvalue = values_fillvalue["no_data_value"]
        values = np.nanmin(multivalues, axis=0, keepdims=True)
        values[np.isnan(values)] = fillvalue 
        return {"values": values, "no_data_value": fillvalue}


class Product(FillNoData):
    """Compare multiple rasters to return the element-wise product (partial NoData are set to 1)."""
    @staticmethod
    def process(self, *args):
        values_fillvalue, multivalues = FillNoData.preprocess(*args)
        fillvalue = values_fillvalue["no_data_value"]
        values = np.nanprod(multivalues, axis=0, keepdims=True)
        values[np.isnan(values)] = fillvalue 
        return {"values": values, "no_data_value": fillvalue}


class Std(FillNoData):
    """Compare multiple rasters to return the element-wise standard deviations (ignores partial NoData)."""
    @staticmethod
    def process(self, *args):
        values_fillvalue, multivalues = FillNoData.preprocess(*args)
        fillvalue = values_fillvalue["no_data_value"]
        values = np.nanstd(multivalues, axis=0, keepdims=True)
        values[np.isnan(values)] = fillvalue 
        return {"values": values, "no_data_value": fillvalue}


class Sum(FillNoData):
    """Compare multiple rasters to return the element-wise summed values (ignores partial NoData)."""
    @staticmethod
    def process(self, *args):
        values_fillvalue, multivalues = FillNoData.preprocess(*args)
        fillvalue = values_fillvalue["no_data_value"]
        values = np.nansum(multivalues, axis=0, keepdims=True)
        values[np.isnan(values)] = fillvalue 
        return {"values": values, "no_data_value": fillvalue}


class Variance(FillNoData):
    """Compare multiple rasters to return the element-wise variance values (ignores partial NoData)."""
    @staticmethod
    def process(self, *args):
        values_fillvalue, multivalues = FillNoData.preprocess(*args)
        fillvalue = values_fillvalue["no_data_value"]
        values = np.nanvar(multivalues, axis=0, keepdims=True)
        values[np.isnan(values)] = fillvalue 
        return {"values": values, "no_data_value": fillvalue}
