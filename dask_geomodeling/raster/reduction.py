"""
Module containing reduction raster blocks.
"""
import numpy as np
from dask_geomodeling.utils import filter_none, get_index, Extent
from dask_geomodeling.utils import parse_percentile_statistic
from .base import RasterBlock
from .elemwise import BaseElementwise
from functools import partial

__all__ = ["Max"]

STATISTICS = {
    "first": None,
    "last": None,
    "count": None,
    "sum": np.nansum,
    "mean": np.nanmean,
    "min": np.nanmin,
    "max": np.nanmax,
    "argmin": np.nanargmin,
    "argmax": np.nanargmax,
    "std": np.nanstd,
    "var": np.nanvar,
    "median": np.nanmedian,
    "product": np.nanprod,
    # "p<number>" uses np.nanpercentile
}


def check_statistic(statistic):
    if statistic not in STATISTICS:
        statistic, percentile = parse_percentile_statistic(statistic)
        if percentile is None:
            raise ValueError('Unknown statistic "{}"'.format(statistic))


def reduce_rasters(stack, statistic, no_data_value=None, dtype=None):
    """Apply a statistic (e.g. "mean") to a stack of rasters, skipping
    'no data' values.

    In this context, reduce means that the dimensionality of the input data
    is reduced by one.

    Args:
      stack (list): a list of dicts containing "values" (ndarray)
        and "no_data_value". If the list has zero length or if the ndarrays
        do not have the same shape, a ValueError is raised.
      statistic (str): the applied statistic (no data is ignored). One of:
        {"last", "first", "count", "sum", "mean", "min",
        "max", "argmin", "argmax", "product", "std", "var", "p<number>"}
      no_data_value (number, optional): the 'no data' value in the output
        array. Defaults to the no data value of the first element in the stack.
      dtype (str or dtype): the datatype of the output array. Defaults to the
        dtype of the first element in the stack. If the input
        data cannot be cast to this dtype, a ValueError is raised.

    Returns:
      dict with "values" and "no_data_value"
    """
    if statistic not in STATISTICS:
        statistic, percentile = parse_percentile_statistic(statistic)
        if percentile is None:
            raise KeyError('Unknown statistic "{}"'.format(statistic))

    if len(stack) == 0:
        raise ValueError("Cannot reduce a zero-length stack")

    # get the output array properties (dtype, no_data_value, shape)
    if dtype is None:
        dtype = stack[0]["values"].dtype
    if no_data_value is None:
        no_data_value = stack[0]["no_data_value"]
    shape = stack[0]["values"].shape

    # sum, count and nans output do not contain no data: fill zeroes right away
    if statistic in {"sum", "count", "nans"}:
        fill_value = 0
    else:
        fill_value = no_data_value

    # create the output array
    out = np.full(shape, fill_value, dtype)

    if statistic == "last":
        # populate 'out' with the last value that is not 'no data'
        for data in stack:
            index = get_index(data["values"], data["no_data_value"])
            out[index] = data["values"][index]
    elif statistic == "first":
        # populate 'out' with the first value that is not 'no data'
        for data in stack[::-1]:
            index = get_index(data["values"], data["no_data_value"])
            out[index] = data["values"][index]
    elif statistic == "count":
        # count the number of values that are not 'no data'
        for data in stack:
            out += get_index(data["values"], data["no_data_value"])
    else:
        if statistic == "percentile":
            func = partial(np.nanpercentile, q=percentile)
        else:
            func = STATISTICS[statistic]
        # transform 'no data' into 'nan' to be able to use numpy functions
        # NB: the dtype is at least float16 to accomodate NaN
        stack_array = np.full(
            (len(stack),) + shape, np.nan, np.result_type(dtype, np.float16)
        )
        for i, data in enumerate(stack):
            index = get_index(data["values"], data["no_data_value"])
            stack_array[i, index] = data["values"][index]

        # protect against all-NaN slice warnings and errors
        not_all_nan = ~np.all(np.isnan(stack_array), axis=0)

        # perform the math
        out[not_all_nan] = func(stack_array[:, not_all_nan], axis=0)

    return {"values": out, "no_data_value": no_data_value}


class BaseReduction(BaseElementwise):
    def __init__(self, *args):
        for arg in args:
            if not isinstance(arg, RasterBlock):
                raise TypeError("'{}' object is not allowed".format(type(arg)))
        super().__init__(*args)

    def get_sources_and_requests(self, **request):
        """
        This method is more strict than the one from BaseElementWise, it is not
        possible to get data from this block when its period is None, meaning
        that either the sources have no common period, or one of the sources'
        periods is None.
        """
        period = self.period
        process_kwargs = {
            "dtype": self.dtype.name, "fillvalue": self.fillvalue,
        }
        if period is None:
            return [(process_kwargs, None)]

        # limit request to self.period so that resulting data is aligned
        start = request.get("start", None)
        stop = request.get("stop", None)
        if start is not None:
            if stop is not None:
                request["start"] = max(start, period[0])
                request["stop"] = min(stop, period[1])
            else:  # stop is None but start isn't
                request["start"] = min(max(start, period[0]), period[1])
        else:  # start and stop are both None
            request["start"] = period[1]

        sources_and_requests = [(source, request) for source in self.args]

        return [(process_kwargs, None)] + sources_and_requests

    @property
    def extent(self):
        """ Boundingbox of combined contents in WGS84 projection. """
        extents = filter_none([x.extent for x in self.args])
        if len(extents) == 0:
            return None
        elif len(extents) == 1:
            return extents[0]

        # multiple extents: return the joined box
        x1 = min([e[0] for e in extents])
        y1 = min([e[1] for e in extents])
        x2 = max([e[2] for e in extents])
        y2 = max([e[3] for e in extents])
        return x1, y1, x2, y2

    @property
    def geometry(self):
        """Combined geometry in the projection of the first store geometry. """
        geometries = filter_none([x.geometry for x in self.args])
        if len(geometries) == 0:
            return
        elif len(geometries) == 1:
            return geometries[0]
        extent = Extent.from_geometry(geometries[0])
        for geometry in geometries[1:]:
            extent = extent.union(Extent.from_geometry(geometry))
        return extent.as_geometry()


def wrap_reduction_function(statistic):
    def reduction_function(process_kwargs, *args):
        stack = []
        for arg in args:
            if "time" in arg or "meta" in arg:
                # return the time / meta right away. assumes there are no
                # mixed requests and that time is aligned
                return arg
            if arg is None:
                continue
            stack.append(arg)

        # return None if all source data is None
        if len(stack) == 0:
            return

        # see BaseElementWise.get_sources_and_requests
        return reduce_rasters(
            stack,
            statistic,
            process_kwargs["fillvalue"],
            process_kwargs["dtype"],
        )
    return reduction_function


class Max(BaseReduction):
    """
    Take the maximum value of two or more rasters, ignoring no data.

    Args:
      *args (list of RasterBlocks): list of rasters to be combined.

    Returns:
      RasterBlock with the maximum values
    """
    process = staticmethod(wrap_reduction_function("max"))

    @property
    def dtype(self):
        # skip the default behaviour where we use at least int32 / float32
        return np.result_type(*self.args)
