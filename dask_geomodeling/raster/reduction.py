"""
Module containing reduction raster blocks.
"""
import numpy as np
from dask_geomodeling.utils import get_index, parse_percentile_statistic

from functools import partial

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
        percentile = parse_percentile_statistic(statistic)
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
        percentile = parse_percentile_statistic(statistic)
        if percentile is None:
            raise KeyError('Unknown statistic "{}"'.format(statistic))
        else:
            statistic = "percentile"

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
