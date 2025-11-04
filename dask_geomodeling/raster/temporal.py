"""
Module containing raster blocks for temporal operations.
"""
from functools import partial
import pytz
from datetime import timedelta as Timedelta
from pandas.tseries.frequencies import to_offset

import numpy as np
import pandas as pd
import warnings

from dask_geomodeling.utils import (
    get_dtype_max,
    parse_percentile_statistic,
    dtype_for_statistic,
    find_nearest,
)

from .base import RasterBlock, BaseSingle


__all__ = ["Snap", "Shift", "TemporalSum", "TemporalAggregate", "Cumulative"]


MICROSECOND = Timedelta(microseconds=1)


class Snap(RasterBlock):
    """
    Snap the time structure of a raster to that of another raster.

    This operations allows to take the cell values from one raster ('store')
    and the temporal properties of another raster ('index').

    If the store is not a temporal raster, its cell values are copied to each
    timestep of the index raster. If the store is also a temporal raster, this
    operation looks at each 'index' timestamp and takes the closest 'store'
    timestamp as cell values.

    Args:
      store (RasterBlock): Return cell values from this raster
      index (RasterBlock): Snap values to the timestamps from this raster

    Returns:
      RasterBlock with temporal properties of the index.
    """

    def __init__(self, store, index):
        for x in (store, index):
            if not isinstance(x, RasterBlock):
                raise TypeError("'{}' object is not allowed".format(type(x)))
        super(Snap, self).__init__(store, index)

    @property
    def store(self):
        return self.args[0]

    @property
    def index(self):
        return self.args[1]

    def __len__(self):
        return len(self.index)

    @property
    def dtype(self):
        return self.store.dtype

    @property
    def fillvalue(self):
        return self.store.fillvalue

    @property
    def period(self):
        return self.index.period if self.store else None

    @property
    def timedelta(self):
        return self.index.timedelta

    @property
    def temporal(self):
        return self.index.temporal

    @property
    def extent(self):
        return self.store.extent

    @property
    def geometry(self):
        return self.store.geometry

    @property
    def projection(self):
        return self.store.projection

    @property
    def geo_transform(self):
        return self.store.geo_transform

    def get_sources_and_requests(self, **request):
        # to avoid repeated evaluation of the periods
        store_period = self.store.period
        index_period = self.index.period

        # if any store is empty, Snap will be empty
        if store_period is None or index_period is None:
            return [(None, None)]

        # time requests are easy: just pass them to self.index
        if request["mode"] == "time":
            return [(None, None), (self.index, request)]

        # query the index time
        start = request.get("start")
        stop = request.get("stop")
        index_result = self.index.get_data(mode="time", start=start, stop=stop)
        if index_result is None:
            return [(None, None)]
        index_time = index_result["time"]

        # for single frame results, query the store with the time from the index
        if stop is None:
            request["start"] = index_time[0]
            return [(None, None), (self.store, request)]

        # multiband request; knowledge of the time structure of the store near start,
        # between start and stop, and near stop is required - result frames can be
        # nearest to store frames outside the requested [start, stop] interval.
        if store_period[0] == store_period[1]:
            # there is only one frame in the store
            store_time = [store_period[0]]
        else:
            # obtain time near start, between start and stop, and near stop
            def get_store_time_set(start=None, stop=None):
                result = self.store.get_data(mode="time", start=start, stop=stop)
                if result is None:
                    return set()
                return set(result["time"])
            store_time = sorted(
                get_store_time_set(start=start)
                | get_store_time_set(start=start, stop=stop)
                | get_store_time_set(start=stop)
            )

        # return a requst to query the store; the actual frames to pick
        # for the result (`nearest`) are passed via the `process_kwargs`
        request["start"] = store_time[0]
        request["stop"] = store_time[-1]
        nearest = find_nearest(store_time, index_time)
        process_kwargs = {"nearest": nearest}
        return [(process_kwargs, None), (self.store, request)]

    @staticmethod
    def process(process_kwargs, data=None):
        if process_kwargs is None:
            return data

        nearest = process_kwargs["nearest"]

        if "values" in data:
            data["values"] = data["values"][nearest]
            return data

        if "meta" in data:
            data["meta"] = [data["meta"][i] for i in nearest]
            return data


class Shift(BaseSingle):
    """
    Shift a temporal raster by some timedelta.

    A positive timedelta shifts into the future and a negative timedelta shifts
    into the past.

    Args:
      store (RasterBlock): The store whose timestamps are to be shifted
      time (integer): The timedelta to shift the store, in milliseconds.

    Returns:
      RasterBlock with its timestamps shifted.
    """

    def __init__(self, store, time):
        if isinstance(time, Timedelta):
            time = int(time.total_seconds() * 1000)
        if not isinstance(time, int):
            raise TypeError("'{}' object is not allowed".format(type(time)))
        super(Shift, self).__init__(store, time)

    @property
    def time(self):
        return Timedelta(milliseconds=self.args[1])

    @property
    def period(self):
        start, stop = self.store.period
        return start + self.time, stop + self.time

    def get_sources_and_requests(self, **request):
        # shift request
        start = request.get("start", None)
        stop = request.get("stop", None)

        if start is not None:
            request["start"] = start - self.time
        if stop is not None:
            request["stop"] = stop - self.time

        return [(self.store, request), (self.time, None)]

    @staticmethod
    def process(data, time):
        if data is None:
            return None
        # shift result if necessary
        if "time" in data:
            data["time"] = [t + time for t in data["time"]]

        return data


class TemporalSum(BaseSingle):
    @staticmethod
    def process(data):
        if data is None:
            return data

        if "time" in data:
            return {"time": data["time"][-1:]}

        if "meta" in data:
            return {"meta": data["meta"][-1:]}

        if "values" in data:
            return {
                "values": data["values"].sum(axis=0)[np.newaxis, ...],
                "no_data_value": data["no_data_value"],
            }


def _dt_to_ts(dt, timezone):
    """Convert a naive UTC python datetime to a pandas timestamp"""
    return pd.Timestamp(dt, tz="UTC").tz_convert(timezone)


def _ts_to_dt(timestamp, timezone):
    """Convert a pandas timestamp to a naive UTC python datetime"""
    try:
        timestamp = timestamp.tz_localize(timezone)
    except TypeError:
        pass
    return timestamp.tz_convert("UTC").tz_localize(None).to_pydatetime(warn=False)


def _get_bin_label(dt, frequency, closed, label, timezone):
    """Returns the label of the bin the input dt belongs to.

    :type dt: datetime.datetime without timezone.
    """
    # the strategy is leveraging pandas.Series.resample to put the dt in a bin
    # while there is only 1 sample here, there might be multiple (empty) bins
    # in some cases (see test_issue_5917)
    series = pd.Series([0], index=[_dt_to_ts(dt, timezone)])
    for label, bin in series.resample(frequency, closed=closed, label=label):
        if len(bin) != 0:
            break
    return _ts_to_dt(label, timezone)


def _get_bin_start(dt, frequency, closed, label, timezone):
    """Returns the start (left side) of the bin the input dt belongs to.

    :type dt: datetime.datetime without timezone.
    """
    # go through resample, this is the only function that supports 'closed'
    series = pd.Series([0], index=[_dt_to_ts(dt, timezone)])
    resampled = series.resample(frequency, closed=closed, label="left")
    return resampled.first().index[0]


def _get_closest_label(dt, frequency, closed, label, timezone, side="both"):
    """Get the label that is closest to dt

    Optionally only include labels to the left or to the right using `side`.
    """
    ts = _dt_to_ts(dt, timezone)
    # first get the bin label that dt belongs to
    candidate = _dt_to_ts(
        _get_bin_label(dt, frequency, closed, label, timezone), timezone
    )
    # some custom logic that finds the closest label
    freq = to_offset(frequency)
    # generate the labels around it
    candidates = pd.date_range(candidate - freq, candidate + freq, freq=freq)
    # get the candidate that is closest
    differences = (candidates - ts).to_series()
    differences.index = candidates  # so that argmin returns the candidate
    if side == "right":
        differences = differences[differences >= pd.Timedelta(0)]
    elif side == "left":
        differences = differences[differences <= pd.Timedelta(0)]
    result = differences.abs().idxmin()
    return _ts_to_dt(result, timezone)


def _default_closed_label(frequency, closed, label):
    """To make closed & label not-None"""
    if frequency is None:
        return ("right", "right")
    # Copied from pandas 'TimeGrouper.__init__':
    end_types = {"M", "A", "Q", "BM", "BA", "BQ", "W"}
    rule = to_offset(frequency).rule_code
    if rule in end_types or ("-" in rule and rule[: rule.find("-")] in end_types):
        if closed is None:
            closed = "right"
        if label is None:
            label = "right"
    else:
        if closed is None:
            closed = "left"
        if label is None:
            label = "left"
    return closed, label


def _label_to_bin_start(dt, frequency, closed, label, timezone):
    """Returns the first datetime in bin with label 'dt'"""
    ts = _dt_to_ts(dt, timezone)
    if label == "right":
        ts -= to_offset(frequency)
    if closed == "right":
        ts += MICROSECOND
    return _ts_to_dt(ts, timezone)


def _label_to_bin_end(dt, frequency, closed, label, timezone):
    """Returns the last datetime in bin with label 'dt'"""
    ts = _dt_to_ts(dt, timezone)
    if label == "left":
        ts += to_offset(frequency)
    if closed == "left":
        ts -= MICROSECOND
    return _ts_to_dt(ts, timezone)


def _resampled_period(period, frequency, closed, label, timezone):
    """Given a source (start, stop) and frequency, compute the (start, stop) interval
    that contains data after resampling. The returned start and stop are bin labels.
    """
    if period is None:
        return
    if frequency is None:
        return period[-1], period[-1]
    return tuple(_get_bin_label(x, frequency, closed, label, timezone) for x in period)


def _snap_to_resampled_labels(period, start, stop, frequency, closed, label, timezone):
    """Snap start and stop to the bin labels (so timeframes of a resampled raster).

    The result are 2 labels: a start and a stop label, as python datetimes.
    If the stop label is None, start == stop. If both are None, there is no label in range.
    """
    period = _resampled_period(period, frequency, closed, label, timezone)
    if period is None:  # return early for an empty source
        return None, None

    if start is None:  # start is None means: return the latest
        start = period[1]

    if stop is None:
        # snap start to a label closest to it
        if start <= period[0]:
            start = period[0]
        elif start >= period[1]:
            start = period[1]
        else:
            start = _get_closest_label(
                start, frequency, closed, label, timezone, side="both"
            )
    else:
        # snap start to a label right from it
        if start <= period[0]:
            start = period[0]
        elif start > period[1]:
            return None, None
        else:
            start = _get_closest_label(
                start, frequency, closed, label, timezone, side="right"
            )
        # snap stop to a label left from it
        if stop >= period[1]:
            stop = period[1]
        elif stop < period[0]:
            return None, None
        else:
            stop = _get_closest_label(
                stop, frequency, closed, label, timezone, side="left"
            )

        if start > stop:
            return None, None

    return start, stop


def _labels_to_start_stop(start_label, stop_label, frequency, closed, label, timezone):
    """Given a start and stop label, get the start/stop to request from source"""
    assert frequency is not None
    start = _label_to_bin_start(start_label, frequency, closed, label, timezone)
    stop = _label_to_bin_end(
        stop_label or start_label, frequency, closed, label, timezone
    )
    return start, stop


def count_not_nan(x, *args, **kwargs):
    return np.sum(~np.isnan(x), *args, **kwargs)


class TemporalAggregate(BaseSingle):
    """
    Resample a raster in time.

    This operation performs temporal aggregation of rasters, for example a
    hourly average of data that has a 5 minute resolution.. The timedelta of
    the resulting raster is determined by the 'frequency' parameter.

    Args:
      source (RasterBlock): The input raster whose timesteps are aggregated
      frequency (string or None): The frequency to resample to, as pandas
        offset string (see the references below). If this value is None, this
        block will return the temporal statistic over the complete time range,
        with output timestamp at the end of the source raster period.
        Defaults to None.
      statistic (string): The type of statistic to perform. Can be one of
        ``{"sum", "count", "min", "max", "mean", "median", "std", "var", "p<percentile>"}``.
        Defaults to ``"sum"``.
      closed (string or None): Determines what side of the interval is closed.
        Can be ``"left"`` or ``"right"``. The default depends on the frequency.
      label (string or None): Determines what side of the interval is closed.
        Can be ``"left"`` or ``"right"``. The default depends on the frequency.
      timezone (string): Timezone to perform the resampling in, defaults to
        ``"UTC"``.

    Returns:
      RasterBlock with temporally aggregated data.

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.resample.html
      https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    """

    # extensive (opposite: intensive) means: additive, proportional to size
    STATISTICS = {
        "sum": {"func": np.nansum, "extensive": True},
        "count": {"func": count_not_nan, "extensive": True},
        "min": {"func": np.nanmin, "extensive": False},
        "max": {"func": np.nanmax, "extensive": False},
        "mean": {"func": np.nanmean, "extensive": False},
        "median": {"func": np.nanmedian, "extensive": False},
        "std": {"func": np.nanstd, "extensive": False},
        "var": {"func": np.nanvar, "extensive": False},
        # 'percentile' is hardcoded to np.nanpercentile
    }

    def __init__(
        self,
        source,
        frequency,
        statistic="sum",
        closed=None,
        label=None,
        timezone="UTC",
    ):
        if not isinstance(source, RasterBlock):
            raise TypeError("'{}' object is not allowed.".format(type(source)))
        if frequency is not None:
            if not isinstance(frequency, str):
                raise TypeError("'{}' object is not allowed.".format(type(frequency)))
            frequency = to_offset(frequency).freqstr

            if closed not in {None, "left", "right"}:
                raise ValueError("closed must be None, 'left', or 'right'.")
            if label not in {None, "left", "right"}:
                raise ValueError("label must be None, 'left', or 'right'.")
            if not isinstance(timezone, str):
                raise TypeError("'{}' object is not allowed.".format(type(timezone)))
            timezone = pytz.timezone(timezone).zone
        else:
            closed = None
            label = None
            timezone = None
        if not isinstance(statistic, str):
            raise TypeError("'{}' object is not allowed.".format(type(statistic)))
        # interpret percentile statistic
        statistic, percentile = parse_percentile_statistic(statistic.lower())
        if percentile:
            statistic = "p{0}".format(percentile)
        elif statistic not in self.STATISTICS:
            raise ValueError("Unknown statistic '{}'".format(statistic))
        super(TemporalAggregate, self).__init__(
            source, frequency, statistic, closed, label, timezone
        )

    @property
    def source(self):
        return self.args[0]

    @property
    def frequency(self):
        return self.args[1]

    @property
    def statistic(self):
        return self.args[2]

    @property
    def closed(self):
        return self.args[3]

    @property
    def label(self):
        return self.args[4]

    @property
    def timezone(self):
        return self.args[5]

    @property
    def _snap_kwargs(self):
        # make closed & label not-None
        closed, label = _default_closed_label(self.frequency, self.closed, self.label)
        return {
            "frequency": self.frequency,
            "closed": closed,
            "label": label,
            "timezone": self.timezone,
        }

    @property
    def period(self):
        return _resampled_period(self.source.period, **self._snap_kwargs)

    @property
    def timedelta(self):
        if self.frequency is None:
            return None
        try:
            return pd.Timedelta(to_offset(self.frequency)).to_pytimedelta()
        except ValueError:
            return  # e.g. Month is non-equidistant

    @property
    def temporal(self):
        return self.frequency is not None

    @property
    def dtype(self):
        return dtype_for_statistic(self.source.dtype, self.statistic)

    @property
    def fillvalue(self):
        return get_dtype_max(self.dtype)

    def get_sources_and_requests(self, **request):
        kwargs = self._snap_kwargs
        start = request.get("start")
        stop = request.get("stop")
        mode = request["mode"]

        start_label, stop_label = _snap_to_resampled_labels(
            self.source.period, start, stop, **kwargs
        )
        if start_label is None:
            return [({"empty": True, "mode": mode}, None)]

        # a time request does not involve a request to self.source
        if mode == "time":
            kwargs["mode"] = "time"
            kwargs["start"] = start_label
            kwargs["stop"] = stop_label
            return [(kwargs, None)]

        # vals or source requests do need a request to self.source
        if self.frequency is None:
            request["start"], request["stop"] = self.source.period
        else:
            request["start"], request["stop"] = _labels_to_start_stop(
                start_label, stop_label, **kwargs
            )

        # return sources and requests depending on the mode
        kwargs["mode"] = request["mode"]
        kwargs["start"] = start_label
        kwargs["stop"] = stop_label
        if mode == "vals":
            kwargs["dtype"] = np.dtype(self.dtype).str
            kwargs["statistic"] = self.statistic

        time_request = {
            "mode": "time",
            "start": request["start"],
            "stop": request["stop"],
        }

        # In case the data request dictates a temporal resolution,
        # also set this in temporal request
        if "time_resolution" in request:
            time_request["time_resolution"] = request["time_resolution"]

        return [(kwargs, None), (self.source, time_request), (self.source, request)]

    @staticmethod
    def process(process_kwargs, time_data=None, data=None):
        mode = process_kwargs["mode"]
        # handle empty data
        if process_kwargs.get("empty"):
            return None if mode == "vals" else {mode: []}
        start = process_kwargs["start"]
        stop = process_kwargs["stop"]
        frequency = process_kwargs["frequency"]
        if frequency is None:
            labels = pd.DatetimeIndex([start])
        else:
            labels = pd.date_range(start, stop or start, freq=frequency)
        if mode == "time":
            return {"time": labels.to_pydatetime().tolist()}

        if time_data is None or not time_data.get("time"):
            return None if mode == "vals" else {mode: []}

        timezone = process_kwargs["timezone"]
        closed = process_kwargs["closed"]
        label = process_kwargs["label"]
        times = time_data["time"]

        # convert times to a pandas series
        series = (
            pd.Series(index=times, dtype=float).tz_localize("UTC").tz_convert(timezone)
        )

        # localize the labels so we can use it as an index
        labels = labels.tz_localize("UTC").tz_convert(timezone)

        if frequency is None:
            # the first (and only label) will be the statistic of all frames
            indices = {labels[0]: range(len(times))}
        else:
            # construct a pandas Resampler object to map labels to frames
            resampler = series.resample(frequency, closed=closed, label=label)
            # get the frame indices belonging to each bin
            indices = resampler.indices

        if mode == "meta":
            if data is None or "meta" not in data:
                return {"meta": []}
            meta = data["meta"]
            return {"meta": [[meta[i] for i in indices[ts]] for ts in labels]}

        # mode == 'vals'
        if data is None or "values" not in data:
            return

        values = data["values"]
        if values.shape[0] != len(times):
            raise RuntimeError("Shape of raster does not match number of timestamps")
        statistic, percentile = parse_percentile_statistic(process_kwargs["statistic"])
        if percentile:
            extensive = False
            agg_func = partial(np.nanpercentile, q=percentile)
        else:
            extensive = TemporalAggregate.STATISTICS[statistic]["extensive"]
            agg_func = TemporalAggregate.STATISTICS[statistic]["func"]

        dtype = process_kwargs["dtype"]
        fillvalue = 0 if extensive else get_dtype_max(dtype)

        # cast to at least float32 so that we can fit in NaN (and make copy)
        values = values.astype(np.result_type(np.float32, dtype))
        # put NaN for no data
        values[data["values"] == data["no_data_value"]] = np.nan

        result = np.full(
            shape=(len(labels), values.shape[1], values.shape[2]),
            fill_value=fillvalue,
            dtype=dtype,
        )

        for i, timestamp in enumerate(labels):
            inds = indices[timestamp]
            if len(inds) == 0:
                continue
            with warnings.catch_warnings():
                # the agg_func could give use 'All-NaN slice encountered'
                warnings.simplefilter("ignore", category=RuntimeWarning)
                aggregated = agg_func(values[inds], axis=0)
            # keep track of NaN or inf values before casting to target dtype
            no_data_mask = ~np.isfinite(aggregated)
            # cast to target dtype
            if dtype != aggregated.dtype:
                aggregated = aggregated.astype(dtype)
            # set fillvalue to NaN values
            aggregated[no_data_mask] = fillvalue
            result[i] = aggregated

        return {"values": result, "no_data_value": get_dtype_max(dtype)}


def accumulate_count_not_nan(x, *args, **kwargs):
    return np.cumsum(~np.isnan(x), *args, **kwargs)


class Cumulative(BaseSingle):
    """
    Compute the cumulative of a raster over time.

    Contrary to ``dask_geomodeling.raster.temporal.TemporalAggregate``, in this
    operation the timedelta of the resulting raster equals the timedelta of the
    input raster. Cell values are accumulated over the supplied period. At the
    end of each period the accumulation is reset.

    Args:
      source (RasterBlock): The input raster whose timesteps are accumulated.
      statistic (string): The type of accumulation to perform. Can be ``"sum"``
        or ``"count"``. Defaults to ``"sum"``.
      frequency (string or None): The period over which accumulation is
        performed. Supply a pandas offset string (see the references below). If
        this value is None, the accumulation will continue indefinitely.
        Defaults to None.
      timezone (string): Timezone in which the accumulation is performed,
        defaults to ``"UTC"``.

    Returns:
      RasterBlock with temporally accumulated data.

    See also:
      https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    """

    # extensive (opposite: intensive) means: additive, proportional to size
    STATISTICS = {
        "sum": {"func": np.nancumsum, "extensive": True},
        "count": {"func": accumulate_count_not_nan, "extensive": True},
    }

    def __init__(self, source, statistic="sum", frequency=None, timezone="UTC"):
        if not isinstance(source, RasterBlock):
            raise TypeError("'{}' object is not allowed.".format(type(source)))
        if not isinstance(statistic, str):
            raise TypeError("'{}' object is not allowed.".format(type(statistic)))
        # interpret percentile statistic
        statistic, percentile = parse_percentile_statistic(statistic.lower())
        if percentile:
            statistic = "p{0}".format(percentile)
        elif statistic not in self.STATISTICS:
            raise ValueError("Unknown statistic '{}'".format(statistic))
        if frequency is not None:
            if not isinstance(frequency, str):
                raise TypeError("'{}' object is not allowed.".format(type(frequency)))
            frequency = to_offset(frequency).freqstr
            if not isinstance(timezone, str):
                raise TypeError("'{}' object is not allowed.".format(type(timezone)))
            timezone = pytz.timezone(timezone).zone
        else:
            timezone = None
        super().__init__(source, statistic, frequency, timezone)

    @property
    def source(self):
        return self.args[0]

    @property
    def statistic(self):
        return self.args[1]

    @property
    def frequency(self):
        return self.args[2]

    @property
    def timezone(self):
        return self.args[3]

    @property
    def _snap_kwargs(self):
        return {
            "frequency": self.frequency,
            "closed": "right",
            "label": "right",
            "timezone": self.timezone,
        }

    @property
    def dtype(self):
        return dtype_for_statistic(self.source.dtype, self.statistic)

    @property
    def fillvalue(self):
        return get_dtype_max(self.dtype)

    def get_sources_and_requests(self, **request):
        # a time request does not involve any resampling, so just propagate
        if request["mode"] == "time":
            return [({"mode": "time"}, None), (self.source, request)]

        kwargs = self._snap_kwargs
        start = request.get("start")
        stop = request.get("stop")
        mode = request["mode"]

        # we need to now what times will be returned in order to figure out
        # what times we need to compute the cumulative
        time_data = self.source.get_data(mode="time", start=start, stop=stop)
        if time_data is None or not time_data.get("time"):
            # return early for an empty source
            return [({"empty": True, "mode": mode}, None)]

        # get the periods from the first and last timestamp
        start = time_data["time"][0]
        stop = time_data["time"][-1]

        if self.frequency is None:
            request["start"] = self.period[0]
            request["stop"] = stop
        else:
            # snap request 'start' to the start of the first period
            request["start"] = _ts_to_dt(_get_bin_start(start, **kwargs), self.timezone)
            # snap request 'stop' to the last requested time
            request["stop"] = stop
            if kwargs["closed"] != "left":
                request["stop"] += Timedelta(microseconds=1)

        # return sources and requests depending on the mode
        kwargs["mode"] = request["mode"]
        kwargs["start"] = start
        kwargs["stop"] = stop
        if mode == "vals":
            kwargs["dtype"] = np.dtype(self.dtype).str
            kwargs["statistic"] = self.statistic

        time_request = {
            "mode": "time",
            "start": request["start"],
            "stop": request["stop"],
        }
        return [(kwargs, None), (self.source, time_request), (self.source, request)]

    @staticmethod
    def process(process_kwargs, time_data=None, data=None):
        mode = process_kwargs["mode"]
        # handle empty data
        if process_kwargs.get("empty"):
            return None if mode == "vals" else {mode: []}
        if mode == "time":
            return time_data
        if time_data is None or not time_data.get("time"):
            return None if mode == "vals" else {mode: []}

        start = process_kwargs["start"]
        stop = process_kwargs["stop"]
        frequency = process_kwargs["frequency"]
        timezone = process_kwargs["timezone"]
        closed = process_kwargs["closed"]
        label = process_kwargs["label"]
        times = (
            pd.Series(index=time_data["time"], dtype=float)
            .tz_localize("UTC")
            .tz_convert(timezone)
        )

        if frequency is None:
            # the first (and only label) will be the statistic of all frames
            indices = {None: range(len(times))}
        else:
            # construct a pandas Resampler object to map labels to frames
            resampler = times.resample(frequency, closed=closed, label=label)
            # get the frame indices belonging to each bin
            indices = resampler.indices

        start_ts = _dt_to_ts(start, timezone)
        stop_ts = _dt_to_ts(stop, timezone)

        if mode == "meta":
            if data is None or "meta" not in data:
                return {"meta": []}
            meta = data["meta"]
            result = []
            for indices_in_bin in indices.values():  # [0, 1], [2, 3], ...
                for length in range(1, len(indices_in_bin) + 1):
                    indices_for_cumulative = indices_in_bin[:length]
                    ts = times.index[indices_for_cumulative[-1]]
                    if ts < start_ts or (stop_ts is not None and ts > stop_ts):
                        continue
                    result.append([meta[i] for i in indices_for_cumulative])
            return {"meta": result}

        # mode == 'vals'
        if data is None or "values" not in data:
            return

        values = data["values"]
        if values.shape[0] != len(times):
            raise RuntimeError("Shape of raster does not match number of timestamps")
        statistic, percentile = parse_percentile_statistic(process_kwargs["statistic"])
        if percentile:
            extensive = False
            agg_func = partial(np.nanpercentile, q=percentile)
        else:
            extensive = Cumulative.STATISTICS[statistic]["extensive"]
            agg_func = Cumulative.STATISTICS[statistic]["func"]

        dtype = process_kwargs["dtype"]
        fillvalue = 0 if extensive else get_dtype_max(dtype)

        # cast to at least float32 so that we can fit in NaN (and make copy)
        values = values.astype(np.result_type(np.float32, dtype))
        # put NaN for no data
        values[data["values"] == data["no_data_value"]] = np.nan

        output_mask = (times.index >= start_ts) & (times.index <= stop_ts)
        output_offset = np.where(output_mask)[0][0]
        n_frames = output_mask.sum()
        result = np.full(
            shape=(n_frames, values.shape[1], values.shape[2]),
            fill_value=fillvalue,
            dtype=dtype,
        )

        for indices_in_bin in indices.values():
            mask = output_mask[indices_in_bin]
            data = values[indices_in_bin]
            accumulated = agg_func(data, axis=0)[mask]
            # keep track of NaN or inf values before casting to target dtype
            no_data_mask = ~np.isfinite(accumulated)
            # cast to target dtype
            if dtype != accumulated.dtype:
                accumulated = accumulated.astype(dtype)
            # set fillvalue to NaN values
            accumulated[no_data_mask] = fillvalue
            indices_in_result = np.array(indices_in_bin)[mask] - output_offset
            result[indices_in_result] = accumulated

        return {"values": result, "no_data_value": get_dtype_max(dtype)}
