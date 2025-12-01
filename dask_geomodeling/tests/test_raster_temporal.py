from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
import unittest
import numpy as np
from numpy.testing import assert_equal

from dask_geomodeling.raster import TemporalAggregate, Cumulative, Resample
from dask_geomodeling.raster.temporal import (
    _shift_datetime,
    _snap_to_resampled_labels,
    _labels_to_start_stop,
    _get_closest_label,
)
from dask_geomodeling.tests.factories import MockRaster

import pytest


@pytest.fixture
def raster():
    return MockRaster(
        origin=Datetime(2000, 1, 1),
        value=np.array([[1.0, 0.0, np.nan]]),
        timedelta=Timedelta(days=1),
        bands=3,
    )


dt = Datetime
td = Timedelta


@pytest.mark.parametrize("freq,closed,label,timezone,expected", [
    ("D", "left", "left", "UTC", (dt(2000, 1, 1), dt(2000, 1, 3))),
    ("D", "left", "right", "UTC", (dt(2000, 1, 2), dt(2000, 1, 4))),
    ("D", "right", "left", "UTC", (dt(1999, 12, 31), dt(2000, 1, 2))),
    ("D", "right", "right", "UTC", (dt(2000, 1, 1), dt(2000, 1, 3))),
    # 2000-01-01 00:00 UTC is 2000-01-01 01:00 in Amsterdam
    # 2000-01-01 01:00 falls in the 2000-01-01 bin (still Amsterdam)
    # the 2000-01-01 bin corresponds to 1999-12-31 23:00 UTC
    ("D", "left", "left", "Europe/Amsterdam", (dt(1999, 12, 31, 23), dt(2000, 1, 2, 23))),
    # 2000-01-01 00:00 UTC is 1999-12-31 19:00 in New York
    # 1999-12-31 19:00 falls in the 1999-12-31 bin (still New York)
    # the 1999-12-31 bin corresponds to 1999-12-31 5:00 UTC
    ("D", "left", "left", "America/New_York", (dt(1999, 12, 31, 5), dt(2000, 1, 2, 5))),
    ("h", "left", "left", "UTC", (dt(2000, 1, 1, 0), dt(2000, 1, 3, 0))),
    ("h", "left", "right", "UTC", (dt(2000, 1, 1, 1), dt(2000, 1, 3, 1))),
    ("h", "right", "left", "UTC", (dt(1999, 12, 31, 23), dt(2000, 1, 2, 23))),
    ("h", "right", "right", "UTC", (dt(2000, 1, 1), dt(2000, 1, 3))),
    # 2000-01-01 00:00 UTC is 2000-01-01 01:00 in Amsterdam
    # the 2000-01-01 01:00 bin corresponds to 2000-01-01 00:00 UTC
    # you don't notice the timezone here
    ("h", "left", "left", "Europe/Amsterdam", (dt(2000, 1, 1), dt(2000, 1, 3))),
    ("h", "left", "left", "America/New_York", (dt(2000, 1, 1), dt(2000, 1, 3))),
    (None, "left", "left", "UTC", (dt(2000, 1, 3), dt(2000, 1, 3))),
    # pandas MonthEnd ("M") bin is 1999-12-31 00:00 UTC to 2000-01-31 00:00 UTC
    ("M", "left", "left", "UTC", (dt(1999, 12, 31), dt(1999, 12, 31))),
    ("M", "left", "right", "UTC", (dt(2000, 1, 31), dt(2000, 1, 31))),
    ("M", "right", "left", "UTC", (dt(1999, 12, 31), dt(1999, 12, 31))),
    ("M", "right", "right", "UTC", (dt(2000, 1, 31), dt(2000, 1, 31))),
    ("M", None, None, "UTC", (dt(2000, 1, 31), dt(2000, 1, 31))),  # (right, right)
    # pandas MonthStart ("MS") bin is 2000-01-01 00:00 UTC to 2000-02-01 00:00 UTC
    ("MS", "left", "left", "UTC", (dt(2000, 1, 1), dt(2000, 1, 1))),
    ("MS", "left", "right", "UTC", (dt(2000, 2, 1), dt(2000, 2, 1))),
    ("MS", "right", "left", "UTC", (dt(1999, 12, 1), dt(2000, 1, 1))),
    ("MS", "right", "right", "UTC", (dt(2000, 1, 1), dt(2000, 2, 1))),
    ("MS", None, None, "UTC", (dt(2000, 1, 1), dt(2000, 1, 1))),  # (left, left)
    # businessday (our sample dataset is Saturday, Sunday, Monday); Sat + Sun belong to Friday
    ("B", "left", "left", "UTC", (dt(1999, 12, 31), dt(2000, 1, 3))),
    ("B", "left", "right", "UTC", (dt(2000, 1, 3), dt(2000, 1, 4))),
    # closed=right moves "monday" into the weekend because it is on 00:00
    ("B", "right", "left", "UTC", (dt(1999, 12, 31), dt(1999, 12, 31))),
    ("B", "right", "right", "UTC", (dt(2000, 1, 3), dt(2000, 1, 3))),
    # Aliases deprecated since pandas 2.2 still work
    ("H", "left", "left", "UTC", (dt(2000, 1, 1, 0), dt(2000, 1, 3, 0))),
    ("M", "left", "left", "UTC", (dt(1999, 12, 31), dt(1999, 12, 31))),
])
def test_period(raster, freq, closed, label, timezone, expected):
    view = TemporalAggregate(raster, freq, closed=closed, label=label, timezone=timezone)
    actual = view.period

    assert actual == expected


@pytest.mark.parametrize(
    "start,stop,freq,timezone,expected",
    [
        # (None, None) means 'latest'; expected is period[1] always
        (None, None, "X", "Y", (dt(2000, 2, 1), None)),
        # (start, None) means 'nearest'; expected are the labels of the nearest bins
        # left out-of-bounds
        (dt(1999, 5, 6), None, "MS", "UTC", (dt(2000, 1, 1), None)),
        # right out-of-bounds (equals 'latest')
        (dt(2001, 5, 6), None, "MS", "UTC", (dt(2000, 2, 1), None)),
        # in bounds, snap to nearest (in a situation with 2 bins: 2000-01-01 and 2000-02-01)
        (dt(2000, 1, 1), None, "MS", "UTC", (dt(2000, 1, 1), None)),
        (dt(2000, 1, 16), None, "MS", "UTC", (dt(2000, 1, 1), None)),
        (dt(2000, 1, 17), None, "MS", "UTC", (dt(2000, 2, 1), None)),
        (dt(2000, 2, 1), None, "MS", "UTC", (dt(2000, 2, 1), None)),
        # (start, stop) means a two-sided closed interval
        (dt(2000, 1, 1), dt(2000, 2, 1), "MS", "UTC", (dt(2000, 1, 1), dt(2000, 2, 1))),
        (dt(1999, 5, 6), dt(2001, 5, 6), "MS", "UTC", (dt(2000, 1, 1), dt(2000, 2, 1))),
        (
            dt(2000, 1, 1),
            dt(2000, 1, 31),
            "MS",
            "UTC",
            (dt(2000, 1, 1), dt(2000, 1, 1)),
        ),
        (dt(2000, 1, 2), dt(2000, 2, 1), "MS", "UTC", (dt(2000, 2, 1), dt(2000, 2, 1))),
        (dt(2000, 1, 2), dt(2000, 1, 31), "MS", "UTC", (None, None)),  # no frames
    ],
)
def test_snap_to_resampled_labels(start, stop, freq, timezone, expected):
    actual = _snap_to_resampled_labels(
        (dt(2000, 1, 1), dt(2000, 2, 1)), start, stop, freq, timezone
    )
    assert actual == expected


def test_snap_to_resampled_labels_none():
    actual = _snap_to_resampled_labels(None, dt(2000, 1, 1), dt(2000, 2, 1), "X", "Y")
    assert actual == (None, None)


@pytest.mark.parametrize(
    "dt_input,freq,timezone,side,expected",
    [
        # 'side' parameter
        (dt(2020, 1, 1, 12), "D", "UTC", "both", dt(2020, 1, 1)),
        (dt(2020, 1, 1, 12, microsecond=1), "D", "UTC", "both", dt(2020, 1, 2)),
        (dt(2020, 1, 1, 12), "D", "UTC", "left", dt(2020, 1, 1)),
        (dt(2020, 1, 1, 12), "D", "UTC", "right", dt(2020, 1, 2)),
        # businessday: 2000-1-3 is a Monday (and Fri-Sun is 1 bin)
        # 2000-1-3 00:00 (Monday) snaps to itself
        (dt(2000, 1, 3), "B", "UTC", "both", dt(2000, 1, 3)),
        # 2000-1-2 (Sunday) snaps forward to Monday
        (dt(2000, 1, 2), "B", "UTC", "both", dt(2000, 1, 3)),
        # 2000-1-1 (Saturday) snaps backward to Friday
        (dt(2000, 1, 1), "B", "UTC", "both", dt(1999, 12, 31)),
        # 1999-12-31 (Friday) snaps to itself
        (dt(1999, 12, 31), "B", "UTC", "both", dt(1999, 12, 31)),
    ],
)
def test_get_closest_label(dt_input, freq, timezone, side, expected):
    actual = _get_closest_label(dt_input, freq, timezone, side=side)
    assert actual == expected


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, dt(2000, 1, 1)),
        (1, dt(2000, 1, 1, 1)),
        (2, dt(2000, 1, 1, 2)),
        (-1, dt(1999, 12, 31, 23)),
        (0.5, dt(2000, 1, 1, 0, 30)),
        (-0.5, dt(1999, 12, 31, 23, 30)),
    ],
)
def test_shift_datetime(n, expected):
    assert _shift_datetime(dt(2000, 1, 1), "h", "UTC", n) == expected




@pytest.mark.parametrize(
    "n,expected",
    [
        (0, dt(2000, 2, 1)),
        (1, dt(2000, 3, 1)),
        (-1, dt(2000, 1, 1)),
        (0.5, dt(2000, 2, 15, 12)),  # + 14.5 days
        (-0.5, dt(2000, 1, 16, 12)),  # - 15.5 days
    ],
)
def test_shift_datetime_irregular_bins(n, expected):
    assert _shift_datetime(dt(2000, 2, 1), "MS", "UTC", n) == expected



us = Timedelta(microseconds=1)


@pytest.mark.parametrize("start_label,stop_label,freq,closed,label,timezone,expected", [
    (dt(2000, 1, 1), None, "D", "left", "left", "UTC", (dt(2000, 1, 1), dt(2000, 1, 2) - us)),
    (dt(2000, 1, 1), None, "D", "left", "right", "UTC", (dt(1999, 12, 31), dt(2000, 1, 1) - us)),
    (dt(2000, 1, 1), None, "D", "right", "left", "UTC", (dt(2000, 1, 1) + us, dt(2000, 1, 2))),
    (dt(2000, 1, 1), None, "D", "right", "right", "UTC", (dt(1999, 12, 31) + us, dt(2000, 1, 1))),
    (dt(2000, 1, 1), None, "MS", "left", "left", "UTC", (dt(2000, 1, 1), dt(2000, 2, 1) - us)),
    (dt(2000, 1, 1), None, "MS", "left", "right", "UTC", (dt(1999, 12, 1), dt(2000, 1, 1) - us)),
    (dt(2000, 1, 1), None, "MS", "right", "left", "UTC", (dt(2000, 1, 1) + us, dt(2000, 2, 1))),
    (dt(2000, 1, 1), None, "MS", "right", "right", "UTC", (dt(1999, 12, 1) + us, dt(2000, 1, 1))),
    # with a 'stop_label' it is just more of the same ...
    (dt(2000, 1, 1), dt(2000, 1, 10), "D", "left", "left", "UTC", (dt(2000, 1, 1), dt(2000, 1, 11) - us)),
    (dt(2000, 1, 1), dt(2000, 10, 1), "MS", "left", "left", "UTC", (dt(2000, 1, 1), dt(2000, 11, 1) - us)),
    # businessday: 2000-1-3 is a Monday (and Fri-Sun is 1 bin)
    (dt(2000, 1, 3), None, "B", "left", "left", "UTC", (dt(2000, 1, 3), dt(2000, 1, 4) - us)),
    (dt(2000, 1, 3), None, "B", "left", "right", "UTC", (dt(1999, 12, 31), dt(2000, 1, 3) - us)),
    (dt(2000, 1, 3), None, "B", "right", "left", "UTC", (dt(2000, 1, 3) + us, dt(2000, 1, 4))),
    (dt(2000, 1, 3), None, "B", "right", "right", "UTC", (dt(1999, 12, 31) + us, dt(2000, 1, 3))),
])
def test_labels_to_start_stop(start_label, stop_label, freq, closed, label, timezone, expected):
    actual = _labels_to_start_stop(start_label, stop_label, freq, closed, label, timezone)
    assert actual == expected



class TestTemporalAggregate(unittest.TestCase):
    klass = TemporalAggregate

    def setUp(self):
        self.raster = MockRaster(
            origin=Datetime(2000, 1, 1),
            value=np.array([[1.0, 0.0, np.nan]]),
            timedelta=Timedelta(days=1),
            bands=3,
        )
        self.raster_uint8 = MockRaster(
            origin=Datetime(2000, 1, 1), value=7, timedelta=Timedelta(days=1), bands=3
        )
        self.request = {
            "mode": "vals",
            "bbox": (0, 0, 3, 1),
            "width": 3,
            "height": 1,
            "projection": self.raster.projection,
        }
        self.request_all = {
            "start": Datetime(1970, 1, 1),
            "stop": Datetime(2020, 1, 1),
            **self.request,
        }
        self.request_empty = {
            "start": Datetime(1970, 1, 1),
            "stop": Datetime(1971, 1, 1),
            **self.request,
        }

    def test_period_none(self):
        view = self.klass(self.raster, frequency=None, statistic="sum")

        # test timedelta
        self.assertIsNone(view.timedelta)

        # get time
        self.request["mode"] = "time"
        result = view.get_data(**self.request)["time"]
        self.assertEqual([Datetime(2000, 1, 3)], result)

        # get data
        result = view.get_data(**self.request_all)["values"]
        assert_equal(result, [[[3.0, 0.0, 0.0]]])

    def test_timedelta(self):
        self.assertEqual(Timedelta(seconds=1), self.klass(self.raster, "S").timedelta)
        self.assertEqual(Timedelta(hours=1), self.klass(self.raster, "H").timedelta)
        # months are nonequidistant
        self.assertIsNone(self.klass(self.raster, "M").timedelta)

    def test_temporal(self):
        self.assertTrue(self.klass(self.raster, "D").temporal)  # equidistant
        self.assertTrue(self.klass(self.raster, "M").temporal)  # non-equidistant
        self.assertFalse(self.klass(self.raster, None).temporal)  # non-temporal

    def test_get_data_time_request(self):
        self.view = self.klass(self.raster, "H", closed="left", label="right")
        self.request["mode"] = "time"

        # no start and stop produces the last element
        result = self.view.get_data(**self.request)["time"]
        self.assertEqual([Datetime(2000, 1, 3, 1)], result)

        # only start produces the closest bin (which could be empty)
        result = self.view.get_data(start=Datetime(1980, 1, 1), **self.request)["time"]
        self.assertEqual([Datetime(2000, 1, 1, 1)], result)
        result = self.view.get_data(start=Datetime(2030, 1, 1), **self.request)["time"]
        self.assertEqual([Datetime(2000, 1, 3, 1)], result)

        result = self.view.get_data(start=Datetime(2000, 1, 1, 1), **self.request)[
            "time"
        ]
        self.assertEqual([Datetime(2000, 1, 1, 1)], result)
        result = self.view.get_data(start=Datetime(2000, 1, 1, 1, 29), **self.request)[
            "time"
        ]
        self.assertEqual([Datetime(2000, 1, 1, 1)], result)
        result = self.view.get_data(start=Datetime(2000, 1, 1, 1, 31), **self.request)[
            "time"
        ]
        self.assertEqual([Datetime(2000, 1, 1, 2)], result)

        # start and stop produce all bins that fall in the (closed) interval
        result = self.view.get_data(
            start=Datetime(1980, 1, 1), stop=Datetime(2000, 1, 1, 2), **self.request
        )["time"]
        self.assertEqual([Datetime(2000, 1, 1, 1), Datetime(2000, 1, 1, 2)], result)
        result = self.view.get_data(
            start=Datetime(2000, 1, 3), stop=Datetime(2020, 1, 1), **self.request
        )["time"]
        self.assertEqual([Datetime(2000, 1, 3), Datetime(2000, 1, 3, 1)], result)
        result = self.view.get_data(
            start=Datetime(2000, 1, 2, 10),
            stop=Datetime(2000, 1, 2, 11),
            **self.request,
        )["time"]
        self.assertEqual([Datetime(2000, 1, 2, 10), Datetime(2000, 1, 2, 11)], result)

    def test_get_data_meta_request(self):
        # first two frames fall into a different frame than the last one
        self.view = self.klass(self.raster, "W")
        self.request["mode"] = "meta"

        # only last
        result = self.view.get_data(**self.request)["meta"]
        self.assertEqual([["Testmeta for band 2"]], result)

        # only first
        result = self.view.get_data(start=Datetime(1970, 1, 1), **self.request)["meta"]
        self.assertEqual([["Testmeta for band 0", "Testmeta for band 1"]], result)

        # complete range
        result = self.view.get_data(
            start=Datetime(1970, 1, 1), stop=Datetime(2020, 1, 1), **self.request
        )["meta"]
        self.assertEqual(
            [["Testmeta for band 0", "Testmeta for band 1"], ["Testmeta for band 2"]],
            result,
        )

        # only last
        result = self.view.get_data(
            start=Datetime(2000, 1, 3), stop=Datetime(2020, 1, 4), **self.request
        )["meta"]
        self.assertEqual([["Testmeta for band 2"]], result)

    def test_get_data_meta_day_with_timezone(self):
        self.request["mode"] = "meta"
        view = self.klass(
            self.raster, "D", statistic="sum", timezone="Europe/Amsterdam"
        )
        result = view.get_data(
            start=Datetime(1999, 12, 31, 23),
            stop=Datetime(1999, 12, 31, 23),
            **self.request,
        )
        self.assertListEqual(result["meta"], [["Testmeta for band 0"]])
        result = view.get_data(
            start=Datetime(2000, 1, 1, 0), stop=Datetime(2000, 1, 1, 0), **self.request
        )
        self.assertListEqual(result["meta"], [])

    def test_get_data_sum_hour(self):
        view = self.klass(self.raster, "H", statistic="sum")
        result = view.get_data(
            start=Datetime(2000, 1, 1, 0), stop=Datetime(2000, 1, 1, 1), **self.request
        )
        assert_equal(result["values"], [[[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])

    def test_get_data_sum_day_with_timezone(self):
        view = self.klass(
            self.raster, "D", statistic="sum", timezone="Europe/Amsterdam"
        )
        result = view.get_data(
            start=Datetime(1999, 12, 31, 23),
            stop=Datetime(1999, 12, 31, 23),
            **self.request,
        )
        assert_equal(result["values"], [[[1.0, 0.0, 0.0]]])
        result = view.get_data(
            start=Datetime(2000, 1, 1, 0), stop=Datetime(2000, 1, 1, 0), **self.request
        )
        self.assertIsNone(result)

    def test_get_data_sum_week(self):
        view = self.klass(self.raster, "W", statistic="sum")
        result = view.get_data(**self.request_all)
        assert_equal(result["values"], [[[2.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])

    def test_get_data_sum(self):
        view = self.klass(self.raster, "M", statistic="sum")
        result = view.get_data(**self.request_all)
        assert_equal(result["values"], [[[3.0, 0.0, 0.0]]])

    def test_get_data_count(self):
        view = self.klass(self.raster, "M", statistic="count")
        result = view.get_data(**self.request_all)
        assert_equal(result["values"], [[[3, 3, 0]]])

    def test_get_data_min(self):
        view = self.klass(self.raster, "M", statistic="min")
        result = view.get_data(**self.request_all)
        assert_equal(result["values"], [[[1.0, 0.0, result["no_data_value"]]]])

    def test_get_data_max(self):
        view = self.klass(self.raster, "M", statistic="max")
        result = view.get_data(**self.request_all)
        assert_equal(result["values"], [[[1.0, 0.0, result["no_data_value"]]]])

    def test_get_data_mean(self):
        view = self.klass(self.raster, "M", statistic="mean")
        result = view.get_data(**self.request_all)
        assert_equal(result["values"], [[[1.0, 0.0, result["no_data_value"]]]])

    def test_get_data_median(self):
        view = self.klass(self.raster, "M", statistic="median")
        result = view.get_data(**self.request_all)
        assert_equal(result["values"], [[[1.0, 0.0, result["no_data_value"]]]])

    def test_get_data_std(self):
        view = self.klass(self.raster, "M", statistic="std")
        result = view.get_data(**self.request_all)
        assert_equal(result["values"], [[[0.0, 0.0, result["no_data_value"]]]])

    def test_get_data_var(self):
        view = self.klass(self.raster, "M", statistic="var")
        result = view.get_data(**self.request_all)
        assert_equal(result["values"], [[[0.0, 0.0, result["no_data_value"]]]])

    def test_get_data_percentile(self):
        view = self.klass(self.raster, "M", statistic="p95")
        result = view.get_data(**self.request_all)
        assert_equal(result["values"], [[[1.0, 0.0, result["no_data_value"]]]])

    def test_count_dtype(self):
        # count always becomes np.int32
        view = self.klass(self.raster, "M", statistic="count")
        self.assertEqual(view.dtype, np.int32)

    def test_min_max_dtype(self):
        # min and max propagate dtype
        view = self.klass(self.raster_uint8, "M", statistic="min")
        self.assertEqual(view.dtype, np.uint8)

    def test_sum_dtype(self):
        # sum upcasts
        view = self.klass(self.raster_uint8, "M", statistic="sum")
        self.assertEqual(view.dtype, np.int32)
        view = self.klass(self.raster, "M", statistic="sum")
        self.assertEqual(view.dtype, np.float64)

    def test_other_dtype(self):
        # others upcast to atleast float32
        view = self.klass(self.raster_uint8, "M", statistic="mean")
        self.assertEqual(view.dtype, np.float32)
        view = self.klass(self.raster, "M", statistic="mean")
        self.assertEqual(view.dtype, np.float64)

    def test_int_result_dtype(self):
        # internally, it goes through float32 to have NaNs. check if this works
        view = self.klass(self.raster, "M", statistic="count")
        result = view.get_data(**self.request_all)
        self.assertEqual(result["values"].dtype, np.int32)

    def test_get_data_empty_vals(self):
        view = self.klass(self.raster, "D")
        assert view.get_data(**self.request_empty) is None

    def test_get_data_empty_time(self):
        view = self.klass(self.raster, "D")
        self.request_empty["mode"] = "time"
        assert view.get_data(**self.request_empty) == {"time": []}

    def test_get_data_empty_meta(self):
        view = self.klass(self.raster, "D")
        self.request_empty["mode"] = "meta"
        assert view.get_data(**self.request_empty) == {"meta": []}


class TestCumulative(unittest.TestCase):
    klass = Cumulative

    def setUp(self):
        self.raster = MockRaster(
            origin=Datetime(2000, 1, 1),
            value=np.array([[1.0, 0.0, np.nan]]),
            timedelta=Timedelta(days=1),
            bands=3,
        )
        self.raster_uint8 = MockRaster(
            origin=Datetime(2000, 1, 1), value=7, timedelta=Timedelta(days=1), bands=3
        )
        self.request = {
            "mode": "vals",
            "bbox": (0, 0, 3, 1),
            "width": 3,
            "height": 1,
            "projection": self.raster.projection,
        }
        self.request_first_two = {
            "start": Datetime(2000, 1, 1),
            "stop": Datetime(2000, 1, 2),
            **self.request,
        }
        self.request_second = {"start": Datetime(2000, 1, 2), **self.request}
        self.request_last = self.request
        self.request_all = {
            "start": Datetime(1970, 1, 1),
            "stop": Datetime(2020, 1, 1),
            **self.request,
        }
        self.request_empty = {
            "start": Datetime(1970, 1, 1),
            "stop": Datetime(1971, 1, 1),
            **self.request,
        }

    def test_get_data_meta(self):
        view = self.klass(self.raster, frequency="W", statistic="sum")
        self.request_all["mode"] = "meta"
        result = view.get_data(**self.request_all)
        self.assertListEqual(
            result["meta"],
            [
                ["Testmeta for band 0"],
                ["Testmeta for band 0", "Testmeta for band 1"],
                ["Testmeta for band 2"],
            ],
        )

    def test_get_time(self):
        view = self.klass(self.raster, frequency="W", statistic="sum")
        self.request_all["mode"] = "time"
        self.assertEqual(
            self.raster.get_data(**self.request_all)["time"],
            view.get_data(**self.request_all)["time"],
        )

    def test_get_data_meta_no_freq(self):
        view = self.klass(self.raster, frequency=None, statistic="sum")
        self.request_all["mode"] = "meta"
        result = view.get_data(**self.request_all)
        self.assertListEqual(
            result["meta"],
            [
                ["Testmeta for band 0"],
                ["Testmeta for band {}".format(i) for i in range(2)],
                ["Testmeta for band {}".format(i) for i in range(3)],
            ],
        )

    def test_get_data_sum_day(self):
        view = self.klass(self.raster, frequency="D", statistic="sum")
        result = view.get_data(**self.request_all)["values"]
        assert_equal(result, [[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
        result = view.get_data(**self.request_first_two)["values"]
        assert_equal(result, [[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
        result = view.get_data(**self.request_second)["values"]
        assert_equal(result, [[[1.0, 0.0, 0.0]]])
        result = view.get_data(**self.request_last)["values"]
        assert_equal(result, [[[1.0, 0.0, 0.0]]])

    def test_get_data_sum_week(self):
        view = self.klass(self.raster, frequency="W", statistic="sum")
        result = view.get_data(**self.request_all)["values"]
        assert_equal(result, [[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
        result = view.get_data(**self.request_first_two)["values"]
        assert_equal(result, [[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
        result = view.get_data(**self.request_second)["values"]
        assert_equal(result, [[[2.0, 0.0, 0.0]]])
        result = view.get_data(**self.request_last)["values"]
        assert_equal(result, [[[1.0, 0.0, 0.0]]])

    def test_get_data_sum_month(self):
        view = self.klass(self.raster, frequency="M", statistic="sum")
        result = view.get_data(**self.request_all)["values"]
        assert_equal(result, [[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]])
        result = view.get_data(**self.request_first_two)["values"]
        assert_equal(result, [[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
        result = view.get_data(**self.request_second)["values"]
        assert_equal(result, [[[2.0, 0.0, 0.0]]])
        result = view.get_data(**self.request_last)["values"]
        assert_equal(result, [[[3.0, 0.0, 0.0]]])

    def test_get_data_sum_no_freq(self):
        view = self.klass(self.raster, frequency=None, statistic="sum")
        result = view.get_data(**self.request_all)["values"]
        assert_equal(result, [[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]])
        result = view.get_data(**self.request_first_two)["values"]
        assert_equal(result, [[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
        result = view.get_data(**self.request_second)["values"]
        assert_equal(result, [[[2.0, 0.0, 0.0]]])
        result = view.get_data(**self.request_last)["values"]
        assert_equal(result, [[[3.0, 0.0, 0.0]]])

    def test_get_data_count(self):
        view = self.klass(self.raster, frequency="M", statistic="count")
        result = view.get_data(**self.request_all)
        assert_equal(result["values"], [[[1, 1, 0]], [[2, 2, 0]], [[3, 3, 0]]])

    def test_get_data_empty_vals(self):
        view = self.klass(self.raster, frequency="D", statistic="sum")
        assert view.get_data(**self.request_empty) is None

    def test_get_data_empty_meta(self):
        view = self.klass(self.raster, frequency="D", statistic="sum")
        self.request_empty["mode"] = "meta"
        assert view.get_data(**self.request_empty) == {"meta": []}


@pytest.mark.parametrize(
    "freq,direction,timezone,expected",
    [
        # Source period is [2000-01-01 00:00, 2000-01-01 02:00], inclusive on both sides
        ("90min", "backward", "UTC", (dt(2000, 1, 1), dt(2000, 1, 1, 3))),
        ("90min", "forward", "UTC", (dt(2000, 1, 1), dt(2000, 1, 1, 1, 30))),
        ("90min", "nearest", "UTC", (dt(2000, 1, 1), dt(2000, 1, 1, 1, 30))),
        ("75min", "backward", "UTC", (dt(2000, 1, 1), dt(2000, 1, 1, 2, 30))),
        ("75min", "forward", "UTC", (dt(2000, 1, 1), dt(2000, 1, 1, 1, 15))),
        ("75min", "nearest", "UTC", (dt(2000, 1, 1), dt(2000, 1, 1, 2, 30))),
        ("D", "backward", "UTC", (dt(2000, 1, 1), dt(2000, 1, 2))),
        ("D", "forward", "UTC", (dt(2000, 1, 1), dt(2000, 1, 1))),
        ("D", "nearest", "UTC", (dt(2000, 1, 1), dt(2000, 1, 1))),
        # Source period in Azores (GMT -1) timezone is [1999-12-31 23:00, 2000-01-01 01:00]
        # this affects the binning for daily frequency. With backwards direction, the period
        # will be [2000-01-01, 2000-01-02] Azores midnight. We add one hour to convert to UTC.
        ("H", "backward", "Atlantic/Azores", (dt(2000, 1, 1), dt(2000, 1, 1, 2))),
        ("D", "backward", "Atlantic/Azores", (dt(2000, 1, 1, 1), dt(2000, 1, 2, 1))),
    ],
)
def test_resample_period(freq, direction, timezone, expected, source):
    view = Resample(source, freq, direction, timezone)
    assert view.period == expected


def test_resample_period_empty(empty_source):
    view = Resample(empty_source, "D")
    assert view.period is None


@pytest.mark.parametrize(
    "freq,expected,",
    [
        ("H", Timedelta(hours=1)),
        ("D", Timedelta(days=1)),
        ("15T", Timedelta(minutes=15)),
        ("S", Timedelta(seconds=1)),
        ("MS", None),
        ("M", None),
        ("A", None),
    ],
)
def test_resample_timedelta(freq, expected, source):
    view = Resample(source, freq)
    assert view.timedelta == expected



@pytest.mark.parametrize(
    "frequency,direction,start,stop,expected_time,expected_values",
    [
        # Source period is [2000-01-01 00:00, 2000-01-01 02:00], inclusive on both sides
        #  90min and 75min test different behaviours around directions
        ("90min", "backward", dt(1970, 1, 1), dt(2020, 1, 1), [dt(2000, 1, 1), dt(2000, 1, 1, 1, 30), dt(2000, 1, 1, 3)], [0, 1, 2]),
        ("90min", "forward", dt(1970, 1, 1), dt(2020, 1, 1), [dt(2000, 1, 1), dt(2000, 1, 1, 1, 30)], [0, 2]),
        ("90min", "nearest", dt(1970, 1, 1), dt(2020, 1, 1), [dt(2000, 1, 1), dt(2000, 1, 1, 1, 30)], [0, 1]),
        ("75min", "backward", dt(1970, 1, 1), dt(2020, 1, 1), [dt(2000, 1, 1), dt(2000, 1, 1, 1, 15), dt(2000, 1, 1, 2, 30)], [0, 1, 2]),
        ("75min", "forward", dt(1970, 1, 1), dt(2020, 1, 1), [dt(2000, 1, 1), dt(2000, 1, 1, 1, 15)], [0, 2]),
        ("75min", "nearest", dt(1970, 1, 1), dt(2020, 1, 1), [dt(2000, 1, 1), dt(2000, 1, 1, 1, 15), dt(2000, 1, 1, 2, 30)], [0, 1, 2]),
        ("D", "backward", dt(1970, 1, 1), dt(2020, 1, 1), [dt(2000, 1, 1), dt(2000, 1, 2)], [0, 2]),
        ("D", "forward", dt(1970, 1, 1), dt(2020, 1, 1), [dt(2000, 1, 1)], [0]),
        # Partial requests
        ("90min", "backward", dt(2000, 1, 1), dt(2000, 1, 1, 2), [dt(2000, 1, 1), dt(2000, 1, 1, 1, 30)], [0, 1]),
        ("90min", "forward", dt(2000, 1, 1), dt(2000, 1, 1, 2), [dt(2000, 1, 1), dt(2000, 1, 1, 1, 30)], [0, 2]),
        ("90min", "nearest", dt(2000, 1, 1), dt(2000, 1, 1, 2), [dt(2000, 1, 1), dt(2000, 1, 1, 1, 30)], [0, 1]),
        ("75min", "backward", dt(2000, 1, 1), dt(2000, 1, 1, 2), [dt(2000, 1, 1), dt(2000, 1, 1, 1, 15)], [0, 1]),
        ("75min", "forward", dt(2000, 1, 1), dt(2000, 1, 1, 2), [dt(2000, 1, 1), dt(2000, 1, 1, 1, 15)], [0, 2]),
        ("75min", "nearest", dt(2000, 1, 1), dt(2000, 1, 1, 2), [dt(2000, 1, 1), dt(2000, 1, 1, 1, 15)], [0, 1]),
        # Oversampling
        ("1min", "nearest", dt(2000, 1, 1, 0, 29), dt(2000, 1, 1, 0, 31), [dt(2000, 1, 1, 0, 29), dt(2000, 1, 1, 0, 30), dt(2000, 1, 1, 0, 31)], [0, 0, 1]),
        # Single time request (stop = None, should return time index nearest to start, which snaps to some source frame)
        ("90min", "backward", dt(2000, 1, 1, 1, 30), None, [dt(2000, 1, 1, 1, 30)], [1]),
        ("90min", "backward", dt(2000, 1, 1, 0, 46), None, [dt(2000, 1, 1, 1, 30)], [1]),
        ("90min", "backward", dt(2000, 1, 1, 2, 15), None, [dt(2000, 1, 1, 1, 30)], [1]),
        ("90min", "forward", dt(2000, 1, 1, 1, 30), None, [dt(2000, 1, 1, 1, 30)], [2]),
        ("90min", "forward", dt(2000, 1, 1, 0, 46), None, [dt(2000, 1, 1, 1, 30)], [2]),
        ("90min", "forward", dt(2000, 1, 1, 2, 15), None, [dt(2000, 1, 1, 1, 30)], [2]),
        # Latest time request (both start and stop None, should return latest time index)
        ("90min", "backward", None, None, [dt(2000, 1, 1, 3)], [2]),
        ("90min", "forward", None, None, [dt(2000, 1, 1, 1, 30)], [2]),
        ("90min", "nearest", None, None, [dt(2000, 1, 1, 1, 30)], [1]),
    ],
)
def test_resample_get_data(source,point_request, frequency, direction, start, stop, expected_time, expected_values):
    view = Resample(source, frequency, direction=direction)

    # Time
    result = view.get_data(
        mode="time", start=start, stop=stop
    )
    assert result["time"] == expected_time

    # Meta (same logic as Data; easier to debug)
    result = view.get_data(
        mode="meta", start=start, stop=stop
    )
    assert result["meta"] == ["Testmeta for band {}".format(i) for i in expected_values]

    # Data
    DATA_MAPPING = [1, 7, 255]
    point_request.update({"start": start, "stop": stop})
    result = view.get_data(**point_request)
    if expected_values:
        assert_equal(
          result["values"],
          np.array([[[DATA_MAPPING[x]]] for x in expected_values]),
        )
    else:
        assert result is None
