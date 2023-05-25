from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
import unittest

import numpy as np
from numpy.testing import assert_equal

from dask_geomodeling import raster
from dask_geomodeling.tests.factories import MockRaster


class TestTemporalAggregate(unittest.TestCase):
    klass = raster.TemporalAggregate

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

    def test_period_day_agg(self):
        self.assertEqual(
            (Datetime(2000, 1, 1), Datetime(2000, 1, 3)),
            self.klass(self.raster, "D", closed="left", label="left").period,
        )
        self.assertEqual(
            (Datetime(2000, 1, 2), Datetime(2000, 1, 4)),
            self.klass(self.raster, "D", closed="left", label="right").period,
        )
        self.assertEqual(
            (Datetime(1999, 12, 31), Datetime(2000, 1, 2)),
            self.klass(self.raster, "D", closed="right", label="left").period,
        )
        self.assertEqual(
            (Datetime(2000, 1, 1), Datetime(2000, 1, 3)),
            self.klass(self.raster, "D", closed="right", label="right").period,
        )

        # 2000-01-01 00:00 UTC is 2000-01-01 01:00 in Amsterdam
        # 2000-01-01 01:00 falls in the 2000-01-01 bin (still Amsterdam)
        # the 2000-01-01 bin corresponds to 1999-12-31 23:00 UTC
        self.assertEqual(
            (Datetime(1999, 12, 31, 23), Datetime(2000, 1, 2, 23)),
            self.klass(self.raster, "D", timezone="Europe/Amsterdam").period,
        )
        # 2000-01-01 00:00 UTC is 1999-12-31 19:00 in New York
        # 1999-12-31 19:00 falls in the 1999-12-31 bin (still New York)
        # the 1999-12-31 bin corresponds to 1999-12-31 5:00 UTC
        self.assertEqual(
            (Datetime(1999, 12, 31, 5), Datetime(2000, 1, 2, 5)),
            self.klass(self.raster, "D", timezone="America/New_York").period,
        )

    def test_period_hour_agg(self):
        self.assertEqual(
            (Datetime(2000, 1, 1, 0), Datetime(2000, 1, 3, 0)),
            self.klass(self.raster, "H", closed="left", label="left").period,
        )
        self.assertEqual(
            (Datetime(2000, 1, 1, 1), Datetime(2000, 1, 3, 1)),
            self.klass(self.raster, "H", closed="left", label="right").period,
        )
        self.assertEqual(
            (Datetime(1999, 12, 31, 23), Datetime(2000, 1, 2, 23)),
            self.klass(self.raster, "H", closed="right", label="left").period,
        )
        self.assertEqual(
            (Datetime(2000, 1, 1), Datetime(2000, 1, 3)),
            self.klass(self.raster, "H", closed="right", label="right").period,
        )

        # 2000-01-01 00:00 UTC is 2000-01-01 01:00 in Amsterdam
        # the 2000-01-01 01:00 bin corresponds to 2000-01-01 00:00 UTC
        # you don't notice the timezone here
        self.assertEqual(
            (Datetime(2000, 1, 1, 0), Datetime(2000, 1, 3, 0)),
            self.klass(self.raster, "H", timezone="Europe/Amsterdam").period,
        )
        self.assertEqual(
            (Datetime(2000, 1, 1, 0), Datetime(2000, 1, 3, 0)),
            self.klass(self.raster, "H", timezone="America/New_York").period,
        )

    def test_period_none(self):
        view = self.klass(self.raster, frequency=None, statistic="sum")

        # test period
        self.assertEqual(
            (Datetime(2000, 1, 3, 0), Datetime(2000, 1, 3, 0)), view.period
        )

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
    klass = raster.Cumulative

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