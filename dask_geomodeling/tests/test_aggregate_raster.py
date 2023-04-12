import unittest
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta

from numpy.testing import assert_almost_equal
from shapely.geometry import box, Point
import numpy as np
import pandas as pd

from dask import config

from dask_geomodeling.tests.factories import (
    MockGeometry,
    MockRaster,
)

from dask_geomodeling.raster import MemorySource
from dask_geomodeling.geometry.aggregate import bucketize
from dask_geomodeling.geometry import AggregateRaster, AggregateRasterAboveThreshold

try:
    from pandas.testing import assert_series_equal
except ImportError:
    from pandas.util.testing import assert_series_equal


class TestAggregateRaster(unittest.TestCase):
    def setUp(self):
        self.raster = MockRaster(
            origin=Datetime(2018, 1, 1), timedelta=Timedelta(hours=1), bands=1
        )
        self.source = MockGeometry(
            polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))],
            properties=[{"id": 1}],
        )
        self.view = AggregateRaster(
            source=self.source, raster=self.raster, statistic="sum"
        )
        self.request = dict(
            mode="intersects",
            projection="EPSG:3857",
            geometry=box(0, 0, 10, 10),
        )
        self.default_raster_limit = config.get("geomodeling.raster-limit")

    def tearDown(self):
        config.set({"geomodeling.raster-limit": self.default_raster_limit})

    def test_arg_types(self):
        self.assertRaises(TypeError, AggregateRaster, self.source, None)
        self.assertRaises(TypeError, AggregateRaster, None, self.raster)
        self.assertRaises(
            TypeError,
            AggregateRaster,
            self.source,
            self.raster,
            statistic=None,
        )
        self.assertRaises(
            TypeError,
            AggregateRaster,
            self.source,
            self.raster,
            projection=4326,
        )

        # if no projection / geo_transform specified, take them from raster
        view = AggregateRaster(self.source, self.raster)
        self.assertEqual(self.raster.projection, view.projection)
        self.assertEqual(1.0, view.pixel_size)

        view = AggregateRaster(
            self.source, self.raster, projection="EPSG:28992", pixel_size=0.2
        )
        self.assertEqual("EPSG:28992", view.projection)
        self.assertEqual(0.2, view.pixel_size)

        # 0 pixel size is unsupported
        self.assertRaises(
            ValueError,
            AggregateRaster,
            self.source,
            self.raster,
            pixel_size=0.0,
        )

        # percentile value out of bounds
        self.assertRaises(
            ValueError,
            AggregateRaster,
            self.source,
            self.raster,
            projection="EPSG:28992",
            statistic="p101",
        )

    def test_column_attr(self):
        self.assertSetEqual(
            self.view.columns, self.source.columns | {self.view.column_name}
        )

    def test_statistics(self):
        range_raster = MockRaster(
            origin=Datetime(2018, 1, 1),
            timedelta=Timedelta(hours=1),
            bands=1,
            value=np.indices((10, 10))[0],
        )
        self.request["start"] = Datetime(2018, 1, 1)
        self.request["stop"] = Datetime(2018, 1, 1, 3)

        for statistic, expected in [
            ("sum", 162.0),
            ("count", 36.0),
            ("mean", 4.5),
            ("min", 2.0),
            ("max", 7.0),
            ("median", 4.5),
            ("p75", 6.0),
        ]:
            view = AggregateRaster(
                source=self.source, raster=range_raster, statistic=statistic
            )
            features = view.get_data(**self.request)["features"]
            agg = features.iloc[0]["agg"]
            self.assertEqual(expected, agg)

    def test_statistics_empty(self):
        range_raster = MockRaster(
            origin=Datetime(2018, 1, 1),
            timedelta=Timedelta(hours=1),
            bands=1,
            value=255,  # nodata
        )
        self.request["start"] = Datetime(2018, 1, 1)
        self.request["stop"] = Datetime(2018, 1, 1, 3)

        for statistic, expected in [
            ("sum", 0),
            ("count", 0),
            ("mean", np.nan),
            ("min", np.nan),
            ("max", np.nan),
            ("median", np.nan),
            ("p75", np.nan),
        ]:
            view = AggregateRaster(
                source=self.source, raster=range_raster, statistic=statistic
            )
            features = view.get_data(**self.request)["features"]
            agg = features.iloc[0]["agg"]
            if np.isnan(expected):
                self.assertTrue(np.isnan(agg))
            else:
                self.assertEqual(expected, agg)

    def test_statistics_partial_empty(self):
        values = np.indices((10, 10), dtype=np.uint8)[0]
        values[2:8, 2:8] = 255  # nodata
        range_raster = MockRaster(
            origin=Datetime(2018, 1, 1),
            timedelta=Timedelta(hours=1),
            bands=1,
            value=values,
        )

        for statistic, expected in [
            ("sum", 0),
            ("count", 0),
            ("mean", np.nan),
            ("min", np.nan),
            ("max", np.nan),
            ("median", np.nan),
            ("p75", np.nan),
        ]:
            view = AggregateRaster(
                source=self.source, raster=range_raster, statistic=statistic
            )
            features = view.get_data(**self.request)["features"]
            agg = features.iloc[0]["agg"]
            if np.isnan(expected):
                self.assertTrue(np.isnan(agg))
            else:
                self.assertEqual(expected, agg)

    def test_raster_request(self):
        req = self.request
        for geom in [box(0, 0, 10, 10), box(4, 4, 6, 6), Point(5, 5)]:
            req["geometry"] = geom
            _, (_, request), _ = self.view.get_sources_and_requests(**req)
            np.testing.assert_allclose(request["bbox"], (2, 2, 8, 8))
            self.assertEqual(6, request["width"])
            self.assertEqual(6, request["height"])

    def test_raster_time_resolution(self):
        req = self.request
        req["time_resolution"] = 3600000
        req["geometry"] = box(0, 0, 10, 10)

        # temp_group = GroupTemporal([
        #     MockRaster(timedelta=Timedelta(hours=1)),
        #     MockRaster(timedelta=Timedelta(minutes=1)),
        #     MockRaster(timedelta=Timedelta(seconds=1))
        # ])

        temp_raster = MockRaster(
            origin=Datetime(2018, 1, 1),
            timedelta=Timedelta(hours=1),
            bands=1
        )

        geom_source = MockGeometry(
            polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))],
            properties=[{"id": 1}],
        )
        view = AggregateRaster(
            source=geom_source, raster=temp_raster, statistic="sum"
        )

        _, (source, request), _ = view.get_sources_and_requests(**req)

        self.assertEqual(3600000, request["time_resolution"])

    def test_pixel_size(self):
        # larger
        self.view = AggregateRaster(
            source=self.source, raster=self.raster, statistic="sum", pixel_size=2
        )
        _, (_, request), _ = self.view.get_sources_and_requests(**self.request)
        np.testing.assert_allclose(request["bbox"], (2, 2, 8, 8))
        self.assertEqual(3, request["width"])
        self.assertEqual(3, request["height"])

        # smaller
        self.view = AggregateRaster(
            source=self.source, raster=self.raster, statistic="sum", pixel_size=0.5
        )
        _, (_, request), _ = self.view.get_sources_and_requests(**self.request)
        np.testing.assert_allclose(request["bbox"], (2, 2, 8, 8))
        self.assertEqual(12, request["width"])
        self.assertEqual(12, request["height"])

    def test_max_pixels(self):
        self.view = AggregateRaster(
            source=self.source,
            raster=self.raster,
            statistic="sum",
            max_pixels=9,
            auto_pixel_size=True,
        )
        _, (_, request), _ = self.view.get_sources_and_requests(**self.request)

        np.testing.assert_allclose(request["bbox"], (2, 2, 8, 8))
        self.assertEqual(3, request["width"])
        self.assertEqual(3, request["height"])

    def test_snap_bbox(self):
        for (x1, y1, x2, y2), exp_bbox, (exp_width, exp_height) in (
            [(2.01, 1.99, 7.99, 8.01), (2, 1, 8, 9), (6, 8)],
            [(1.99, 2.01, 8.01, 7.99), (1, 2, 9, 8), (8, 6)],
            [(2.0, 2.0, 8.0, 8.0), (2, 2, 8, 8), (6, 6)],
            [(2.9, 1.1, 8.9, 7.1), (2, 1, 9, 8), (7, 7)],
        ):
            self.view = AggregateRaster(
                MockGeometry([((x1, y1), (x2, y1), (x2, y2), (x1, y2))]), self.raster
            )
            _, (_, request), _ = self.view.get_sources_and_requests(**self.request)

            np.testing.assert_allclose(request["bbox"], exp_bbox)
            self.assertEqual(exp_width, request["width"])
            self.assertEqual(exp_height, request["height"])

    def test_max_pixels_with_snap(self):
        x1, y1, x2, y2 = 2.01, 1.99, 7.99, 8.01

        self.view = AggregateRaster(
            MockGeometry([((x1, y1), (x2, y1), (x2, y2), (x1, y2))]),
            self.raster,
            max_pixels=20,
            auto_pixel_size=True,
        )
        _, (_, request), _ = self.view.get_sources_and_requests(**self.request)

        # The resulting bbox has too many pixels, so the pixel_size increases
        # by an integer factor to 2. Therefore snapping becomes different too.
        np.testing.assert_allclose(request["bbox"], (2, 0, 8, 10))
        self.assertEqual(3, request["width"])
        self.assertEqual(5, request["height"])

    def test_no_auto_scaling(self):
        self.view = AggregateRaster(
            source=self.source, raster=self.raster, statistic="sum", max_pixels=9
        )

        self.assertRaises(
            RuntimeError, self.view.get_sources_and_requests, **self.request
        )

    def test_max_pixels_fallback(self):
        config.set({"geomodeling.raster-limit": 9})
        self.view = AggregateRaster(
            source=self.source, raster=self.raster, statistic="sum"
        )

        self.assertRaises(
            RuntimeError, self.view.get_sources_and_requests, **self.request
        )

    def test_extensive_scaling(self):
        # if a requested resolution is lower than the base resolution, the
        # sum aggregation requires scaling. sum is an extensive variable.

        # this is, of course, approximate. we choose a geo_transform so that
        # scaled-down geometries still precisely match the original
        view1 = self.view
        view2 = AggregateRaster(
            self.source,
            self.raster,
            statistic="sum",
            pixel_size=0.1,
            max_pixels=6 ** 2,
            auto_pixel_size=True,
        )
        agg1 = view1.get_data(**self.request)["features"].iloc[0]["agg"]
        agg2 = view2.get_data(**self.request)["features"].iloc[0]["agg"]
        self.assertEqual(agg1 * (10 ** 2), agg2)

    def test_intensive_scaling(self):
        # if a requested resolution is lower than the base resolution, the
        # mean aggregation does not require scaling.

        view1 = AggregateRaster(self.source, self.raster, statistic="mean")
        view2 = AggregateRaster(
            self.source,
            self.raster,
            statistic="mean",
            pixel_size=0.1,
            max_pixels=6 ** 2,
            auto_pixel_size=True,
        )
        agg1 = view1.get_data(**self.request)["features"].iloc[0]["agg"]
        agg2 = view2.get_data(**self.request)["features"].iloc[0]["agg"]
        self.assertEqual(agg1, agg2)

    def test_different_projection(self):
        view = AggregateRaster(
            source=self.source,
            raster=self.raster,
            statistic="mean",
            projection="EPSG:3857",
        )

        self.request["projection"] = "EPSG:4326"
        self.request["geometry"] = box(-180, -90, 180, 90)

        _, (_, request), _ = view.get_sources_and_requests(**self.request)
        self.assertEqual(request["projection"], "EPSG:3857")

        result = view.get_data(**self.request)
        self.assertEqual(result["projection"], "EPSG:4326")
        self.assertEqual(result["features"].iloc[0]["agg"], 1.0)

    def test_time(self):
        raster = MockRaster(
            origin=Datetime(2018, 1, 1), timedelta=Timedelta(hours=1), bands=3
        )
        view = AggregateRaster(
            source=self.source, raster=raster, statistic="mean"
        )
        request = self.request.copy()

        # full range
        request["start"], request["stop"] = raster.period
        data = view.get_data(**request)
        value = data["features"].iloc[0]["agg"][0]
        self.assertEqual(3, len(value))

        # single frame
        request["stop"] = None
        data = view.get_data(**request)
        value = data["features"].iloc[0]["agg"]
        self.assertEqual(1.0, value)

        # out of range
        request["start"] = raster.period[0] + Timedelta(days=1)
        request["stop"] = raster.period[1] + Timedelta(days=1)
        data = view.get_data(**request)
        value = data["features"].iloc[0]["agg"]
        self.assertTrue(np.isnan(value))

    def test_chained_aggregation(self):
        raster2 = MockRaster(
            origin=Datetime(2018, 1, 1), timedelta=Timedelta(hours=1), bands=1, value=7
        )

        chained = AggregateRaster(
            self.view, raster2, statistic="mean", column_name="agg2"
        )
        result = chained.get_data(**self.request)
        feature = result["features"].iloc[0]

        self.assertEqual(36.0, feature["agg"])
        self.assertEqual(7.0, feature["agg2"])

    def test_overlapping_geometries(self):
        source = MockGeometry(
            polygons=[
                ((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)),
                ((2.0, 2.0), (8.0, 2.0), (8.0, 5.0), (2.0, 5.0)),
            ],
            properties=[{"id": 1}, {"id": 2}],
        )
        view = AggregateRaster(
            source=source, raster=self.raster, statistic="sum"
        )
        result = view.get_data(**self.request)
        self.assertEqual(result["features"]["agg"].values.tolist(), [36.0, 18.0])

    def test_aggregate_percentile_one_empty(self):
        # if there are only nodata pixels in the geometries, we expect the
        # statistic of mean, min, max, median and percentile to be NaN.
        for agg in ["mean", "min", "max", "median", "p90.0"]:
            data = np.ones((1, 10, 10), dtype=np.uint8)
            data[:, :5, :] = 255
            raster = MemorySource(
                data, 255, "EPSG:3857", pixel_size=1, pixel_origin=(0, 10)
            )
            source = MockGeometry(
                polygons=[
                    ((2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)),
                    ((6.0, 6.0), (8.0, 6.0), (8.0, 8.0), (6.0, 8.0)),
                ],
                properties=[{"id": 1}, {"id": 2}],
            )
            view = AggregateRaster(source=source, raster=raster, statistic=agg)
            result = view.get_data(**self.request)
            assert np.isnan(result["features"]["agg"].values[1])

    def test_empty_dataset(self):
        source = MockGeometry(polygons=[], properties=[])
        view = AggregateRaster(
            source=source, raster=self.raster, statistic="sum"
        )
        result = view.get_data(**self.request)
        self.assertEqual(0, len(result["features"]))

    def test_aggregate_above_threshold(self):
        range_raster = MockRaster(
            origin=Datetime(2018, 1, 1),
            timedelta=Timedelta(hours=1),
            bands=1,
            value=np.indices((10, 10))[0],
        )
        source = MockGeometry(
            polygons=[
                ((2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)),  # contains 7, 8
                ((2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)),  # contains 7, 8
                ((7.0, 7.0), (9.0, 7.0), (9.0, 9.0), (7.0, 9.0)),  # contains 2, 3
                ((6.0, 6.0), (8.0, 6.0), (8.0, 8.0), (6.0, 8.0)),  # contains 3, 4
            ],
            properties=[
                {"id": 1, "threshold": 8.0},  # threshold halfway
                {"id": 3, "threshold": 3.0},  # threshold below
                {"id": 2000000, "threshold": 4.0},  # threshold above
                {"id": 9},
            ],  # no threshold
        )
        self.request["start"] = Datetime(2018, 1, 1)
        self.request["stop"] = Datetime(2018, 1, 1, 3)

        for statistic, expected in [
            ("sum", [16.0, 30.0, 0.0, 0.0]),
            ("count", [2, 4, 0, 0]),
            ("mean", [8.0, 7.5, np.nan, np.nan]),
        ]:
            view = AggregateRasterAboveThreshold(
                source=source,
                raster=range_raster,
                statistic=statistic,
                threshold_name="threshold",
            )
            features = view.get_data(**self.request)["features"]
            assert_series_equal(
                features["agg"],
                pd.Series(expected, index=[1, 3, 2000000, 9], dtype=np.float32),
                check_names=False,
            )

    def test_aggregate_no_interaction(self):
        for dx in [0.0, 0.1, 0.4999, 0.50001, 0.9, 0.99999]:
            raster = MockRaster(
                origin=Datetime(2018, 1, 1),
                timedelta=Timedelta(hours=1),
                bands=1,
                value=np.indices((10, 10))[1],
            )
            source = MockGeometry(
                polygons=[
                    ((2.0 + dx, 2.0), (4.0 + dx, 2.0), (4.0 + dx, 4.0), (2.0 + dx, 4.0)),
                    ((3.0, 6.0), (5, 6.0), (5, 8.0), (3, 8.0)),  # exactly contains 3, 4
                ],
                properties=[{"id": 1}, {"id": 2}],
            )
            view = AggregateRaster(source=source, raster=raster, statistic="min")
            result = view.get_data(**self.request)
            assert result["features"]["agg"][2] == 3

    def test_small_geometry(self):
        # Rasterize only takes pixels into account whose center is
        # inside the polygon. 
        raster = MockRaster(
            origin=Datetime(2018, 1, 1),
            timedelta=Timedelta(hours=1),
            bands=1,
            value=np.indices((10, 10))[1],  # increasing in x only
        )
        source = MockGeometry(
            polygons=[
                ((2.0, 2.0), (2.1, 2.0), (2.1, 3.0), (2.0, 3.0)),
                ((4.9, 1.0), (5.0, 1.0), (5.0, 2.0), (4.9, 2.0)),
            ],
            properties=[{"id": 1}, {"id": 2}],
        )
        view = AggregateRaster(
            source=source, raster=raster, statistic="max"
        )
        result = view.get_data(**self.request)
        assert_almost_equal(result["features"]["agg"].values, [2., 4.])

    def test_small_geometry_threshold(self):
        # Rasterize only takes pixels into account whose center is
        # inside the polygon. 
        raster = MockRaster(
            origin=Datetime(2018, 1, 1),
            timedelta=Timedelta(hours=1),
            bands=1,
            value=np.indices((10, 10))[1],  # increasing in x only
        )
        source = MockGeometry(
            polygons=[
                ((2.0, 2.0), (2.1, 2.0), (2.1, 3.0), (2.0, 3.0)),
                ((4.9, 1.0), (5.0, 1.0), (5.0, 2.0), (4.9, 2.0)),
            ],
            properties=[{"id": 1, "threshold": 3.}, {"id": 2, "threshold": 3.}],
        )
        view = AggregateRasterAboveThreshold(
            source=source, raster=raster, statistic="max", threshold_name="threshold"
        )
        result = view.get_data(**self.request)
        assert_almost_equal(result["features"]["agg"].values, [np.nan, 4.])

    def test_small_geometry_temporal(self):
        raster = MockRaster(
            origin=Datetime(2018, 1, 1), timedelta=Timedelta(hours=1), bands=3
        )
        source = MockGeometry(
            polygons=[
                ((2.0, 2.0), (2.1, 2.0), (2.1, 3.0), (2.0, 3.0)),
            ],
            properties=[{"id": 1}, {"id": 2}],
        )
        view = AggregateRaster(
            source=source, raster=raster, statistic="max"
        )
        request = self.request.copy()
        request["start"], request["stop"] = raster.period
        result = view.get_data(**request)
        assert_almost_equal(result["features"]["agg"].loc[1][0], [1., 1., 1.])


class TestBucketize(unittest.TestCase):
    def test_bucketize(self):
        bboxes = [
            (0, 0, 2, 2),  # new bucket
            (2, 2, 4, 4),  # new bucket because of overlap with previous bucket
            (0, 0, 3, 3),  # new bucket because of size
            (5, 5, 7, 7),  # same as first
        ]
        expected = [[0, 3], [1], [2]]
        buckets = bucketize(bboxes)
        self.assertEqual([0, 1, 2, 3], sorted(i for b in buckets for i in b))
        self.assertEqual(expected, sorted(buckets))
