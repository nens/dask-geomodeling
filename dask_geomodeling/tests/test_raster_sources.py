from datetime import datetime, timedelta
import unittest
import os

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from dask_geomodeling import utils
from dask_geomodeling.raster import MemorySource, RasterFileSource

from dask_geomodeling.tests.factories import (
    setup_temp_root,
    teardown_temp_root,
    create_tif,
)


class TstRasterSourceBase:
    def test_period(self):
        self.assertEqual(datetime(2000, 1, 1), self.source.period[0])
        self.assertEqual(datetime(2000, 1, 2), self.source.period[1])

    def test_timedelta(self):
        self.assertEqual(timedelta(days=1), self.source.timedelta)

    def test_len(self):
        self.assertEqual(2, len(self.source))

    def test_projection(self):
        self.assertEqual("EPSG:28992", self.source.projection)

    def test_dtype(self):
        self.assertEqual(np.uint8, self.source.dtype)

    def test_fillvalue(self):
        self.assertEqual(np.uint8(255), self.source.fillvalue)

    def test_extent(self):
        expected = (
            utils.Extent((136700, 455795, 136705, 455800), utils.get_sr("EPSG:28992"))
            .transformed(utils.get_sr("EPSG:4326"))
            .bbox
        )
        assert_allclose(self.source.extent, expected, atol=1e-10)

    def test_geometry(self):
        expected = (
            utils.Extent((136700, 455795, 136705, 455800), utils.get_sr("EPSG:28992"))
            .as_geometry()
            .ExportToWkt()
        )
        self.assertEqual(expected, self.source.geometry.ExportToWkt())

    def test_point_single_pixel(self):
        # data is defined at [136700, 136705) and (455795, 455800]
        for dx, dy in ((0, 0), (0, -4.99), (4.99, 0), (4.99, -4.99)):
            data = self.source.get_data(
                mode="vals",
                projection="EPSG:28992",
                bbox=(136700 + dx, 455800 + dy, 136700 + dx, 455800 + dy),
                width=1,
                height=1,
            )
            self.assertEqual(data["values"].shape, (1, 1, 1))
            assert_equal(data["values"], 5)

    def test_point_single_pixel_nodata(self):
        # data is defined at [136700, 136705) and (455795, 455800]
        for dx, dy in ((0, -5.0), (5.0, 0), (-5.0, 5.0), (-0.01, 0), (0, 0.01)):
            data = self.source.get_data(
                mode="vals",
                projection="EPSG:28992",
                bbox=(136700 + dx, 455800 + dy, 136700 + dx, 455800 + dy),
                width=1,
                height=1,
            )
            self.assertEqual(data["values"].shape, (1, 1, 1))
            assert_equal(data["values"], data["no_data_value"])

    def test_bbox_1x1(self):
        data = self.source.get_data(
            mode="vals",
            projection="EPSG:28992",
            bbox=(136700, 455800 - 5, 136700 + 5, 455800),
            width=1,
            height=1,
        )
        self.assertEqual(data["values"].shape, (1, 1, 1))
        assert_equal(data["values"], 5)

    def test_bbox_1x1_nodata(self):
        for dx, dy in ((0, -5), (-5, 0), (0, 5), (5, 0)):
            data = self.source.get_data(
                mode="vals",
                projection="EPSG:28992",
                bbox=(136700 + dx, 455800 - 5 + dy, 136705 + dx, 455800 + dy),
                width=1,
                height=1,
            )
            self.assertEqual(data["values"].shape, (1, 1, 1))
            assert_equal(data["values"], data["no_data_value"])

    def test_bbox_2x1(self):
        data = self.source.get_data(
            mode="vals",
            projection="EPSG:28992",
            bbox=(136700, 455800 - 5, 136710, 455800),
            width=2,
            height=1,
        )
        self.assertEqual(data["values"].shape, (1, 1, 2))
        assert_equal(data["values"], [[[5, data["no_data_value"]]]])

    def test_bbox_1x2(self):
        # y axis swapping: expect nodata on the low-y, so high-index side
        data = self.source.get_data(
            mode="vals",
            projection="EPSG:28992",
            bbox=(136700, 455800 - 10, 136705, 455800),
            width=1,
            height=2,
        )
        self.assertEqual(data["values"].shape, (1, 2, 1))
        assert_equal(data["values"], [[[5], [data["no_data_value"]]]])

    def test_bbox_4x2(self):
        data = self.source.get_data(
            mode="vals",
            projection="EPSG:28992",
            bbox=(136700, 455800 - 5, 136710, 455800),
            width=4,
            height=2,
        )
        self.assertEqual(data["values"].shape, (1, 2, 4))
        n = data["no_data_value"]
        assert_equal(data["values"], [[[5, 4, n, n], [5, 4, n, n]]])

    def test_bbox_single_pixel_zoom_in(self):
        data = self.source.get_data(
            mode="vals",
            projection="EPSG:28992",
            bbox=(136700, 455800 - 5, 136705, 455800),
            width=5,
            height=5,
        )
        self.assertEqual(data["values"].shape, (1, 5, 5))
        assert_equal(data["values"], 5)

    def test_get_time_last(self):
        data = self.source.get_data(mode="time")
        self.assertEqual(data["time"], [self.source.period[1]])

    def test_get_time_nearest(self):
        for start, expected in [
            (datetime(1970, 1, 1), datetime(2000, 1, 1)),
            (datetime(2000, 1, 1), datetime(2000, 1, 1)),
            (datetime(2000, 1, 1, 12), datetime(2000, 1, 1)),
            (datetime(2000, 1, 1, 12, 1), datetime(2000, 1, 2)),
            (datetime(2000, 1, 2), datetime(2000, 1, 2)),
            (datetime(2018, 1, 1), datetime(2000, 1, 2)),
        ]:
            data = self.source.get_data(mode="time", start=start)
            self.assertEqual(data["time"], [expected])

    def test_get_time_range(self):
        for start, stop in [
            (datetime(1970, 1, 1), datetime(1999, 12, 31, 12, 59)),
            (datetime(2000, 1, 2, 0, 1), datetime(2018, 1, 1)),
        ]:
            data = self.source.get_data(mode="time", start=start, stop=stop)
            self.assertEqual(data["time"], [])

        for start, stop in [
            (datetime(1970, 1, 1), datetime(2000, 1, 1)),
            (datetime(2000, 1, 1), datetime(2000, 1, 1)),
            (datetime(2000, 1, 1), datetime(2000, 1, 1, 23, 59)),
        ]:
            data = self.source.get_data(mode="time", start=start, stop=stop)
            self.assertEqual(data["time"], [datetime(2000, 1, 1)])

        for start, stop in [
            (datetime(1970, 1, 1), datetime(2010, 1, 1)),
            (datetime(2000, 1, 1), datetime(2000, 1, 2)),
        ]:
            data = self.source.get_data(mode="time", start=start, stop=stop)
            self.assertEqual(data["time"], [datetime(2000, 1, 1), datetime(2000, 1, 2)])


class TestMemorySource(TstRasterSourceBase, unittest.TestCase):
    def setUp(self):
        self.source = MemorySource(
            data=np.array([[[4]], [[5]]], dtype=np.uint8),
            no_data_value=255,
            projection="EPSG:28992",
            pixel_size=5,
            pixel_origin=(136700, 455800),
            time_first=datetime(2000, 1, 1),
            time_delta=timedelta(days=1),
            metadata=["meta 1", "meta 2"],
        )

    def test_get_meta_last(self):
        self.assertListEqual(
            self.source.get_data(mode="meta")["meta"], self.source.metadata[1:]
        )

    def test_get_meta_first(self):
        self.assertListEqual(
            self.source.get_data(mode="meta", start=datetime(1970, 1, 1))["meta"],
            self.source.metadata[:1],
        )

    def test_get_meta_all(self):
        self.assertListEqual(
            self.source.get_data(
                mode="meta", start=datetime(1970, 1, 1), stop=datetime(2010, 1, 1)
            )["meta"],
            self.source.metadata,
        )

    def test_get_meta_empty(self):
        self.assertListEqual(
            self.source.get_data(
                mode="meta", start=datetime(1970, 1, 1), stop=datetime(1971, 1, 1)
            )["meta"],
            [],
        )


class TestGeoTIFFSource(TstRasterSourceBase, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = setup_temp_root()
        cls.single_pixel_tif = os.path.join(cls.path, "test01.tiff")
        create_tif(
            cls.single_pixel_tif,
            bands=2,
            base_level=5,
            dtype="u1",
            no_data_value=255,
            projection="EPSG:28992",
            geo_transform=(136700.0, 5.0, 0.0, 455800.0, 0.0, -5.0),
            shape=(1, 1),
        )

    @classmethod
    def tearDownClass(cls):
        teardown_temp_root(cls.path)

    def setUp(self):
        self.source = RasterFileSource(
            url=self.single_pixel_tif,
            time_first=datetime(2000, 1, 1),
            time_delta=timedelta(days=1),
        )

    def tearDown(self):
        self.source.close_dataset()  # needed for the tearDownClass
