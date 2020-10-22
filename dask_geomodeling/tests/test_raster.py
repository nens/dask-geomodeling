from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
import unittest

import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy import ndimage

from dask_geomodeling import raster
from dask_geomodeling.utils import EPSG4326, EPSG3857
from dask_geomodeling.utils import Extent, get_dtype_max, get_epsg_or_wkt
from dask_geomodeling.raster import RasterBlock
from dask_geomodeling.tests.factories import MockRaster, MockGeometry


class MockRasterWithGeotransform(MockRaster):
    def __init__(self, *args, **kwargs):
        self._geo_transform = kwargs.pop("geo_transform")
        super().__init__(*args, **kwargs)

    @property
    def geo_transform(self):
        return self._geo_transform


class TestRasterBlockAttrs(unittest.TestCase):
    """Tests properties that all raster blocks share"""

    def test_attrs(self):
        """ Check compulsory attributes for all views in raster.py """
        missing = []
        for name, klass in raster.__dict__.items():
            try:
                if not issubclass(klass, RasterBlock):
                    continue  # skip non-RasterBlock objects
                if klass is RasterBlock:
                    continue  # skip the baseclass
            except TypeError:
                continue  # also skip non-classes
            for attr in (
                "period",
                "timedelta",
                "extent",
                "dtype",
                "fillvalue",
                "geometry",
                "projection",
                "geo_transform",
            ):
                if not hasattr(klass, attr):
                    print(name, attr)
                    missing.append([name, attr])
        if len(missing) > 0:
            print(missing)

        self.assertEqual(0, len(missing))


class TestElementwise(unittest.TestCase):
    klass = raster.elemwise.BaseElementwise

    def test_differing_timedelta(self):
        storage1 = MockRaster(
            origin=Datetime(2018, 4, 1), timedelta=Timedelta(hours=1), bands=6
        )
        storage2 = MockRaster(
            origin=Datetime(2018, 4, 1), timedelta=Timedelta(hours=2), bands=3
        )

        # differing timedeltas are not allowed
        self.assertRaises(ValueError, self.klass, storage1, storage2)

    def test_propagate_timedelta(self):
        storage1 = MockRaster(timedelta=Timedelta(hours=1))

        for args in [
            (storage1, "something"),
            ("something", storage1),
            (storage1, storage1),
        ]:
            elemwise = self.klass(*args)
            self.assertEqual(elemwise.timedelta, storage1.timedelta)

    def test_propagate_none_timedelta(self):
        storage1 = MockRaster(timedelta=Timedelta(hours=1))
        storage2 = MockRaster(timedelta=None)

        # None timedelta precedes
        for args in [(storage1, storage2), (storage2, storage1)]:
            elemwise = self.klass(*args)
            self.assertIsNone(elemwise.timedelta)

    def test_propagate_period(self):
        storage1 = MockRaster(
            origin=Datetime(2018, 4, 1), timedelta=Timedelta(hours=1), bands=6
        )
        storage2 = MockRaster(
            origin=Datetime(2018, 4, 1, 2), timedelta=Timedelta(hours=1), bands=6
        )

        for args in [
            (storage1, "something"),
            ("something", storage1),
            (storage1, storage1),
        ]:
            elemwise = self.klass(*args)
            self.assertEqual(storage1.period, elemwise.period)

        # the elemwise period equals the intersection between the two
        elemwise = self.klass(storage1, storage2)
        self.assertEqual(elemwise.period[0], storage2.period[0])
        self.assertEqual(elemwise.period[1], storage1.period[1])

        # period is None if there is no intersection
        storage3 = MockRaster(
            origin=Datetime(2018, 4, 2), timedelta=Timedelta(hours=1), bands=6
        )
        elemwise = self.klass(storage1, storage3)
        self.assertIsNone(elemwise.period)

    def test_propagate_none_period(self):
        storage1 = MockRaster(origin=None)
        storage2 = MockRaster(
            origin=Datetime(2018, 4, 1), timedelta=Timedelta(hours=1), bands=6
        )

        for args in [
            (storage1, "something"),
            ("something", storage1),
            (storage1, storage1),
        ]:
            elemwise = self.klass(*args)
            self.assertIsNone(elemwise.period)

        # None period precedes
        for args in [(storage1, storage2), (storage2, storage1)]:
            elemwise = self.klass(*args)
            self.assertIsNone(elemwise.period)

    def test_propagate_extent(self):
        storage1 = MockRaster(value=np.empty((1, 2)))
        storage2 = MockRaster(value=np.empty((3, 4)))

        for args in [
            (storage1, "something"),
            ("something", storage1),
            (storage1, storage1),
        ]:
            elemwise = self.klass(*args)
            self.assertEqual(storage1.extent, elemwise.extent)

        # the elemwise extent equals the intersection between the two
        elemwise = self.klass(storage1, storage2)
        self.assertEqual(elemwise.extent, (0, 0, 2, 1))

    def test_propagate_none_extent(self):
        storage1 = MockRaster(value=None)
        storage2 = MockRaster(value=np.empty((1, 2)))

        for args in [
            (storage1, "something"),
            ("something", storage1),
            (storage1, storage1),
        ]:
            elemwise = self.klass(*args)
            self.assertIsNone(elemwise.extent)

        # None extent precedes
        for args in [(storage1, storage2), (storage2, storage1)]:
            elemwise = self.klass(*args)
            self.assertIsNone(elemwise.extent)

    def test_propagate_geometry(self):
        storage1 = MockRaster(value=np.empty((1, 2)))
        storage2 = MockRaster(value=np.empty((3, 4)))

        # the combined extent equals the intersected bbox
        for args in [(storage1, storage2), (storage2, storage1)]:
            combined = self.klass(*args)
            x1, x2, y1, y2 = combined.geometry.GetEnvelope()
            self.assertEqual((x1, y1, x2, y2), (0.0, 0.0, 2.0, 1.0))

    def test_propagate_geometry_different_projection(self):
        storage1 = MockRaster(projection="EPSG:3857")
        storage2 = MockRaster(projection="EPSG:4326")

        # the combined extent equals the joined bbox in the first store proj
        for args in [(storage1, storage2), (storage2, storage1)]:
            geometry = self.klass(*args).geometry
            self.assertEqual(
                args[0].projection, get_epsg_or_wkt(geometry.GetSpatialReference())
            )

    def test_propagate_projection(self):
        self.assertEqual(
            self.klass(MockRaster(value=1, projection="EPSG:3857"), 1).projection,
            "EPSG:3857",
        )

        self.assertEqual(
            self.klass(1, MockRaster(value=1, projection="EPSG:3857")).projection,
            "EPSG:3857",
        )

        self.assertEqual(
            self.klass(
                MockRaster(value=1, projection="EPSG:3857"),
                MockRaster(value=2, projection="EPSG:3857"),
            ).projection,
            "EPSG:3857",
        )

        self.assertIsNone(
            self.klass(
                MockRaster(value=1, projection="EPSG:3857"),
                MockRaster(value=2, projection="EPSG:4326"),
            ).projection
        )

        self.assertIsNone(
            self.klass(
                MockRaster(value=1, projection="EPSG:3857"),
                MockRaster(value=2, projection=None),
            ).projection
        )

        self.assertIsNone(
            self.klass(
                MockRaster(value=1, projection=None),
                MockRaster(value=2, projection=None),
            ).projection
        )

    def test_propagate_geo_transform(self):
        # single geo-transform propagates
        self.assertTupleEqual(
            self.klass(
                MockRasterWithGeotransform(geo_transform=(0, 1, 0, 1, 0, -1)), 1
            ).geo_transform,
            (0, 1, 0, 1, 0, -1),
        )

        self.assertTupleEqual(
            self.klass(
                1, MockRasterWithGeotransform(geo_transform=(0, 1, 0, 1, 0, -1))
            ).geo_transform,
            (0, 1, 0, 1, 0, -1),
        )

        # matching geotransform propagates
        self.assertTupleEqual(
            self.klass(
                MockRasterWithGeotransform(geo_transform=(0, 1, 0, 1, 0, -1)),
                MockRasterWithGeotransform(geo_transform=(5, 1, 0, -8, 0, -1)),
            ).geo_transform,
            (0, 1, 0, 1, 0, -1),
        )

        # non-matching results in None
        self.assertIsNone(
            self.klass(
                MockRasterWithGeotransform(geo_transform=(0, 1, 0, 1, 0, -1)),
                MockRasterWithGeotransform(geo_transform=(0, 2, 0, 1, 0, -2)),
            ).geo_transform
        )

        # check None propagation
        self.assertIsNone(
            self.klass(
                MockRasterWithGeotransform(geo_transform=None),
                MockRasterWithGeotransform(geo_transform=(0, 1, 0, 1, 0, -1)),
            ).geo_transform
        )

        self.assertIsNone(
            self.klass(
                MockRasterWithGeotransform(geo_transform=(0, 1, 0, 1, 0, -1)),
                MockRasterWithGeotransform(geo_transform=None),
            ).geo_transform
        )


class TestMath(unittest.TestCase):
    klass = raster.elemwise.BaseMath

    def setUp(self):
        self.storage = MockRaster(
            origin=Datetime(2000, 1, 1), timedelta=Timedelta(hours=1), bands=3
        )
        self.bool_storage = MockRaster(
            origin=Datetime(2000, 1, 1),
            timedelta=Timedelta(hours=1),
            bands=1,
            value=[[1, 1], [7, 7], [255, 255]],  # 2x3 shape. 255 is nodata
        )
        self.logexp_storage = MockRaster(
            origin=Datetime(2000, 1, 1),
            timedelta=Timedelta(hours=1),
            bands=1,
            value=np.array(
                [[-1, 0], [np.e, 10], [999, get_dtype_max("f8")]], dtype="f8"
            ),
        )
        self.vals_request = dict(
            mode="vals",
            start=Datetime(2000, 1, 1),
            stop=Datetime(2010, 1, 1, 2),
            width=2,
            height=3,
        )
        self.time_request = dict(
            mode="time", start=Datetime(2000, 1, 1), stop=Datetime(2001, 1, 1)
        )
        self.expected_time = [
            Datetime(2000, 1, 1) + i * Timedelta(hours=1) for i in range(3)
        ]
        self.meta_request = dict(
            mode="meta", start=Datetime(2000, 1, 1), stop=Datetime(2001, 1, 1)
        )
        self.expected_meta = ["Testmeta for band {}".format(i) for i in range(3)]
        self.all_requests = (self.vals_request, self.time_request, self.meta_request)

    def test_math_init(self):
        # we can't init with a non-number
        self.assertRaises(TypeError, self.klass, self.storage, "not-a-number")

    def test_add_dtype(self):
        for dtype, expected in [
            ("bool", "i4"),
            ("u1", "i4"),
            ("i8", "i8"),
            ("f2", "f4"),
            ("f8", "f8"),
        ]:
            view = self.storage + np.ones(1, dtype=dtype)
            self.assertEqual(np.dtype(expected), view.dtype)
            data = view.get_data(**self.vals_request)["values"]
            self.assertEqual(np.dtype(expected), data.dtype)

    def test_divide_dtype(self):
        for dtype, expected in [
            ("bool", "f4"),
            ("u1", "f4"),
            ("i8", "f8"),
            ("f2", "f4"),
            ("f8", "f8"),
        ]:
            view = self.storage / np.ones(1, dtype=dtype)
            self.assertEqual(np.dtype(expected), view.dtype)
            data = view.get_data(**self.vals_request)["values"]
            self.assertEqual(np.dtype(expected), data.dtype)

    def test_add(self):
        view = self.storage + 5
        assert_equal(view.get_data(**self.vals_request)["values"], 6)

    def test_subtract(self):
        view = self.storage - 1
        assert_equal(view.get_data(**self.vals_request)["values"], 0)

    def test_multiply(self):
        view = self.storage * 10
        assert_equal(view.get_data(**self.vals_request)["values"], 10)

    def test_negate(self):
        view = -self.storage
        assert_equal(view.get_data(**self.vals_request)["values"], -1)

    def test_divide(self):
        view = self.storage / 10
        assert_equal(view.get_data(**self.vals_request)["values"], 0.1)

    def test_power(self):
        storage7 = self.storage * 7
        view = storage7 ** 1
        assert_equal(view.get_data(**self.vals_request)["values"], 7)
        view = storage7 ** 2
        assert_equal(view.get_data(**self.vals_request)["values"], 49)
        view = storage7 ** 0.5
        assert_equal(view.get_data(**self.vals_request)["values"], np.sqrt(7))
        view = storage7 ** -1
        assert_equal(view.get_data(**self.vals_request)["values"], 1 / 7)
        view = storage7 ** 0
        assert_equal(view.get_data(**self.vals_request)["values"], 1)

    def test_equal(self):
        view = self.bool_storage == 7
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :, 0], [False, True, False]
        )

        # nodata == nodata evaluates to False
        view = self.bool_storage == 255
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :, 0], [False, False, False]
        )

    def test_notequal(self):
        view = self.bool_storage != 7
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :, 0], [True, False, True]
        )

        # nodata != nodata evaluates to True
        view = self.bool_storage != 255
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :, 0], [True, True, True]
        )

    def test_greater(self):
        view = self.bool_storage > 1
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :, 0], [False, True, False]
        )

    def test_greater_equal(self):
        view = self.bool_storage >= 7
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :, 0], [False, True, False]
        )

    def test_less(self):
        view = self.bool_storage < 7
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :, 0], [True, False, False]
        )

    def test_less_equal(self):
        view = self.bool_storage <= 1
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :, 0], [True, False, False]
        )

    def test_invert(self):
        view = ~(self.bool_storage == 7)  # == 7 gives [False, True]
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :2, 0], [True, False]
        )

    def test_and(self):
        view = (self.bool_storage == 7) & True  # [False, True] & True
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :2, 0], [False, True]
        )
        # [False, True] & [False, True]
        view = (self.bool_storage == 7) & (self.bool_storage == 7)
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :2, 0], [False, True]
        )
        # [False, True] & [True, False]
        view = (self.bool_storage == 7) & (self.bool_storage != 7)
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :2, 0], [False, False]
        )

    def test_or(self):
        view = (self.bool_storage == 7) | True  # [False, True] | True
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :2, 0], [True, True]
        )
        # [False, True] | [False, True]
        view = (self.bool_storage == 7) | (self.bool_storage == 7)
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :2, 0], [False, True]
        )
        # [False, True] | [True, False]
        view = (self.bool_storage == 7) | (self.bool_storage != 7)
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :2, 0], [True, True]
        )

    def test_xor(self):
        view = (self.bool_storage == 7) ^ True  # [False, True] ^ True
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :2, 0], [True, False]
        )
        # [False, True] ^ [False, True]
        view = (self.bool_storage == 7) ^ (self.bool_storage == 7)
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :2, 0], [False, False]
        )
        # [False, True] ^ [True, False]
        view = (self.bool_storage == 7) ^ (self.bool_storage != 7)
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :2, 0], [True, True]
        )

    def test_isdata(self):
        view = raster.IsData(self.bool_storage)
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :3, 0], [True, True, False]
        )

        # cannot take IsData from boolean storage
        self.assertRaises(TypeError, raster.IsData, self.bool_storage == 7)

    def test_isnodata(self):
        view = raster.IsNoData(self.bool_storage)
        assert_equal(
            view.get_data(**self.vals_request)["values"][0, :3, 0], [False, False, True]
        )

        # cannot take IsData from boolean storage
        self.assertRaises(TypeError, raster.IsNoData, self.bool_storage == 7)

    def test_math_vals(self):
        view = raster.Add(self.storage, 2)
        vals = view.get_data(**self.vals_request)
        assert_equal(vals["values"], 3)
        assert_equal(vals["no_data_value"], view.fillvalue)

        view = raster.Add(self.storage, self.storage)
        vals = view.get_data(**self.vals_request)
        assert_equal(vals["values"], 2)
        assert_equal(vals["no_data_value"], view.fillvalue)

    def test_math_time(self):
        view = raster.Add(self.storage, 2)
        time = view.get_data(**self.time_request)
        self.assertEqual(time["time"], self.expected_time)

    def test_math_meta(self):
        view = raster.Add(self.storage, 2)
        meta = view.get_data(**self.meta_request)
        self.assertEqual(meta["meta"], self.expected_meta)

    def test_math_none(self):
        view = raster.Add(self.storage, 2)
        for mode in ["vals", "meta", "time"]:
            self.assertIsNone(
                view.get_data(
                    mode=mode, start=Datetime(2018, 1, 1), stop=Datetime(2018, 2, 2)
                )
            )

    def test_math_nodata(self):
        nodata = MockRaster(
            origin=self.storage.origin,
            timedelta=self.storage.timedelta,
            bands=self.storage.bands,
            value=255,
        )

        # nodata should propagate always
        for args in [(nodata, 2), (nodata, self.storage), (self.storage, nodata)]:
            view = raster.Divide(*args)
            result = view.get_data(**self.vals_request)
            assert_equal(result["values"], result["no_data_value"])

    def test_base_log_exp_init(self):
        # cannot take Exp from boolean storage
        arg = self.logexp_storage == 7
        self.assertRaises(TypeError, raster.elemwise.BaseLogExp, arg)

    def test_exp(self):
        view = raster.Exp(self.logexp_storage)
        n = view.fillvalue
        expected = [[1 / np.e, 1], [np.exp(np.e), np.exp(10)], [n, n]]
        assert_allclose(view.get_data(**self.vals_request)["values"][0], expected)

    def test_log_e(self):
        view = raster.Log(self.logexp_storage)
        n = view.fillvalue
        expected = [[n, n], [1, np.log(10)], [np.log(999), n]]
        assert_allclose(view.get_data(**self.vals_request)["values"][0], expected)

    def test_log_10(self):
        view = raster.Log10(self.logexp_storage)
        n = view.fillvalue
        expected = [[n, n], [np.log10(np.e), 1], [np.log10(999), n]]
        assert_allclose(view.get_data(**self.vals_request)["values"][0], expected)


class TestFillNoData(unittest.TestCase):
    klass = raster.FillNoData

    def setUp(self):
        self.storage_kwargs = dict(
            origin=Datetime(2000, 1, 1), timedelta=Timedelta(hours=1), bands=3
        )
        self.vals_request = dict(
            mode="vals",
            start=Datetime(2000, 1, 1),
            stop=Datetime(2010, 1, 1, 2),
            width=2,
            height=3,
        )

    def test_fill_nodata(self):
        storage = MockRaster(**self.storage_kwargs)
        nodata = MockRaster(value=255, **self.storage_kwargs)

        # nodata should be shadowed always
        for args in [(nodata, storage), (storage, nodata)]:
            view = self.klass(*args)
            result = view.get_data(**self.vals_request)
            assert_equal(result["values"], 1)

    def test_fill_priority(self):
        storage1 = MockRaster(value=1, **self.storage_kwargs)

        storage2 = MockRaster(value=2, **self.storage_kwargs)

        # the highest priority is on the right
        view = self.klass(storage2, storage1)
        result = view.get_data(**self.vals_request)
        assert_equal(result["values"], 1)

        # the highest priority is on the right
        view = self.klass(storage1, storage2)
        result = view.get_data(**self.vals_request)
        assert_equal(result["values"], 2)

    def test_fill_nodata_none_data(self):
        vals_request = dict(
            mode="vals",
            start=Datetime(2000, 1, 1),
            stop=Datetime(2010, 1, 1, 2),
            width=2,
            height=3,
        )
        args = (MockRaster(None), MockRaster(None))
        view = self.klass(*args)
        result = view.get_data(**vals_request)
        self.assertIsNone(result)


class TestCombine(unittest.TestCase):
    klass = raster.combine.BaseCombine

    def test_only_view_sources(self):
        storage1 = MockRaster(timedelta=Timedelta(hours=1))

        self.assertRaises(TypeError, self.klass, storage1, 1)
        self.assertRaises(TypeError, self.klass, storage1, "no storage")
        self.assertRaises(TypeError, self.klass, storage1, np.zeros(1))

    def test_propagate_timedelta(self):
        # equal timedeltas are propagated
        storage1 = MockRaster(
            origin=Datetime(2018, 4, 1), timedelta=Timedelta(hours=1), bands=3
        )
        storage2 = MockRaster(
            origin=Datetime(2018, 5, 1), timedelta=Timedelta(hours=1), bands=3
        )

        combined = self.klass(storage1, storage2)
        self.assertEqual(combined.timedelta, storage1.timedelta)

        # different timedeltas are allowed and result in None timedelta
        storage3 = MockRaster(
            origin=Datetime(2018, 4, 1), timedelta=Timedelta(hours=2), bands=3
        )
        combined = self.klass(storage1, storage3)
        self.assertIsNone(combined.timedelta)

        # empty rasters (with no period/timedelta) are ignored
        storage4 = MockRaster(origin=None, timedelta=None, bands=3)
        combined = self.klass(storage1, storage2, storage4)
        self.assertEqual(combined.timedelta, storage1.timedelta)

        # equal timedeltas are not propagated if origins are not aligned
        # an integer number of timedelta apart
        storage5 = MockRaster(
            origin=Datetime(2018, 4, 1, 0, 10), timedelta=Timedelta(hours=1), bands=3
        )

        combined = self.klass(storage1, storage5)
        self.assertIsNone(combined.timedelta)

    def test_propagate_period(self):
        storage1 = MockRaster(
            origin=Datetime(2018, 4, 1), timedelta=Timedelta(hours=1), bands=6
        )
        storage2 = MockRaster(
            origin=Datetime(2018, 4, 1, 2), timedelta=Timedelta(hours=1), bands=6
        )

        combined = self.klass(storage1, storage1)
        self.assertEqual(storage1.period, combined.period)

        # the period is combined
        combined = self.klass(storage1, storage2)
        self.assertEqual(combined.period[0], storage1.period[0])
        self.assertEqual(combined.period[1], storage2.period[1])

    def test_propagate_none_period(self):
        storage1 = MockRaster(origin=None)
        storage2 = MockRaster(
            origin=Datetime(2018, 4, 1), timedelta=Timedelta(hours=1), bands=6
        )

        combined = self.klass(storage1, storage1)
        self.assertIsNone(combined.period)

        # a period with a value precedes
        for args in [(storage1, storage2), (storage2, storage1)]:
            combined = self.klass(*args)
            self.assertEqual(combined.period, storage2.period)

    def test_propagate_extent(self):
        storage1 = MockRaster(value=np.empty((1, 2)))
        storage2 = MockRaster(value=np.empty((3, 4)))

        combined = self.klass(storage1, storage1)
        self.assertEqual(combined.extent, storage1.extent)

        # the combined extent equals the joined bbox
        for args in [(storage1, storage2), (storage2, storage1)]:
            combined = self.klass(*args)
            self.assertEqual(combined.extent, (0, 0, 4, 3))

    def test_propagate_geometry(self):
        storage1 = MockRaster(value=np.empty((1, 2)))
        storage2 = MockRaster(value=np.empty((3, 4)))

        # the combined extent equals the joined bbox
        for args in [(storage1, storage2), (storage2, storage1)]:
            combined = self.klass(*args)
            x1, x2, y1, y2 = combined.geometry.GetEnvelope()
            self.assertEqual((x1, y1, x2, y2), (0.0, 0.0, 4.0, 3.0))

    def test_propagate_geometry_different_projection(self):
        storage1 = MockRaster(projection="EPSG:3857")
        storage2 = MockRaster(projection="EPSG:4326")

        # the combined extent equals the joined bbox in the first store proj
        for args in [(storage1, storage2), (storage2, storage1)]:
            geometry = self.klass(*args).geometry
            self.assertEqual(
                args[0].projection, get_epsg_or_wkt(geometry.GetSpatialReference())
            )

    def test_propagate_projection(self):
        combined = self.klass(
            MockRaster(value=1, projection="EPSG:3857"),
            MockRaster(value=2, projection="EPSG:3857"),
        )
        self.assertEqual(combined.projection, "EPSG:3857")

        combined = self.klass(
            MockRaster(value=1, projection="EPSG:3857"),
            MockRaster(value=2, projection="EPSG:4326"),
        )
        self.assertIsNone(combined.projection)

        combined = self.klass(
            MockRaster(value=1, projection="EPSG:3857"),
            MockRaster(value=2, projection=None),
        )
        self.assertIsNone(combined.projection)

        combined = self.klass(
            MockRaster(value=1, projection=None), MockRaster(value=2, projection=None)
        )
        self.assertIsNone(combined.projection)

    def test_propagate_geo_transform(self):
        # matching geotransform propagates
        self.assertTupleEqual(
            self.klass(
                MockRasterWithGeotransform(geo_transform=(0, 1, 0, 1, 0, -1)),
                MockRasterWithGeotransform(geo_transform=(5, 1, 0, -8, 0, -1)),
            ).geo_transform,
            (0, 1, 0, 1, 0, -1),
        )

        # non-matching results in None
        self.assertIsNone(
            self.klass(
                MockRasterWithGeotransform(geo_transform=(0, 1, 0, 1, 0, -1)),
                MockRasterWithGeotransform(geo_transform=(0, 2, 0, 1, 0, -2)),
            ).geo_transform
        )

        # check None propagation
        self.assertIsNone(
            self.klass(
                MockRasterWithGeotransform(geo_transform=None),
                MockRasterWithGeotransform(geo_transform=(0, 1, 0, 1, 0, -1)),
            ).geo_transform
        )

        self.assertIsNone(
            self.klass(
                MockRasterWithGeotransform(geo_transform=(0, 1, 0, 1, 0, -1)),
                MockRasterWithGeotransform(geo_transform=None),
            ).geo_transform
        )

    def test_propagate_none_extent(self):
        storage1 = MockRaster(value=None)
        storage2 = MockRaster(value=np.empty((1, 2)))

        # an extent with a value precedes
        for args in [(storage1, storage2), (storage2, storage1)]:
            combined = self.klass(*args)
            self.assertEqual(combined.extent, storage2.extent)

    def test_empty_length(self):
        storage1 = MockRaster(origin=None)
        combined = self.klass(storage1)

        self.assertEqual(len(combined), 0)


class TestGroup(TestCombine, unittest.TestCase):
    klass = raster.Group

    def setUp(self):
        self.storage1 = MockRaster(
            origin=Datetime(2000, 1, 1), timedelta=Timedelta(minutes=5), bands=3
        )
        self.storage2 = MockRaster(
            origin=Datetime(2000, 1, 1), timedelta=Timedelta(minutes=3), bands=6
        )
        self.storage3 = MockRaster(
            origin=Datetime(2000, 1, 1), timedelta=Timedelta(minutes=5), bands=3
        )
        self.storage4 = MockRaster(origin=None)

        self.nodatastorage = MockRaster(
            origin=Datetime(2000, 1, 1),
            timedelta=Timedelta(minutes=5),
            bands=3,
            value=255,
        )
        self.vals_request = dict(
            mode="vals",
            start=Datetime(2000, 1, 1),
            width=2,
            stop=Datetime(2010, 1, 1, 2),
            height=3,
        )

        super(TestGroup, self).setUp()

    def test_group_by_time(self):
        view = self.klass(self.storage1, self.storage2, self.storage3, self.storage4)
        time = view.get_data(
            mode="time", start=Datetime(2000, 1, 1), stop=Datetime(2001, 1, 1)
        )["time"]
        self.assertEqual(
            time,
            [
                Datetime(2000, 1, 1, 0, 0),
                Datetime(2000, 1, 1, 0, 3),
                Datetime(2000, 1, 1, 0, 5),
                Datetime(2000, 1, 1, 0, 6),
                Datetime(2000, 1, 1, 0, 9),
                Datetime(2000, 1, 1, 0, 10),
                Datetime(2000, 1, 1, 0, 12),
                Datetime(2000, 1, 1, 0, 15),
            ],
        )

        meta = view.get_data(
            mode="meta", start=Datetime(2000, 1, 1), stop=Datetime(2001, 1, 1)
        )["meta"]
        expected = ["Testmeta for band {}".format(i) for i in (0, 1, 1, 2, 3, 2, 4, 5)]
        self.assertEqual(meta, expected)

        view.get_data(
            mode="vals",
            start=Datetime(2000, 1, 1),
            stop=Datetime(2001, 1, 1),
            width=1,
            height=1,
        )

    def test_group_by_bands(self):
        timedelta = self.storage1.timedelta
        storage5 = MockRaster(
            origin=self.storage1.origin + timedelta,
            timedelta=timedelta,
            bands=2,
            value=7,
        )
        # equal periods
        view = self.klass(self.storage1, self.storage4, storage5, self.nodatastorage)
        request = dict(start=Datetime(2000, 1, 1), stop=Datetime(2001, 1, 1))
        _requests = view.get_sources_and_requests(mode="meta", **request)
        self.assertEqual(_requests[0][0]["combine_mode"], "by_bands")

        time = view.get_data(mode="time", **request)["time"]
        self.assertEqual(
            time,
            [
                Datetime(2000, 1, 1, 0, 0),
                Datetime(2000, 1, 1, 0, 5),
                Datetime(2000, 1, 1, 0, 10),
            ],
        )
        meta = view.get_data(mode="meta", **request)["meta"]
        expected = ["Testmeta for band {}".format(i) for i in range(3)]
        self.assertEqual(meta, expected)

        data = view.get_data(mode="vals", width=1, height=1, **request)
        self.assertEqual(data["values"].tolist(), [[[1]], [[7]], [[7]]])

    def test_group_no_start(self):
        """Picks the lastmost frame"""
        view = self.klass(self.storage1, self.storage2, self.storage2)

        # data
        data = view.get_data(mode="vals", width=1, height=1)
        self.assertEqual(data["values"].tolist(), [[[1]]])

        # meta
        meta = view.get_data(mode="meta")["meta"]
        self.assertEqual(meta, ["Testmeta for band 5"])

        # time
        time = view.get_data(mode="time")["time"]
        self.assertEqual(time, [Datetime(2000, 1, 1, 0, 15)])

    def test_group_no_stop(self):
        """Picks the nearest frame"""
        view = self.klass(self.storage1, self.storage2)

        # data
        data = view.get_data(
            mode="vals", width=1, height=1, start=Datetime(2000, 1, 1, 0, 4)
        )
        self.assertEqual(data["values"].tolist(), [[[1]]])

        # data outside
        data = view.get_data(mode="vals", width=1, height=1, start=Datetime(2012, 1, 1))
        self.assertEqual(data["values"].tolist(), [[[1]]])

        # meta
        meta = view.get_data(mode="meta", start=Datetime(2000, 1, 1, 0, 13))["meta"]
        self.assertEqual(meta, ["Testmeta for band 4"])

        meta = view.get_data(mode="meta", start=Datetime(2012, 1, 1))["meta"]
        self.assertEqual(meta, ["Testmeta for band 5"])

        # time
        time = view.get_data(mode="time", start=Datetime(2000, 1, 1, 0, 7))["time"]
        self.assertEqual(time, [Datetime(2000, 1, 1, 0, 6)])

        # time outside
        time = view.get_data(mode="time", start=Datetime(2012, 1, 1))["time"]
        self.assertEqual(time, [Datetime(2000, 1, 1, 0, 15)])

    def test_group_no_result(self):
        view = self.klass(self.storage1, self.storage2, self.storage3, self.storage4)

        # data
        data = view.get_data(
            mode="vals",
            width=1,
            height=1,
            start=Datetime(2001, 1, 1),
            stop=Datetime(2002, 1, 1),
        )
        self.assertIsNone(data)

    def test_empty_group(self):
        view = self.klass(self.storage4)

        # data
        data = view.get_data(
            mode="vals",
            width=1,
            height=1,
            start=Datetime(2001, 1, 1),
            stop=Datetime(2002, 1, 1),
        )
        self.assertIsNone(data)

    def test_fill_nodata(self):
        storage = self.storage1
        nodata = self.nodatastorage

        # nodata should be shadowed always
        for args in [(nodata, storage), (storage, nodata)]:
            view = self.klass(*args)
            result = view.get_data(**self.vals_request)
            assert_equal(result["values"], 1)

    def test_fill_priority(self):
        storage1 = self.storage1  # value=1
        storage2 = MockRaster(
            origin=self.storage1.origin,
            timedelta=self.storage1.timedelta,
            bands=self.storage1.bands,
            value=2,
        )

        # the highest priority is on the right
        view = self.klass(storage2, storage1)
        result = view.get_data(**self.vals_request)
        assert_equal(result["values"], 1)

        # the highest priority is on the right
        view = self.klass(storage1, storage2)
        result = view.get_data(**self.vals_request)
        assert_equal(result["values"], 2)


class TestSnap(unittest.TestCase):
    klass = raster.Snap

    def setUp(self):
        self.raster = MockRaster(
            origin=Datetime(2000, 1, 1),
            value=7,
            timedelta=Timedelta(minutes=5),
            bands=3,
        )
        self.index = MockRaster(
            origin=Datetime(2000, 1, 1), timedelta=Timedelta(minutes=3), bands=6
        )
        self.empty = MockRaster(origin=None)

        self.vals_request = dict(
            mode="vals",
            start=Datetime(2000, 1, 1),
            stop=Datetime(2010, 1, 1, 2),
            width=2,
            height=3,
        )
        self.time_request = dict(
            mode="time", start=Datetime(2000, 1, 1), stop=Datetime(2001, 1, 1)
        )
        self.expected_time = [
            Datetime(2000, 1, 1) + i * Timedelta(hours=1) for i in range(3)
        ]
        self.meta_request = dict(
            mode="meta", start=Datetime(2000, 1, 1), stop=Datetime(2001, 1, 1)
        )
        self.expected_meta = ["Testmeta for band {}".format(i) for i in range(3)]

        self.view = self.klass(self.raster, self.index)

    def test_snap_attributes(self):
        # time related properties should come from index
        self.assertEqual(self.view.period, self.index.period)
        self.assertEqual(self.view.timedelta, self.index.timedelta)
        self.assertEqual(len(self.view), len(self.index))

    def test_snap_empty_store_or_index(self):
        view = self.klass(self.raster, self.empty)
        data = view.get_data(**self.vals_request)
        self.assertIsNone(data)

    def test_snap_no_result(self):
        for mode in ["vals", "meta", "time"]:
            data = self.view.get_data(
                mode=mode, start=Datetime(2001, 1, 1), stop=Datetime(2002, 1, 1)
            )
            self.assertIsNone(data)

    def test_snap_single_band(self):
        # data
        data = self.view.get_data(mode="vals", width=1, height=1)
        self.assertEqual(data["values"].tolist(), [[[7]]])

        # meta
        data = self.view.get_data(mode="meta")
        self.assertEqual(data["meta"], ["Testmeta for band 2"])

        # time
        data = self.view.get_data(mode="time")
        self.assertEqual(data["time"], [Datetime(2000, 1, 1, 0, 15)])

    def test_snap_multiband_data(self):
        view = self.view

        def t(x):
            return Datetime(2000, 1, 1, 0, x)

        # dig deep in one case
        data = view.get_data(mode="time", start=t(6), stop=t(9))
        self.assertEqual(
            data["time"], [Datetime(2000, 1, 1, 0, 6), Datetime(2000, 1, 1, 0, 9)]
        )
        data = view.get_data(mode="vals", start=t(6), stop=t(9))
        self.assertEqual(data["values"].tolist(), [[[7]], [[7]]])

        data = view.get_data(mode="meta", start=t(6), stop=t(9))
        self.assertEqual(data["meta"], ["Testmeta for band 1", "Testmeta for band 2"])

        # test only meta in the other cases
        # expand left
        data = view.get_data(start=t(6), stop=t(7), mode="meta")
        self.assertEqual(data["meta"], ["Testmeta for band 1"])

        # expand right
        data = view.get_data(start=t(8), stop=t(9), mode="meta")
        self.assertEqual(data["meta"], ["Testmeta for band 2"])

        # expand left repeat
        data = view.get_data(start=t(12), stop=t(15), mode="meta")
        self.assertEqual(data["meta"], ["Testmeta for band 2", "Testmeta for band 2"])

        # both right and left
        data = view.get_data(start=t(5), stop=t(10), mode="meta")
        self.assertEqual(data["meta"], ["Testmeta for band 1", "Testmeta for band 2"])

        # left time, no data
        data = view.get_data(start=t(7), stop=t(9), mode="meta")
        self.assertEqual(data["meta"], ["Testmeta for band 2"])

        # right time, no data
        data = view.get_data(start=t(6), stop=t(8), mode="meta")
        self.assertEqual(data["meta"], ["Testmeta for band 1"])

        # inner time, no data
        # needs view inversed
        view = self.klass(self.index, self.raster)
        data = view.get_data(start=t(3), stop=t(5), mode="meta")
        self.assertEqual(data["meta"], ["Testmeta for band 2"])

    def test_snap_repeat(self):
        origin1 = Datetime(2000, 1, 1)
        timedelta = Timedelta(minutes=5)

        # repeat a store with a single frame
        store1 = MockRaster(origin=origin1, timedelta=timedelta, bands=1)
        store2 = MockRaster(origin=origin1, timedelta=timedelta, bands=3)
        view = self.klass(store1, store2)
        data = view.get_data(
            mode="meta", start=Datetime(2000, 1, 1), stop=Datetime(2001, 1, 1)
        )
        self.assertEqual(data["meta"], ["Testmeta for band 0"] * 3)


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


class TestBase(unittest.TestCase):
    def setUp(self):
        self.raster = MockRaster(
            origin=Datetime(2000, 1, 1),
            value=7,
            timedelta=Timedelta(minutes=5),
            bands=3,
        )
        self.raster_nodata = MockRaster(
            origin=Datetime(2000, 1, 1),
            timedelta=Timedelta(minutes=5),
            bands=3,
            value=255,
        )
        self.raster_none = MockRaster(origin=None)
        self.raster_array = MockRaster(
            origin=Datetime(2000, 1, 1),
            value=np.arange(0.0, 1.6, 0.1).reshape(4, 4),
            timedelta=Timedelta(minutes=5),
            bands=3,
        )

        self.point_request = dict(
            mode="vals",
            start=Datetime(2000, 1, 1),
            stop=Datetime(2000, 1, 1),
            width=1,
            height=1,
            bbox=(0, 0, 0, 0),
            projection="EPSG:3857",
        )
        self.vals_request = dict(
            mode="vals",
            start=Datetime(2000, 1, 1),
            stop=Datetime(2010, 1, 1, 2),
            width=2,
            height=3,
            bbox=(0, 0, 2, 3),
            projection="EPSG:3857",
        )
        self.none_request = dict(
            mode="vals",
            start=Datetime(2001, 1, 1),
            stop=Datetime(2001, 1, 1, 2),
            width=2,
            height=3,
            projection="EPSG:3857",
        )
        self.time_request = dict(
            mode="time", start=Datetime(2000, 1, 1), stop=Datetime(2001, 1, 1)
        )
        self.expected_time = [
            Datetime(2000, 1, 1) + i * Timedelta(minutes=5) for i in range(3)
        ]
        self.meta_request = dict(
            mode="meta", start=Datetime(2000, 1, 1), stop=Datetime(2001, 1, 1)
        )
        self.expected_meta = ["Testmeta for band {}".format(i) for i in range(3)]

    def test_base_view(self):
        # store and view
        original = self.raster
        view = raster.base.BaseSingle(store=original)

        # properties
        self.assertEqual(original.extent, view.extent)
        self.assertEqual(original.period, view.period)
        self.assertEqual(original.timedelta, view.timedelta)

    def test_shift(self):
        # store and view
        original = self.raster
        time = original.timedelta
        view = raster.Shift(store=original, time=time)
        self.assertEqual(original.timedelta, view.timedelta)
        self.assertEqual(view.period[0] - original.period[0], time)

        # query original
        start, stop = original.period
        original_data = original.get_data(mode="vals", start=start, stop=stop)
        original_meta = original.get_data(mode="meta", start=start, stop=stop)
        original_time = original.get_data(mode="time", start=start, stop=stop)

        # query view
        start, stop = view.period
        view_data = view.get_data(mode="vals", start=start, stop=stop)
        view_meta = view.get_data(mode="meta", start=start, stop=stop)
        view_time = view.get_data(mode="time", start=start, stop=stop)

        # check assertions
        self.assertTrue(np.equal(view_data["values"], original_data["values"]).all())
        self.assertEqual(view_meta["meta"], original_meta["meta"])
        self.assertEqual(view_time["time"], [t + time for t in original_time["time"]])

        # check construction with milliseconds
        view2 = raster.Shift(
            store=original, time=int(original.timedelta.total_seconds() * 1000)
        )
        self.assertEqual(view2.time, view.time)

    def test_mask(self):
        view = raster.Mask(store=self.raster, value=8)
        data = view.get_data(**self.vals_request)
        self.assertEqual(str(view.dtype), "uint8")
        assert_equal(data["values"], 8)

        # nodata is not masked to 0
        view = raster.Mask(store=self.raster_nodata, value=8)
        data = view.get_data(**self.vals_request)
        self.assertEqual(view.fillvalue, 0)
        assert_equal(data["values"], 0)
        assert_equal(data["no_data_value"], 0)

        # unless value is 0, then it becomes 1
        view = raster.Mask(store=self.raster_nodata, value=0)
        data = view.get_data(**self.vals_request)
        self.assertEqual(view.fillvalue, 1)
        assert_equal(data["values"], 1)
        assert_equal(data["no_data_value"], 1)

        self.assertEqual(view.get_data(**self.meta_request)["meta"], self.expected_meta)
        self.assertEqual(view.get_data(**self.time_request)["time"], self.expected_time)

        # the 'value' determines the dtype. 1000 becomes uint16.
        view = raster.Mask(store=self.raster, value=1000)
        data = view.get_data(**self.vals_request)
        self.assertEqual(str(view.dtype), "uint16")
        assert_equal(data["values"], 1000)

        # -1000 becomes int16.
        view = raster.Mask(store=self.raster, value=-1000)
        data = view.get_data(**self.vals_request)
        self.assertEqual(str(view.dtype), "int16")
        assert_equal(data["values"], -1000)

        # 3.14159 becomes float32.
        view = raster.Mask(store=self.raster, value=3.14159)
        data = view.get_data(**self.vals_request)
        self.assertEqual(str(view.dtype), "float32")
        assert_equal(data["values"], 3.14159)

    def test_mask_below(self):
        # filled result
        view = raster.MaskBelow(store=self.raster, value=0)
        data = view.get_data(**self.vals_request)
        assert_equal(data["values"], 7)

        # empty result
        data = view.get_data(**self.none_request)
        self.assertIsNone(data)

        # masked result
        view = raster.MaskBelow(store=self.raster, value=10)
        data = view.get_data(**self.vals_request)
        assert_equal(data["values"], 255)

        self.assertEqual(view.get_data(**self.meta_request)["meta"], self.expected_meta)
        self.assertEqual(view.get_data(**self.time_request)["time"], self.expected_time)

    def test_step(self):
        view = raster.Step(store=self.raster, value=0)
        view.get_data(**self.meta_request)
        view.get_data(**self.time_request)

        # empty result
        data = view.get_data(**self.none_request)
        self.assertIsNone(data)

        # right value result (store returns 7)
        view = raster.Step(store=self.raster, left=3, right=10, value=6)
        data = view.get_data(**self.vals_request)
        assert_equal(data["values"], 10)

        # left value result
        view = raster.Step(store=self.raster, left=3, right=10, value=8)
        data = view.get_data(**self.vals_request)
        assert_equal(data["values"], 3)

        # at value result
        view = raster.Step(store=self.raster, at=15, value=7)
        data = view.get_data(**self.vals_request)
        assert_equal(data["values"], 15)

        self.assertEqual(view.get_data(**self.meta_request)["meta"], self.expected_meta)
        self.assertEqual(view.get_data(**self.time_request)["time"], self.expected_time)

    def test_classify_meta_time(self):
        view = raster.Classify(store=self.raster, bins=[1, 2, 3])
        self.assertEqual(view.get_data(**self.meta_request)["meta"], self.expected_meta)
        self.assertEqual(view.get_data(**self.time_request)["time"], self.expected_time)

    def test_classify(self):
        values = np.array([[1, 5], [7, 10], [255, 255]], dtype=np.uint8)
        mockraster = MockRaster(
            origin=Datetime(2000, 1, 1),
            value=values,
            timedelta=Timedelta(minutes=5),
            bands=1,
        )
        view = raster.Classify(store=mockraster, bins=[3, 8])
        data = view.get_data(**self.vals_request)
        assert_equal(data["values"][0, :2], [[0, 1], [1, 2]])
        assert_equal(data["values"][0, 2], data["no_data_value"])
        self.assertEqual(view.fillvalue, data["no_data_value"])

    def test_classify_dtype(self):
        # 254 edges, 255 bins, 256 values: uint8
        view = raster.Classify(store=self.raster, bins=np.arange(254))
        self.assertEqual(view.dtype, np.uint8)
        # one more: uint16
        view = raster.Classify(store=self.raster, bins=np.arange(255))
        self.assertEqual(view.dtype, np.uint16)

    def test_dilate(self):
        values = np.array([[0, 2], [0, 0], [0, 0]])
        store = MockRaster(
            origin=Datetime(2000, 1, 1),
            value=values,
            timedelta=Timedelta(minutes=5),
            bands=1,
        )

        view = raster.Dilate(store=store, values=[2])

        # skip dilation
        data = view.get_data(**self.point_request)
        self.assertEqual(data["values"].tolist(), [[[0]]])

        # perform dilation
        data = view.get_data(**self.vals_request)
        self.assertEqual(data["values"].shape, (1, 3, 2))
        self.assertEqual(data["values"].tolist(), [[[2, 2], [0, 2], [0, 0]]])

        # dilate outside of bbox
        request = self.vals_request.copy()
        request["bbox"] = (1, 1, 2, 2)
        request["height"] = 1
        request["width"] = 1
        data = view.get_data(**request)
        self.assertEqual(data["values"].shape, (1, 1, 1))
        self.assertEqual(data["values"].tolist(), [[[2]]])

        # perform no dilation
        view = raster.Dilate(store=store, values=[1])
        data = view.get_data(**self.vals_request)
        self.assertEqual(data["values"].tolist(), [store.value.tolist()])

        # meta and time requests
        view = raster.Dilate(self.raster, values=[2])
        self.assertEqual(view.get_data(**self.meta_request)["meta"], self.expected_meta)
        self.assertEqual(view.get_data(**self.time_request)["time"], self.expected_time)

    def test_moving_max(self):
        values = np.array([[0, 2], [0, 0], [0, 0]])
        store = MockRaster(
            origin=Datetime(2000, 1, 1),
            value=values,
            timedelta=Timedelta(minutes=5),
            bands=1,
        )

        view = raster.MovingMax(store=store, size=3)

        # skip moving max
        data = view.get_data(**self.point_request)
        self.assertEqual(data["values"].tolist(), [[[0]]])

        # perform moving max
        data = view.get_data(**self.vals_request)
        self.assertEqual(data["values"].shape, (1, 3, 2))
        self.assertEqual(data["values"].tolist(), [[[2, 2], [2, 2], [0, 0]]])

        # moving max outside of bbox
        request = self.vals_request.copy()
        request["bbox"] = (1, 1, 2, 2)
        request["height"] = 1
        request["width"] = 1
        data = view.get_data(**request)
        self.assertEqual(data["values"].shape, (1, 1, 1))
        self.assertEqual(data["values"].tolist(), [[[2]]])

        # meta and time requests
        view = raster.MovingMax(self.raster, size=3)
        self.assertEqual(view.get_data(**self.meta_request)["meta"], self.expected_meta)
        self.assertEqual(view.get_data(**self.time_request)["time"], self.expected_time)

    def test_smooth(self):
        values = np.zeros((101, 101), dtype=np.float32)
        # 5x5 square in the center
        peak = 1000
        values[48:53, 48:53] = peak
        sigma = 5
        raster1 = MockRaster(
            origin=Datetime(2000, 1, 1),
            value=values,
            timedelta=Timedelta(minutes=5),
            bands=1,
        )
        view = raster.Smooth(store=raster1, size=sigma * 3)
        expected = ndimage.gaussian_filter(values, sigma=sigma, mode="constant", cval=0)

        request = self.vals_request.copy()

        # margins are large: approximate result
        request["bbox"] = (0, 0, 101, 101)
        request["height"] = 101
        request["width"] = 101
        data = view.get_data(**request)
        # Large tolerance for Scipy 0.17. Scipy 1.0 has better results because
        # of a solved rounding issue in zoom and affine_transform.
        assert_allclose(data["values"][0], expected, atol=peak * 0.1)

        # margins are small: exact result
        sigma = 1
        view = raster.Smooth(store=raster1, size=sigma * 3)
        expected = ndimage.gaussian_filter(values, sigma=sigma, mode="constant", cval=0)
        for bbox in (
            (0, 0, 101, 101),
            (0, 0, 48, 48),  # nonzero value is outside of bbox
            (50, 50, 60, 60),
        ):  # partial
            request["bbox"] = bbox
            request["height"] = bbox[3] - bbox[1]
            request["width"] = bbox[2] - bbox[0]
            data = view.get_data(**request)
            _expected = expected[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            assert_allclose(data["values"][0], _expected, atol=peak * 0.0001)

        # use EPSG4326 boxes
        request["projection"] = "EPSG:4326"
        for bbox in (
            (0, 0, 101, 101),
            (0, 0, 48, 48),  # nonzero value is outside of bbox
            (50, 50, 60, 60),
        ):  # partial
            request["height"] = bbox[3] - bbox[1]
            request["width"] = bbox[2] - bbox[0]
            extent = Extent(bbox, EPSG3857)
            request["bbox"] = extent.transformed(EPSG4326).bbox
            data = view.get_data(**request)
            _expected = expected[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            assert_allclose(data["values"][0], _expected, atol=peak * 0.0001)

        # meta and time requests
        view = raster.Smooth(self.raster, size=10)
        self.assertEqual(view.get_data(**self.meta_request)["meta"], self.expected_meta)
        self.assertEqual(view.get_data(**self.time_request)["time"], self.expected_time)

        # point request
        request["bbox"] = (50, 50, 50, 50)
        request["height"] = 1
        request["width"] = 1
        data = view.get_data(**request)
        data["values"][0, 0, 0] == peak

    def test_hill_shade(self):
        view = raster.HillShade(store=self.raster)
        self.assertEqual(view.dtype, "u1")

        # skip hillshade
        view.get_data(**self.point_request)

        # perform hillshade
        data = view.get_data(**self.vals_request)
        self.assertEqual(data["values"].shape, (3, 3, 2))

        # meta and time requests
        self.assertEqual(view.get_data(**self.meta_request)["meta"], self.expected_meta)
        self.assertEqual(view.get_data(**self.time_request)["time"], self.expected_time)

    def test_temporal_sum(self):
        view = raster.TemporalSum(store=self.raster)

        data = view.get_data(**self.none_request)
        self.assertIsNone(data)

        data = view.get_data(**self.vals_request)
        self.assertEqual(data["values"].shape, (1, 3, 2))
        self.assertEqual(data["values"][0, 0, 0].tolist(), 21)

        data = view.get_data(**self.time_request)
        self.assertEqual(self.expected_time[-1:], data["time"])

        data = view.get_data(**self.meta_request)
        self.assertEqual(self.expected_meta[-1:], data["meta"])


class TestRasterize(unittest.TestCase):
    def setUp(self):
        self.point_request = dict(
            mode="vals", width=1, height=1, bbox=(0, 0, 0, 0), projection="EPSG:3857"
        )
        self.vals_request = dict(
            mode="vals", width=2, height=3, bbox=(0, 0, 2, 3), projection="EPSG:3857"
        )
        squares = [
            ((0.0, 1.0), (0.0, 2.0), (1.0, 2.0), (1.0, 1.0)),  # 1 pixel inside
            ((10.0, 2.0), (10.0, 3.0), (20.0, 3.0), (20.0, 2.0)),  # outside
            ((1.0, 2.0), (1.0, 13.0), (12.0, 13.0), (12.0, 2.0)),  # partially inside
        ]
        properties = [{"id": x, "value": x / 3} for x in (51, 212, 512)]
        self.geometry_source = MockGeometry(squares, properties)
        self.view = raster.Rasterize(self.geometry_source, "id")

    def test_vals_request(self):
        data = self.view.get_data(**self.vals_request)

        # invert vertical axis so that x, y corresponds to j, i
        values = data["values"][0, ::-1]

        self.assertEqual(values[1, 0], 51)
        self.assertEqual(values[2, 1], 512)
        self.assertEqual(np.sum(values == data["no_data_value"]), 4)

    def test_overlapping(self):
        # last polygon is on top
        squares = [
            ((0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0)),  # full bbox
            ((0.0, 1.0), (0.0, 2.0), (1.0, 2.0), (1.0, 1.0)),  # 1 pixel
        ]
        view = raster.Rasterize(MockGeometry(squares), "id")
        data = view.get_data(**self.vals_request)
        values = data["values"][0]
        self.assertEqual(values[1, 0], 1)
        self.assertEqual(np.sum(values == 0), 5)

    def test_shifting_pixel(self):
        # we don't test the edge case (edge at 0.5) because it is ill-defined
        pixel = np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))

        # horizontal shift
        for offset in (0.0, 0.49, 0.51, 1.0):
            shifted = pixel + [offset, 0.0]
            view = raster.Rasterize(MockGeometry([shifted]), "id")
            data = view.get_data(**self.vals_request)

            if offset < 0.5:
                self.assertEqual(0, data["values"][0, 2, 0])
            else:
                self.assertEqual(0, data["values"][0, 2, 1])
            self.assertEqual(1, np.sum(data["values"] == 0))

        # vertical shift
        for offset in (0.0, 0.49, 0.51, 1.0):
            shifted = pixel + [0.0, offset]
            view = raster.Rasterize(MockGeometry([shifted]), "id")
            data = view.get_data(**self.vals_request)

            if offset < 0.5:
                self.assertEqual(0, data["values"][0, 2, 0])
            else:
                self.assertEqual(0, data["values"][0, 1, 0])
            self.assertEqual(1, np.sum(data["values"] == 0))

    def test_point_request(self):
        # a point requests returns the value of the last geometry that is
        # received by Rasterize from the source
        pixel = np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))

        # no geometry
        view = raster.Rasterize(MockGeometry([]), "id")
        data = view.get_data(**self.point_request)
        self.assertEqual([[[data["no_data_value"]]]], data["values"].tolist())

        # 2 geometries (numbered 0 and 1)
        view = raster.Rasterize(MockGeometry([pixel, pixel]), "id")
        data = view.get_data(**self.point_request)
        self.assertEqual([[[1]]], data["values"].tolist())

        # 2 geometries, with id_field
        view = raster.Rasterize(
            MockGeometry([pixel, pixel], [{"id": x} for x in (51, 212)]), "id"
        )
        data = view.get_data(**self.point_request)
        self.assertEqual([[[212]]], data["values"].tolist())

    def test_meta_time(self):
        # meta and time requests
        data = self.view.get_data(mode="time")
        self.assertEqual([Datetime(1970, 1, 1)], data["time"])

        data = self.view.get_data(mode="meta")
        self.assertEqual([None], data["meta"])

    def test_limit(self):
        # with limit
        view = raster.Rasterize(self.geometry_source, "id", limit=1)
        data = view.get_data(**self.vals_request)
        self.assertEqual(np.sum(data["values"] == data["no_data_value"]), 5)

    def test_rasterize_id(self):
        view = raster.Rasterize(self.geometry_source, column_name="id")
        data = view.get_data(**self.vals_request)
        # invert vertical axis so that x, y corresponds to j, i
        values = data["values"][0, ::-1]

        self.assertEqual(values.dtype, np.int32)
        self.assertEqual(values[1, 0], 51)
        self.assertEqual(values[2, 1], 512)
        self.assertEqual(np.sum(values == data["no_data_value"]), 4)

    def test_rasterize_id_as_uint(self):
        view = raster.Rasterize(self.geometry_source, column_name="id", dtype="uint8")
        data = view.get_data(**self.vals_request)
        # invert vertical axis so that x, y corresponds to j, i
        values = data["values"][0, ::-1]

        self.assertEqual(values.dtype, np.uint8)
        self.assertEqual(data["no_data_value"], 255)
        self.assertEqual(values[1, 0], np.uint8(51))
        self.assertEqual(values[2, 1], np.uint8(512))
        self.assertEqual(np.sum(values == data["no_data_value"]), 4)

    def test_rasterize_value(self):
        view = raster.Rasterize(
            self.geometry_source, column_name="value", dtype="float"
        )
        data = view.get_data(**self.vals_request)
        # invert vertical axis so that x, y corresponds to j, i
        values = data["values"][0, ::-1]

        self.assertEqual(values.dtype, np.float64)
        self.assertEqual(values[1, 0], 51 / 3)
        self.assertEqual(values[2, 1], 512 / 3)
        self.assertEqual(np.sum(values == data["no_data_value"]), 4)

    def test_rasterize_value_as_float16(self):
        view = raster.Rasterize(
            self.geometry_source, column_name="value", dtype="float16"
        )
        data = view.get_data(**self.vals_request)
        # invert vertical axis so that x, y corresponds to j, i
        values = data["values"][0, ::-1]

        self.assertEqual(values.dtype, np.float16)
        self.assertEqual(values[1, 0], np.float16(51 / 3))
        self.assertEqual(values[2, 1], np.float16(512 / 3))
        self.assertEqual(np.sum(values == data["no_data_value"]), 4)

    def test_geometry_request(self):
        (_, req), _ = self.view.get_sources_and_requests(
            mode="vals",
            width=256,
            height=100,
            bbox=(0, 0, 10, 10),
            projection="EPSG:3857",
            start=Datetime(2018, 1, 1),
            stop=Datetime(2019, 1, 1),
        )

        self.assertEqual("intersects", req["mode"])
        self.assertEqual(100.0, req["geometry"].area)
        self.assertEqual("EPSG:3857", req["projection"])
        self.assertEqual(10 / 256, req["min_size"])
        self.assertEqual(Datetime(2018, 1, 1), req["start"])
        self.assertEqual(Datetime(2019, 1, 1), req["stop"])
