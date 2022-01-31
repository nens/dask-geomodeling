from unittest import mock
import unittest
import pytest
import sys

from dask import config
from osgeo import osr
from shapely import geometry
from shapely.geometry import box
from numpy.testing import assert_almost_equal, assert_array_equal

import numpy as np
import pandas as pd
import geopandas as gpd

from dask_geomodeling import utils


class TestUtils(unittest.TestCase):
    def test_get_index(self):
        index = utils.get_index(values=np.array([0, 1]), no_data_value=1)
        self.assertTrue((index == np.array([True, False])).all())

    def test_extent(self):
        sr = utils.get_sr("EPSG:4326")
        extent = utils.Extent(sr=sr, bbox=(0, 0, 1, 1))
        geometry = extent.as_geometry()
        self.assertEqual(str(geometry), "POLYGON ((0 0,1 0,1 1,0 1,0 0))")
        self.assertEqual(str(geometry.GetSpatialReference()), str(sr))

    def test_extent_has_repr(self):
        sr = utils.get_sr("EPSG:4326")
        extent = utils.Extent(sr=sr, bbox=(0, 0, 1, 1))
        self.assertTrue(repr(extent))

    def test_get_dtype_max(self):
        self.assertIsInstance(utils.get_dtype_max("f4"), float)
        self.assertIsInstance(utils.get_dtype_max("u4"), int)

    def test_get_dtype_min(self):
        self.assertIsInstance(utils.get_dtype_min("f4"), float)
        self.assertIsInstance(utils.get_dtype_min("u4"), int)

    def test_get_int_dtype(self):
        for dtype in ["i1", "i2", "i4", "i8"]:
            hi = np.iinfo(dtype).max
            lo = np.iinfo(dtype).max
            self.assertEqual(utils.get_int_dtype(hi - 1), dtype)
            self.assertEqual(utils.get_int_dtype(lo), dtype)

    def test_get_uint_dtype(self):
        self.assertRaises(ValueError, utils.get_uint_dtype, -1)
        for dtype in ["u1", "u2", "u4", "u8"]:
            hi = np.iinfo(dtype).max
            self.assertEqual(utils.get_uint_dtype(hi - 1), dtype)

    def test_get_projection(self):
        projection_rd = str("EPSG:28992")
        projection_wgs = str("EPSG:4326")
        rd = osr.SpatialReference(osr.GetUserInputAsWKT(projection_rd))
        self.assertEqual(utils.get_projection(rd), projection_rd)
        wgs = osr.SpatialReference(osr.GetUserInputAsWKT(projection_wgs))
        self.assertEqual(utils.get_projection(wgs), projection_wgs)

    def test_get_sr(self):
        wkt = """GEOGCS["GCS_WGS_1984",
                     DATUM["D_WGS_1984",SPHEROID[
                         "WGS_1984",6378137,298.257223563]],
                     PRIMEM["Greenwich",0],
                     UNIT["Degree",
                         0.017453292519943295]]"""
        self.assertTrue(utils.get_sr(wkt).ExportToProj4())

    def test_get_epsg_or_wkt(self):
        # epsg
        wkt = osr.GetUserInputAsWKT(str("EPSG:3857"))
        out = utils.get_epsg_or_wkt(wkt)
        self.assertEqual(out, "EPSG:3857")
        # no epsg
        wkt = """GEOGCS["GCS_WGS_1984",
                        DATUM["D_WGS_1984",SPHEROID[
                            "WGS_1984",6378137,298.257223563]],
                        PRIMEM["Greenwich",0],
                        UNIT["Degree",
                            0.017453292519943295]]"""
        out = utils.get_epsg_or_wkt(wkt)
        # do not test exact equality: different GDAL versions give different
        # results
        assert out.startswith("GEOGCS")
        assert 'PRIMEM["Greenwich",0]' in out
        assert 'UNIT["Degree"' in out

    def test_get_footprint(self):
        output = utils.get_footprint(size=5)
        reference = np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ],
            dtype="b1",
        )
        self.assertTrue(np.equal(output, reference).all())

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_safe_file_url(self):
        f = utils.safe_file_url
        if not sys.platform.startswith("win"):
            # prepends file:// if necessary
            assert f("/tmp") == "file:///tmp"
            assert f("/tmp", "/") == "file:///tmp"

            # absolute input
            assert f("file:///tmp") == "file:///tmp"
            assert f("file:///tmp", "/") == "file:///tmp"
            assert f("file://tmp", "/") == "file:///tmp"

            # relative input
            assert f("path", "/tmp/abs") == "file:///tmp/abs/path"
            assert f("../abs/path", "/tmp/abs") == "file:///tmp/abs/path"

            # raise on unknown protocol
            with pytest.raises(NotImplementedError):
                f("unknown://tmp")

            # paths outside of 'start'
            assert f("file://../x", "/tmp") == "file:///x"
            assert f("/etc/abs", "/tmp") == "file:///etc/abs"
            assert f("../", "/tmp") == "file:///"

            # raise on path outside start when strict-file-paths=True
            with config.set({"geomodeling.strict-file-paths": True}):
                with pytest.raises(IOError):
                    f("file://../x", "/tmp")
                with pytest.raises(IOError):
                    f("/etc/abs", "/tmp")
                with pytest.raises(IOError):
                    f("../", "/tmp")
        else:
            # prepends file:// if necessary
            assert f("C:\\tmp") == "file://C:\\tmp"
            assert f("C:\\tmp", "C:\\") == "file://C:\\tmp"

            # absolute input
            assert f("file://C:\\tmp") == "file://C:\\tmp"
            assert f("file://C:\\tmp", "C:\\") == "file://C:\\tmp"
            assert f("file://tmp", "C:\\") == "file://C:\\tmp"

            # relative input
            assert f("path", "C:\\tmp\\abs") == "file://C:\\tmp\\abs\\path"
            assert f("..\\abs\\path", "C:\\tmp\\abs") == "file://C:\\tmp\\abs\\path"

            # raise on unknown protocol
            with pytest.raises(NotImplementedError):
                f("unknown://tmp")

            # paths outside of 'start'
            assert f("file://..\\x", "C:\\tmp") == "file://C:\\x"
            assert f("D:\\tmp", "C:\\tmp") == "file://D:\\tmp"
            assert f("..\\", "C:\\tmp") == "file://C:\\"

            # raise on path outside start when strict-file-paths=True
            with config.set({"geomodeling.strict-file-paths": True}):
                with pytest.raises(IOError):
                    f("file://..\\x", "C:\\tmp")
                with pytest.raises(IOError):
                    f("D:\\tmp", "C:\\tmp")
                with pytest.raises(IOError):
                    f("..\\", "C:\\tmp")

    def test_get_crs(self):
        if utils.GEOPANDAS_GTE_0_7_0:
            from pyproj import CRS

            expected_type = CRS
        else:
            expected_type = dict

        # from EPSG
        epsg = "EPSG:28992"
        crs = utils.get_crs(epsg)
        self.assertIsInstance(crs, expected_type)

        # from proj4
        proj4 = """
            +proj=sterea +lat_0=52.15616055555555 +lon_0=5.38763888888889
            +k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel
            +towgs84=565.2369,50.0087,465.658,-0.406857,0.350733,-1.87035,4.0812
            +units=m +no_defs
        """
        crs = utils.get_crs(proj4)
        self.assertIsInstance(crs, expected_type)

    def test_shapely_transform(self):
        src_srs = "EPSG:28992"
        dst_srs = "EPSG:4326"
        box28992 = geometry.box(100000, 400000, 200000, 500000)
        box4326 = utils.shapely_transform(box28992, src_srs=src_srs, dst_srs=dst_srs)
        assert_almost_equal((5.4, 52.0), list(box4326.centroid.coords)[0], decimal=1)

    def test_shapely_transform_invalid(self):
        src_srs = "EPSG:4326"
        dst_srs = "EPSG:28992"
        box4326 = geometry.box(100000, 400000, 200000, 500000)
        with pytest.raises(utils.TransformException):
            utils.shapely_transform(box4326, src_srs=src_srs, dst_srs=dst_srs)

    @mock.patch("dask_geomodeling.utils.shapely_transform")
    def test_min_size_transform(self, shapely_transform):
        min_size = 100
        src_srs = "some_srs"
        dst_srs = "another_srs"
        box = geometry.box(0, 0, 100, 200)
        shapely_transform.return_value = box
        result = utils.transform_min_size(
            min_size=min_size, geometry=box, src_srs=src_srs, dst_srs=dst_srs
        )
        shapely_transform.assert_called_with(
            box.centroid.buffer(min_size / 2), src_srs=src_srs, dst_srs=dst_srs
        )
        self.assertEqual(200, result)

    @mock.patch("dask_geomodeling.utils.shapely_transform")
    def test_transform_extent(self, shapely_transform):
        src_srs = "some_fiona_crs"
        dst_srs = "another_fiona_crs"
        extent = 0, 0, 1, 1
        expected = 2, 2, 3, 3
        shapely_transform.return_value = geometry.box(*expected)
        result = utils.transform_extent(extent, src_srs=src_srs, dst_srs=dst_srs)
        shapely_transform.assert_called_with(
            geometry.box(*extent), src_srs=src_srs, dst_srs=dst_srs
        )
        self.assertEqual(expected, result)


class TestGeoTransform(unittest.TestCase):
    def setUp(self):
        self.geotransform = utils.GeoTransform((190000, 1, 0, 450000, 0, -1))

    def test_get_indices_for_bbox(self):
        indices = self.geotransform.get_indices_for_bbox(
            (190001, 449996, 190006, 450000)
        )

        # X ranges between 1 and 6, measured from 190000
        # Y ranges between 0 and -4, measured from 450000, but note
        # that dy in the geotransform is -1 so this comes out as 1 in
        # indices.

        # indices must be of the form ((y1, y2), (x1, x2))
        self.assertEqual(indices, ((0, 4), (1, 6)))

    def test_invalid_raises(self):
        # with zero pixel size
        self.assertRaises(ValueError, utils.GeoTransform, (0, 1, 0, 0, 0, 0))
        self.assertRaises(ValueError, utils.GeoTransform, (0, 0, 0, 0, 0, 1))
        # with a tilt
        self.assertRaises(ValueError, utils.GeoTransform, (0, 1, 1, 0, 0, 1))
        self.assertRaises(ValueError, utils.GeoTransform, (0, 1, 0, 0, 1, 1))
        # allow some space for float32 imprecision
        self.assertRaises(ValueError, utils.GeoTransform, (0, 1, 0, 0, 1e-6, 1))
        self.assertRaises(ValueError, utils.GeoTransform, (0, 1, 1e-6, 0, 0, 1))
        utils.GeoTransform((0, 1, 1e-8, 0, -1e-8, 1))

    def test_compare(self):
        for matching in (
            self.geotransform,
            (0, 1, 0, 0, 0, -1),
            (0, 1, 0, 0, 0, 1),
            (251, 1, 0, -4926, 0, -1),
        ):
            self.assertTrue(self.geotransform.aligns_with(matching))

        for non_matching in (
            (0, 2, 0, 0, 0, -1),
            (0, 1, 0, 0, 0, 2),
            (251.1, 1, 0, 0, 0, -1),
            (251.1, 1, 0, 1.6, 0, -1),
        ):
            self.assertFalse(self.geotransform.aligns_with(non_matching))


class TestRasterize(unittest.TestCase):
    def setUp(self):
        self.geoseries = gpd.GeoSeries([box(2, 2, 4, 4), box(6, 6, 8, 8)])
        self.box = dict(
            bbox=(0, 0, 10, 10), projection="EPSG:28992", width=10, height=10
        )
        self.point_in = dict(
            bbox=(3, 3, 3, 3), projection="EPSG:28992", width=1, height=1
        )
        self.point_out = dict(
            bbox=(5, 5, 5, 5), projection="EPSG:28992", width=1, height=1
        )

    def test_rasterize(self):
        raster = utils.rasterize_geoseries(self.geoseries, **self.box)
        values = raster["values"]
        self.assertEqual(bool, values.dtype)
        assert_array_equal(values[2:4, 2:4], True)
        assert_array_equal(values[6:8, 6:8], True)
        self.assertEqual(2 * 2 * 2, values.sum())

    def test_rasterize_point_true(self):
        raster = utils.rasterize_geoseries(self.geoseries, **self.point_in)
        self.assertTupleEqual(raster["values"].shape, (1, 1, 1))
        assert_array_equal(raster["values"], True)

    def test_rasterize_point_false(self):
        raster = utils.rasterize_geoseries(self.geoseries, **self.point_out)
        self.assertTupleEqual(raster["values"].shape, (1, 1, 1))
        assert_array_equal(raster["values"], False)

    def test_rasterize_none_geometry(self):
        self.geoseries.iloc[1] = None
        raster = utils.rasterize_geoseries(self.geoseries, **self.box)
        self.assertEqual(2 * 2, raster["values"].sum())

    def test_rasterize_int(self):
        raster = utils.rasterize_geoseries(
            self.geoseries, values=pd.Series([1, 2]), **self.box
        )
        values = raster["values"]
        self.assertEqual(np.int32, values.dtype)
        assert_array_equal(values[2:4, 2:4], 1)
        assert_array_equal(values[6:8, 6:8], 2)
        self.assertEqual(2 * 2 * 2, (values != raster["no_data_value"]).sum())

    def test_rasterize_int_point(self):
        raster = utils.rasterize_geoseries(
            self.geoseries, values=pd.Series([1, 2]), **self.point_in
        )
        self.assertTupleEqual(raster["values"].shape, (1, 1, 1))
        assert_array_equal(raster["values"], 1)

    def test_rasterize_int_point_nodata(self):
        raster = utils.rasterize_geoseries(
            self.geoseries, values=pd.Series([1, 2]), **self.point_out
        )
        self.assertTupleEqual(raster["values"].shape, (1, 1, 1))
        assert_array_equal(raster["values"], raster["no_data_value"])

    def test_rasterize_float(self):
        raster = utils.rasterize_geoseries(
            self.geoseries, values=pd.Series([1.2, 2.4]), **self.box
        )
        values = raster["values"]
        self.assertEqual(np.float64, values.dtype)
        assert_array_equal(values[2:4, 2:4], 1.2)
        assert_array_equal(values[6:8, 6:8], 2.4)
        self.assertEqual(2 * 2 * 2, (values != raster["no_data_value"]).sum())

    def test_rasterize_float_point(self):
        raster = utils.rasterize_geoseries(
            self.geoseries, values=pd.Series([1.2, 2.4]), **self.point_in
        )
        self.assertTupleEqual(raster["values"].shape, (1, 1, 1))
        assert_array_equal(raster["values"], 1.2)

    def test_rasterize_float_point_nodata(self):
        raster = utils.rasterize_geoseries(
            self.geoseries, values=pd.Series([1.2, 2.4]), **self.point_out
        )
        self.assertTupleEqual(raster["values"].shape, (1, 1, 1))
        assert_array_equal(raster["values"], raster["no_data_value"])

    def test_rasterize_float_nan_inf(self):
        raster = utils.rasterize_geoseries(
            self.geoseries, values=pd.Series([np.nan, np.inf]), **self.box
        )
        values = raster["values"]
        self.assertEqual(np.float64, values.dtype)
        self.assertEqual(0, (values != raster["no_data_value"]).sum())

    def test_rasterize_bool(self):
        raster = utils.rasterize_geoseries(
            self.geoseries, values=pd.Series([True, False]), **self.box
        )
        values = raster["values"]
        self.assertEqual(bool, values.dtype)
        assert_array_equal(values[2:4, 2:4], True)
        assert_array_equal(values[6:8, 6:8], False)
        self.assertEqual(2 * 2, values.sum())

    def test_rasterize_bool_empty(self):
        raster = utils.rasterize_geoseries(
            self.geoseries, values=pd.Series([False, False]), **self.box
        )
        values = raster["values"]
        self.assertEqual(bool, values.dtype)
        self.assertEqual(0, values.sum())

    def test_rasterize_categorical_int(self):
        raster = utils.rasterize_geoseries(
            self.geoseries, values=pd.Series([1, 2], dtype="category"), **self.box
        )
        self.assertEqual(np.int32, raster["values"].dtype)

    def test_rasterize_categorical_float(self):
        raster = utils.rasterize_geoseries(
            self.geoseries, values=pd.Series([1.2, 2.4], dtype="category"), **self.box
        )
        self.assertEqual(np.float64, raster["values"].dtype)
