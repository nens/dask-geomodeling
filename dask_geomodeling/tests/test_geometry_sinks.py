import os
import unittest
import sys

import geopandas as gpd
import pytest
import fiona
from pandas.util.testing import assert_frame_equal
from shapely.geometry import box

from dask_geomodeling.geometry import parallelize, sinks
from dask_geomodeling.tests.factories import (MockGeometry, setup_temp_root,
                                              teardown_temp_root)


class TestGeometryFileSink(unittest.TestCase):
    klass = sinks.GeometryFileSink

    @classmethod
    def setUpClass(cls):
        cls.root = setup_temp_root()

    @classmethod
    def tearDownClass(cls):
        teardown_temp_root(cls.root)

    def setUp(self):
        self.request = {
            "mode": "intersects",
            "projection": "EPSG:3857",
            "geometry": box(0, 0, 2, 2),
        }
        self.request_2 = {
            "mode": "intersects",
            "projection": "EPSG:3857",
            "geometry": box(10, 10, 12, 12),
        }
        self.path = os.path.join(self.root, self._testMethodName)
        self.polygons = [
            ((0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)),
            ((10.0, 10.0), (10.0, 12.0), (12.0, 12.0), (12.0, 10.0)),
        ]
        self.properties = [
            {"int": 5, "float": 3.2, "str": "bla"},
            {"int": 7, "float": 5.2, "str": "bla2"},
        ]
        self.source = MockGeometry(
            self.polygons, projection="EPSG:3857", properties=self.properties
        )
        self.expected = self.source.get_data(**self.request)["features"]
        self.expected_combined = self.source.get_data(
            mode="intersects", projection="EPSG:3857", geometry=box(0, 0, 12, 12)
        )["features"]

    def test_non_available_extension(self):
        with pytest.raises(ValueError):
            self.klass(self.source, self.path, "bmp")

    def test_geojson(self):
        block = self.klass(self.source, self.path, "geojson")
        block.get_data(**self.request)

        filename = [x for x in os.listdir(self.path) if x.endswith(".geojson")][0]
        actual = gpd.read_file(os.path.join(self.path, filename))

        # compare dataframes without checking the order of records / columns
        assert_frame_equal(actual, self.expected, check_like=True)
        # compare projections ('init' contains the EPSG code)
        assert actual.crs["init"] == self.expected.crs["init"]

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="The geopackage driver crashes on Win32"
    )
    def test_geopackage(self):
        block = self.klass(self.source, self.path, "gpkg")
        block.get_data(**self.request)

        filename = [x for x in os.listdir(self.path) if x.endswith(".gpkg")][0]
        actual = gpd.read_file(os.path.join(self.path, filename))

        # compare dataframes without checking the order of records / columns
        assert_frame_equal(actual, self.expected, check_like=True)
        # compare projections ('init' contains the EPSG code)
        assert actual.crs["init"] == self.expected.crs["init"]

    def test_shapefile(self):
        block = self.klass(self.source, self.path, "shp")
        block.get_data(**self.request)

        filename = [x for x in os.listdir(self.path) if x.endswith(".shp")][0]
        actual = gpd.read_file(os.path.join(self.path, filename))

        # compare dataframes without checking the order of records / columns
        assert_frame_equal(actual, self.expected, check_like=True)
        # compare projections ('init' contains the EPSG code)
        assert actual.crs["init"] == self.expected.crs["init"]

    @pytest.mark.skipif(
        "GML" not in fiona.supported_drivers,
        reason="This version of GDAL does not support GML writing."
    )
    def test_gml(self):
        block = self.klass(self.source, self.path, "gml")
        block.get_data(**self.request)

        filename = [x for x in os.listdir(self.path) if x.endswith(".gml")][0]
        actual = gpd.read_file(os.path.join(self.path, filename))

        # GML writer adds an 'fid' column
        del actual["fid"]

        # compare dataframes without checking the order of records / columns
        assert_frame_equal(actual, self.expected, check_like=True)
        # GML does not contain a CRS

    def test_fields_non_available(self):
        with pytest.raises(ValueError):
            self.klass(self.source, self.path, "shp", fields={"target": "nonexisting"})

    def test_fields(self):
        block = self.klass(
            self.source,
            self.path,
            "geojson",
            fields={"target": "str", "int1": "int", "int2": "int"},
        )
        block.get_data(**self.request)

        actual = gpd.read_file(os.path.join(self.path, os.listdir(self.path)[0]))
        assert set(actual.columns) == {"geometry", "target", "int1", "int2"}

    def test_merge_files(self):
        block = self.klass(self.source, self.path, "geojson")
        block.get_data(**self.request)
        block.get_data(**self.request_2)

        assert len(os.listdir(self.path)) == 2
        filename = os.path.join(self.root, "combined.geojson")
        sinks.GeometryFileSink.merge_files(self.path, filename)
        actual = gpd.read_file(filename)

        # compare dataframes without checking the order of records / columns
        assert_frame_equal(actual, self.expected_combined, check_like=True)
        # compare projections ('init' contains the EPSG code)
        assert actual.crs["init"] == self.expected_combined.crs["init"]

    def test_merge_files_cleanup(self):
        block = self.klass(self.source, self.path, "geojson")
        block.get_data(**self.request)
        block.get_data(**self.request_2)

        assert len(os.listdir(self.path)) == 2
        filename = os.path.join(self.root, "combined2.geojson")
        sinks.GeometryFileSink.merge_files(self.path, filename, remove_source=True)
        assert not os.path.isdir(self.path)

    def test_with_tiler(self):
        block = parallelize.GeometryTiler(
            self.klass(self.source, self.path, "geojson"),
            size=10.0,
            projection="EPSG:3857",
        )

        # this generates 4 tiles
        block.get_data(
            mode="centroid", projection="EPSG:3857", geometry=box(0, 0, 20, 20)
        )

        # but only 2 of them contain data
        assert len(os.listdir(self.path)) == 2

        for filename in os.listdir(self.path):
            df = gpd.read_file(os.path.join(self.path, filename))
            assert len(df) == 1
            assert df.crs["init"] == "epsg:3857"
