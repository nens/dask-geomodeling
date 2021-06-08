import os
import unittest
import json

import geopandas as gpd
import pytest
from shapely.geometry import box

from dask_geomodeling import utils
from dask_geomodeling.geometry import parallelize, sinks
from dask_geomodeling.geometry import Classify
from dask_geomodeling.tests.factories import (
    MockGeometry,
    setup_temp_root,
    teardown_temp_root,
)

try:
    from pandas.testing import assert_frame_equal
except ImportError:
    from pandas.util.testing import assert_frame_equal


def compare_crs(actual, expected):
    if utils.GEOPANDAS_GTE_0_7_0:
        assert actual == expected
    else:
        assert actual["init"] == expected["init"]


def assert_frame_equal_ignore_index(actual, expected, sort_col):
    assert_frame_equal(
        actual.set_index(sort_col).sort_index(),
        expected.set_index(sort_col).sort_index(),
        check_like=True,
    )


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
        self.request_tiled = {
            "mode": "centroid",
            "projection": "EPSG:3857",
            "geometry": box(0, 0, 20, 20),
        }
        self.path = os.path.join(self.root, self._testMethodName)
        self.polygons = [
            ((0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)),
            ((10.0, 10.0), (10.0, 12.0), (12.0, 12.0), (12.0, 10.0)),
        ]
        self.properties = [
            {"int": 5, "float": 3.2, "str": "bla", "lst": [1], "dct": {"a": "b"}},
            {"int": 7, "float": 5.2, "str": "bla2", "lst": [], "dct": {}},
        ]
        self.source = MockGeometry(
            self.polygons, projection="EPSG:3857", properties=self.properties
        )
        self.expected = self.source.get_data(**self.request)["features"]
        self.expected_combined = self.source.get_data(
            mode="intersects", projection="EPSG:3857", geometry=box(0, 0, 12, 12)
        )["features"]

        # we expect lists and dicts to be serialized to JSON
        self.expected["lst"] = self.expected["lst"].map(json.dumps)
        self.expected["dct"] = self.expected["dct"].map(json.dumps)
        self.expected_combined["lst"] = self.expected_combined["lst"].map(json.dumps)
        self.expected_combined["dct"] = self.expected_combined["dct"].map(json.dumps)

    @staticmethod
    def expected_to_geojson(expected):
        """The result from gpd.read_file for a geojson is different from the
        others for list and dict typed values:

        - lists are omitted
        - dicts are deserialized

        Call this function to transform self.expected for GeoJSON testing.
        """
        del expected["lst"]
        expected["dct"] = expected["dct"].map(json.loads)
        return utils.geodataframe_transform(expected, "EPSG:3857", "EPSG:4326")

    def test_non_available_extension(self):
        with pytest.raises(ValueError):
            self.klass(self.source, self.path, "bmp")

    def test_geojson(self):
        block = self.klass(self.source, self.path, "geojson")
        block.get_data(**self.request)

        filename = [x for x in os.listdir(self.path) if x.endswith(".geojson")][0]
        path = os.path.join(self.path, filename)
        actual = gpd.read_file(path)

        # gpd.read_file doesn't read list values. test it separately:
        with open(path) as f:
            data = json.load(f)
            assert data["features"][0]["properties"]["lst"] == [1]
        expected = self.expected_to_geojson(self.expected)

        # compare dataframes without checking the order of records / columns
        assert_frame_equal_ignore_index(actual, expected, "int")
        # compare projections
        compare_crs(actual.crs, expected.crs)

    @pytest.mark.skipif(
        "gpkg" not in sinks.GeometryFileSink.supported_extensions,
        reason="This version of Fiona/GDAL does not support geopackages.",
    )
    def test_geopackage(self):
        block = self.klass(self.source, self.path, "gpkg")
        block.get_data(**self.request)

        filename = [x for x in os.listdir(self.path) if x.endswith(".gpkg")][0]
        actual = gpd.read_file(os.path.join(self.path, filename))

        # compare dataframes without checking the order of records / columns
        assert_frame_equal_ignore_index(actual, self.expected, "int")
        # compare projections
        compare_crs(actual.crs, self.expected.crs)

    def test_shapefile(self):
        block = self.klass(self.source, self.path, "shp")
        block.get_data(**self.request)

        filename = [x for x in os.listdir(self.path) if x.endswith(".shp")][0]
        actual = gpd.read_file(os.path.join(self.path, filename))

        # compare dataframes without checking the order of records / columns
        assert_frame_equal_ignore_index(actual, self.expected, "int")
        # compare projections
        compare_crs(actual.crs, self.expected.crs)

    @pytest.mark.skipif(
        "gml" not in sinks.GeometryFileSink.supported_extensions,
        reason="This version of Fiona/GDAL does not support GML.",
    )
    def test_gml(self):
        block = self.klass(self.source, self.path, "gml")
        block.get_data(**self.request)

        filename = [x for x in os.listdir(self.path) if x.endswith(".gml")][0]
        actual = gpd.read_file(os.path.join(self.path, filename))

        # GML writer adds an 'fid' column
        del actual["fid"]

        # compare dataframes without checking the order of records / columns
        assert_frame_equal_ignore_index(actual, self.expected, "int")

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

        expected = self.expected_to_geojson(self.expected_combined)

        # compare dataframes without checking the order of records / columns
        assert_frame_equal_ignore_index(actual, expected, "int")
        # compare projections
        compare_crs(actual.crs, expected.crs)

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
        block.get_data(**self.request_tiled)

        # but only 2 of them contain data
        assert len(os.listdir(self.path)) == 2

        for filename in os.listdir(self.path):
            df = gpd.read_file(os.path.join(self.path, filename))
            assert len(df) == 1

    def test_categorical_column(self):
        with_categorical = self.source.set(
            "categorical", Classify(self.source["float"], bins=[6], labels=["A", "B"])
        )
        block = self.klass(
            with_categorical, self.path, "geojson", fields={"label": "categorical"}
        )
        block.get_data(**self.request)

        actual = gpd.read_file(os.path.join(self.path, os.listdir(self.path)[0]))
        assert actual["label"].tolist() == ["A"]

    def test_to_file_geojson(self):
        self.source.to_file(self.path + ".geojson", **self.request)
        actual = gpd.read_file(self.path + ".geojson")

        expected = self.expected_to_geojson(self.expected)

        # compare dataframes without checking the order of records / columns
        assert_frame_equal_ignore_index(actual, expected, "int")

    def test_to_file_shapefile(self):
        self.source.to_file(self.path + ".shp", **self.request)
        actual = gpd.read_file(self.path + ".shp")
        # compare dataframes without checking the order of records / columns
        assert_frame_equal_ignore_index(actual, self.expected, "int")

    def test_to_file_with_tiling_geojson(self):
        self.source.to_file(self.path + ".geojson", tile_size=10, **self.request_tiled)
        actual = gpd.read_file(self.path + ".geojson")
        # because we lose the index in the saving process, just check the len
        assert len(actual) == 2

    def test_to_file_dry_run(self):
        self.source.to_file(self.path + ".geojson", dry_run=True, **self.request)
        assert not os.path.exists(self.path)

    def test_to_file_with_tiling_shapefile(self):
        self.source.to_file(self.path + ".shp", tile_size=10, **self.request_tiled)
        actual = gpd.read_file(self.path + ".shp")
        # because we lose the index in the saving process, just check the len
        assert len(actual) == 2
