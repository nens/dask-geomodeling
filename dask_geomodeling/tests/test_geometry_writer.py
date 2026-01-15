import os
import unittest
import json
import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import box
from geopandas.testing import assert_geodataframe_equal
from dask_geomodeling import utils
from dask_geomodeling.geometry import GeometryFileWriter
from dask_geomodeling.tests.factories import (
    MockGeometry,
    setup_temp_root,
    teardown_temp_root,
)



class TestGeometryFileWriter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = setup_temp_root()

    @classmethod
    def tearDownClass(cls):
        teardown_temp_root(cls.root)

    def setUp(self):
        self.path = os.path.join(self.root, self._testMethodName)
        
        # Construct GeoDataFrame directly
        from shapely.geometry import Polygon
        import pandas as pd
        
        self.features = gpd.GeoDataFrame(
            {
                "int": [5, 7],
                "float": [3.2, 5.2],
                "str": ["bla", "bla2"],
                "lst": [[1], [2]],
                "dct": [{"a": "b"}, {}],
                "cat": pd.Categorical(["A", "B"], categories=["A", "B"]),
                "geometry": [
                    Polygon([(0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)]),
                    Polygon([(10.0, 10.0), (10.0, 12.0), (12.0, 12.0), (12.0, 10.0)]),
                ],
            },
            index=pd.Index([2, 15], name="fid"),
            crs="EPSG:3857",
        )
        
        self.features_tiled = self.features
        self.features_categorical = self.features
        
        # Expected data for assertions
        self.expected = self.features.copy()
        # Categorical columns are converted to their underlying dtype when written
        self.expected["cat"] = self.expected["cat"].astype(str)

    def test_non_available_extension(self):
        with pytest.raises(KeyError):
            writer = GeometryFileWriter(self.path + ".bmp")
            writer.write(self.features)

    def test_geojson(self):
        writer = GeometryFileWriter(self.path + ".geojson")
        writer.write(self.features)

        actual = gpd.read_file(self.path + ".geojson", fid_as_index=True)

        self.expected = self.expected.to_crs("EPSG:4326")
        # Geopandas/pyogrio reads 'id' as index and as field
        self.expected.insert(6, "id", self.expected.index)
        # GeoJSON is read as int32
        for int_col in ["id", "int"]:
            self.expected[int_col] = self.expected[int_col].astype('int32')

        # Because of the reprojection, use a precision comparison for geometries
        actual.geometry = actual.geometry.set_precision(0.000001)
        self.expected.geometry = self.expected.geometry.set_precision(0.000001)
        assert_geodataframe_equal(actual, self.expected)
        assert actual.crs == "EPSG:4326"

    @pytest.mark.skipif(
        "gpkg" not in GeometryFileWriter.supported_extensions,
        reason="This version of Pyogrio/GDAL does not support geopackages.",
    )
    def test_geopackage(self):
        writer = GeometryFileWriter(self.path + ".gpkg")
        writer.write(self.features)

        actual = gpd.read_file(self.path + ".gpkg", fid_as_index=True)

        # we expect lists and dicts to be serialized to JSON
        self.expected["lst"] = self.expected["lst"].map(json.dumps)
        self.expected["dct"] = self.expected["dct"].map(json.dumps)

        assert_geodataframe_equal(actual, self.expected)
        # compare projections
        assert actual.crs == self.expected.crs

    def test_shapefile(self):
        writer = GeometryFileWriter(self.path + ".shp")
        writer.write(self.features)

        actual = gpd.read_file(self.path + ".shp", fid_as_index=True)
        # We don't write the index for shapefiles, instead it is stored as 'fid' field
        self.expected.insert(6, "fid", self.expected.index)
        self.expected.index = [0, 1]
        self.expected.index.names = ["fid"]
        # we expect lists and dicts to be serialized to JSON
        self.expected["lst"] = self.expected["lst"].map(json.dumps)
        self.expected["dct"] = self.expected["dct"].map(json.dumps)

        # compare dataframes without checking the order of records / columns
        assert_geodataframe_equal(actual, self.expected)
        # compare projections
        assert actual.crs == self.expected.crs

    @pytest.mark.skipif(
        "gml" not in GeometryFileWriter.supported_extensions,
        reason="This version of Pyogrio/GDAL does not support GML.",
    )
    def test_gml(self):
        writer = GeometryFileWriter(self.path + ".gml")
        writer.write(self.features)

        actual = gpd.read_file(self.path + ".gml", fid_as_index=True)
        del actual["gml_id"]
        # We don't write the index for shapefiles, instead it is stored as 'fid' field
        self.expected.insert(6, "fid", self.expected.index)
        self.expected.index = [0, 1]
        self.expected.index.names = ["fid"]
        # we expect lists and dicts to be serialized to JSON
        self.expected["lst"] = self.expected["lst"].map(json.dumps)
        self.expected["dct"] = self.expected["dct"].map(json.dumps)

        # compare dataframes without checking the order of records / columns
        assert_geodataframe_equal(actual, self.expected)

    def test_fields_non_available(self):
        with pytest.raises(KeyError):
            writer = GeometryFileWriter(self.path + ".shp", fields={"target": "nonexisting"})
            writer.write(self.features)

    def test_custom_fields(self):
        writer = GeometryFileWriter(
            self.path + ".geojson",
            fields={"target": "str", "int1": "int", "int2": "int"},
        )
        writer.write(self.features)

        actual = gpd.read_file(self.path + ".geojson", fid_as_index=True)
        assert set(actual.columns) == {"geometry", "target", "int1", "int2", "id"}

    def test_custom_fields_refers_index(self):
        writer = GeometryFileWriter(
            self.path + ".geojson",
            fields={"target": "fid"},
        )
        writer.write(self.features)

        actual = gpd.read_file(self.path + ".geojson", fid_as_index=True)
        assert set(actual.columns) == {"geometry", "target", "id"}

    def test_custom_fields_overwrite_index(self):
        writer = GeometryFileWriter(
            self.path + ".geojson",
            fields={"id": "int"},
        )
        writer.write(self.features)

        actual = gpd.read_file(self.path + ".geojson", fid_as_index=True)
        assert set(actual.columns) == {"geometry", "id"}
        assert actual["id"].tolist() == [5, 7]




class TestToFile(unittest.TestCase):
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
        self.request_tiled = {
            "mode": "centroid",
            "projection": "EPSG:3857",
            "geometry": box(0, 0, 20, 20),
        }
        self.path = os.path.join(self.root, self._testMethodName)
        
        polygons = [
            ((0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)),
            ((10.0, 10.0), (10.0, 12.0), (12.0, 12.0), (12.0, 10.0)),
        ]
        properties = [
            {"int": 5, "float": 3.2, "str": "bla", "lst": [1], "dct": {"a": "b"}},
            {"int": 7, "float": 5.2, "str": "bla2", "lst": [2], "dct": {}},
        ]
        
        self.source = MockGeometry(
            polygons, projection="EPSG:3857", properties=properties
        )
        
        # Expected data for assertions
        self.expected = self.source.get_data(**self.request)["features"].copy()
        # The index is written both as FID and as a field
        self.expected["index"] = self.expected.index.astype('int32')
        # we expect lists and dicts to be serialized to JSON
        self.expected["lst"] = self.expected["lst"].map(json.dumps)
        self.expected["dct"] = self.expected["dct"].map(json.dumps)

    @staticmethod
    def read_file_geojson(path):
        """The result from gpd.read_file for a geojson is different from the
        others for list and dict typed values:

        - lists are omitted
        - dicts are deserialized differently
        - the projection is always EPSG:4326
        """
        df = gpd.read_file(path)

        with open(path) as f:
            data = json.load(f)
            for i, feature in enumerate(data["features"]):
                df.loc[i, "lst"] = json.dumps(feature["properties"].get("lst"))
                df.loc[i, "dct"] = json.dumps(feature["properties"].get("dct"))

        return utils.geodataframe_transform(df, "EPSG:4326", "EPSG:3857")

    def test_to_file_geojson(self):
        self.source.to_file(self.path + ".geojson", **self.request)
        actual = self.read_file_geojson(self.path + ".geojson")

        # compare dataframes without checking the order of records / columns
        assert_frame_equal_ignore_index(actual, self.expected, "int")

    def test_to_file_shapefile(self):
        self.source.to_file(self.path + ".shp", **self.request)
        actual = gpd.read_file(self.path + ".shp", fid_as_index=True)
        # compare dataframes without checking the order of records / columns
        assert_frame_equal_ignore_index(actual, self.expected, "int")

    def test_to_file_with_tiling_geojson(self):
        self.source.to_file(self.path + ".geojson", tile_size=10, **self.request_tiled)
        actual = gpd.read_file(self.path + ".geojson", fid_as_index=True)
        # because we lose the index in the saving process, just check the len
        assert len(actual) == 2

    def test_to_file_dry_run(self):
        self.source.to_file(self.path + ".geojson", dry_run=True, **self.request)
        assert not os.path.exists(self.path)

    def test_to_file_with_tiling_shapefile(self):
        self.source.to_file(self.path + ".shp", tile_size=10, **self.request_tiled)
        actual = gpd.read_file(self.path + ".shp", fid_as_index=True)
        # because we lose the index in the saving process, just check the len
        assert len(actual) == 2
