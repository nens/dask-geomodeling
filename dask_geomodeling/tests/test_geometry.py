import os
import unittest
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta

from numpy.testing import assert_almost_equal
from osgeo import ogr
from shapely.geometry import box, Point, Polygon
import geopandas as gpd
import numpy as np
import pandas as pd

from dask import config

from dask_geomodeling.utils import Extent, get_sr, shapely_transform
from dask_geomodeling.tests.factories import (
    setup_temp_root,
    teardown_temp_root,
    MockGeometry,
    MockRaster,
)

from dask_geomodeling.raster import MemorySource
from dask_geomodeling.geometry import aggregate
from dask_geomodeling.geometry import set_operations
from dask_geomodeling.geometry import field_operations
from dask_geomodeling.geometry import geom_operations
from dask_geomodeling.geometry import parallelize
from dask_geomodeling.geometry import merge
from dask_geomodeling.geometry import text
from dask_geomodeling import geometry

try:
    from pandas.testing import assert_series_equal
except ImportError:
    from pandas.util.testing import assert_series_equal


def create_geojson(abspath, polygons=10, bbox=None, ndim=2, projection="EPSG:4326"):
    """Create random triangle polygons inside bbox"""
    driver = ogr.GetDriverByName(str("GeoJSON"))
    driver.DeleteDataSource(abspath)
    datasource = driver.CreateDataSource(abspath)
    layer = datasource.CreateLayer(
        str("results"), get_sr(projection), geom_type=ogr.wkbPolygon
    )
    field_definition = ogr.FieldDefn(str("name"), ogr.OFTString)
    layer.CreateField(field_definition)
    field_definition = ogr.FieldDefn(str("id"), ogr.OFTInteger)
    layer.CreateField(field_definition)
    layer_definition = layer.GetLayerDefn()

    if np.isscalar(polygons):
        polygons = np.random.random((polygons, 3, ndim))
        bbox_min = np.asarray(bbox[:ndim])
        bbox_max = np.asarray(bbox[-ndim:])
        polygons = polygons * (bbox_max - bbox_min) + bbox_min

    for feature_id, coords in enumerate(polygons):
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for coord in coords:
            ring.AddPoint_2D(*coord)
        ring.AddPoint_2D(*coords[0])  # close the ring
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        feature = ogr.Feature(layer_definition)
        feature.SetGeometry(polygon)
        feature.SetField(str("name"), str("test"))
        feature.SetField(str("id"), feature_id + 10)
        layer.CreateFeature(feature)
    layer.SyncToDisk()
    datasource.SyncToDisk()

    return polygons


class TestGeometryBlockAttrs(unittest.TestCase):
    """Tests properties that all geometry blocks share"""

    def test_attrs(self):
        """ Check compulsory attributes for all views in geometry.py """
        missing = []
        for name, klass in geometry.__dict__.items():
            try:
                if not issubclass(klass, geometry.GeometryBlock):
                    continue  # skip non-RasterBlock objects
                if klass is geometry.GeometryBlock:
                    continue  # skip the baseclass
            except TypeError:
                continue  # also skip non-classes
            for attr in ("columns",):
                if not hasattr(klass, attr):
                    print(name, attr)
                    missing.append([name, attr])
        if len(missing) > 0:
            print(missing)

        self.assertEqual(0, len(missing))


class TestGeometryFileSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = setup_temp_root()

        # paths
        cls.relpath = "test.json"
        cls.abspath = os.path.join(cls.root, "test.json")
        cls.url = "file://" + cls.abspath

    @classmethod
    def tearDownClass(cls):
        teardown_temp_root(cls.root)

    def setUp(self):
        self.bbox = (0, 0, 1, 1)
        self.projection = "EPSG:4326"
        self.polygons = create_geojson(
            self.abspath, bbox=(0, 0, 1, 1), polygons=10, ndim=2, projection="EPSG:4326"
        )
        self.id_field = "id"
        self.source = geometry.GeometryFileSource(self.url, id_field="id")

    def test_attr(self):
        self.assertEqual(self.source.url, self.url)
        self.assertEqual(self.source.path, self.abspath)
        self.assertEqual(self.source.id_field, self.id_field)

    def test_columns(self):
        self.assertSetEqual(self.source.columns, {"id", "name", "geometry"})

    def test_get_data(self):
        result = self.source.get_data(geometry=box(*self.bbox), projection="EPSG:4326")
        self.assertEqual(self.projection, result["projection"])
        self.assertEqual(10, len(result["features"]))

    def test_get_data_centroid_mode(self):
        triangle = [[[0.8, 0.8], [2.0, 0.8], [2.0, 2.0]]]
        self.polygons = create_geojson(
            self.abspath,
            bbox=self.bbox,
            polygons=triangle,
            ndim=2,
            projection="EPSG:4326",
        )
        result = self.source.get_data(
            geometry=box(*self.bbox), projection="EPSG:4326", mode="centroid"
        )
        self.assertTrue(Polygon(*self.polygons).intersection(box(*self.bbox)))
        self.assertFalse(Polygon(*self.polygons).centroid.within(box(*self.bbox)))
        self.assertEqual(self.projection, result["projection"])
        self.assertEqual(0, len(result["features"]))

    def test_reproject(self):
        extent = Extent(self.bbox, get_sr(self.projection))
        bbox3857 = extent.transformed(get_sr("EPSG:3857")).bbox
        result = self.source.get_data(geometry=box(*bbox3857), projection="EPSG:3857")
        self.assertEqual("EPSG:3857", result["projection"])
        actual = result["features"].crs
        if isinstance(actual, dict):  # pre-geopandas 0.7
            self.assertEqual("epsg:3857", actual["init"])
        else:  # geopandas >=0.7
            self.assertEqual("EPSG:3857", actual.to_string())
        self.assertEqual(10, len(result["features"]))

    def test_limit(self):
        result = self.source.get_data(
            geometry=box(*self.bbox), projection="EPSG:4326", limit=3
        )
        self.assertEqual(3, len(result["features"]))

    def test_bbox(self):
        square = np.array([(0.5, 0.5), (0.5, 0.6), (0.6, 0.6), (0.6, 0.5)])
        outside = square + (1, 0)
        edge = square + (0.45, 0.0)
        # L shape just outside standard bbox (but envelope overlaps)
        corner = np.array(
            [(0.0, 2.0), (2.0, 2.0), (2.0, 0.0), (1.1, 0.0), (1.01, 1.1), (0.0, 1.1)]
        )

        create_geojson(
            self.abspath,
            polygons=(square, outside, edge, corner),
            projection="EPSG:4326",
        )

        # square and edge
        result = self.source.get_data(
            geometry=box(0.0, 0.0, 1.0, 1.0), projection="EPSG:4326"
        )
        self.assertEqual(2, len(result["features"]))

        # only square
        result = self.source.get_data(
            geometry=box(0.0, 0.0, 0.9, 1.0), projection="EPSG:4326"
        )
        self.assertEqual(1, len(result["features"]))

        # point request, check all 4 corners
        for x, y in [(0.5, 0.5), (0.5, 0.6), (0.6, 0.5), (0.6, 0.6)]:
            result = self.source.get_data(
                geometry=box(x, y, x, y), projection="EPSG:4326"
            )
            self.assertEqual(1, len(result["features"]))

        # point request, check just outside all 4 edges
        for x, y in [(0.49, 0.55), (0.61, 0.6), (0.55, 0.49), (0.6, 0.61)]:
            result = self.source.get_data(
                geometry=box(x, y, x, y), projection="EPSG:4326"
            )
            self.assertEqual(0, len(result["features"]))

    def test_size_filter(self):
        full = (0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)
        small = (0.0, 0.0), (0.0, 0.1), (0.0, 0.1), (0.1, 0.0)
        create_geojson(self.abspath, (full, small))

        result = self.source.get_data(
            geometry=box(*self.bbox), projection="EPSG:4326", min_size=1.1
        )
        self.assertEqual(0, len(result["features"]))

        result = self.source.get_data(
            geometry=box(*self.bbox), projection="EPSG:4326", min_size=0.9
        )
        self.assertEqual(1, len(result["features"]))

        # no filter
        result = self.source.get_data(
            geometry=box(*self.bbox), projection="EPSG:4326", min_size=0
        )
        self.assertEqual(2, len(result["features"]))

    def test_index(self):
        # the index column is named source.id_field
        result = self.source.get_data(
            geometry=box(*self.bbox), projection="EPSG:4326", limit=1
        )
        self.assertEqual(self.source.id_field, result["features"].index.name)

    def test_properties(self):
        # all properties are produced from the file
        result = self.source.get_data(
            geometry=box(*self.bbox), projection="EPSG:4326", limit=1
        )
        self.assertIn("name", result["features"].columns)

    def test_filters(self):
        # filtering returns the matching features
        result = self.source.get_data(
            geometry=box(*self.bbox), projection="EPSG:4326", filters=dict(name="test")
        )
        self.assertEqual(10, len(result["features"]))

        # and does not return non-matching features
        result = self.source.get_data(
            geometry=box(*self.bbox), projection="EPSG:4326", filters=dict(name="a")
        )
        self.assertEqual(0, len(result["features"]))

        # filters on non-existing fields are ignored
        result = self.source.get_data(
            geometry=box(*self.bbox), projection="EPSG:4326", filters=dict(a=1)
        )
        self.assertEqual(10, len(result["features"]))

        # attempting to use a django ORM expression raises ValueError
        request = dict(geometry=box(*self.bbox), filters={"name__in": ["tst"]})
        self.assertRaises(ValueError, self.source.get_data, **request)

    def test_extent_mode(self):
        result = self.source.get_data(geometry=box(*self.bbox), projection="EPSG:4326")
        expected_extent = tuple(result["features"].total_bounds)

        # extent matches the one obtained from the normal 'intersects' request
        result = self.source.get_data(
            mode="extent", geometry=box(*self.bbox), projection="EPSG:4326"
        )
        self.assertEqual("EPSG:4326", result["projection"])
        self.assertTupleEqual(expected_extent, result["extent"])

        # limit does not influence the extent
        result = self.source.get_data(
            mode="extent", geometry=box(*self.bbox), projection="EPSG:4326", limit=1
        )
        self.assertTupleEqual(expected_extent, result["extent"])


class TestSetOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = setup_temp_root()

        # paths
        cls.relpath = "test.json"
        cls.abspath = os.path.join(cls.root, "test.json")
        cls.url = "file://" + cls.abspath

    @classmethod
    def tearDownClass(cls):
        teardown_temp_root(cls.root)

    def setUp(self):
        self.request = {
            "mode": "intersects",
            "projection": "EPSG:3857",
            "geometry": box(0, 0, 1, 1),
        }
        self.polygons = [((0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0))]
        self.source = MockGeometry(self.polygons)
        self.empty = MockGeometry(polygons=[])

    def test_intersect_with_request(self):
        view = set_operations.Intersection(self.source, None)

        # return only the intersection with the bbox
        result = view.get_data(**self.request)
        self.assertAlmostEqual(1.0, result["features"]["geometry"].iloc[0].area)

        # return the intersected extent
        self.request["mode"] = "extent"
        result = view.get_data(**self.request)
        self.assertTupleEqual((0.0, 0.0, 1.0, 1.0), result["extent"])

    def test_difference(self):
        # Define a second datasource to use in the difference operation
        other = MockGeometry(
            polygons=[((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))]
        )
        view = set_operations.Difference(self.source, other)

        # the request to other should have the source's extent as geometry
        _, (_, other_req) = view.get_sources_and_requests(**self.request)
        self.assertAlmostEqual(4.0, other_req["geometry"].area)

        # the result should be the difference between the two polygons
        result = view.get_data(**self.request)
        self.assertEqual(1, len(result["features"]))
        self.assertAlmostEqual(3.0, result["features"]["geometry"].iloc[0].area)

    def test_difference_with_empty_source(self):
        view = set_operations.Difference(self.empty, self.source)

        # there should be no requests as source is empty
        sources_and_requests = view.get_sources_and_requests(**self.request)
        self.assertEqual(1, len(sources_and_requests))
        self.assertIsNone(sources_and_requests[0][1])

        # the result should be empty
        result = view.get_data(**self.request)
        self.assertEqual(0, len(result["features"]))

    def test_difference_with_empty_other(self):
        view = set_operations.Difference(self.source, self.empty)

        # there should be requests as source was non-empty
        sources_and_requests = view.get_sources_and_requests(**self.request)
        self.assertEqual(2, len(sources_and_requests))
        self.assertIsNotNone(sources_and_requests[0][1])
        self.assertIsNotNone(sources_and_requests[1][1])

        # but the result should be unchanged
        result = view.get_data(**self.request)
        self.assertEqual(1, len(result["features"]))
        self.assertAlmostEqual(4.0, result["features"]["geometry"].iloc[0].area)

    def test_difference_different_id(self):
        # Define a second datasource that produces a geometry with different ID
        other = MockGeometry(
            polygons=[((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))],
            properties=[{"id": 21}],
        )
        view = set_operations.Difference(self.source, other)

        # the result should contain the original geometry, unchanged
        result = view.get_data(**self.request)
        self.assertEqual(1, len(result["features"]))
        self.assertAlmostEqual(4.0, result["features"]["geometry"].iloc[0].area)

    def test_area(self):
        view = geom_operations.Area(self.source, projection="EPSG:3857")
        result = view.get_data(**self.request)
        self.assertListEqual(result.tolist(), [Polygon(x).area for x in self.polygons])

    def test_area_reproject(self):
        view = geom_operations.Area(self.source, projection="EPSG:3857")
        self.request["projection"] = "EPSG:4326"
        result = view.get_data(**self.request)
        np.testing.assert_almost_equal(
            result.tolist(), [Polygon(x).area for x in self.polygons]
        )

    def test_area_empty(self):
        view = geom_operations.Area(self.empty, projection="EPSG:3857")
        result = view.get_data(**self.request)
        self.assertEqual(0, len(result))


class TestGeometryWktSource(unittest.TestCase):
    def setUp(self):
        self.projection = "EPSG:28992"
        self.geometry = shapely_transform(
            box(135000.5, 455998, 135001.5, 455999.5), "EPSG:28992", self.projection
        )
        self.request = dict(
            mode="intersects",
            geometry=box(135000.5, 455998, 135001.5, 455999.5),
            projection=self.projection,
        )

    def test_geometry_wkt_source_vals_wrong_mode(self):
        self.request["mode"] = "jose"
        view = geometry.GeometryWKTSource(self.geometry.wkt, self.projection)
        with self.assertRaises(ValueError) as ctx:
            view.get_data(**self.request)
        self.assertEqual("Unknown mode 'jose'", str(ctx.exception))

    def test_geometry_wkt_source_vals(self):
        self.request["mode"] = "intersects"
        assert self.geometry.intersects(self.request["geometry"])
        assert self.geometry.centroid.intersects(self.request["geometry"])
        view = geometry.GeometryWKTSource(self.geometry.wkt, self.projection)
        actual = view.get_data(**self.request)
        assert actual["features"]["geometry"][0].wkt == self.geometry.wkt

    def test_geometry_wkt_source_vals_intersects_not_centroid(self):
        self.request["mode"] = "intersects"
        self.geometry = shapely_transform(
            box(135001, 455998, 135002.5, 455999.5), "EPSG:28992", self.projection
        )
        assert self.geometry.intersects(self.request["geometry"])
        assert not self.geometry.centroid.intersects(self.request["geometry"])
        view = geometry.GeometryWKTSource(self.geometry.wkt, self.projection)
        actual = view.get_data(**self.request)
        assert actual["features"]["geometry"][0].wkt == self.geometry.wkt

    def test_geometry_wkt_source_vals_empty(self):
        self.request["mode"] = "intersects"
        self.geometry = shapely_transform(
            box(135100.5, 455998, 135101.5, 455999.5), "EPSG:28992", self.projection
        )
        assert not self.geometry.intersects(self.request["geometry"])
        assert not self.geometry.centroid.intersects(self.request["geometry"])
        view = geometry.GeometryWKTSource(self.geometry.wkt, self.projection)
        actual = view.get_data(**self.request)
        assert actual["features"].empty

    def test_geometry_wkt_source_vals_mode_centroid(self):
        self.request["mode"] = "centroid"
        assert self.geometry.intersects(self.request["geometry"])
        assert self.geometry.centroid.intersects(self.request["geometry"])
        view = geometry.GeometryWKTSource(self.geometry.wkt, self.projection)
        actual = view.get_data(**self.request)
        assert actual["features"]["geometry"][0].wkt == self.geometry.wkt

    def test_geometry_wkt_source_vals_intersects_centroid_empty(self):
        self.request["mode"] = "centroid"
        self.geometry = shapely_transform(
            box(135001, 455998, 135002.5, 455999.5), "EPSG:28992", self.projection
        )
        assert self.geometry.intersects(self.request["geometry"])
        assert not self.geometry.centroid.intersects(self.request["geometry"])
        view = geometry.GeometryWKTSource(self.geometry.wkt, self.projection)
        actual = view.get_data(**self.request)
        assert actual["features"].empty

    def test_geometry_wkt_source_vals_centroid_empty(self):
        self.request["mode"] = "centroid"
        self.geometry = shapely_transform(
            box(135100.5, 455998, 135101.5, 455999.5), "EPSG:28992", self.projection
        )
        assert not self.geometry.intersects(self.request["geometry"])
        assert not self.geometry.centroid.intersects(self.request["geometry"])
        view = geometry.GeometryWKTSource(self.geometry.wkt, self.projection)
        actual = view.get_data(**self.request)
        assert actual["features"].empty

    def test_geometry_wkt_source_vals_mode_extent(self):
        self.request["mode"] = "extent"
        view = geometry.GeometryWKTSource(self.geometry.wkt, self.projection)
        actual = view.get_data(**self.request)
        assert actual == {
            "extent": (135000.5, 455998.0, 135001.5, 455999.5),
            "projection": "EPSG:28992",
        }

    def test_geometry_wkt_source_vals_extent_empty(self):
        self.request["mode"] = "extent"
        self.geometry = shapely_transform(
            box(135100.5, 455998, 135101.5, 455999.5), "EPSG:28992", self.projection
        )
        view = geometry.GeometryWKTSource(self.geometry.wkt, self.projection)
        actual = view.get_data(**self.request)
        assert actual == {"projection": "EPSG:28992", "extent": None}

    def test_geometry_wkt_source_vals_min_size(self):
        self.request["mode"] = "intersects"
        self.request["min_size"] = 2.0
        view = geometry.GeometryWKTSource(self.geometry.wkt, self.projection)
        actual = view.get_data(**self.request)
        assert actual["features"].empty


class TestConstructive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = setup_temp_root()

        # paths
        cls.relpath = "test.json"
        cls.abspath = os.path.join(cls.root, "test.json")
        cls.url = "file://" + cls.abspath

    @classmethod
    def tearDownClass(cls):
        teardown_temp_root(cls.root)

    def setUp(self):
        self.bbox = (0, 0, 1, 1)
        self.projection = "EPSG:4326"
        self.id_field = "id"
        self.source = geometry.Simplify(
            geometry.GeometryFileSource(self.url, id_field="id"),
            tolerance=None,
            preserve_topology=False,
        )

    def test_min_size_simplify(self):
        trapezoid1 = (0.0, 0.0), (0.49, 1.0), (0.51, 1.0), (1.0, 0.0)
        trapezoid2 = (0.0, 0.0), (0.4, 1.0), (0.6, 1.0), (1.0, 0.0)
        create_geojson(self.abspath, (trapezoid1, trapezoid2))

        # min_size = None does not simplify
        result = self.source.get_data(
            geometry=box(*self.bbox), projection="EPSG:4326", min_size=None
        )
        self.assertEqual(2, len(result["features"]))
        geoms = result["features"].geometry.values
        self.assertEqual(5, len(geoms[0].exterior.coords))
        self.assertEqual(5, len(geoms[1].exterior.coords))

        # min_size = 0.05 should simplify only the first trapezoid
        result = self.source.get_data(
            geometry=box(*self.bbox), projection="EPSG:4326", min_size=0.05
        )
        self.assertEqual(2, len(result["features"]))
        geoms = result["features"].geometry.values
        self.assertEqual(4, len(geoms[0].exterior.coords))
        self.assertEqual(5, len(geoms[1].exterior.coords))

        # min_size = 0.2 should simplify both
        result = self.source.get_data(
            geometry=box(*self.bbox), projection="EPSG:4326", min_size=0.2
        )
        self.assertEqual(2, len(result["features"]))
        geoms = result["features"].geometry.values
        self.assertEqual(4, len(geoms[0].exterior.coords))
        self.assertEqual(4, len(geoms[1].exterior.coords))


class BufferTestCase(unittest.TestCase):
    def test_buffer(self):
        polygons = [((1, 1), (2, 1), (2, 2), (1, 2))]
        source = geometry.Buffer(
            MockGeometry(polygons), distance=1.0, projection="EPSG:3857", resolution=1
        )
        request = dict(
            mode="intersects", projection="EPSG:3857", geometry=box(0, 0, 10, 10)
        )
        data = source.get_data(**request)
        actual = data["features"].geometry.area
        expected = pd.Series(7.0)
        assert_series_equal(expected, actual, check_names=False)

    def test_buffer_transform(self):
        # Define a polygon in RD New (1 square meter).
        polygon = (
            (155000, 463000),
            (155001, 463000),
            (155001, 463001),
            (155000, 463001),
        )
        # Apply a buffer of 10 cm.
        distance = 0.1
        source = geometry.Buffer(
            MockGeometry([polygon], projection="EPSG:28992"),
            distance=distance,
            projection="EPSG:28992",
            resolution=1,  # For easy area calculation.
        )
        # Request WGS 84 to force a transformation of the buffered geometry.
        request = dict(
            mode="intersects", projection="EPSG:4326", geometry=box(4, 51, 7, 53)
        )
        data = source.get_data(**request)
        # Transform back to RD New for square meters.
        actual = (
            data["features"]
            .geometry.apply(shapely_transform, args=("EPSG:4326", "EPSG:28992"))
            .area
        )
        width = height = 1 + 2 * distance
        expected = pd.Series(width * height - 2 * distance * distance)
        assert_series_equal(expected, actual, check_names=False)

    def test_extent_mode(self):
        # Define a polygon in RD New (1 square kilometer).
        polygon = (
            (155000, 463000),
            (156000, 463000),
            (156000, 464000),
            (155000, 464000),
        )
        # Apply a buffer of 10 m.
        source = geometry.Buffer(
            MockGeometry([polygon], projection="EPSG:28992"),
            distance=10,
            projection="EPSG:28992",
        )
        request = dict(
            mode="extent", projection="EPSG:4326", geometry=box(4, 51, 7, 53)
        )
        data = source.get_data(**request)
        # NB: This is not 1020 * 1020 meter, because a square in EPSG:28992
        # will no longer be a square in EPSG:4326. In other words, such a
        # box can no longer be defined by lower left and upper right.
        expected = (
            5.38705742335229,
            52.15508055768161,
            5.401968302292468,
            52.16425103224139,
        )
        actual = data["extent"]
        assert_almost_equal(expected, actual)

    def test_buffer_empty(self):
        source = geometry.Buffer(
            MockGeometry([]), distance=1.0, projection="EPSG:3857", resolution=1
        )
        request = dict(
            mode="intersects", projection="EPSG:3857", geometry=box(0, 0, 10, 10)
        )
        data = source.get_data(**request)
        self.assertEqual(0, len(data["features"]))

        request["mode"] = "extent"
        data = source.get_data(**request)
        self.assertIsNone(data["extent"])


class TestAggregateRaster(unittest.TestCase):
    def setUp(self):
        self.raster = MockRaster(
            origin=Datetime(2018, 1, 1), timedelta=Timedelta(hours=1), bands=1
        )
        self.source = MockGeometry(
            polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))],
            properties=[{"id": 1}],
        )
        self.view = geometry.AggregateRaster(
            source=self.source, raster=self.raster, statistic="sum"
        )
        self.request = dict(
            mode="intersects",
            projection="EPSG:3857",
            geometry=box(0, 0, 10, 10),
            min_size=1.0,
        )
        self.default_raster_limit = config.get("geomodeling.raster-limit")

    def tearDown(self):
        config.set({"geomodeling.raster-limit": self.default_raster_limit})

    def test_arg_types(self):
        self.assertRaises(TypeError, geometry.AggregateRaster, self.source, None)
        self.assertRaises(TypeError, geometry.AggregateRaster, None, self.raster)
        self.assertRaises(
            TypeError,
            geometry.AggregateRaster,
            self.source,
            self.raster,
            statistic=None,
        )
        self.assertRaises(
            TypeError,
            geometry.AggregateRaster,
            self.source,
            self.raster,
            projection=4326,
        )

        # if no projection / geo_transform specified, take them from raster
        view = geometry.AggregateRaster(self.source, self.raster)
        self.assertEqual(self.raster.projection, view.projection)
        self.assertEqual(1.0, view.pixel_size)

        view = geometry.AggregateRaster(
            self.source, self.raster, projection="EPSG:28992", pixel_size=0.2
        )
        self.assertEqual("EPSG:28992", view.projection)
        self.assertEqual(0.2, view.pixel_size)

        # 0 pixel size is unsupported
        self.assertRaises(
            ValueError,
            geometry.AggregateRaster,
            self.source,
            self.raster,
            pixel_size=0.0,
        )

        # percentile value out of bounds
        self.assertRaises(
            ValueError,
            geometry.AggregateRaster,
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
            view = geometry.AggregateRaster(
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
            view = geometry.AggregateRaster(
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
            view = geometry.AggregateRaster(
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

    def test_pixel_size(self):
        # larger
        self.view = geometry.AggregateRaster(
            source=self.source, raster=self.raster, statistic="sum", pixel_size=2
        )
        _, (_, request), _ = self.view.get_sources_and_requests(**self.request)
        np.testing.assert_allclose(request["bbox"], (2, 2, 8, 8))
        self.assertEqual(3, request["width"])
        self.assertEqual(3, request["height"])

        # smaller
        self.view = geometry.AggregateRaster(
            source=self.source, raster=self.raster, statistic="sum", pixel_size=0.5
        )
        _, (_, request), _ = self.view.get_sources_and_requests(**self.request)
        np.testing.assert_allclose(request["bbox"], (2, 2, 8, 8))
        self.assertEqual(12, request["width"])
        self.assertEqual(12, request["height"])

    def test_max_pixels(self):
        self.view = geometry.AggregateRaster(
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
            self.view = geometry.AggregateRaster(
                MockGeometry([((x1, y1), (x2, y1), (x2, y2), (x1, y2))]), self.raster
            )
            _, (_, request), _ = self.view.get_sources_and_requests(**self.request)

            np.testing.assert_allclose(request["bbox"], exp_bbox)
            self.assertEqual(exp_width, request["width"])
            self.assertEqual(exp_height, request["height"])

    def test_max_pixels_with_snap(self):
        x1, y1, x2, y2 = 2.01, 1.99, 7.99, 8.01

        self.view = geometry.AggregateRaster(
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
        self.view = geometry.AggregateRaster(
            source=self.source, raster=self.raster, statistic="sum", max_pixels=9
        )

        self.assertRaises(
            RuntimeError, self.view.get_sources_and_requests, **self.request
        )

    def test_max_pixels_fallback(self):
        config.set({"geomodeling.raster-limit": 9})
        self.view = geometry.AggregateRaster(
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
        view2 = geometry.AggregateRaster(
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

        view1 = geometry.AggregateRaster(self.source, self.raster, statistic="mean")
        view2 = geometry.AggregateRaster(
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
        view = geometry.AggregateRaster(
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
        view = geometry.AggregateRaster(
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

        chained = geometry.AggregateRaster(
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
        view = geometry.AggregateRaster(
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
            view = geometry.AggregateRaster(source=source, raster=raster, statistic=agg)
            result = view.get_data(**self.request)
            assert np.isnan(result["features"]["agg"].values[1])

    def test_empty_dataset(self):
        source = MockGeometry(polygons=[], properties=[])
        view = geometry.AggregateRaster(
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
            view = geometry.AggregateRasterAboveThreshold(
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


class TestBucketize(unittest.TestCase):
    def test_bucketize(self):
        bboxes = [
            (0, 0, 2, 2),  # new bucket
            (2, 2, 4, 4),  # new bucket because of overlap with previous bucket
            (0, 0, 3, 3),  # new bucket because of size
            (5, 5, 7, 7),  # same as first
        ]
        expected = [[0, 3], [1], [2]]
        buckets = aggregate.bucketize(bboxes)
        self.assertEqual([0, 1, 2, 3], sorted(i for b in buckets for i in b))
        self.assertEqual(expected, sorted(buckets))


class TestSetGetSeries(unittest.TestCase):
    def setUp(self):
        self.N = 10
        properties = [{"id": i, "col_1": i * 2} for i in range(self.N)]
        polygons = [((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))] * self.N
        self.source1 = MockGeometry(polygons=polygons, properties=properties)
        properties = [
            {"id": i, "col_2": i * 3, "col_3": i * 4, "col_4": i if i % 2 else np.nan}
            for i in range(self.N)
        ]
        self.source2 = MockGeometry(polygons=polygons, properties=properties)
        self.request = dict(
            mode="intersects", projection="EPSG:3857", geometry=box(0, 0, 10, 10)
        )

    def test_get_series(self):
        series = geometry.GetSeriesBlock(self.source1, "col_1")
        data = series.get_data(**self.request)
        assert_almost_equal(data.values, [i * 2 for i in range(self.N)])

    def test_get_not_available(self):
        self.assertRaises(
            KeyError, geometry.GetSeriesBlock, self.source1, "not_available"
        )

    def test_get_series_by_indexing(self):
        series = self.source1["col_1"]
        self.assertIsInstance(series, geometry.GetSeriesBlock)
        self.assertIs(series.args[0], self.source1)
        self.assertIs(series.args[1], "col_1")

    def test_set_series(self):
        source = geometry.SetSeriesBlock(self.source1, "added", self.source2["col_2"])
        data = source.get_data(**self.request)
        added_values = data["features"]["added"].values
        assert_almost_equal(added_values, [i * 3 for i in range(self.N)])
        self.assertSetEqual({"geometry", "col_1", "added"}, source.columns)

    def test_set_series_overwrite(self):
        source = geometry.SetSeriesBlock(self.source1, "col_1", self.source2["col_2"])
        data = source.get_data(**self.request)
        added_values = data["features"]["col_1"].values
        assert_almost_equal(added_values, [i * 3 for i in range(self.N)])
        self.assertSetEqual({"geometry", "col_1"}, source.columns)

    def test_set_series_multiple(self):
        source = geometry.SetSeriesBlock(
            self.source1,
            "added",
            self.source2["col_2"],
            "added2",
            self.source2["col_3"],
        )
        data = source.get_data(**self.request)
        added_values = data["features"]["added"].values
        assert_almost_equal(added_values, [i * 3 for i in range(self.N)])
        added_values = data["features"]["added2"].values
        assert_almost_equal(added_values, [i * 4 for i in range(self.N)])
        self.assertSetEqual({"geometry", "col_1", "added", "added2"}, source.columns)

    def test_set_series_by_set_method(self):
        args = ("a1", self.source2["col_2"], "a2", self.source2["col_3"])
        source = self.source1.set(*args)
        self.assertIsInstance(source, geometry.SetSeriesBlock)
        self.assertIs(source.args[0], self.source1)
        self.assertTupleEqual(source.args[1:], args)

    def test_set_series_float(self):
        source = geometry.SetSeriesBlock(self.source1, "constant", 2.1)
        data = source.get_data(**self.request)["features"]["constant"]
        self.assertTrue(np.issubdtype(data.dtype, np.floating))
        self.assertTrue((data == 2.1).all())

    def test_set_series_int(self):
        source = geometry.SetSeriesBlock(self.source1, "constant", 2)
        data = source.get_data(**self.request)
        data = source.get_data(**self.request)["features"]["constant"]
        self.assertTrue(np.issubdtype(data.dtype, np.integer))
        self.assertTrue((data == 2).all())

    def test_set_series_bool(self):
        source = geometry.SetSeriesBlock(self.source1, "constant", True)
        data = source.get_data(**self.request)["features"]["constant"]
        self.assertTrue(data.dtype == bool)
        self.assertTrue(data.all())

    def test_set_series_string(self):
        source = geometry.SetSeriesBlock(self.source1, "constant", "string")
        data = source.get_data(**self.request)["features"]["constant"]
        self.assertTrue((data == "string").all())


class TestWhere(unittest.TestCase):
    def setUp(self):
        values = [-float("inf"), -2, 1.2, 5.0, float("inf"), float("nan")]
        self.properties = [
            {
                "id": i,
                "col_1": x,
                "bool_filter": True if x >= 0 else False,
                "extra": x * 10,
            }
            for i, x in enumerate(values)
        ]
        self.source = MockGeometry(
            polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))] * len(values),
            properties=self.properties,
        )
        self.prop_df = pd.DataFrame(self.properties)
        self.request = dict(
            mode="intersects", projection="EPSG:3857", geometry=box(0, 0, 10, 10)
        )

    def test_where(self):
        series = field_operations.Where(
            self.source["col_1"], cond=self.source["bool_filter"], other="Hola!"
        )
        view = self.source.set("result", series)

        result = view.get_data(**self.request)
        expected = pd.Series(["Hola!", "Hola!", 1.2, 5, float("inf"), "Hola!"])
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_where_with_float_filter(self):
        series = field_operations.Where(
            self.source["col_1"], cond=self.source["col_1"], other="Hola!"
        )
        view = self.source.set("result", series)

        result = view.get_data(**self.request)
        expected = pd.Series([-float("inf"), -2, 1.2, 5, float("inf"), "Hola!"])
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_where_with_other_column(self):
        series = field_operations.Where(
            self.source["col_1"],
            cond=self.source["bool_filter"],
            other=self.source["extra"],
        )
        view = self.source.set("result", series)

        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"].where(
            self.prop_df["bool_filter"], self.prop_df["extra"]
        )
        expected = pd.Series([-float("inf"), -20, 1.2, 5, float("inf"), float("nan")])
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_mask(self):
        series = field_operations.Mask(
            self.source["col_1"], cond=self.source["bool_filter"], other="Hola!"
        )
        view = self.source.set("result", series)

        result = view.get_data(**self.request)
        expected = pd.Series([-float("inf"), -2, "Hola!", "Hola!", "Hola!", float("nan")])
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_mask_with_float_filter(self):
        series = field_operations.Mask(
            self.source["col_1"], cond=self.source["col_1"], other="Hola!"
        )
        view = self.source.set("result", series)

        result = view.get_data(**self.request)
        expected = pd.Series(["Hola!", "Hola!", "Hola!", "Hola!", "Hola!", float("nan")])
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_mask_with_other_column(self):
        series = field_operations.Mask(
            self.source["col_1"],
            cond=self.source["bool_filter"],
            other=self.source["extra"],
        )
        view = self.source.set("result", series)

        result = view.get_data(**self.request)
        expected = pd.Series([-float("inf"), -2, 12.0, 50, float("inf"), float("nan")])
        self.assertTrue(result["features"]["result"].equals(expected))


class TestMerge(unittest.TestCase):
    def setUp(self):
        polygon_1 = [((2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0))]
        polygon_2 = [((3.0, 3.0), (4.0, 3.0), (4.0, 4.0), (3.0, 4.0))]
        polygon_4 = [((5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0))]
        self.source_1 = MockGeometry(polygons=polygon_1)
        self.source_2 = MockGeometry(polygons=polygon_2)
        self.source_3 = MockGeometry([])
        self.source_4 = MockGeometry(polygons=polygon_4)
        self.request = dict(
            geometry=box(0, 0, 10, 10), mode="intersects", projection="EPSG:3857"
        )

    def test_merge_dask_geomodeling(self):
        view = merge.MergeGeometryBlocks(
            left=self.source_1,
            right=self.source_2,
            how="inner",
            suffixes=("", "_right"),
        )
        result = view.get_data(**self.request)
        expected_columns = {"geometry", "geometry_right"}
        self.assertSetEqual(set(result["features"].columns), expected_columns)
        self.assertSetEqual(view.columns, expected_columns)
        self.assertIsInstance(result["features"], gpd.GeoDataFrame)

    def test_merge_dask_geomodeling_empty_source(self):
        view = merge.MergeGeometryBlocks(
            left=self.source_1,
            right=self.source_3,
            how="inner",
            suffixes=("", "_right"),
        )
        result = view.get_data(**self.request)
        self.assertTrue(result["features"].empty)

    def test_merge_dask_geomodeling_extent_mode(self):
        self.request["mode"] = "extent"
        view = merge.MergeGeometryBlocks(
            left=self.source_1, right=self.source_2, how="inner"
        )
        result = view.get_data(**self.request)
        self.assertTupleEqual(result["extent"], (3.0, 3.0, 3.0, 3.0))

    def test_merge_dask_geomodeling_extent_mode_no_intersect(self):
        self.request["mode"] = "extent"
        view = merge.MergeGeometryBlocks(
            left=self.source_1, right=self.source_4, how="inner"
        )
        result = view.get_data(**self.request)
        self.assertIsNone(result["extent"])

    def test_merge_dask_geomodeling_extent_mode_no_intersect_outer_join(self):
        self.request["mode"] = "extent"
        view = merge.MergeGeometryBlocks(
            left=self.source_1, right=self.source_4, how="outer"
        )
        result = view.get_data(**self.request)
        self.assertTupleEqual(result["extent"], (2.0, 2.0, 6.0, 6.0))

    def test_merge_dask_geomodeling_source_empty(self):
        self.request["mode"] = "extent"
        view = merge.MergeGeometryBlocks(
            left=self.source_1, right=self.source_3, how="inner"
        )
        result = view.get_data(**self.request)
        self.assertIsNone(result["extent"])

    def test_merge_dask_geomodeling_no_intersect_outer_join_source_empty(self):
        self.request["mode"] = "extent"
        view = merge.MergeGeometryBlocks(
            left=self.source_1, right=self.source_3, how="outer"
        )
        result = view.get_data(**self.request)
        self.assertTupleEqual(result["extent"], (2.0, 2.0, 3.0, 3.0))


class TestFieldOperations(unittest.TestCase):
    def setUp(self):
        values = [-float("inf"), -2, 1.2, 5.0, float("inf"), float("nan")]
        self.properties = [
            {
                "id": i,
                "id_value": float(i),
                "col_1": x,
                "col_2": 2 * x,
                "bool_1": x > 0,
                "bool_2": x > 2,
                "col_source": float(i * 2 + 1),
                "col_choice_1": chr(i + 65),  # 'A'
                "col_choice_2": chr(i + 70),  # 'F'
                "none": None,
            }
            for i, x in enumerate(values)
        ]
        self.prop_df = pd.DataFrame(self.properties)
        self.source = MockGeometry(
            polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))] * len(values),
            properties=self.properties,
        )
        self.request = dict(
            mode="intersects", projection="EPSG:3857", geometry=box(0, 0, 10, 10)
        )

    def test_choose(self):
        series = field_operations.Choose(
            self.source["id_value"],
            self.source["col_1"],
            self.source["col_2"],
            self.source["bool_1"],
        )
        result = series.get_data(**self.request)
        values = result.values
        self.assertEqual(-float("inf"), values[0])  # -inf
        self.assertEqual(-4.0, values[1])  # -4.0
        self.assertEqual(1.0, values[2])  # 1.0
        self.assertTrue(np.isnan(values[3]))  # nan
        self.assertTrue(np.isnan(values[4]))  # nan
        self.assertTrue(np.isnan(values[5]))  # nan

    def test_choose_values_neq_index(self):
        series = field_operations.Choose(
            self.source["col_source"],
            self.source["col_1"],
            self.source["col_2"],
            self.source["bool_1"],
        )
        result = series.get_data(**self.request)
        values = result.values
        self.assertEqual(-float("inf"), values[0])

    def test_choice_dtype_str(self):
        series = field_operations.Choose(
            self.source["id_value"],
            self.source["col_choice_1"],
            self.source["col_choice_2"],
        )
        result = series.get_data(**self.request)
        values = result.values
        self.assertEqual("A", values[0])
        self.assertEqual("G", values[1])

    def test_choose_different_length(self):
        val = [-float("inf"), -2, 1.2, 5.0, float("inf"), float("nan"), 1]
        properties = [{"id_value": float(i)} for i, x in enumerate(val)]
        source_2 = MockGeometry(
            polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))] * len(val),
            properties=properties,
        )
        series = field_operations.Choose(
            source_2["id_value"],
            self.source["col_choice_1"],
            self.source["col_2"],
            self.source["bool_1"],
        )
        result = series.get_data(**self.request)
        values = result.values
        self.assertEqual("A", values[0])
        self.assertEqual(-4, values[1])
        self.assertTrue(values[2])
        self.assertEqual(len(values), len(val))

    def test_classify_field(self):
        series = field_operations.Classify(
            self.source["col_1"], bins=[0, 1.2, 5.0], labels=["A", "B"]
        )
        result = series.get_data(**self.request)
        values = result.values
        self.assertTrue(np.isnan(values[0]))  # -inf
        self.assertTrue(np.isnan(values[1]))  # -2
        self.assertEqual("A", values[2])  # 1.2
        self.assertEqual("B", values[3])  # 5.
        self.assertTrue(np.isnan(values[4]))  # inf
        self.assertTrue(np.isnan(values[5]))  # nan

    def test_classify_field_left(self):
        series = field_operations.Classify(
            self.source["col_1"], bins=[0, 1.2, 10.0], labels=["A", "B"], right=False
        )
        result = series.get_data(**self.request)
        values = result.values
        self.assertTrue(np.isnan(values[0]))  # -inf
        self.assertTrue(np.isnan(values[1]))  # -2
        self.assertEqual("B", values[2])  # 1.2
        self.assertEqual("B", values[3])  # 5.
        self.assertTrue(np.isnan(values[4]))  # inf
        self.assertTrue(np.isnan(values[5]))  # nan

    def test_classify_field_open_bounds(self):
        series = field_operations.Classify(
            self.source["col_1"], bins=[1.2, 5], labels=["A", "B", "C"]
        )
        result = series.get_data(**self.request)
        values = result.values
        self.assertEqual("A", values[0])  # -inf
        self.assertEqual("A", values[1])  # -2
        self.assertEqual("A", values[2])  # 1.2
        self.assertEqual("B", values[3])  # 5.
        self.assertEqual("C", values[4])  # inf
        self.assertTrue(np.isnan(values[5]))  # nan

    def test_classify_field_open_bounds_left(self):
        series = field_operations.Classify(
            self.source["col_1"], bins=[1.2, 5], labels=["A", "B", "C"], right=False
        )
        result = series.get_data(**self.request)
        values = result.values
        self.assertEqual("A", values[0])  # -inf
        self.assertEqual("A", values[1])  # -2
        self.assertEqual("B", values[2])  # 1.2
        self.assertEqual("C", values[3])  # 5.
        self.assertEqual("C", values[4])  # inf
        self.assertTrue(np.isnan(values[5]))  # nan.

    def test_classify_none(self):
        series = field_operations.Classify(
            self.source["none"], bins=[0, 0.2], labels=["A"]
        )
        result = series.get_data(**self.request)
        values = result.values
        self.assertTrue(all([np.isnan(x) for x in values]))

    def test_classify_from_columns_empty(self):
        view = field_operations.ClassifyFromColumns(
            self.source, "col_1", ["id_value"], labels=["A", "B"]
        )
        result = view.get_data(
            mode="intersects", projection="EPSG:3857", geometry=box(0, 0, 0, 0)
        )
        self.assertEqual(0, len(result))

    def test_classify_from_columns_none(self):
        series = field_operations.ClassifyFromColumns(
            self.source, "none", ["id_value"], labels=["A", "B"]
        )
        result = series.get_data(**self.request)
        values = result.values
        self.assertTrue(all([np.isnan(x) for x in values]))

    def test_classify_from_columns_varying_bin(self):
        series = field_operations.ClassifyFromColumns(
            self.source,
            "col_1",
            ["id_value"],
            labels=["lower_than_id", "higher_than_id"],
        )
        result = series.get_data(**self.request)
        values = result.values
        self.assertEqual("lower_than_id", values[0])  # -inf < 0
        self.assertEqual("lower_than_id", values[1])  # -2 < 1
        self.assertEqual("lower_than_id", values[2])  # 1.2 < 2
        self.assertEqual("higher_than_id", values[3])  # 5. > 3
        self.assertEqual("higher_than_id", values[4])  # inf > 4
        self.assertTrue(np.isnan(values[5]))  # nan

    def test_classify_from_columns(self):
        source_with_bins = self.source.set("bin_1", 0, "bin_2", 1.2, "bin_3", 5.0)
        series = field_operations.ClassifyFromColumns(
            source_with_bins, "col_1", ["bin_1", "bin_2", "bin_3"], labels=["A", "B"]
        )
        result = series.get_data(**self.request)
        expected = field_operations.Classify(
            self.source["col_1"], bins=[0, 1.2, 5.0], labels=["A", "B"]
        ).get_data(**self.request)
        assert_series_equal(result, expected, check_names=False)

    def test_classify_int_labels_as_float(self):
        actual = field_operations.Classify(
            self.source["col_source"], bins=[0, 1.0, 5.0], labels=[2, 3]
        ).get_data(**self.request)
        self.assertEqual(actual.dtype, float)

    def test_classify_not_categorical(self):
        actual = field_operations.Classify(
            self.source["col_source"], bins=[0, 0.5, 1.0], labels=["A", "B", "C", "D"]
        ).get_data(**self.request)
        self.assertEqual(actual.dtype.name, "object")

    def test_classify_from_columns_left(self):
        source_with_bins = self.source.set("bin_1", 0, "bin_2", 1.2, "bin_3", 5.0)
        series = field_operations.ClassifyFromColumns(
            source_with_bins,
            "col_1",
            ["bin_1", "bin_2", "bin_3"],
            labels=["A", "B"],
            right=False,
        )
        result = series.get_data(**self.request)
        expected = field_operations.Classify(
            self.source["col_1"], bins=[0, 1.2, 5.0], labels=["A", "B"], right=False
        ).get_data(**self.request)
        assert_series_equal(result, expected, check_names=False)

    def test_classify_from_columns_open_bounds(self):
        source_with_bins = self.source.set("bin_1", 1.2, "bin_2", 5)
        series = field_operations.ClassifyFromColumns(
            source_with_bins, "col_1", ["bin_1", "bin_2"], labels=["A", "B", "C"]
        )
        result = series.get_data(**self.request)
        expected = field_operations.Classify(
            self.source["col_1"], bins=[1.2, 5.0], labels=["A", "B", "C"]
        ).get_data(**self.request)
        assert_series_equal(result, expected, check_names=False)

    def test_classify_from_columns_open_bounds_left(self):
        source_with_bins = self.source.set("bin_1", 1.2, "bin_2", 5)
        series = field_operations.ClassifyFromColumns(
            source_with_bins,
            "col_1",
            ["bin_1", "bin_2"],
            labels=["A", "B", "C"],
            right=False,
        )
        result = series.get_data(**self.request)
        expected = field_operations.Classify(
            self.source["col_1"], bins=[1.2, 5.0], labels=["A", "B", "C"], right=False
        ).get_data(**self.request)
        assert_series_equal(result, expected, check_names=False)

    def test_classify_from_columns_int_labels_as_float(self):
        source_with_bins = self.source.set("bin_1", 1, "bin_2", 2)
        series = field_operations.ClassifyFromColumns(
            source_with_bins,
            "col_1",
            ["bin_1", "bin_2"],
            labels=[200],
            right=False,
        )
        result = series.get_data(**self.request)
        self.assertEqual(result.dtype, float)

    def test_add_fields(self):
        series_block = self.source["col_1"] + self.source["col_2"]
        view = self.source.set("result", series_block)

        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"] + self.prop_df["col_2"]
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_add_int_fields(self):
        series_block = self.source["col_1"] + 2
        view = self.source.set("result", series_block)

        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"] + 2
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_sub_fields(self):
        series_block = self.source["col_1"] - self.source["col_2"]
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"].sub(self.prop_df["col_2"])
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_sub_float_fields(self):
        series_block = self.source["col_1"] - 2.1
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"].sub(2.1)
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_mul_fields(self):
        series_block = self.source["col_1"] * self.source["col_2"]
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"].mul(self.prop_df["col_2"])
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_mul_inf_fields(self):
        series_block = self.source["col_1"] * float("inf")
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"].mul(float("inf"))
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_mul_none_fields(self):
        series_block = self.source["col_1"] + float("nan")
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        self.assertTrue(np.all(np.isnan(result["features"]["result"])))

    def test_div_fields(self):
        series_block = self.source["col_1"] / self.source["col_2"]
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"].truediv(self.prop_df["col_2"])
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_floor_div_divide_fields(self):
        series_block = self.source["col_1"] // self.source["col_2"]
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"].floordiv(self.prop_df["col_2"])
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_pow_inverse_fields(self):
        series_block = self.source["col_1"] ** -1
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"].pow(float(-1))
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_square_root_fields(self):
        series_block = self.source["col_1"] ** 0.5
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"].pow(0.5)
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_mod_fields(self):
        series_block = self.source["col_1"] % self.source["col_2"]
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"].mod(self.prop_df["col_2"])
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_lt_fields(self):
        series_block = self.source["col_1"] < self.source["col_2"]
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"].lt(self.prop_df["col_2"])
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_le_fields(self):
        series_block = self.source["col_1"] <= self.source["col_2"]
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"].le(self.prop_df["col_2"])
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_eq_fields(self):
        series_block = self.source["col_1"] == self.source["col_2"]
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"].eq(self.prop_df["col_2"])
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_ne_fields(self):
        series_block = self.source["col_1"] != self.source["col_2"]
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"].ne(self.prop_df["col_2"])
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_gt_fields(self):
        series_block = self.source["col_1"] > self.source["col_2"]
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"].gt(self.prop_df["col_2"])
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_ge_fields(self):
        series_block = self.source["col_1"] >= self.source["col_2"]
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["col_1"].ge(self.prop_df["col_2"])
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_and_fields(self):
        series_block = self.source["bool_1"] & self.source["bool_2"]
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["bool_1"] & self.prop_df["bool_2"]
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_or_fields(self):
        series_block = self.source["bool_1"] | self.source["bool_2"]
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["bool_1"] | self.prop_df["bool_2"]
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_xor_fields(self):
        series_block = self.source["bool_1"] ^ self.source["bool_2"]
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = self.prop_df["bool_1"] ^ self.prop_df["bool_2"]
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_not_field(self):
        series_block = ~self.source["bool_1"]
        view = self.source.set("result", series_block)
        result = view.get_data(**self.request)
        expected = ~self.prop_df["bool_1"]
        self.assertTrue(result["features"]["result"].equals(expected))

    def test_set_multiple_columns(self):
        series_sum = self.source["col_1"] + self.source["col_2"]
        series_product = self.source["col_1"] * self.source["col_2"]
        view = self.source.set("sum", series_sum, "product", series_product)

        result = view.get_data(**self.request)
        self.assertTrue(
            result["features"]["sum"].equals(
                self.prop_df["col_1"] + self.prop_df["col_2"]
            )
        )
        self.assertTrue(
            result["features"]["product"].equals(
                self.prop_df["col_1"] * self.prop_df["col_2"]
            )
        )

    def test_round(self):
        self.assertRaises(TypeError, field_operations.Round, self.source, "s")

        view = field_operations.Round(self.source["col_1"] / 3, 2)
        result = view.get_data(**self.request)
        expected = (self.prop_df["col_1"] / 3).round(2)
        self.assertTrue(result.equals(expected))

    def test_interp(self):
        # interpolate on 2 x = y
        view = field_operations.Interp(
            self.source["col_1"], [0.0, 5.0], [0.0, 10.0], left=-1.0, right=11.0
        )
        result = view.get_data(**self.request)
        values = result.values

        self.assertEqual(-1.0, values[0])  # -inf
        self.assertEqual(-1.0, values[1])  # -2
        self.assertEqual(2.4, values[2])  # 1.2
        self.assertEqual(10.0, values[3])  # 5.
        self.assertEqual(11.0, values[4])  # inf
        self.assertTrue(np.isnan(values[5]))  # nan


class TestParallelize(unittest.TestCase):
    def setUp(self):
        self.projection = "EPSG:3857"
        self.source = MockGeometry(
            polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))],
            properties=[{"id": 1}],
        )
        self.request = dict(
            mode="centroid", projection=self.projection, geometry=box(0, 0, 10, 5)
        )

    def test_extent(self):
        self.request["mode"] = "extent"
        view = parallelize.GeometryTiler(self.source, 2.5, self.projection)
        data = view.get_data(**self.request)
        self.assertListEqual([2, 2, 8, 8], list(data["extent"]))

    def test_mode_intersects(self):
        self.request["mode"] = "intersects"
        view = parallelize.GeometryTiler(self.source, 2.5, self.projection)
        with self.assertRaises(NotImplementedError):
            view.get_data(**self.request)

    def test_empty(self):
        view = parallelize.GeometryTiler(MockGeometry([]), 5.0, self.projection)
        data = view.get_data(**self.request)
        self.assertEqual(0, len(data["features"]))

    def test_some_tiles_empty(self):
        self.request["geometry"] = box(0, 0, 10, 50)
        view = parallelize.GeometryTiler(self.source, 10.0, self.projection)
        data = view.get_data(**self.request)
        self.assertEqual(1, len(data["features"]))

    def test_no_tiling(self):
        view = parallelize.GeometryTiler(self.source, 10, self.projection)
        requests = [x[1] for x in view.get_sources_and_requests(**self.request)]
        self.assertEqual(1, len(requests))
        self.assertEqual(50.0, requests[0]["geometry"].area)

    def test_two_tiles(self):
        view = parallelize.GeometryTiler(self.source, 5, self.projection)
        requests = [x[1] for x in view.get_sources_and_requests(**self.request)]
        self.assertEqual(2, len(requests))
        for request in requests:
            self.assertEqual(25.0, request["geometry"].area)

    def test_skip_empty_tiles(self):
        # set-up an L-shaped polygon request. the empty space should not result
        # in tiles. In the 2.5x2.5 tiling in this example, it should skip 3
        self.request["geometry"] = Polygon(
            ((0, 0), (10, 0), (10, 5), (9, 5), (9, 1), (0, 1), (0, 0))
        )
        view = parallelize.GeometryTiler(self.source, 2.5, self.projection)
        requests = [x[1] for x in view.get_sources_and_requests(**self.request)]
        self.assertEqual(5, len(requests))
        # the tiles are all intersected with the requested geometry
        for request in requests:
            self.assertLess(request["geometry"].area, 25.0)

    def test_eight_tiles(self):
        view = parallelize.GeometryTiler(self.source, 2.5, self.projection)
        requests = [x[1] for x in view.get_sources_and_requests(**self.request)]
        self.assertEqual(8, len(requests))
        for request in requests:
            self.assertEqual(2.5 * 2.5, request["geometry"].area)

    def test_tile_resize(self):
        view = parallelize.GeometryTiler(self.source, 8, self.projection)
        requests = [x[1] for x in view.get_sources_and_requests(**self.request)]
        self.assertEqual(2, len(requests))
        for request in requests:
            self.assertEqual(25.0, request["geometry"].area)

    def test_tile_different_projection(self):
        # NB we add 0.00001 to the tile size to account for projection errors
        view = parallelize.GeometryTiler(self.source, 5.00001, self.projection)

        # transform the requested geometry to WGS84
        geometry_wgs84 = shapely_transform(
            self.request["geometry"], "EPSG:3857", "EPSG:4326"
        )
        self.request["projection"] = "EPSG:4326"
        self.request["geometry"] = geometry_wgs84

        requests = [x[1] for x in view.get_sources_and_requests(**self.request)]
        self.assertEqual(2, len(requests))
        for request in requests:
            self.assertEqual("EPSG:3857", request["projection"])
            self.assertAlmostEqual(25.0, request["geometry"].area)

    def test_merge_results(self):
        source = MockGeometry(
            polygons=[
                ((2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)),
                ((6.0, 2.0), (7.0, 2.0), (7.0, 3.0), (6.0, 3.0)),
            ],
            properties=[{"id": 1}, {"id": 2}],
        )
        view = parallelize.GeometryTiler(source, 5, self.projection)
        data = view.get_data(**self.request)
        self.assertEqual(2, len(data["features"]))


class TestText(unittest.TestCase):
    def setUp(self):
        self.key_mapping = {
            "modelname": "model_name",
            "duration": "rainfall_duration",
            "strength": "rainfall_strength",
            "ahn2": "ahn2_used",
        }
        self.description = (
            "\nSimulation of Rotterdam"
            "\n\nmodelname=rotterdam 01"
            "\nduration=120\nstrength=70\nahn2=true"
        )
        self.expected = {
            "model_name": "rotterdam 01",
            "rainfall_duration": 120,
            "rainfall_strength": 70,
            "ahn2_used": True,
        }
        self.source = MockGeometry(
            polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))],
            properties=[{"id": 1, "description": self.description}],
        )
        self.view = text.ParseTextColumn(self.source, "description", self.key_mapping)
        self.request = dict(
            mode="intersects", projection="EPSG:3857", geometry=box(0, 0, 10, 10)
        )

    def test_parser_columns(self):
        data = self.view.get_data(**self.request)
        self.assertTrue(
            set(self.key_mapping.values()).issubset(data["features"].columns.tolist())
        )
        self.assertSetEqual(set(data["features"].columns), self.view.columns)

    def test_parser_results(self):
        data = self.view.get_data(**self.request)["features"]
        for col in self.expected:
            assert data.loc[1, col] == self.expected[col]

    def test_parser_empty_description(self):
        source = MockGeometry(
            polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))],
            properties=[{"id": 1, "description": None}],
        )
        view = text.ParseTextColumn(source, "description", self.key_mapping)
        data = view.get_data(**self.request)["features"]
        for col in self.expected:
            assert np.isnan(data.loc[1, col])

    def test_parser_empty_one_description(self):
        source = MockGeometry(
            polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))] * 2,
            properties=[
                {"id": 1, "description": None},
                {"id": 2, "description": self.description},
            ],
        )
        view = text.ParseTextColumn(source, "description", self.key_mapping)
        data = view.get_data(**self.request)["features"]
        for col in self.expected:
            assert np.isnan(data.loc[1, col])
            assert data.loc[2, col] == self.expected[col]

    def test_parser_empty_all_descriptions(self):
        source = MockGeometry(
            polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))] * 2,
            properties=[{"id": 1, "description": None}, {"id": 2, "description": None}],
        )
        view = text.ParseTextColumn(source, "description", self.key_mapping)
        data = view.get_data(**self.request)["features"]
        for col in self.expected:
            assert np.isnan(data.loc[1, col])
            assert np.isnan(data.loc[2, col])

    def test_parser_two_same(self):
        source = MockGeometry(
            polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))] * 2,
            properties=[
                {"id": 1, "description": self.description},
                {"id": 2, "description": self.description},
            ],
        )
        view = text.ParseTextColumn(source, "description", self.key_mapping)
        data = view.get_data(**self.request)["features"]
        self.assertEqual("object", str(data["model_name"].dtype))
        for col in self.expected:
            assert data.loc[1, col] == self.expected[col]
            assert data.loc[2, col] == self.expected[col]

    def test_parser_two_different(self):
        other_description = (
            "\nSimulation of Groningen"
            "\n\nmodelname=groningen 01"
            "\nduration=60\nstrength=120\nahn2=false"
        )
        source = MockGeometry(
            polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))] * 2,
            properties=[
                {"id": 1, "description": self.description},
                {"id": 2, "description": other_description},
            ],
        )
        view = text.ParseTextColumn(source, "description", self.key_mapping)
        data = view.get_data(**self.request)["features"]

        expected2 = {
            "model_name": "groningen 01",
            "rainfall_duration": 60,
            "rainfall_strength": 120,
            "ahn2_used": False,
        }

        for col in self.expected:
            assert data.loc[1, col] == self.expected[col]
            assert data.loc[2, col] == expected2[col]

    def test_parser_missing_and_null_keys(self):
        description = (
            "\nSimulation of Utrecht" "\n\nmodelname=null" "\nduration=60\nstrength=120"
        )
        source = MockGeometry(
            polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))],
            properties=[{"id": 1, "description": description}],
        )
        view = text.ParseTextColumn(source, "description", self.key_mapping)
        data = view.get_data(**self.request)
        record = data["features"].iloc[0]

        self.assertTrue(pd.isnull(record["model_name"]))
        self.assertEqual(record["rainfall_duration"], 60)
        self.assertEqual(record["rainfall_strength"], 120)
        self.assertTrue(pd.isnull(record["ahn2_used"]))

    def test_parser_into_same_column(self):
        view = text.ParseTextColumn(
            self.source, "description", {"modelname": "description"}
        )
        data = view.get_data(**self.request)["features"]
        self.assertEqual("object", str(data["description"].dtype))
        assert data.loc[1, "description"] == "rotterdam 01"

    def test_parser_into_same_column_non_existing(self):
        view = text.ParseTextColumn(
            self.source, "description", {"nonexisting": "description"}
        )
        data = view.get_data(**self.request)["features"]
        assert np.isnan(data.loc[1, "description"])

    def test_parser_into_same_column_empty(self):
        source = MockGeometry(
            polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))] * 2,
            properties=[{"id": 1, "model_name": None}],
        )
        view = text.ParseTextColumn(source, "model_name", self.key_mapping)
        data = view.get_data(**self.request)["features"]
        for col in self.expected:
            assert np.isnan(data.loc[1, col])
