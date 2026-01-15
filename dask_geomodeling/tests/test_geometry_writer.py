import os
import json
import geopandas as gpd
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from shapely.geometry import box, Polygon
from geopandas.testing import assert_geodataframe_equal
from dask_geomodeling import utils
from dask_geomodeling.geometry import GeometryFileWriter
from dask_geomodeling.geometry.writer import to_file
from dask_geomodeling.tests.factories import (
    MockGeometry,
    setup_temp_root,
    teardown_temp_root,
)


@pytest.fixture(scope="module")
def temp_root():
    """Module-level fixture for temporary root directory"""
    root = setup_temp_root()
    yield root
    teardown_temp_root(root)


@pytest.fixture
def test_features():
    """Fixture providing test GeoDataFrame"""
    return gpd.GeoDataFrame(
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


@pytest.fixture
def expected_features(test_features):
    """Fixture providing expected test data"""
    expected = test_features.copy()
    expected["cat"] = expected["cat"].astype(str)
    return expected


@pytest.fixture
def test_path(temp_root, request):
    """Fixture providing unique test path for each test"""
    return os.path.join(temp_root, request.node.name)


# Tests for GeometryFileWriter


def test_writer_non_available_extension(test_path, test_features):
    with pytest.raises(KeyError):
        writer = GeometryFileWriter(test_path + ".bmp")
        writer.write(test_features)


def test_writer_geojson(test_path, test_features, expected_features):
    writer = GeometryFileWriter(test_path + ".geojson")
    writer.write(test_features)

    actual = gpd.read_file(test_path + ".geojson", fid_as_index=True)

    expected = expected_features.to_crs("EPSG:4326")
    # Geopandas/pyogrio reads 'id' as index and as field
    expected.insert(6, "id", expected.index)
    # GeoJSON is read as int32
    for int_col in ["id", "int"]:
        expected[int_col] = expected[int_col].astype('int32')

    # Because of the reprojection, use a precision comparison for geometries
    actual.geometry = actual.geometry.set_precision(0.000001)
    expected.geometry = expected.geometry.set_precision(0.000001)
    assert_geodataframe_equal(actual, expected)
    assert actual.crs == "EPSG:4326"


@pytest.mark.skipif(
    "gpkg" not in GeometryFileWriter.supported_extensions,
    reason="This version of Pyogrio/GDAL does not support geopackages.",
)
def test_writer_geopackage(test_path, test_features, expected_features):
    writer = GeometryFileWriter(test_path + ".gpkg")
    writer.write(test_features)

    actual = gpd.read_file(test_path + ".gpkg", fid_as_index=True)

    # we expect lists and dicts to be serialized to JSON
    expected = expected_features.copy()
    expected["lst"] = expected["lst"].map(json.dumps)
    expected["dct"] = expected["dct"].map(json.dumps)

    assert_geodataframe_equal(actual, expected)
    # compare projections
    assert actual.crs == expected.crs


def test_writer_shapefile(test_path, test_features, expected_features):
    writer = GeometryFileWriter(test_path + ".shp")
    writer.write(test_features)

    actual = gpd.read_file(test_path + ".shp", fid_as_index=True)
    # We don't write the index for shapefiles, instead it is stored as 'fid' field
    expected = expected_features.copy()
    expected.insert(6, "fid", expected.index)
    expected.index = [0, 1]
    expected.index.names = ["fid"]
    # we expect lists and dicts to be serialized to JSON
    expected["lst"] = expected["lst"].map(json.dumps)
    expected["dct"] = expected["dct"].map(json.dumps)

    # compare dataframes without checking the order of records / columns
    assert_geodataframe_equal(actual, expected)
    # compare projections
    assert actual.crs == expected.crs


@pytest.mark.skipif(
    "gml" not in GeometryFileWriter.supported_extensions,
    reason="This version of Pyogrio/GDAL does not support GML.",
)
def test_writer_gml(test_path, test_features, expected_features):
    writer = GeometryFileWriter(test_path + ".gml")
    writer.write(test_features)

    actual = gpd.read_file(test_path + ".gml", fid_as_index=True)
    del actual["gml_id"]
    # We don't write the index for shapefiles, instead it is stored as 'fid' field
    expected = expected_features.copy()
    expected.insert(6, "fid", expected.index)
    expected.index = [0, 1]
    expected.index.names = ["fid"]
    # we expect lists and dicts to be serialized to JSON
    expected["lst"] = expected["lst"].map(json.dumps)
    expected["dct"] = expected["dct"].map(json.dumps)

    # compare dataframes without checking the order of records / columns
    assert_geodataframe_equal(actual, expected)


def test_writer_fields_non_available(test_path, test_features):
    with pytest.raises(KeyError):
        writer = GeometryFileWriter(test_path + ".shp", fields={"target": "nonexisting"})
        writer.write(test_features)


def test_writer_custom_fields(test_path, test_features):
    writer = GeometryFileWriter(
        test_path + ".geojson",
        fields={"target": "str", "int1": "int", "int2": "int"},
    )
    writer.write(test_features)

    actual = gpd.read_file(test_path + ".geojson", fid_as_index=True)
    assert set(actual.columns) == {"geometry", "target", "int1", "int2", "id"}


def test_writer_custom_fields_refers_index(test_path, test_features):
    writer = GeometryFileWriter(
        test_path + ".geojson",
        fields={"target": "fid"},
    )
    writer.write(test_features)

    actual = gpd.read_file(test_path + ".geojson", fid_as_index=True)
    assert set(actual.columns) == {"geometry", "target", "id"}


def test_writer_custom_fields_overwrite_index(test_path, test_features):
    writer = GeometryFileWriter(
        test_path + ".geojson",
        fields={"id": "int"},
    )
    writer.write(test_features)

    actual = gpd.read_file(test_path + ".geojson", fid_as_index=True)
    assert set(actual.columns) == {"geometry", "id"}
    assert actual["id"].tolist() == [5, 7]



@pytest.fixture
def mock_geometry_source():
    """Fixture providing MockGeometry source"""
    polygons = [
        ((0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)),
        ((10.0, 10.0), (10.0, 12.0), (12.0, 12.0), (12.0, 10.0)),
    ]
    properties = [
        {"int": 5, "float": 3.2, "str": "bla", "lst": [1], "dct": {"a": "b"}},
        {"int": 7, "float": 5.2, "str": "bla2", "lst": [2], "dct": {}},
    ]
    return MockGeometry(polygons, projection="EPSG:3857", properties=properties)


@pytest.fixture
def request_params():
    """Fixture providing request parameters"""
    return {
        "mode": "intersects",
        "projection": "EPSG:3857",
        "geometry": box(0, 0, 2, 2),
    }


@patch('dask_geomodeling.geometry.writer.GeometryFileWriter')
def test_to_file_basic(mock_writer_class, test_path, mock_geometry_source, request_params):
    """Test basic to_file call without tiling"""
    mock_writer = Mock()
    mock_writer_class.return_value = mock_writer
    
    to_file(mock_geometry_source, test_path + ".geojson", **request_params)
    
    # Check that GeometryFileWriter was instantiated with correct args
    mock_writer_class.assert_called_once_with(
        utils.safe_abspath(test_path + ".geojson"), 
        fields=None
    )
    
    # Check that write was called once
    assert mock_writer.write.call_count == 1
    
    # Get the features that were passed to write
    call_args = mock_writer.write.call_args[0]
    features = call_args[0]
    
    # Verify it's a GeoDataFrame with expected data
    assert isinstance(features, gpd.GeoDataFrame)
    # The request filters to only features within box(0, 0, 2, 2)
    assert len(features) == 1
    assert "int" in features.columns
    assert "geometry" in features.columns


@patch('dask_geomodeling.geometry.writer.GeometryFileWriter')
def test_to_file_with_fields(mock_writer_class, test_path, mock_geometry_source, request_params):
    """Test to_file with custom field mapping"""
    mock_writer = Mock()
    mock_writer_class.return_value = mock_writer
    
    fields = {"output_int": "int", "output_str": "str"}
    to_file(mock_geometry_source, test_path + ".geojson", fields=fields, **request_params)
    
    # Check that fields were passed to GeometryFileWriter
    mock_writer_class.assert_called_once_with(
        utils.safe_abspath(test_path + ".geojson"), 
        fields=fields
    )
    
    # Check that write was called
    mock_writer.write.assert_called_once()


@patch('dask_geomodeling.geometry.writer.GeometryFileWriter')
@patch('dask_geomodeling.geometry.writer.GeometryTiler')
def test_to_file_with_tiling(mock_tiler_class, mock_writer_class, test_path, mock_geometry_source, request_params):
    """Test to_file with tiling enabled"""
    mock_writer = Mock()
    mock_writer_class.return_value = mock_writer
    
    mock_tiler = Mock()
    mock_tiler.get_data.return_value = {"features": gpd.GeoDataFrame()}
    mock_tiler_class.return_value = mock_tiler
    
    tile_size = 10
    to_file(mock_geometry_source, test_path + ".geojson", tile_size=tile_size, **request_params)
    
    # Check that GeometryTiler was instantiated
    mock_tiler_class.assert_called_once_with(
        mock_geometry_source, 
        tile_size, 
        request_params["projection"]
    )
    
    # Check that get_data was called on the tiler, not the source
    mock_tiler.get_data.assert_called_once()
    
    # Check that write was called
    mock_writer.write.assert_called_once()


@patch('dask_geomodeling.geometry.writer.GeometryFileWriter')
def test_to_file_dry_run(mock_writer_class, test_path, mock_geometry_source, request_params):
    """Test dry_run doesn't write anything"""
    mock_writer = Mock()
    mock_writer_class.return_value = mock_writer
    
    to_file(mock_geometry_source, test_path + ".geojson", dry_run=True, **request_params)
    
    # Check that GeometryFileWriter was still instantiated (validation happens)
    mock_writer_class.assert_called_once()
    
    # Check that write was NOT called
    mock_writer.write.assert_not_called()


def test_to_file_target_exists(test_path, mock_geometry_source, request_params):
    """Test that FileExistsError is raised if target exists"""
    # Create the target file
    with open(test_path + ".geojson", "w") as f:
        f.write("{}")
    
    with pytest.raises(FileExistsError, match="already exists"):
        to_file(mock_geometry_source, test_path + ".geojson", **request_params)


def test_to_file_target_dir_not_exists(temp_root, mock_geometry_source, request_params):
    """Test that FileNotFoundError is raised if target directory doesn't exist"""
    nonexistent_path = os.path.join(temp_root, "nonexistent", "file.geojson")
    
    with pytest.raises(FileNotFoundError, match="does not exist"):
        to_file(mock_geometry_source, nonexistent_path, **request_params)
