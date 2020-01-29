from datetime import datetime, timedelta

import numpy as np
import pytest
from numpy.testing import assert_equal
from shapely.geometry import box

from dask_geomodeling import raster
from dask_geomodeling.raster.sources import MemorySource


@pytest.fixture
def source():
    bands = 3
    time_first = datetime(2000, 1, 1)
    time_delta = timedelta(hours=1)
    yield MemorySource(
        data=[
            np.full((10, 10), 1, dtype=np.uint8),
            np.full((10, 10), 7, dtype=np.uint8),
            np.full((10, 10), 255, dtype=np.uint8),
        ],
        no_data_value=255,
        projection="EPSG:28992",
        pixel_size=0.5,
        pixel_origin=(135000, 456000),
        time_first=time_first,
        time_delta=time_delta,
        metadata=["Testmeta for band {}".format(i) for i in range(bands)],
    )


@pytest.fixture
def empty_source():
    yield MemorySource(
        data=np.empty((0, 0, 0), dtype=np.uint8),
        no_data_value=255,
        projection="EPSG:28992",
        pixel_size=0.5,
        pixel_origin=(135000, 456000),
    )


@pytest.fixture
def nodata_source():
    time_first = datetime(2000, 1, 1)
    time_delta = timedelta(hours=1)
    yield MemorySource(
        data=np.full((3, 10, 10), 255, dtype=np.uint8),
        no_data_value=255,
        projection="EPSG:28992",
        pixel_size=0.5,
        pixel_origin=(135000, 456000),
        time_first=time_first,
        time_delta=time_delta,
    )


@pytest.fixture
def vals_request():
    bands = 3
    time_first = datetime(2000, 1, 1)
    time_delta = timedelta(hours=1)
    yield {
        "mode": "vals",
        "start": time_first,
        "stop": time_first + bands * time_delta,
        "width": 4,
        "height": 6,
        "bbox": (135000, 456000 - 3, 135000 + 2, 456000),
        "projection": "EPSG:28992",
    }


@pytest.fixture
def point_request():
    bands = 3
    time_first = datetime(2000, 1, 1)
    time_delta = timedelta(hours=1)
    yield {
        "mode": "vals",
        "start": time_first,
        "stop": time_first + bands * time_delta,
        "width": 1,
        "height": 1,
        "bbox": (135001, 455999, 135001, 455999),
        "projection": "EPSG:28992",
    }


@pytest.fixture
def vals_request_none():
    bands = 3
    time_first = datetime(2001, 1, 1)
    time_delta = timedelta(hours=1)
    yield {
        "mode": "vals",
        "start": time_first,
        "stop": time_first + bands * time_delta,
        "width": 4,
        "height": 6,
        "bbox": (135000, 456000 - 3, 135000 + 2, 456000),
        "projection": "EPSG:28992",
    }


@pytest.fixture
def expected_meta():
    bands = 3
    return ["Testmeta for band {}".format(i) for i in range(bands)]


@pytest.fixture
def expected_time():
    bands = 3
    time_first = datetime(2000, 1, 1)
    time_delta = timedelta(hours=1)
    return [time_first + i * time_delta for i in range(bands)]


def test_clip_attrs_store_empty(source, empty_source):
    # clip should propagate the (empty) extent of the store
    clip = raster.Clip(empty_source, source)
    assert clip.extent is None
    assert clip.geometry is None


def test_clip_attrs_mask_empty(source, empty_source):
    # clip should propagate the (empty) extent of the clipping mask
    clip = raster.Clip(source, empty_source)
    assert clip.extent is None
    assert clip.geometry is None


def test_clip_attrs_intersects(source, empty_source):
    # create a raster in that only partially overlaps the store
    clipping_mask = MemorySource(
        data=source.data,
        no_data_value=source.no_data_value,
        projection="EPSG:28992",
        pixel_size=source.pixel_size,
        pixel_origin=[o + 3 for o in source.pixel_origin],
        time_first=source.time_first,
        time_delta=source.time_delta,
    )
    clip = raster.Clip(source, clipping_mask)
    expected_extent = (
        clipping_mask.extent[0],
        clipping_mask.extent[1],
        source.extent[2],
        source.extent[3],
    )
    expected_geometry = source.geometry.Intersection(clipping_mask.geometry)
    assert clip.extent == expected_extent
    assert clip.geometry.ExportToWkt() == expected_geometry.ExportToWkt()


def test_clip_attrs_with_reprojection(source, empty_source):
    # create a raster in WGS84 that contains the store
    clipping_mask = MemorySource(
        data=source.data,
        no_data_value=source.no_data_value,
        projection="EPSG:4326",
        pixel_size=1,
        pixel_origin=(4, 54),
        time_first=source.time_first,
        time_delta=source.time_delta,
    )
    clip = raster.Clip(source, clipping_mask)
    assert clip.extent == source.extent
    assert clip.geometry.GetEnvelope() == source.geometry.GetEnvelope()


def test_clip_attrs_no_intersection(source, empty_source):
    # create a raster in that does not overlap the store
    clipping_mask = MemorySource(
        data=source.data,
        no_data_value=source.no_data_value,
        projection="EPSG:28992",
        pixel_size=source.pixel_size,
        pixel_origin=[o + 5 for o in source.pixel_origin],
        time_first=source.time_first,
        time_delta=source.time_delta,
    )
    clip = raster.Clip(source, clipping_mask)
    assert clip.extent is None
    assert clip.geometry is None


def test_clip_empty_source(source, empty_source, vals_request):
    clip = raster.Clip(empty_source, source)
    assert clip.get_data(**vals_request) is None


def test_clip_with_empty_mask(source, empty_source, vals_request):
    clip = raster.Clip(source, empty_source)
    assert clip.get_data(**vals_request) is None


def test_clip_with_nodata(source, nodata_source, vals_request):
    # the clipping mask has nodata everywhere (everything will be masked)
    clip = raster.Clip(source, nodata_source)
    assert_equal(clip.get_data(**vals_request)["values"], 255)


def test_clip_with_data(source, nodata_source, vals_request):
    # the clipping mask has data everywhere (nothing will be masked)
    clip = raster.Clip(source, source)
    assert_equal(clip.get_data(**vals_request)["values"][:, 0, 0], [1, 7, 255])


def test_clip_with_bool(source, vals_request):
    clip = raster.Clip(source, source == 7)
    assert_equal(clip.get_data(**vals_request)["values"][:, 0, 0], [255, 7, 255])


def test_clip_meta_request(source, vals_request, expected_meta):
    clip = raster.Clip(source, source)
    vals_request["mode"] = "meta"
    assert clip.get_data(**vals_request)["meta"] == expected_meta


def test_clip_time_request(source, vals_request, expected_time):
    clip = raster.Clip(source, source)
    vals_request["mode"] = "time"
    assert clip.get_data(**vals_request)["time"] == expected_time


def test_reclassify(source, vals_request):
    view = raster.Reclassify(store=source, data=[[7, 1000]])
    data = view.get_data(**vals_request)

    assert_equal(data["values"][:, 0, 0], [1, 1000, data["no_data_value"]])


def test_reclassify_select(source, vals_request):
    view = raster.Reclassify(store=source, data=[[7, 1000]], select=True)
    data = view.get_data(**vals_request)

    expected = [data["no_data_value"], 1000, data["no_data_value"]]
    assert_equal(data["values"][:, 0, 0], expected)


def test_reclassify_to_float(source, vals_request):
    view = raster.Reclassify(store=source, data=[[7, 8.2]])
    data = view.get_data(**vals_request)

    assert_equal(data["values"][:, 0, 0], [1.0, 8.2, data["no_data_value"]])


def test_reclassify_bool(source, vals_request):
    source_bool = source == 7
    view = raster.Reclassify(store=source_bool, data=[[True, 1000]])
    data = view.get_data(**vals_request)

    assert_equal(data["values"][:, 0, 0], [0, 1000, 0])


def test_reclassify_int32(source, vals_request):
    # this will have a high fillvalue that may lead to a MemoryError
    source_int32 = source * 1
    assert source_int32.dtype == np.int32

    view = raster.Reclassify(store=source_int32, data=[[7, 1000]])
    data = view.get_data(**vals_request)

    assert_equal(data["values"][:, 0, 0], [1, 1000, data["no_data_value"]])


def test_reclassify_float_raster(source):
    source_float = source / 2
    assert source_float.dtype == np.float32
    with pytest.raises(TypeError):
        raster.Reclassify(store=source_float, data=[[7.0, 1000]])


def test_reclassify_float_data(source):
    with pytest.raises(TypeError):
        raster.Reclassify(store=source, data=[[7.4, 1000]])


def test_reclassify_wrong_mapping_shape(source):
    with pytest.raises(ValueError):
        raster.Reclassify(store=source, data=[[[7, 1000]], [1, 100]])


def test_reclassify_meta_request(source, vals_request, expected_meta):
    view = raster.Reclassify(store=source, data=[[7, 1000]])
    vals_request["mode"] = "meta"
    assert view.get_data(**vals_request)["meta"] == expected_meta


def test_reclassify_time_request(source, vals_request, expected_time):
    view = raster.Reclassify(store=source, data=[[7, 1000]])
    vals_request["mode"] = "time"
    assert view.get_data(**vals_request)["time"] == expected_time


def test_rasterize_wkt_vals(vals_request):
    # vals_request has width=4, height=6 and cell size of 0.5
    # we place a rectangle of 2 x 3 with corner at x=1, y=2
    view = raster.RasterizeWKT(
        box(135000.5, 455998, 135001.5, 455999.5).wkt, "EPSG:28992"
    )
    vals_request["start"] = vals_request["stop"] = None
    actual = view.get_data(**vals_request)
    assert actual["values"][0].astype(int).tolist() == [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]


def test_rasterize_wkt_vals_no_intersection(vals_request):
    view = raster.RasterizeWKT(box(135004, 455995, 135004.5, 455996).wkt, "EPSG:28992")
    vals_request["start"] = vals_request["stop"] = None
    actual = view.get_data(**vals_request)
    assert ~actual["values"].any()


@pytest.mark.parametrize(
    "bbox,expected",
    [
        [(135000.5, 455998, 135001.5, 455999.5), True],
        [(135000.5, 455998, 135000.9, 455998.9), False],
    ],
)
def test_rasterize_wkt_point(point_request, bbox, expected):
    view = raster.RasterizeWKT(box(*bbox).wkt, "EPSG:28992")
    point_request["start"] = point_request["stop"] = None
    actual = view.get_data(**point_request)
    assert actual["values"].tolist() == [[[expected]]]
