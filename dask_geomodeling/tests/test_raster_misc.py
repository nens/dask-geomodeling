from datetime import datetime, timedelta

import numpy as np
import pytest
from numpy.testing import assert_equal

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


def test_clip_extent_attr(source, empty_source):
    # clip should propagate the extent of the clipping mask
    clip = raster.Clip(empty_source, source)
    assert clip.extent == source.extent
    assert clip.extent != empty_source.extent


def test_clip_geometry_attr(source, empty_source):
    # clip should propagate the geometry of the clipping mask
    clip = raster.Clip(empty_source, source)
    assert clip.geometry.ExportToWkt() == source.geometry.ExportToWkt()
    assert clip.geometry.ExportToWkt() != empty_source.geometry.ExportToWkt()


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
