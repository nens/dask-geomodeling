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
def expected_meta():
    bands = 3
    return ["Testmeta for band {}".format(i) for i in range(bands)]


@pytest.fixture
def expected_time():
    bands = 3
    time_first = datetime(2000, 1, 1)
    time_delta = timedelta(hours=1)
    return [time_first + i * time_delta for i in range(bands)]


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
