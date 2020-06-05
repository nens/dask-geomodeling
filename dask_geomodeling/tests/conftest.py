from datetime import datetime, timedelta

import numpy as np
import pytest

from dask_geomodeling.raster.sources import MemorySource


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def empty_source():
    yield MemorySource(
        data=np.empty((0, 0, 0), dtype=np.uint8),
        no_data_value=255,
        projection="EPSG:28992",
        pixel_size=0.5,
        pixel_origin=(135000, 456000),
    )


@pytest.fixture(scope="session")
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
