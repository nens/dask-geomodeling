from datetime import datetime, timedelta

import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from shapely.geometry import box

from dask_geomodeling import raster
from dask_geomodeling.utils import shapely_transform, get_sr
from dask_geomodeling.raster.sources import MemorySource


@pytest.fixture
def source():
    yield MemorySource(
        data=np.full((1, 10, 10), 7, dtype=np.uint8),
        no_data_value=255,
        projection="EPSG:28992",
        pixel_size=10,
        pixel_origin=(135000, 456000),
    )

@pytest.fixture
def center():
    # the source raster has data at x 135000..135100 and y 456000..455900
    yield 135050, 455950


def test_place_attrs(source):
    # clip should propagate the (empty) extent of the store
    place = raster.Place(source, "EPSG:28992", (135000, 456000), [(0, 0)])
    assert place.extent is None  # TODO shifted extent
    assert place.geometry is None  # TODO shifted geometry


def test_place_exact(source, center):
    place = raster.Place(source, "EPSG:28992", center, [(50, 50)])
    values = place.get_data(
        mode="vals",
        bbox=(0, 0, 100, 100),
        projection="EPSG:28992",
        width=10,
        height=10,
    )["values"]
    assert (values == 7).all()


def test_place_half(source, center):
    place = raster.Place(source, "EPSG:28992", center, [(0, 50)])
    values = place.get_data(
        mode="vals",
        bbox=(0, 0, 100, 100),
        projection="EPSG:28992",
        width=5,
        height=5,
    )["values"]
    assert (values[:, :, :4] == 7).all()
    assert (values[:, :, 6:] == 255).all()