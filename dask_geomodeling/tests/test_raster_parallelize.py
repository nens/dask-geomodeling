from datetime import datetime, timedelta

import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal

from dask_geomodeling import raster


@pytest.fixture
def empty_source():
    return raster.MemorySource(
        data=np.empty((0, 0, 0), dtype=np.uint8),
        no_data_value=255,
        projection="EPSG:28992",
        pixel_size=0.5,
        pixel_origin=(135000, 456000),
    )


def test_tiler_defaults(empty_source):
    block = raster.RasterTiler(empty_source, 10, "EPSG:28992")
    assert block.store is empty_source
    assert block.size == [10.0, 10.0]
    assert block.projection == "EPSG:28992"
    assert block.topleft == [0.0, 0.0]


def test_tiler_source_validation(empty_source):
    with pytest.raises(TypeError):
        raster.RasterTiler("a", 10, "EPSG:28992")


def test_tiler_size_validation(empty_source):
    with pytest.raises(ValueError):
        raster.RasterTiler(empty_source, "a", "EPSG:28992")
    with pytest.raises(ValueError):
        raster.RasterTiler(empty_source, 0, "EPSG:28992")
    with pytest.raises(ValueError):
        raster.RasterTiler(empty_source, [1], "EPSG:28992")
    with pytest.raises(ValueError):
        raster.RasterTiler(empty_source, [2, 3, 3], "EPSG:28992")


def test_tiler_projection_validation(empty_source):
    with pytest.raises(TypeError):
        raster.RasterTiler(empty_source, 10, 2)
    with pytest.raises(ValueError):
        raster.RasterTiler(empty_source, 10, "not-a-projection")


def test_tiler_topleft_validation(empty_source):
    with pytest.raises(TypeError):
        raster.RasterTiler(empty_source, 10, "EPSG:28992", 2)
    with pytest.raises(ValueError):
        raster.RasterTiler(empty_source, 10, "EPSG:28992", [2])
    with pytest.raises(ValueError):
        raster.RasterTiler(empty_source, 10, "EPSG:28992", [2, 3, 4])
    with pytest.raises(ValueError):
        raster.RasterTiler(empty_source, 10, "EPSG:28992", [2, "a"])
