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


@pytest.mark.parametrize(
    "bbox,expected_tiles",
    [
        ((0.0, 0.0, 7.0, 7.0), [(0.0, 0.0, 7.0, 7.0)]),  # exact
        ((-7.0, 7.0, 0.0, 14.0), [(-7.0, 7.0, 0.0, 14.0)]),  # exact, shifted
        ((2.0, 7.0, 7.0, 14.0), [(2.0, 7.0, 7.0, 14.0)]),  # smaller in x1
        ((0.0, 7.0, 5.0, 14.0), [(0.0, 7.0, 5.0, 14.0)]),  # smaller in x2
        ((0.0, 9.0, 7.0, 14.0), [(0.0, 9.0, 7.0, 14.0)]),  # smaller in y1
        ((0.0, 7.0, 7.0, 12.0), [(0.0, 7.0, 7.0, 12.0)]),  # smaller in y2
        ((0.0, 0.0, 14.0, 7.0), [(0.0, 0.0, 7.0, 7.0), (7.0, 0.0, 14.0, 7.0)]),
        ((0.0, 0.0, 7.0, 14.0), [(0.0, 0.0, 7.0, 7.0), (0.0, 7.0, 7.0, 14.0)]),
        ((2.0, 0.0, 9.0, 7.0), [(2.0, 0.0, 7.0, 7.0), (7.0, 0.0, 9.0, 7.0)]),
        ((0.0, -2.0, 7.0, 5.0), [(0.0, -2.0, 7.0, 0.0), (0.0, 0.0, 7.0, 5.0)]),
        (
            (2.0, -2.0, 9.0, 5.0),
            [
                (2.0, -2.0, 7.0, 0.0),
                (2.0, 0.0, 7.0, 5.0),
                (7.0, -2.0, 9.0, 0.0),
                (7.0, 0.0, 9.0, 5.0),
            ],
        ),
    ],
)
def test_tiling(empty_source, bbox, expected_tiles):
    block = raster.RasterTiler(empty_source, 7, "EPSG:28992")
    s_r = block.get_sources_and_requests(
        mode="vals",
        bbox=bbox,
        width=int(bbox[2] - bbox[0]),
        height=int(bbox[3] - bbox[1]),
        projection="EPSG:28992",
    )
    assert [x[1]["bbox"] for x in s_r] == expected_tiles


@pytest.mark.parametrize(
    "cellsize,expected_tiles",
    [
        ((1, 1), [(3.0, 3.0, 15.0, 15.0)]),
        ((2, 2), [(2.0, 2.0, 16.0, 16.0)]),
        ((3, 3), [(3.0, 3.0, 15.0, 15.0)]),
        ((4, 4), [(0.0, 0.0, 16.0, 16.0)]),
        ((2, 3), [(2.0, 3.0, 16.0, 15.0)]),
        ((1, 4), [(3.0, 0.0, 15.0, 16.0)]),
    ],
)
def test_tiling_cellsize(empty_source, cellsize, expected_tiles):
    block = raster.RasterTiler(empty_source, 24, "EPSG:28992")
    s_r = block.get_sources_and_requests(
        mode="vals",
        bbox=(3.0, 3.0, 15.0, 15.0),
        width=int(12 / cellsize[0]),
        height=int(12 / cellsize[1]),
        projection="EPSG:28992",
    )
    assert [x[1]["bbox"] for x in s_r] == expected_tiles


@pytest.mark.parametrize(
    "topleft,expected_tiles",
    [
        ((0, 0), [(2.0, 2.0, 5.0, 5.0)]),
        ((0.1, 0), [(1.1, 2.0, 5.1, 5.0)]),
        ((0, 0.1), [(2.0, 1.1, 5.0, 5.1)]),
        ((0.1, 0.1), [(1.1, 1.1, 5.1, 5.1)]),
    ],
)
def test_tiling_topleft(empty_source, topleft, expected_tiles):
    block = raster.RasterTiler(empty_source, 7, "EPSG:28992", topleft)
    s_r = block.get_sources_and_requests(
        mode="vals",
        bbox=(2.0, 2.0, 5.0, 5.0),
        width=3,
        height=3,
        projection="EPSG:28992",
    )
    assert [x[1]["bbox"] for x in s_r] == expected_tiles
