from datetime import datetime

import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal

from dask_geomodeling import raster


def check_sources_and_requests(sources_and_requests, expected_bboxes, cellsize=(1, 1)):
    for (_, req), expected in zip(list(sources_and_requests)[1:], expected_bboxes):
        assert req["bbox"] == expected
        assert req["width"] == int((expected[2] - expected[0]) / cellsize[0])
        assert req["height"] == int((expected[3] - expected[1]) / cellsize[1])


def test_tiler_defaults(empty_source):
    block = raster.RasterTiler(empty_source, 10, "EPSG:28992")
    assert block.store is empty_source
    assert block.size == [10.0, 10.0]
    assert block.projection == "EPSG:28992"
    assert block.corner == [0.0, 0.0]


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


def test_tiler_corner_validation(empty_source):
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
    check_sources_and_requests(s_r, expected_tiles)


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
    check_sources_and_requests(s_r, expected_tiles, cellsize)


@pytest.mark.parametrize(
    "corner,expected_tiles",
    [
        ((0, 0), [(2.0, 2.0, 5.0, 5.0)]),
        ((0.1, 0), [(1.1, 2.0, 5.1, 5.0)]),
        ((0, 0.1), [(2.0, 1.1, 5.0, 5.1)]),
        ((0.1, 0.1), [(1.1, 1.1, 5.1, 5.1)]),
    ],
)
def test_tiling_corner(empty_source, corner, expected_tiles):
    block = raster.RasterTiler(empty_source, 7, "EPSG:28992", corner)
    s_r = block.get_sources_and_requests(
        mode="vals",
        bbox=(2.0, 2.0, 5.0, 5.0),
        width=3,
        height=3,
        projection="EPSG:28992",
    )
    check_sources_and_requests(s_r, expected_tiles)


def test_cell_tile_mismatch(empty_source):
    block = raster.RasterTiler(empty_source, 7, "EPSG:28992")
    s_r = block.get_sources_and_requests(
        mode="vals",
        bbox=(0.0, 0.0, 6.0, 6.0),
        width=2,  # cellsize_x = 3 will be adjusted with round(7 / 3) to 3.5
        height=3,  # cellsize_y = 2 will be adjusted with round(7 / 2) to 1.75
        projection="EPSG:28992",
    )
    s_r = list(s_r)[1:]
    assert len(s_r) == 1
    assert s_r[0][1]["width"] == 2
    assert s_r[0][1]["height"] == 4


@pytest.mark.parametrize(
    "bbox_offset",
    [
        (0, -5, 5, 0),  # covers precisely the 5x5 meters of data in the source
        (0, -5, 6, 0),
        (0, -6, 5, 0),
        (-1, -5, 5, 0),
        (0, -5, 5, 1),
    ],
)
def test_tiling_process(source, bbox_offset):
    # piece back together tiles with nodata at the negative y edge
    block = raster.RasterTiler(source, 2, "EPSG:28992")
    request = dict(
        mode="vals",
        bbox=(
            source.pixel_origin[0] + bbox_offset[0],
            source.pixel_origin[1] + bbox_offset[1],
            source.pixel_origin[0] + bbox_offset[2],
            source.pixel_origin[1] + bbox_offset[3],
        ),
        width=(bbox_offset[2] - bbox_offset[0]) * 2,  # 0.5 m resolution
        height=(bbox_offset[3] - bbox_offset[1]) * 2,  # 0.5 m resolution
        projection="EPSG:28992",
        start=source.period[0],
    )
    actual = block.get_data(**request)
    expected = source.get_data(**request)
    assert_equal(actual["values"], expected["values"])
    assert actual["no_data_value"] == expected["no_data_value"]
