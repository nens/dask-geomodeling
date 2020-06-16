import pytest
from numpy.testing import assert_equal

from dask_geomodeling import raster

## See conftest.py for the common fixtures


def check_sources_and_requests(sources_and_requests, expected_bboxes, cellsize=(1, 1)):
    for (_, req), expected in zip(list(sources_and_requests)[1:], expected_bboxes):
        assert req["bbox"] == expected
        assert req["width"] == int((expected[2] - expected[0]) / cellsize[0])
        assert req["height"] == int((expected[3] - expected[1]) / cellsize[1])


def test_tiler_defaults(empty_source):
    block = raster.RasterTiler(empty_source, 10)
    assert block.store is empty_source
    assert block.tile_size == [10.0, 10.0]


def test_tiler_source_validation(empty_source):
    with pytest.raises(TypeError):
        raster.RasterTiler("a", 10)


def test_tiler_tile_size_validation(empty_source):
    with pytest.raises(ValueError):
        raster.RasterTiler(empty_source, "a")
    with pytest.raises(ValueError):
        raster.RasterTiler(empty_source, 0)
    with pytest.raises(ValueError):
        raster.RasterTiler(empty_source, [1])
    with pytest.raises(ValueError):
        raster.RasterTiler(empty_source, [2, 3, 3])


@pytest.mark.parametrize(
    "bbox,expected_tiles",
    [
        ((0.0, 0.0, 7.0, 7.0), [(0.0, 0.0, 7.0, 7.0)]),  # exact
        ((2.0, -1.0, 9.0, 6.0), [(2.0, -1.0, 9.0, 6.0)]),  # exact, shifted
        ((2.0, 7.0, 7.0, 14.0), [(2.0, 7.0, 7.0, 14.0)]),  # smaller in x1
        ((0.0, 7.0, 5.0, 14.0), [(0.0, 7.0, 5.0, 14.0)]),  # smaller in x2
        ((0.0, 9.0, 7.0, 14.0), [(0.0, 9.0, 7.0, 14.0)]),  # smaller in y1
        ((0.0, 7.0, 7.0, 12.0), [(0.0, 7.0, 7.0, 12.0)]),  # smaller in y2
        ((0.0, 0.0, 14.0, 7.0), [(0.0, 0.0, 7.0, 7.0), (7.0, 0.0, 14.0, 7.0)]),
        ((0.0, 0.0, 7.0, 14.0), [(0.0, 0.0, 7.0, 7.0), (0.0, 7.0, 7.0, 14.0)]),
        (
            (10.0, -10.0, 20.0, 2.0),
            [
                (10.0, -10.0, 17.0, -3.0),
                (10.0, -3.0, 17.0, 2.0),
                (17.0, -10.0, 20.0, -3.0),
                (17.0, -3.0, 20.0, 2.0),
            ],
        ),
    ],
)
def test_tiler(empty_source, bbox, expected_tiles):
    block = raster.RasterTiler(empty_source, 7)
    s_r = block.get_sources_and_requests(
        mode="vals",
        bbox=bbox,
        width=int(bbox[2] - bbox[0]),
        height=int(bbox[3] - bbox[1]),
        projection="EPSG:28992",
    )
    check_sources_and_requests(s_r, expected_tiles)


@pytest.mark.parametrize(
    "cellsize", [((1, 1)), ((2, 2)), ((3, 3)), ((4, 4)), ((2, 3)), ((1, 4))]
)
def test_tiler_cellsize(empty_source, cellsize):
    block = raster.RasterTiler(empty_source, 24)
    s_r = block.get_sources_and_requests(
        mode="vals",
        bbox=(3.0, 3.0, 15.0, 15.0),
        width=int(12 / cellsize[0]),
        height=int(12 / cellsize[1]),
        projection="EPSG:28992",
    )
    check_sources_and_requests(s_r, [(3.0, 3.0, 15.0, 15.0)], cellsize)


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
def test_tiler_process(source, bbox_offset):
    # piece back together tiles with nodata at the negative y edge
    block = raster.RasterTiler(source, 2)
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


def test_tiler_point_request(source, point_request):
    view = raster.RasterTiler(source, 2)
    actual = view.get_data(**point_request)
    assert actual["values"].tolist() == [[[1]], [[7]], [[255]]]


def test_tiler_meta_request(source, vals_request, expected_meta):
    tiler = raster.RasterTiler(source, 2)
    vals_request["mode"] = "meta"
    assert tiler.get_data(**vals_request)["meta"] == expected_meta


def test_tiler_time_request(source, vals_request, expected_time):
    tiler = raster.RasterTiler(source, 2)
    vals_request["mode"] = "time"
    assert tiler.get_data(**vals_request)["time"] == expected_time
