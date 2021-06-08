from datetime import datetime

import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from shapely.geometry import box

from dask_geomodeling import raster
from dask_geomodeling.utils import shapely_transform, get_sr
from dask_geomodeling.raster.sources import MemorySource


def test_clip_attrs_store_empty(source, empty_source):
    # clip should propagate the (empty) extent of the store
    clip = raster.Clip(empty_source, raster.Snap(source, empty_source))
    assert clip.extent is None
    assert clip.geometry is None


def test_clip_attrs_mask_empty(source, empty_source):
    # clip should propagate the (empty) extent of the clipping mask
    clip = raster.Clip(source, raster.Snap(empty_source, source))
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


def test_clip_attrs_no_intersection(source):
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


def test_clip_matching_timedelta(source):
    clip = raster.Clip(source, source == 7)
    assert clip.timedelta == source.timedelta


def test_clip_unequal_timedelta(source, empty_source):
    # clip checks for matching timedeltas; test that here
    # NB: note that `source` is temporal and `empty_source` is not
    with pytest.raises(ValueError, match=".*resolution of the clipping.*"):
        raster.Clip(source, empty_source)
    with pytest.raises(ValueError, match=".*resolution of the clipping.*"):
        raster.Clip(empty_source, source)


def test_clip_empty_source(source, empty_source, vals_request):
    clip = raster.Clip(empty_source, raster.Snap(source, empty_source))
    assert clip.get_data(**vals_request) is None


def test_clip_with_empty_mask(source, empty_source, vals_request):
    clip = raster.Clip(source, raster.Snap(empty_source, source))
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


def test_clip_partial_temporal_overlap(source, vals_request):
    # create a clipping mask in that temporally does not overlap the store
    clipping_mask = MemorySource(
        data=source.data,
        no_data_value=source.no_data_value,
        projection=source.projection,
        pixel_size=source.pixel_size,
        pixel_origin=source.pixel_origin,
        time_first=source.time_first + source.time_delta,
        time_delta=source.time_delta,
    )
    clip = raster.Clip(source, clipping_mask)
    assert clip.period == (clipping_mask.period[0], source.period[1])
    assert clip.get_data(**vals_request)["values"][:, 0, 0].tolist() == [7, 255]


def test_clip_no_temporal_overlap(source, vals_request):
    # create a clipping mask in that temporally does not overlap the store
    clipping_mask = MemorySource(
        data=source.data,
        no_data_value=source.no_data_value,
        projection=source.projection,
        pixel_size=source.pixel_size,
        pixel_origin=source.pixel_origin,
        time_first=source.time_first + 10 * source.time_delta,
        time_delta=source.time_delta,
    )
    clip = raster.Clip(source, clipping_mask)
    assert clip.period is None
    assert clip.get_data(**vals_request) is None


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


@pytest.mark.parametrize("projection", ["EPSG:28992", "EPSG:4326", "EPSG:3857"])
def test_rasterize_wkt_vals(vals_request, projection):
    # vals_request has width=4, height=6 and cell size of 0.5
    # we place a rectangle of 2 x 3 with corner at x=1, y=2
    view = raster.RasterizeWKT(
        shapely_transform(
            box(135000.5, 455998, 135001.5, 455999.5), "EPSG:28992", projection
        ).wkt,
        projection,
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


def test_rasterize_wkt_attrs():
    geom = box(135004, 455995, 135004.5, 455996)
    view = raster.RasterizeWKT(geom.wkt, "EPSG:28992")
    assert view.projection == "EPSG:28992"
    assert_almost_equal(view.geometry.GetEnvelope(), [135004, 135004.5, 455995, 455996])
    assert view.geometry.GetSpatialReference().IsSame(get_sr("EPSG:28992"))
    assert view.dtype == bool
    assert view.fillvalue is None
    assert_almost_equal(
        view.extent, shapely_transform(geom, "EPSG:28992", "EPSG:4326").bounds
    )
    assert view.timedelta is None
    assert view.period == (datetime(1970, 1, 1), datetime(1970, 1, 1))
