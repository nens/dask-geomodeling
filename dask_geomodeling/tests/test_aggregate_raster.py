from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
import pytest
from numpy.testing import assert_almost_equal
from shapely.geometry import box, Point
import numpy as np
import pandas as pd

from dask import config

from dask_geomodeling.tests.factories import (
    MockGeometry,
    MockRaster,
)

from dask_geomodeling.raster import MemorySource
from dask_geomodeling.geometry.aggregate import bucketize
from dask_geomodeling.geometry import AggregateRaster, AggregateRasterAboveThreshold

try:
    from pandas.testing import assert_series_equal
except ImportError:
    from pandas.util.testing import assert_series_equal


@pytest.fixture
def constant_raster():
    return MockRaster(
        origin=Datetime(2018, 1, 1), timedelta=Timedelta(hours=1), bands=1
    )


@pytest.fixture
def geometry_source():
    return MockGeometry(
        polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))],
        properties=[{"id": 1}],
    )


@pytest.fixture
def geometry_request():
    return dict(
        mode="intersects",
        projection="EPSG:3857",
        geometry=box(0, 0, 10, 10),
    )


@pytest.fixture
def aggregate_raster(geometry_source, constant_raster):
    return AggregateRaster(
        source=geometry_source, raster=constant_raster, statistic="sum"
    )


@pytest.fixture
def range_raster():
    return MockRaster(
        origin=Datetime(2018, 1, 1),
        timedelta=Timedelta(hours=1),
        bands=1,
        value=np.indices((10, 10))[0].astype(float),
    )


@pytest.fixture
def nodata_raster():
    return MockRaster(
        origin=Datetime(2018, 1, 1),
        timedelta=Timedelta(hours=1),
        bands=1,
        value=255,  # nodata
    )


def test_arg_types(geometry_source, constant_raster):
    with pytest.raises(TypeError):
        AggregateRaster(geometry_source, None)

    with pytest.raises(TypeError):
        AggregateRaster(None, constant_raster)

    with pytest.raises(TypeError):
        AggregateRaster(
            geometry_source,
            constant_raster,
            statistic=None,
        )

    with pytest.raises(TypeError):
        AggregateRaster(
            geometry_source,
            constant_raster,
            projection=4326,
        )


def test_projection_gt_from_raster(geometry_source, constant_raster):
    view = AggregateRaster(geometry_source, constant_raster)
    assert constant_raster.projection == view.projection
    assert 1.0 == view.pixel_size


def test_projection_gt_not_from_raster(geometry_source, constant_raster):
    view = AggregateRaster(
        geometry_source, constant_raster, projection="EPSG:28992", pixel_size=0.2
    )
    assert "EPSG:28992" == view.projection
    assert 0.2 == view.pixel_size


def test_0_pixel_size_unsupported(geometry_source, constant_raster):
    with pytest.raises(ValueError):
        AggregateRaster(
            geometry_source,
            constant_raster,
            pixel_size=0.0,
        )


def test_percentile_out_of_bounds(geometry_source, constant_raster):
    with pytest.raises(ValueError):
        AggregateRaster(
            geometry_source,
            constant_raster,
            projection="EPSG:28992",
            statistic="p101",
        )


def test_column_attr(aggregate_raster, geometry_source):
    assert aggregate_raster.columns == (
        geometry_source.columns | {aggregate_raster.column_name}
    )


@pytest.mark.parametrize(
    "statistic,expected",
    [
        ("sum", 162.0),
        ("count", 36.0),
        ("mean", 4.5),
        ("min", 2.0),
        ("max", 7.0),
        ("median", 4.5),
        ("p75", 6.0),
    ],
)
def test_statistics(
    range_raster, geometry_source, geometry_request, statistic, expected
):
    geometry_request["start"] = Datetime(2018, 1, 1)
    geometry_request["stop"] = Datetime(2018, 1, 1, 3)

    view = AggregateRaster(
        source=geometry_source, raster=range_raster, statistic=statistic
    )
    features = view.get_data(**geometry_request)["features"]
    agg = features.iloc[0]["agg"]
    assert expected == agg


@pytest.mark.parametrize(
    "statistic,expected",
    [
        ("sum", 0),
        ("count", 0),
        ("mean", np.nan),
        ("min", np.nan),
        ("max", np.nan),
        ("median", np.nan),
        ("p75", np.nan),
    ],
)
def test_statistics_empty(
    geometry_source, nodata_raster, geometry_request, statistic, expected
):
    geometry_request["start"] = Datetime(2018, 1, 1)
    geometry_request["stop"] = Datetime(2018, 1, 1, 3)

    view = AggregateRaster(
        source=geometry_source, raster=nodata_raster, statistic=statistic
    )
    features = view.get_data(**geometry_request)["features"]
    agg = features.iloc[0]["agg"]
    assert_almost_equal(agg, expected)


@pytest.mark.parametrize(
    "statistic,expected",
    [
        ("sum", 0),
        ("count", 0),
        ("mean", np.nan),
        ("min", np.nan),
        ("max", np.nan),
        ("median", np.nan),
        ("p75", np.nan),
    ],
)
def test_statistics_partial_empty(
    geometry_source, geometry_request, statistic, expected
):
    values = np.indices((10, 10), dtype=np.uint8)[0]
    values[2:8, 2:8] = 255  # nodata
    raster = MockRaster(
        origin=Datetime(2018, 1, 1),
        timedelta=Timedelta(hours=1),
        bands=1,
        value=values,
    )

    view = AggregateRaster(source=geometry_source, raster=raster, statistic=statistic)
    features = view.get_data(**geometry_request)["features"]
    agg = features.iloc[0]["agg"]
    assert_almost_equal(agg, expected)


@pytest.mark.parametrize("geom", [box(0, 0, 10, 10), box(4, 4, 6, 6), Point(5, 5)])
def test_raster_request(geometry_request, aggregate_raster, geom):
    geometry_request["geometry"] = geom
    _, (_, request), _ = aggregate_raster.get_sources_and_requests(**geometry_request)
    assert_almost_equal(request["bbox"], (2, 2, 8, 8))
    assert 6 == request["width"]
    assert 6 == request["height"]


def test_raster_time_resolution(geometry_request):
    req = geometry_request
    req["time_resolution"] = 3600000
    req["geometry"] = box(0, 0, 10, 10)

    temp_raster = MockRaster(
        origin=Datetime(2018, 1, 1), timedelta=Timedelta(hours=1), bands=1
    )

    geom_source = MockGeometry(
        polygons=[((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0))],
        properties=[{"id": 1}],
    )
    view = AggregateRaster(source=geom_source, raster=temp_raster, statistic="sum")

    _, (source, request), _ = view.get_sources_and_requests(**req)

    assert 3600000 == request["time_resolution"]


def test_pixel_size_larger(geometry_source, geometry_request, constant_raster):
    # larger
    aggregate_raster = AggregateRaster(
        source=geometry_source, raster=constant_raster, statistic="sum", pixel_size=2
    )
    _, (_, request), _ = aggregate_raster.get_sources_and_requests(**geometry_request)
    assert_almost_equal(request["bbox"], (2, 2, 8, 8))
    assert 3 == request["width"]
    assert 3 == request["height"]


def test_pixel_size_smaller(geometry_source, geometry_request, constant_raster):
    aggregate_raster = AggregateRaster(
        source=geometry_source, raster=constant_raster, statistic="sum", pixel_size=0.5
    )
    _, (_, request), _ = aggregate_raster.get_sources_and_requests(**geometry_request)
    assert_almost_equal(request["bbox"], (2, 2, 8, 8))
    assert 12 == request["width"]
    assert 12 == request["height"]


def test_max_pixels(geometry_source, constant_raster, geometry_request):
    aggregate_raster = AggregateRaster(
        source=geometry_source,
        raster=constant_raster,
        statistic="sum",
        max_pixels=9,
        auto_pixel_size=True,
    )
    _, (_, request), _ = aggregate_raster.get_sources_and_requests(**geometry_request)

    assert_almost_equal(request["bbox"], (2, 2, 8, 8))
    assert 3 == request["width"]
    assert 3 == request["height"]


@pytest.mark.parametrize(
    "box,exp_bbox,exp_shape",
    (
        [(2.01, 1.99, 7.99, 8.01), (2, 1, 8, 9), (6, 8)],
        [(1.99, 2.01, 8.01, 7.99), (1, 2, 9, 8), (8, 6)],
        [(2.0, 2.0, 8.0, 8.0), (2, 2, 8, 8), (6, 6)],
        [(2.9, 1.1, 8.9, 7.1), (2, 1, 9, 8), (7, 7)],
        [(2.0, 1.0, 3.0, 2.0), (2.5, 1.5, 2.5, 1.5), (1, 1)],  # 1 cell
        [(2.0, 1.1, 3.0, 2.1), (2, 1, 3, 3), (1, 2)],  # 1 cell only in x
        [(1.1, 1.0, 3., 2.0), (1, 1, 3, 2), (2, 1)],  # 1 cell only in y
    ),
)
def test_snap_bbox(constant_raster, geometry_request, box, exp_bbox, exp_shape):
    (x1, y1, x2, y2) = box

    aggregate_raster = AggregateRaster(
        MockGeometry([((x1, y1), (x2, y1), (x2, y2), (x1, y2))]), constant_raster
    )
    _, (_, request), _ = aggregate_raster.get_sources_and_requests(**geometry_request)

    assert_almost_equal(request["bbox"], exp_bbox)
    assert exp_shape[0] == request["width"]
    assert exp_shape[1] == request["height"]


def test_max_pixels_with_snap(constant_raster, geometry_request):
    x1, y1, x2, y2 = 2.01, 1.99, 7.99, 8.01

    aggregate_raster = AggregateRaster(
        MockGeometry([((x1, y1), (x2, y1), (x2, y2), (x1, y2))]),
        constant_raster,
        max_pixels=20,
        auto_pixel_size=True,
    )
    _, (_, request), _ = aggregate_raster.get_sources_and_requests(**geometry_request)

    # The resulting bbox has too many pixels, so the pixel_size increases
    # by an integer factor to 2. Therefore snapping becomes different too.
    assert_almost_equal(request["bbox"], (2, 0, 8, 10))
    assert 3 == request["width"]
    assert 5 == request["height"]


def test_no_auto_scaling(geometry_source, constant_raster, geometry_request):
    aggregate_raster = AggregateRaster(
        source=geometry_source, raster=constant_raster, statistic="sum", max_pixels=9
    )

    with pytest.raises(RuntimeError):
        aggregate_raster.get_sources_and_requests(**geometry_request)


@pytest.fixture
def lower_raster_limit():
    original = config.get("geomodeling.raster-limit")
    try:
        config.set({"geomodeling.raster-limit": 9})
        yield
    finally:
        config.set({"geomodeling.raster-limit": original})


def test_max_pixels_fallback(
    geometry_source, constant_raster, geometry_request, lower_raster_limit
):
    aggregate_raster = AggregateRaster(
        source=geometry_source, raster=constant_raster, statistic="sum"
    )
    with pytest.raises(RuntimeError):
        aggregate_raster.get_sources_and_requests(**geometry_request)


def test_extensive_scaling(
    geometry_source, constant_raster, geometry_request, aggregate_raster
):
    # if a requested resolution is lower than the base resolution, the
    # sum aggregation requires scaling. sum is an extensive variable.

    # this is, of course, approximate. we choose a geo_transform so that
    # scaled-down geometries still precisely match the original
    view1 = aggregate_raster
    view2 = AggregateRaster(
        geometry_source,
        constant_raster,
        statistic="sum",
        pixel_size=0.1,
        max_pixels=6**2,
        auto_pixel_size=True,
    )
    agg1 = view1.get_data(**geometry_request)["features"].iloc[0]["agg"]
    agg2 = view2.get_data(**geometry_request)["features"].iloc[0]["agg"]
    assert agg1 * (10**2) == agg2


def test_intensive_scaling(geometry_source, constant_raster, geometry_request):
    # if a requested resolution is lower than the base resolution, the
    # mean aggregation does not require scaling.

    view1 = AggregateRaster(geometry_source, constant_raster, statistic="mean")
    view2 = AggregateRaster(
        geometry_source,
        constant_raster,
        statistic="mean",
        pixel_size=0.1,
        max_pixels=6**2,
        auto_pixel_size=True,
    )
    agg1 = view1.get_data(**geometry_request)["features"].iloc[0]["agg"]
    agg2 = view2.get_data(**geometry_request)["features"].iloc[0]["agg"]
    assert agg1 == agg2


def test_different_projection(geometry_source, constant_raster, geometry_request):
    view = AggregateRaster(
        source=geometry_source,
        raster=constant_raster,
        statistic="mean",
        projection="EPSG:3857",
    )

    geometry_request["projection"] = "EPSG:4326"
    geometry_request["geometry"] = box(-180, -90, 180, 90)

    _, (_, request), _ = view.get_sources_and_requests(**geometry_request)
    assert request["projection"] == "EPSG:3857"

    result = view.get_data(**geometry_request)
    assert result["projection"] == "EPSG:4326"
    assert result["features"].iloc[0]["agg"] == 1.0


def test_time(geometry_source, constant_raster, geometry_request):
    raster = MockRaster(
        origin=Datetime(2018, 1, 1), timedelta=Timedelta(hours=1), bands=3
    )
    view = AggregateRaster(source=geometry_source, raster=raster, statistic="mean")
    request = geometry_request

    # full range
    request["start"], request["stop"] = raster.period
    data = view.get_data(**request)
    value = data["features"].iloc[0]["agg"][0]
    assert 3 == len(value)

    # single frame
    request["stop"] = None
    data = view.get_data(**request)
    value = data["features"].iloc[0]["agg"]
    assert 1.0 == value

    # out of range
    request["start"] = raster.period[0] + Timedelta(days=1)
    request["stop"] = raster.period[1] + Timedelta(days=1)
    data = view.get_data(**request)
    value = data["features"].iloc[0]["agg"]
    assert np.isnan(value)


def test_chained_aggregation(aggregate_raster, geometry_request):
    raster2 = MockRaster(
        origin=Datetime(2018, 1, 1), timedelta=Timedelta(hours=1), bands=1, value=7
    )

    chained = AggregateRaster(
        aggregate_raster, raster2, statistic="mean", column_name="agg2"
    )
    result = chained.get_data(**geometry_request)
    feature = result["features"].iloc[0]

    assert 36.0 == feature["agg"]
    assert 7.0 == feature["agg2"]


def test_overlapping_geometries(constant_raster, geometry_request):
    source = MockGeometry(
        polygons=[
            ((2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)),
            ((2.0, 2.0), (8.0, 2.0), (8.0, 5.0), (2.0, 5.0)),
        ],
        properties=[{"id": 1}, {"id": 2}],
    )
    view = AggregateRaster(source=source, raster=constant_raster, statistic="sum")
    result = view.get_data(**geometry_request)
    assert result["features"]["agg"].values.tolist() == [36.0, 18.0]


@pytest.mark.parametrize("agg", ["mean", "min", "max", "median", "p90.0"])
def test_aggregate_percentile_one_empty(geometry_request, agg):
    # if there are only nodata pixels in the geometries, we expect the
    # statistic of mean, min, max, median and percentile to be NaN.
    data = np.ones((1, 10, 10), dtype=np.uint8)
    data[:, :5, :] = 255
    raster = MemorySource(data, 255, "EPSG:3857", pixel_size=1, pixel_origin=(0, 10))
    source = MockGeometry(
        polygons=[
            ((2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)),
            ((6.0, 6.0), (8.0, 6.0), (8.0, 8.0), (6.0, 8.0)),
        ],
        properties=[{"id": 1}, {"id": 2}],
    )
    view = AggregateRaster(source=source, raster=raster, statistic=agg)
    result = view.get_data(**geometry_request)
    assert np.isnan(result["features"]["agg"].values[1])


def test_empty_dataset(constant_raster, geometry_request):
    source = MockGeometry(polygons=[], properties=[])
    view = AggregateRaster(source=source, raster=constant_raster, statistic="sum")
    result = view.get_data(**geometry_request)
    assert 0 == len(result["features"])


@pytest.mark.parametrize(
    "statistic,expected",
    [
        ("sum", [16.0, 30.0, 0.0, 0.0]),
        ("count", [2, 4, 0, 0]),
        ("mean", [8.0, 7.5, np.nan, np.nan]),
    ],
)
def test_aggregate_above_threshold(range_raster, geometry_request, statistic, expected):
    source = MockGeometry(
        polygons=[
            ((2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)),  # contains 7, 8
            ((2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)),  # contains 7, 8
            ((7.0, 7.0), (9.0, 7.0), (9.0, 9.0), (7.0, 9.0)),  # contains 2, 3
            ((6.0, 6.0), (8.0, 6.0), (8.0, 8.0), (6.0, 8.0)),  # contains 3, 4
        ],
        properties=[
            {"id": 1, "threshold": 8.0},  # threshold halfway
            {"id": 3, "threshold": 3.0},  # threshold below
            {"id": 2000000, "threshold": 4.0},  # threshold above
            {"id": 9},
        ],  # no threshold
    )
    geometry_request["start"] = Datetime(2018, 1, 1)
    geometry_request["stop"] = Datetime(2018, 1, 1, 3)

    view = AggregateRasterAboveThreshold(
        source=source,
        raster=range_raster,
        statistic=statistic,
        threshold_name="threshold",
    )
    features = view.get_data(**geometry_request)["features"]
    assert_series_equal(
        features["agg"],
        pd.Series(expected, index=[1, 3, 2000000, 9], dtype=np.float32),
        check_names=False,
    )


@pytest.mark.parametrize("dx", [0.0, 0.1, 0.4999, 0.50001, 0.9, 0.99999])
def test_aggregate_no_interaction(geometry_request, dx):
    raster = MockRaster(
        origin=Datetime(2018, 1, 1),
        timedelta=Timedelta(hours=1),
        bands=1,
        value=np.indices((10, 10))[1],
    )
    source = MockGeometry(
        polygons=[
            ((2.0 + dx, 2.0), (4.0 + dx, 2.0), (4.0 + dx, 4.0), (2.0 + dx, 4.0)),
            ((3.0, 6.0), (5, 6.0), (5, 8.0), (3, 8.0)),  # exactly contains 3, 4
        ],
        properties=[{"id": 1}, {"id": 2}],
    )
    view = AggregateRaster(source=source, raster=raster, statistic="min")
    result = view.get_data(**geometry_request)
    assert result["features"]["agg"][2] == 3


@pytest.fixture
def raster_2x3():
    return MemorySource(
        np.arange(6).reshape(2, 3).astype(float),
        255,
        "EPSG:3857",
        pixel_size=2.0,
        pixel_origin=(0, 4),
    )


@pytest.mark.parametrize(
    "polygons,expected",
    [
        ([((2, 2), (1.9, 2), (2, 1.9))], [3.0]),
        ([((2, 2), (2.1, 2), (2, 1.9))], [4.0]),
        ([((2, 2), (2.1, 2), (2, 2.1))], [1.0]),
        ([((2, 2), (1.9, 2), (2, 2.1))], [0.0]),
        ([((2, 2), (1.9, 2), (2, 1.9)), ((2, 2), (2.1, 2), (2, 2.1))], [3.0, 1.0]),
    ],
)
def test_small_geometry(geometry_request, polygons, expected, raster_2x3):
    # Rasterize only takes pixels into account whose center is
    # inside the polygon
    source = MockGeometry(
        polygons=polygons,
        properties=[{"id": i + 1} for i in range(len(polygons))],
    )
    view = AggregateRaster(source=source, raster=raster_2x3, statistic="max")
    result = view.get_data(**geometry_request)
    assert_almost_equal(result["features"]["agg"].values, expected)


@pytest.mark.parametrize(
    "statistic,expected",
    [
        ("max", 3.0),
        ("min", 3.0),
        ("sum", 3.0),
        ("count", 1.0),
        ("mean", 3.0),
        ("p95", 3.0),
    ],
)
def test_small_geometry_statistics(geometry_request, statistic, expected, raster_2x3):
    # Rasterize only takes pixels into account whose center is
    # inside the polygon
    source = MockGeometry(
        polygons=[((2, 2), (1.9, 2), (2, 1.9))],
        properties=[{"id": 1}],
    )
    view = AggregateRaster(source=source, raster=raster_2x3, statistic=statistic)
    result = view.get_data(**geometry_request)
    assert_almost_equal(result["features"]["agg"].values, expected)


@pytest.mark.parametrize(
    "threshold,expected",
    [(2.0, 3.0), (3.0, 3.0), (4.0, np.nan)],
)
def test_small_geometry_threshold(geometry_request, raster_2x3, threshold, expected):
    source = MockGeometry(
        polygons=[((2, 2), (1.9, 2), (2, 1.9))],
        properties=[{"id": 1, "threshold": threshold}],
    )
    view = AggregateRasterAboveThreshold(
        source=source, raster=raster_2x3, statistic="max", threshold_name="threshold"
    )
    result = view.get_data(**geometry_request)
    assert_almost_equal(result["features"]["agg"].values, [expected])


def test_small_geometry_temporal(geometry_request):
    raster = MockRaster(
        origin=Datetime(2018, 1, 1), timedelta=Timedelta(hours=1), bands=3
    )
    source = MockGeometry(
        polygons=[
            ((2.0, 2.0), (2.1, 2.0), (2.1, 3.0), (2.0, 3.0)),
        ],
        properties=[{"id": 1}, {"id": 2}],
    )
    view = AggregateRaster(source=source, raster=raster, statistic="max")
    request = geometry_request.copy()
    request["start"], request["stop"] = raster.period
    result = view.get_data(**request)
    assert_almost_equal(result["features"]["agg"].loc[1][0], [1.0, 1.0, 1.0])


def test_bucketize():
    bboxes = [
        (0, 0, 2, 2),  # new bucket
        (2, 2, 4, 4),  # new bucket because of overlap with previous bucket
        (0, 0, 3, 3),  # new bucket because of size
        (5, 5, 7, 7),  # same as first
    ]
    expected = [[0, 3], [1], [2]]
    buckets = bucketize(bboxes)
    assert [0, 1, 2, 3] == sorted(i for b in buckets for i in b)
    assert expected == sorted(buckets)
