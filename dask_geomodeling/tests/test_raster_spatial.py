import numpy as np
import pytest
from shapely.geometry import box, Point

from dask_geomodeling import raster
from dask_geomodeling.utils import shapely_transform
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


@pytest.fixture(scope="module", params=["exact", "zoomed_in", "zoomed_out"])
def vals_request(request):
    if request.param == "exact":
        bbox = (0, 0, 100, 80)
    elif request.param == "zoomed_in":
        bbox = (0, 0, 50, 40)
    elif request.param == "zoomed_out":
        bbox = (0, 0, 200, 160)
    return dict(
        mode="vals",
        bbox=bbox,
        projection="EPSG:28992",
        width=int(bbox[2] / 10),
        height=int(bbox[3] / 10),
    )


@pytest.fixture
def empty():
    yield MemorySource(
        data=np.full((0, 0, 0), 7, dtype=np.uint8),
        no_data_value=255,
        projection="EPSG:28992",
        pixel_size=20,
        pixel_origin=(0, 0),
    )


@pytest.fixture
def center():
    # the source raster has data at x 135000..135100 and y 456000..455900
    yield 135050, 455950


@pytest.fixture
def center_epsg3857(center):
    yield shapely_transform(Point(*center), "EPSG:28992", "EPSG:3857").coords[0]


def test_place_attrs(source, center):
    place = raster.Place(source, "EPSG:28992", center, [(50, 50)])

    # Place should not adapt these attributes:
    assert place.period == source.period
    assert place.timedelta == source.timedelta
    assert place.dtype == source.dtype
    assert place.fillvalue == source.fillvalue

    # If the place projection equals the store projection:
    assert place.projection == source.projection
    assert place.geo_transform == source.geo_transform

    extent_epsg28992 = (0, 0, 100, 100)
    extent_epsg4326 = shapely_transform(
        box(*extent_epsg28992), "EPSG:28992", "EPSG:4326"
    ).bounds
    x1, x2, y1, y2 = place.geometry.GetEnvelope()
    assert (x1, y1, x2, y2) == extent_epsg28992
    assert place.extent == pytest.approx(extent_epsg4326, rel=1e-4)


def test_place_invalid_statistic(source, center):
    with pytest.raises(ValueError):
        place = raster.Place(
            source, "EPSG:28992", center, [(50, 50)], statistic="nonexisting"
        )


def test_place_attrs_reproject(source, center_epsg3857):
    # Place should adapt the spatial attributes (extent & geometry)
    place = raster.Place(
        source, "EPSG:3857", center_epsg3857, [(572050, 6812050), (570050, 6811050)]
    )
    # The native projection != store projection:
    assert place.projection is None
    assert place.geo_transform is None

    extent_epsg3857 = 570000, 6811000, 572100, 6812100
    extent_epsg4326 = shapely_transform(
        box(*extent_epsg3857), "EPSG:3857", "EPSG:4326"
    ).bounds
    x1, x2, y1, y2 = place.geometry.GetEnvelope()
    assert (x1, y1, x2, y2) == pytest.approx(extent_epsg3857, rel=1e-4)
    assert place.extent == pytest.approx(extent_epsg4326, rel=1e-4)


def test_place_empty(empty, center, vals_request):
    place = raster.Place(empty, "EPSG:28992", center, [(50, 50)])

    assert place.geometry is None
    assert place.extent is None
    assert place.get_data(**vals_request) is None


def test_place_exact(source, center, vals_request):
    place = raster.Place(source, "EPSG:28992", center, [(50, 50)])
    values = place.get_data(**vals_request)["values"]
    # swap Y axis for readable test
    values = values[:, ::-1, :]
    assert (values[:, :10, :10] == 7).all()


def test_place_reproject(source, center_epsg3857):
    target = 572050, 6812050  # EPSG3857 coords somewhere inside RD validity
    place = raster.Place(source, "EPSG:3857", center_epsg3857, [target])
    x, y = shapely_transform(Point(*target), "EPSG:3857", "EPSG:28992").coords[0]
    values = place.get_data(
        mode="vals",
        bbox=(x - 40, y - 40, x + 40, y + 40),
        projection="EPSG:28992",
        width=8,
        height=8,
    )["values"]
    assert (values == 7).all()


def test_place_horizontal_shift(source, center, vals_request):
    # shift 1 cell (10 meters) to the right
    place = raster.Place(source, "EPSG:28992", center, [(60, 50)])
    values = place.get_data(**vals_request)["values"]
    # swap Y axis for readable test
    values = values[:, ::-1, :]
    assert (values[:, :10, 1:11] == 7).all()
    assert (values[:, :, 0] == 255).all()


def test_place_vertical_shift(source, center, vals_request):
    # shift 1 cell (10 meters) up
    place = raster.Place(source, "EPSG:28992", center, [(50, 60)])
    values = place.get_data(**vals_request)["values"]
    # swap Y axis for readable test
    values = values[:, ::-1, :]
    assert (values[:, 1:11, :10] == 7).all()
    assert (values[:, 0, :] == 255).all()


@pytest.mark.parametrize("statistic,expected", [
    ("first", (255, 7, 7, 7)),  # (no features, first, second, both)
    ("last", (255, 7, 7, 7)),  # the default
    ("count", (0, 1, 1, 2)),
    ("nans", (2, 1, 1, 0)),
    ("sum", (0, 7, 7, 14)),
    ("mean", (255, 7, 7, 7)),
    ("min", (255, 7, 7, 7)),
    ("max", (255, 7, 7, 7)),
    ("argmin", (255, 0, 1, 0)),
    ("argmax", (255, 0, 1, 0)),
    ("std", (255, 0, 0, 0)),
    ("var", (255, 0, 0, 0)),
    ("median", (255, 7, 7, 7)),
    ("p99", (255, 7, 7, 7)),

])
def test_place_multiple(source, center, vals_request, statistic, expected):
    # place such that only the left and bottom ridges have values
    place = raster.Place(
        source, "EPSG:28992", center, [(-40, 50), (50, -40)], statistic
    )
    values = place.get_data(**vals_request)["values"]
    # swap Y axis for readable test
    values = values[:, ::-1, :]
    # zero features
    assert (values[:, 1:, 1:] == expected[0]).all()
    # one feature
    assert (values[:, 1:10, 0] == expected[1]).all()
    assert (values[:, 0, 1:10] == expected[2]).all()
    # two features
    assert (values[:, 0, 0] == expected[3]).all()


def test_place_outside(source, center, vals_request):
    x1, y1, x2, y2 = vals_request["bbox"]
    coordinates = [(x1 - 50, y1), (x1, y1 - 50), (x2 + 50, y2), (x2, y2 + 50)]
    place = raster.Place(source, "EPSG:28992", center, coordinates)
    values = place.get_data(**vals_request)["values"]
    assert (values[:, :, :] == 255).all()


def test_place_time_request(source, center):
    place = raster.Place(source, "EPSG:28992", center, [(150, 50)])
    assert source.get_data(mode="time") == place.get_data(mode="time")


def test_place_meta_request(source, center):
    place = raster.Place(source, "EPSG:28992", center, [(150, 50)])
    assert source.get_data(mode="meta") == place.get_data(mode="meta")
