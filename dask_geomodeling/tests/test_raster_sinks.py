import os
import numpy as np
import pytest
from datetime import datetime, timedelta
from osgeo import gdal
import shutil
from dask_geomodeling.raster.sinks import RasterFileSink, to_file
from dask_geomodeling.raster.parallelize import RasterTiler
from dask_geomodeling.tests.factories import MockRaster, setup_temp_root, teardown_temp_root


@pytest.fixture(scope="module")
def root():
    path = setup_temp_root()
    yield path
    teardown_temp_root(path)


@pytest.fixture
def source():
    return MockRaster(
        origin=datetime(2000, 1, 1),
        timedelta=timedelta(hours=1),
        bands=1,
        value=np.full((100, 100), 7, dtype=np.uint8),
        projection="EPSG:3857",
    )


@pytest.fixture
def request_kwargs():
    return {
        "mode": "vals",
        "bbox": (0, 0, 100, 100),
        "projection": "EPSG:3857",
        "width": 4,
        "height": 4,
        "start": datetime(2000, 1, 1),
        "stop": datetime(2000, 1, 1),
    }


@pytest.fixture
def tiled_output(source, root, request_kwargs):
    """Write two tiles via RasterTiler + RasterFileSink and return the dir."""
    path = os.path.join(root, "tiled_output")
    sink = RasterFileSink(source, path)
    tiler = RasterTiler(sink, 2)
    # Use a larger request so tiling actually splits into multiple tiles
    large_request = dict(request_kwargs, width=8, height=8)
    tiler.get_data(**large_request)
    return path


@pytest.fixture
def sink(source, root):
    path = os.path.join(root, "sink_fixture")
    yield RasterFileSink(source, path)
    shutil.rmtree(path, ignore_errors=True)


def test_init(source, root):
    path = os.path.join(root, "test_init")
    sink = RasterFileSink(source, path)
    assert sink.store is source
    assert sink.url == f"file://{path}"


def test_init_non_raster(root):
    with pytest.raises(TypeError):
        RasterFileSink("not_a_raster", os.path.join(root, "test"))


def test_process(sink, request_kwargs):
    path = sink.url.replace("file://", "")

    result = sink.get_data(**request_kwargs)

    # process returns None
    assert result is None

    files = [f for f in os.listdir(path) if f.endswith(".tif")]
    assert len(files) == 1

    tif_path = os.path.join(path, files[0])
    ds = gdal.Open(tif_path)
    assert ds is not None
    assert ds.RasterXSize == 4
    assert ds.RasterYSize == 4

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    assert arr.shape == (4, 4)
    assert (arr == 7).all()
    assert band.GetNoDataValue() == 255.0

    # Check geotransform: bbox (0,0,100,100), width=4, height=4
    gt = ds.GetGeoTransform()
    assert gt[0] == pytest.approx(0)    # x origin
    assert gt[1] == pytest.approx(25)   # pixel width
    assert gt[3] == pytest.approx(100)  # y origin (top)
    assert gt[5] == pytest.approx(-25)  # pixel height (negative)


@pytest.mark.parametrize("request_overrides", [{"start": datetime(2099, 1, 1), "stop": datetime(2099, 1, 1)}, {"bbox": (1000, 1000, 1100, 1100)}])
def test_process_no_data_creates_no_files(sink, request_kwargs, request_overrides):
    """Request outside the time range should return None."""
    path = sink.url.replace("file://", "")

    result = sink.get_data(**{**request_kwargs, **request_overrides})
    assert result is None

    assert not os.path.exists(path)


def test_non_vals_mode_forwards(sink, root):
    """Non-vals modes (e.g. 'time') should forward data unchanged."""
    result = sink.get_data(
        mode="time",
        bbox=(0, 0, 100, 100),
        projection="EPSG:3857",
        width=4,
        height=4,
        start=datetime(2000, 1, 1),
        stop=datetime(2000, 1, 1),
    )
    assert "time" in result
    assert len(result["time"]) == 1


def test_tiled_sink(tiled_output):
    """RasterFileSink combined with RasterTiler should create multiple tiles."""
    files = [f for f in os.listdir(tiled_output) if f.endswith(".tif")]
    assert len(files) > 1


def test_merge_files(tiled_output, root):
    target = os.path.join(root, "merged.vrt")
    RasterFileSink.merge_files(tiled_output, target)
    assert os.path.exists(target)

    ds = gdal.Open(target)
    assert ds is not None
    assert ds.RasterXSize == 8
    assert ds.RasterYSize == 8


def test_merge_files_target_exists(tiled_output, root):
    target = os.path.join(root, "exists.vrt")
    RasterFileSink.merge_files(tiled_output, target)

    with pytest.raises(IOError):
        RasterFileSink.merge_files(tiled_output, target)


def test_merge_files_no_sources(root):
    path = os.path.join(root, "empty_dir")
    os.makedirs(path, exist_ok=True)
    target = os.path.join(root, "no_sources.vrt")
    with pytest.raises(IOError):
        RasterFileSink.merge_files(path, target)


def test_to_file(source, root, request_kwargs):
    """to_file should create a VRT at the target path."""
    target = os.path.join(root, "to_file_output.vrt")
    kwargs = {k: v for k, v in request_kwargs.items() if k != "mode"}
    to_file(source, target, tile_size=2, **kwargs)
    assert os.path.exists(target)

    ds = gdal.Open(target)
    assert ds is not None
    assert ds.RasterXSize == 4
    assert ds.RasterYSize == 4


def test_rasterblock_to_file(source, root, request_kwargs):
    """RasterBlock.to_file convenience method."""
    target = os.path.join(root, "block_to_file.vrt")
    kwargs = {k: v for k, v in request_kwargs.items() if k != "mode"}
    source.to_file(target, tile_size=2, **kwargs)
    assert os.path.exists(target)

    ds = gdal.Open(target)
    assert ds is not None
