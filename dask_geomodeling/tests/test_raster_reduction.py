import numpy as np
from numpy.testing import assert_array_equal
import pytest

from dask_geomodeling.raster.reduction import reduce_rasters


@pytest.fixture
def stack():
    m = np.iinfo(np.uint64).max
    return [
        {
            "values": np.array([[[1, 1, 1], [0, 5, 0]]], dtype=np.uint8),
            "no_data_value": 0,
        },
        {
            "values": np.array([[[2, 2, 2], [4, 5, m]]], dtype=np.uint64),
            "no_data_value": m,
        },
        {
            "values": np.array([[[3, 3, 3], [4, 42.0, 42.0]]], dtype=np.float32),
            "no_data_value": 42.0,
        },
    ]


@pytest.fixture
def stack_nodata_only():
    return [
        {"values": np.zeros((1, 2, 3), dtype=np.uint8), "no_data_value": 0},
        {"values": np.ones((1, 2, 3), dtype=np.uint64), "no_data_value": 1},
        {"values": np.full((1, 2, 3), 42.0, dtype=np.float32), "no_data_value": 42.0},
    ]


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    "statistic, expected",
    [
        ("first", [[[1, 1, 1], [4, 5, 255]]]),
        ("last", [[[3, 3, 3], [4, 5, 255]]]),
        ("count", [[[3, 3, 3], [2, 2, 0]]]),
        ("sum", [[[6, 6, 6], [8, 10, 0]]]),
        ("mean", [[[2, 2, 2], [4, 5, 255]]]),
        ("min", [[[1, 1, 1], [4, 5, 255]]]),
        ("max", [[[3, 3, 3], [4, 5, 255]]]),
        ("argmin", [[[0, 0, 0], [1, 0, 255]]]),
        ("argmax", [[[2, 2, 2], [1, 0, 255]]]),
        ("std", [[[np.sqrt(2 / 3), np.sqrt(2 / 3), np.sqrt(2 / 3)], [0, 0, 255]]]),
        ("var", [[[2 / 3, 2 / 3, 2 / 3], [0, 0, 255]]]),
        ("median", [[[2, 2, 2], [4, 5, 255]]]),
        ("product", [[[6, 6, 6], [16, 25, 255]]]),
        ("p99", [[[2.98, 2.98, 2.98], [4, 5, 255]]]),
    ],
)
def test_reduce(statistic, expected, dtype, stack):
    actual = reduce_rasters(stack, statistic, no_data_value=255, dtype=dtype)
    expected = np.array(expected, dtype=dtype)
    assert_array_equal(actual["values"], expected)


@pytest.mark.parametrize(
    "statistic, expected_value",
    [
        ("first", 255),
        ("last", 255),
        ("count", 0),
        ("sum", 0),
        ("mean", 255),
        ("min", 255),
        ("max", 255),
        ("argmin", 255),
        ("argmax", 255),
        ("std", 255),
        ("var", 255),
        ("median", 255),
        ("product", 255),
        ("p99", 255),
    ],
)
def test_reduce_nan_input(statistic, expected_value, stack_nodata_only):
    actual = reduce_rasters(
        stack_nodata_only, statistic, no_data_value=255, dtype=np.uint8
    )
    expected = np.full((1, 2, 3), expected_value, dtype=np.uint8)
    assert_array_equal(actual["values"], expected)


@pytest.mark.parametrize("statistic", ["first", "sum"])
def test_reduce_defaults(statistic, stack):
    actual = reduce_rasters(stack, statistic)
    assert actual["values"].dtype == stack[0]["values"].dtype
    assert actual["no_data_value"] == stack[0]["no_data_value"]


def test_reduce_raises_zero_length(stack):
    with pytest.raises(ValueError):
        reduce_rasters([], "first")
