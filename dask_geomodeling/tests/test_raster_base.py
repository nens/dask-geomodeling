from dask_geomodeling.raster import RasterBlock
from datetime import timedelta, datetime

import pytest


class MockRaster(RasterBlock):
    def __init__(self, timedelta, period):
        self.timedelta = timedelta
        self.period = period


DT1 = datetime(1970, 1, 1)
DT2 = datetime(1970, 1, 10)


@pytest.mark.parametrize("timedelta, period, expected", [
    (timedelta(days=1), None, True),  # temporal, empty
    (timedelta(days=1), (DT1, DT1), True),  # temporal, length-1
    (timedelta(days=1), (DT1, DT2), True),  # temporal
    (None, None, False),  # non-temporal raster-service, empty
    (None, (DT1, DT1), False),  # non-temporal raster-service
    (timedelta(minutes=5), None, False),  # non-temporal raster-store, empty
    (timedelta(minutes=5), (DT1, DT1), False),  # non-temporal raster-store
    (None, (DT1, DT2), True),  # temporal non-equidistant
])
def test_temporal_guess(timedelta, period, expected):
    block = MockRaster(timedelta, period)
    assert block.temporal == expected
