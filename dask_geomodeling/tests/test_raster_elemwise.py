from dask_geomodeling.raster.elemwise import BaseElementwise
from dask_geomodeling.tests.factories import MockRaster
from datetime import datetime, timedelta
import pytest




@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("temporal1,delta1,temporal2,delta2,ok",
                         [
        # nontemporal - nontemporal
        (False, None, False, None, True),  # old: True
        (False, timedelta(minutes=5), False, timedelta(minutes=5), True),   # old: True
        (False, None, False, timedelta(minutes=5), True),  # old: True

        # nontemporal - temporal
        # note that before "None" delta meant only "temporal nonequidistant"
        # so we can safely have new behaviour for (False, None) cases
        (False, None, True, None, False),  # old: True
        (False, None, True, timedelta(hours=1), False),  # old: True
        (False, timedelta(minutes=5), True, None, False),  # old: True
        (False, timedelta(minutes=5), True, timedelta(hours=1), False),  # old: False

        # temporal - temporal
        (True, timedelta(hours=1), True, timedelta(hours=1), True),  # old: True
        (True, timedelta(hours=1), True, timedelta(hours=2), False),  # old: False
        (True, timedelta(hours=1), True, None, True),  # old: True
        (True, None, True, None, True),  # old: True
    ]
)
def test_raster_elemwise_init_ok(delta1,temporal1,delta2,temporal2,inverse,ok):
    raster1 = MockRaster(origin=datetime(2000, 1, 1), timedelta=delta1, temporal=temporal1)
    raster2 = MockRaster(origin=datetime(2000, 1, 1), timedelta=delta2, temporal=temporal2)

    if inverse:
        raster1, raster2 = raster2, raster1

    if ok:
        BaseElementwise(raster1, raster2)
    else:
        with pytest.raises(ValueError):
            BaseElementwise(raster1, raster2)

