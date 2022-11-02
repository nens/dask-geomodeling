import numpy as np
from osgeo import gdal
from pathlib import Path
from skyview2 import sky_view_factor as sky_view_factor_old
from skyview3 import sky_view_factor as sky_view_factor_new, shadow_factor
import time


gdal.UseExceptions()

data_dir = Path(__file__).parent


def input_data(path):
    ds = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    no_data = band.GetNoDataValue()
    data = band.ReadAsArray().astype(np.float32)
    data[data == no_data] = -9999.0
    return data, ds.GetGeoTransform(), ds.GetProjection()


def output_data(path, data, bbox, transform, projection):
    cols = data.shape[1]
    rows = data.shape[0]

    transform = list(transform)
    transform[0] += bbox[0] * transform[1]
    transform[3] += bbox[1] * transform[5]

    data = data.astype(np.float32)
    # if transform[5] < 0:
    #     transform[3] = transform[3] + (transform[5] * rows)
    #     transform[5] = -1 * transform[5]
    #     data = data[::-1]

    driver = gdal.GetDriverByName("GTiff")
    outRaster = driver.Create(str(path), cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform(transform)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(data)
    outRaster.SetProjection(projection)
    outband.FlushCache()


w, h = 256, 257
m = 200  # pixels
raster, transform, projection = input_data(data_dir / "ahn/tiles/31h/31hz2_02.tif")


# before = time.time()
# old = sky_view_factor_old(raster[:h + 2 * m,:w + 2 * m], resolution=0.5, svf_r_max=m, svf_n_dir=128)["svf"][m: -m, m:-m]
# output_data(data_dir / "31hz2_02_skyview_ref.tif", old, (m, m, w + m, h + m), transform, projection)
# print(f"Computed skyview (old method) in {time.time() - before:.2f}s")


# old, _, _ = input_data(data_dir / "31hz2_02_skyview_ref.tif")
# old[old == -9999.0] = 0.0


# def sky_view_factor_err(n_points, jitter=0.0):
#     before = time.time()
#     new = sky_view_factor_new(
#         raster[: h + 2 * m, : w + 2 * m],
#         resolution=0.5,
#         svf_r_max=m,
#         svf_n_dir=16,
#         svf_n_points=n_points,
#         svf_jitter=jitter
#     )["svf"]
#     print(f"Computed skyview (new method, n={n_points}) in {time.time() - before:.2f}s")
#     diff = old - new
#     print(
#         f"n_points={n_points}: mean err={np.mean(diff)}, rms err={(np.sum(diff**2)**0.5) / new.size}, p90={np.percentile(np.abs(diff), 90)}"
#     )
#     output_data(
#         data_dir / f"31hz2_02_skyview_new_{n_points}.tif",
#         new,
#         (m, m, w + m, h + m),
#         transform,
#         projection,
#     )


# for n_points in (50,):
#     sky_view_factor_err(n_points, jitter=0.0)

before = time.time()
shadow = shadow_factor(raster[:h + 2 * m,:w + 2 * m], resolution=0.5, svf_r_max=m, svf_n_dir=16)["svf"]
output_data(data_dir / "31hz2_02_shadow.tif", shadow, (m, m, w + m, h + m), transform, projection)


# from memory_profiler import memory_usage
# mem_usage = memory_usage(lambda: sky_view_factor_old(raster, resolution=0.5, svf_r_max=200))
# print('Maximum memory usage: %s' % max(mem_usage))

# mem_usage = memory_usage(lambda: sky_view_factor_new(raster, resolution=0.5, svf_r_max=200))
# print('Maximum memory usage: %s' % max(mem_usage))
