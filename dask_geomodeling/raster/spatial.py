"""
Module containing raster blocks for spatial operations.
"""
import math

from scipy import ndimage
import numpy as np

from dask_geomodeling.utils import (
    EPSG3857,
    get_sr,
    Extent,
    get_dtype_min,
    get_footprint,
)

from .base import BaseSingle

__all__ = ["Dilate", "Smooth", "MovingMax", "HillShade"]


def expand_request_pixels(request, radius=1):
    """ Expand request by `radius` pixels. Returns None for non-vals requests
    or point requests. """
    if request["mode"] != "vals":  # do nothing with time and meta requests
        return None

    width, height = request["width"], request["height"]
    x1, y1, x2, y2 = request["bbox"]
    pwidth, pheight = x2 - x1, y2 - y1

    if pwidth == 0 or pheight == 0:  # cannot dilate a point request
        return None

    amount_x = pwidth / width * radius
    amount_y = pheight / height * radius

    new_request = request.copy()
    new_request["bbox"] = (x1 - amount_x, y1 - amount_x, x2 + amount_y, y2 + amount_y)
    new_request["width"] += 2 * radius
    new_request["height"] += 2 * radius
    return new_request


def expand_request_meters(request, radius_m=1):
    """
    Expand request by `radius_m` meters, rounded so that an integer number of
    pixels is added to all sides.

    Returns a tuple of:
     - new request with adapted bbox, width and height
     - the radius transformed to pixels as a (y, x) tuple of floats
     - the added margins as a (y, x) tuple of integers
    """
    sr = get_sr(request["projection"])
    bbox = request["bbox"]

    # throughout, variables in the projected unit ( = meters, mostly) are
    # suffixed by _m, in pixels by _px
    if sr.IsGeographic():
        # expand geographic bbox in EPSG3857
        extent_geom = Extent(bbox, sr)
        bbox = extent_geom.transformed(EPSG3857).bbox
    else:
        # most Projected projections are in meters, but to be sure:
        radius_m /= sr.GetLinearUnits()

    # compute the initial zoom factors: how much to expand the bbox to obtain
    # margins of exactly 'radius' (in meters)
    x1, y1, x2, y2 = bbox
    shape_m = y2 - y1, x2 - x1

    # omit the +1 in zoom for efficiency: zoom=0.2 is a zoom factor of 1.2
    zoom = [2 * radius_m / s for s in shape_m]

    # compute the size in pixels, and compute the margins (rounded size)
    shape_px = request["height"], request["width"]
    radius_px = [z * s / 2 for (z, s) in zip(zoom, shape_px)]
    margins_px = [int(round(sz)) for sz in radius_px]

    # use these (int-valued) margins to compute the actual zoom and margins
    zoom = [2 * m / s for (s, m) in zip(shape_px, margins_px)]
    margins_m = [z * s / 2 for (z, s) in zip(zoom, shape_m)]

    # assemble the request
    new_request = request.copy()
    new_request["bbox"] = (
        x1 - margins_m[1],
        y1 - margins_m[0],
        x2 + margins_m[1],
        y2 + margins_m[0],
    )
    if sr.IsGeographic():
        # transform back to original projection
        extent_proj = Extent(new_request["bbox"], EPSG3857)
        new_request["bbox"] = extent_proj.transformed(sr).bbox
    new_request["height"] += 2 * margins_px[0]
    new_request["width"] += 2 * margins_px[1]

    return new_request, radius_px


class Dilate(BaseSingle):
    """
    Perform binary dilation on specific values. Dilation is done in the
    order of the values parameter.

    :param store: raster to perform dilation on
    :param values: only dilate pixels that have these values

    :type store: RasterBlock
    :type values: list
    """

    def __init__(self, store, values):
        values = np.asarray(values, dtype=store.dtype)
        super(Dilate, self).__init__(store, values)

    @property
    def values(self):
        return self.args[1]

    def get_sources_and_requests(self, **request):
        new_request = expand_request_pixels(request, radius=1)
        if new_request is None:  # not an expandable request: do nothing
            return [(self.store, request)]
        else:
            return [(self.store, new_request), (self.values, None)]

    @staticmethod
    def process(data, values=None):
        if data is None or values is None or "values" not in data:
            return data
        original = data["values"]
        dilated = original.copy()
        for value in values:
            dilated[ndimage.binary_dilation(original == value)] = value
        dilated = dilated[:, 1:-1, 1:-1]
        return {"values": dilated, "no_data_value": data["no_data_value"]}


class MovingMax(BaseSingle):
    """
    Apply a spatial maximum filter to the data using a circular footprint.

    :param store: raster to apply the maximum filter on
    :param size: diameter of the circular footprint in pixels

    :type store: RasterBlock
    :type store: int

    This block can be used for visualization of sparse data.
    """

    def __init__(self, store, size):
        # round size to nearest odd integer
        size = int(2 * round((size - 1) / 2) + 1)
        if size < 3:
            raise ValueError("The size should be odd and larger than 1")
        super(MovingMax, self).__init__(store, size)

    @property
    def size(self):
        return self.args[1]

    def get_sources_and_requests(self, **request):
        size = self.size
        new_request = expand_request_pixels(request, radius=int(size // 2))
        if new_request is None:  # not an expandable request: do nothing
            return [(self.store, request)]
        else:
            return [(self.store, new_request), (size, None)]

    @staticmethod
    def process(data, size=None):
        if data is None or size is None or "values" not in data:
            return data
        radius = int(size // 2)
        footprint = get_footprint(size)[np.newaxis]

        # put absolute minimum on no data pixels
        array = data["values"].copy()
        minimum = get_dtype_min(array.dtype)
        no_data_mask = array == data["no_data_value"]
        array[no_data_mask] = minimum

        # apply maximum filter
        filtered = ndimage.maximum_filter(array, footprint=footprint)

        # replace absolute minimum with original fillvalue
        filtered[(filtered == minimum) & no_data_mask] = data["no_data_value"]

        # cut out the result
        filtered = filtered[:, radius:-radius, radius:-radius]
        return {"values": filtered, "no_data_value": data["no_data_value"]}


class Smooth(BaseSingle):
    """
    Smooth the values from a raster spatially using gaussian smoothing.

    :param store: raster to smooth
    :param size: size of the smoothing in meters. the 'sigma' of the
        gaussian kernel equals ``size / 3``.
    :param fill: fill value to be used for 'no data' values during
        smoothing. The output will not have 'no data' values.

    :type store: Store
    :type size: scalar
    :type fill: scalar

    The challenge is to mitigate the edge effects whilst remaining
    performant. If the necessary margin for smoothing is more than 6 pixels in
    any direction, the approach used here is requesting a zoomed out geometry,
    smooth that and zoom back in to the original region of interest.
    """

    MARGIN_THRESHOLD = 6

    def __init__(self, store, size, fill=0):
        for x in (size, fill):
            if not isinstance(x, (int, float)):
                raise TypeError("'{}' object is not allowed".format(type(x)))
        super(Smooth, self).__init__(store, size, fill)

    @property
    def size(self):
        return self.args[1]

    @property
    def fill(self):
        return self.args[2]

    def get_sources_and_requests(self, **request):
        if request["mode"] != "vals":  # do nothing with time and meta requests
            return [(self.store, request)]

        new_request, size = expand_request_meters(request, self.size)

        # check how many pixels will be added by the request
        if any([s > self.MARGIN_THRESHOLD for s in size]):
            smooth_mode = "zoom"
            # rescale the size
            zoom = [new_request[x] / request[x] for x in ("height", "width")]
            size = [s / z for s, z in zip(size, zoom)]
            # request the original (not expanded) shape
            new_request["height"] = request["height"]
            new_request["width"] = request["width"]
        else:
            smooth_mode = "exact"

        process_kwargs = dict(smooth_mode=smooth_mode, fill=self.fill, size=size)

        return [(self.store, new_request), (process_kwargs, None)]

    @staticmethod
    def process(data, process_kwargs=None):
        if data is None or process_kwargs is None:
            return data
        smooth_mode = process_kwargs["smooth_mode"]
        size_px = process_kwargs["size"]
        fill = process_kwargs["fill"]

        # fill in nodata values
        values = data["values"].copy()
        no_data_value = data["no_data_value"]
        values[values == no_data_value] = fill

        # compute the sigma
        sigma = 0, size_px[0] / 3, size_px[1] / 3
        ndimage.gaussian_filter(
            values, sigma, output=values, mode="constant", cval=fill
        )

        # remove the margins
        if smooth_mode == "exact":
            my, mx = [int(round(s)) for s in size_px]
            values = values[:, my : values.shape[1] - my, mx : values.shape[2] - mx]
        else:
            _, ny, nx = values.shape
            zy, zx = [1 - 2 * size_px[0] / ny, 1 - 2 * size_px[1] / nx]

            values = ndimage.affine_transform(
                values,
                order=0,
                matrix=np.diag([1, zy, zx]),
                offset=[0, size_px[0], size_px[1]],
            )

        return {"values": values, "no_data_value": no_data_value}


class HillShade(BaseSingle):
    """
    Calculate a hillshade from the raster values.

    :param store: RasterBlock object
    :param size: size of the effect, in projected units
    :param altitude: Light source altitude in degrees. Default 45.
    :param azimuth: Light source azimuth in degrees. Default 315.
    :param fill: fill value to be used for 'no data' values during hillshading

    :type store: RasterBlock
    :type size: scalar
    :type altitude: scalar
    :type azimuth: scalar
    :type fill: scalar

    Hillshade, adapted from gdal algorithm. Smooths prior to shading to
    keep it good looking even when zoomed beyond 1:1.
    """

    def __init__(self, store, altitude=45, azimuth=315, fill=0):
        for x in (altitude, azimuth, fill):
            if not isinstance(x, (int, float)):
                raise TypeError("'{}' object is not allowed".format(type(x)))
        super(HillShade, self).__init__(store, float(altitude), float(azimuth), fill)

    @property
    def altitude(self):
        return self.args[1]

    @property
    def azimuth(self):
        return self.args[2]

    @property
    def fill(self):
        return self.args[3]

    @property
    def dtype(self):
        return np.dtype("u1")

    @property
    def fillvalue(self):
        return 256  # on purpose, it does not exist in bytes

    @staticmethod
    def process(data, process_kwargs=None):
        """
        Adapted from:
        https://github.com/OSGeo/gdal/blob/2.0/gdal/apps/gdaldem.cpp#L481

        Edges are not implemented, result clips one pixel from array.
        """
        if process_kwargs is None:
            return data

        array = data["values"].copy()
        array[array == data["no_data_value"]] = process_kwargs["fill"]

        xres, yres = process_kwargs["resolution"]
        alt = math.radians(process_kwargs["altitude"])
        az = math.radians(process_kwargs["azimuth"])
        zsf = 1 / 8  # vertical scale factor
        square_zsf = zsf * zsf

        # gradient
        s0 = slice(None, None), slice(None, -2), slice(None, -2)
        s1 = slice(None, None), slice(None, -2), slice(1, -1)
        s2 = slice(None, None), slice(None, -2), slice(2, None)
        s3 = slice(None, None), slice(1, -1), slice(None, -2)
        s4 = slice(None, None), slice(1, -1), slice(1, -1)
        s5 = slice(None, None), slice(1, -1), slice(2, None)
        s6 = slice(None, None), slice(2, None), slice(None, -2)
        s7 = slice(None, None), slice(2, None), slice(1, -1)
        s8 = slice(None, None), slice(2, None), slice(2, None)

        # angle calculation
        y = np.empty(array.shape, dtype="f4")
        y[s4] = (
            array[s0]
            + 2 * array[s1]
            + array[s2]
            - array[s6]
            - 2 * array[s7]
            - array[s8]
        ) / yres

        x = np.empty(array.shape, dtype="f4")
        x[s4] = (
            array[s0]
            + 2 * array[s3]
            + array[s6]
            - array[s2]
            - 2 * array[s5]
            - array[s8]
        ) / xres

        with np.errstate(all="ignore"):
            xx_plus_yy = x * x + y * y
            aspect = np.arctan2(y, x)

            # shading
            cang = (
                math.sin(alt)
                - math.cos(alt) * zsf * np.sqrt(xx_plus_yy) * np.sin(aspect - az)
            ) / np.sqrt(1 + square_zsf * xx_plus_yy)

        cang = cang[..., 1:-1, 1:-1]
        result = np.where(cang <= 0, 0, 255 * cang).astype("u1")
        return {"values": result, "no_data_value": 256}

    def get_sources_and_requests(self, **request):
        new_request = expand_request_pixels(request, radius=1)
        if new_request is None:  # not an expandable request: do nothing
            return [(self.store, request)]

        # determine resolution
        bbox = request["bbox"]
        resolution = (
            (bbox[2] - bbox[0]) / request["width"],
            (bbox[3] - bbox[1]) / request["height"],
        )

        process_kwargs = dict(
            resolution=resolution,
            altitude=self.altitude,
            azimuth=self.azimuth,
            fill=self.fill,
        )

        return [(self.store, new_request), (process_kwargs, None)]
