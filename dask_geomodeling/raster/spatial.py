"""
Module containing raster blocks for spatial operations.
"""
import math

from scipy import ndimage
import numpy as np
from osgeo import ogr

from dask_geomodeling.utils import (
    EPSG3857,
    EPSG4326,
    POLYGON,
    get_sr,
    Extent,
    get_dtype_min,
    get_footprint,
    get_index,
    shapely_transform,
)
from dask_geomodeling.raster.reduction import reduce_rasters, check_statistic

from .base import BaseSingle, RasterBlock
from shapely.geometry import Point

__all__ = ["Dilate", "Smooth", "MovingMax", "HillShade", "Place"]


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

    if shape_m[0] > 0 and shape_m[1] > 0:
        # Resolution in pixels per meter:
        resolution = request["height"] / shape_m[0], request["width"] / shape_m[1]
        # How many pixels to add:
        radius_px = [radius_m * res for res in resolution]
        # How many pixels to add, rounded to integers:
        margins_px = [int(round(r)) for r in radius_px]
        # How many meters to add (based on rounded pixels):
        margins_m = [m / res for m, res in zip(margins_px, resolution)]
    else:
        # There is no resolution. Add MARGIN_THRESHOLD pixels to the request.
        radius_px = margins_px = [Smooth.MARGIN_THRESHOLD] * 2
        # Expand the request with radius_m exactly.
        margins_m = [radius_m] * 2

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
    Perform spatial dilation on specific cell values.

    Cells with values in the supplied list are spatially dilated by one cell
    in each direction, including diagonals.

    Dilation is performed in the order of the values parameter.

    Args:
      store (RasterBlock): Raster to perform dilation on.
      values (list): Only cells with these values are dilated.

    Returns:
      RasterBlock where cells in values list are dilated.

    See also:
      https://en.wikipedia.org/wiki/Dilation_%28morphology%29
    """

    def __init__(self, store, values):
        values = np.asarray(values, dtype=store.dtype)
        super().__init__(store, values.tolist())

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
        for value in np.asarray(values, dtype=original.dtype):
            dilated[ndimage.binary_dilation(original == value)] = value
        dilated = dilated[:, 1:-1, 1:-1]
        return {"values": dilated, "no_data_value": data["no_data_value"]}


class MovingMax(BaseSingle):
    """
    Apply a spatial maximum filter to the data using a circular footprint.

    This can be used for visualization of sparse data.

    Args:
      store (RasterBlock): Raster to which the filter is applied
      size (integer): Diameter of the circular footprint. This should always be
        an odd number larger than 1.

    Returns:
      RasterBlock with maximum values inside the footprint of each input cell.
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
    Smooth the values from a raster spatially using Gaussian smoothing.

    Args:
      store (RasterBlock): Raster to be smoothed
      size (number): The extent of the smoothing in meters. The 'sigma' value
        for the Gaussian kernal equals ``size / 3``.
      fill (number): 'no data' are replaced by this value during smoothing,
        defaults to 0.

    Returns:
      RasterBlock with spatially smoothed values.

    See Also:
      https://en.wikipedia.org/wiki/Gaussian_blur

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

    Args:
      store (RasterBlock): Raster to which the hillshade algorithm is applied.
      altitude (number): Light source altitude in degrees, defaults to 45.
      azimuth (number): Light source azimuth in degrees, defaults to 315.
      fill (number): Fill value to be used for 'no data' values.

    Returns:
      Hillshaded raster

    See also:
      https://pro.arcgis.com/en/pro-app/tool-reference/3d-analyst/how-hillshade-works.htm
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


class Place(BaseSingle):
    """Place an input raster at given coordinates

    Note that if the store's projection is different from the requested one,
    the data will be reprojected before placing it at a different position.

    Args:
      store (RasterBlock): Raster that will be placed.
      place_projection (str): The projection in which this operation is done.
        This also specifies the projection of the ``anchor`` and
        ``coordinates`` args.
      anchor (list of 2 numbers): The anchor into the source raster that will
        be placed at given coordinates.
      coordinates (list of lists of 2 numbers): The target coordinates. The
        center of the bbox will be placed on each of these coordinates.
      statistic (str): What method to use to merge overlapping rasters. One of:
        {"last", "first", "count", "sum", "mean", "min",
        "max", "argmin", "argmax", "product", "std", "var", "p<number>"}

    Returns:
      RasterBlock with the source raster placed

    """

    def __init__(self, store, place_projection, anchor, coordinates, statistic="last"):
        if not isinstance(store, RasterBlock):
            raise TypeError("'{}' object is not allowed".format(type(store)))
        try:
            get_sr(place_projection)
        except RuntimeError:
            raise ValueError(
                "'{}' is not a valid projection string".format(place_projection)
            )
        anchor = list(anchor)
        if len(anchor) != 2:
            raise ValueError("Expected 2 numbers in the 'anchor' parameter")
        for x in anchor:
            if not isinstance(x, (int, float)):
                raise TypeError("'{}' object is not allowed".format(type(x)))
        if coordinates is None or len(coordinates) == 0:
            coordinates = []
        else:
            coordinates = np.asarray(coordinates, dtype=float)
            if coordinates.ndim != 2 or coordinates.shape[1] != 2:
                raise ValueError(
                    "Expected a list of lists of 2 numbers in the "
                    "'coordinates' parameter"
                )
            coordinates = coordinates.tolist()
        check_statistic(statistic)
        super().__init__(store, place_projection, anchor, coordinates, statistic)

    @property
    def place_projection(self):
        return self.args[1]

    @property
    def anchor(self):
        return self.args[2]

    @property
    def coordinates(self):
        return self.args[3]

    @property
    def statistic(self):
        return self.args[4]

    @property
    def projection(self):
        """The native projection of this block.

        Only returns something if the place projection equals the store
        projection"""
        store_projection = self.store.projection
        if store_projection is None:
            return
        if get_sr(self.place_projection).IsSame(get_sr(store_projection)):
            return store_projection

    @property
    def geo_transform(self):
        """The native geo_transform of this block

        Returns None if the store projection and place projections differ."""
        if self.projection is not None:
            return self.store.geo_transform

    @property
    def extent(self):
        geometry = self.geometry
        if geometry is None:
            return
        if not geometry.GetSpatialReference().IsSame(EPSG4326):
            geometry = geometry.Clone()
            geometry.TransformTo(EPSG4326)
        x1, x2, y1, y2 = geometry.GetEnvelope()
        return x1, y1, x2, y2

    @property
    def geometry(self):
        """Combined geometry in this block's native projection. """
        store_geometry = self.store.geometry
        if store_geometry is None:
            return
        sr = get_sr(self.place_projection)
        if not store_geometry.GetSpatialReference().IsSame(sr):
            store_geometry = store_geometry.Clone()
            store_geometry.TransformTo(sr)
        _x1, _x2, _y1, _y2 = store_geometry.GetEnvelope()
        p, q = self.anchor
        P, Q = zip(*self.coordinates)
        x1, x2 = _x1 + min(P) - p, _x2 + max(P) - p
        y1, y2 = _y1 + min(Q) - q, _y2 + max(Q) - q
        return ogr.CreateGeometryFromWkt(POLYGON.format(x1, y1, x2, y2), sr)

    def get_sources_and_requests(self, **request):
        if request["mode"] != "vals":
            return ({"mode": request["mode"]}, None), (self.store, request)

        # transform the anchor and coordinates into the requested projection
        anchor = shapely_transform(
            Point(self.anchor), self.place_projection, request["projection"]
        ).coords[0]
        coordinates = [
            shapely_transform(
                Point(coord), self.place_projection, request["projection"]
            ).coords[0]
            for coord in self.coordinates
        ]

        # transform the source's extent
        extent_geometry = self.store.geometry
        if extent_geometry is None:
            # no geometry means: no data
            return (({"mode": "null"}, None),)
        sr = get_sr(request["projection"])
        if not extent_geometry.GetSpatialReference().IsSame(sr):
            extent_geometry = extent_geometry.Clone()
            extent_geometry.TransformTo(sr)
        xmin, xmax, ymin, ymax = extent_geometry.GetEnvelope()

        # compute the requested cellsize
        x1, y1, x2, y2 = request["bbox"]
        size_x = (x2 - x1) / request["width"]
        size_y = (y2 - y1) / request["height"]

        # point requests: never request the full source extent
        if size_x > 0 and size_y > 0:
            # check what the full source extent would require
            full_height = math.ceil((ymax - ymin) / size_y)
            full_width = math.ceil((xmax - xmin) / size_x)
            if full_height * full_width <= request["width"] * request["height"]:
                _request = request.copy()
                _request["width"] = full_width
                _request["height"] = full_height
                _request["bbox"] = (
                    xmin,
                    ymin,
                    xmin + full_width * size_x,
                    ymin + full_height * size_y,
                )
                process_kwargs = {
                    "mode": "warp",
                    "anchor": anchor,
                    "coordinates": coordinates,
                    "src_bbox": _request["bbox"],
                    "dst_bbox": request["bbox"],
                    "cellsize": (size_x, size_y),
                    "statistic": self.statistic,
                }
                return [(process_kwargs, None), (self.store, _request)]

        # generate a new (backwards shifted) bbox for each coordinate
        sources_and_requests = []
        filtered_coordinates = []
        for _x, _y in coordinates:
            bbox = [
                x1 + anchor[0] - _x,
                y1 + anchor[1] - _y,
                x2 + anchor[0] - _x,
                y2 + anchor[1] - _y,
            ]
            # check the overlap with the source's extent
            # Note that raster cells are defined [xmin, xmax) and (ymin, ymax]
            # so points precisely at xmax or ymin certainly do not have data.
            if bbox[0] >= xmax or bbox[1] > ymax or bbox[2] < xmin or bbox[3] <= ymin:
                continue
            filtered_coordinates.append((_x, _y))
            _request = request.copy()
            _request["bbox"] = bbox
            sources_and_requests.append((self.store, _request))
        if len(sources_and_requests) == 0:
            # No coordinates inside: we still need to return an array
            # of the correct shape. Send a time request to get the depth.
            _request = request.copy()
            _request["mode"] = "time"
            process_kwargs = {
                "mode": "empty",
                "dtype": self.dtype,
                "fillvalue": self.fillvalue,
                "width": request["width"],
                "height": request["height"],
                "statistic": self.statistic,
            }
            return [(process_kwargs, None), (self.store, _request)]
        process_kwargs = {"mode": "group", "statistic": self.statistic}
        return [(process_kwargs, None)] + sources_and_requests

    @staticmethod
    def process(process_kwargs, *multi):
        if process_kwargs["mode"] in {"meta", "time"}:
            return multi[0]
        if process_kwargs["mode"] == "null":
            return
        if process_kwargs["mode"] == "empty":
            data = multi[0]
            if data is None:
                return
            out_shape = (
                len(data["time"]),
                process_kwargs["height"],
                process_kwargs["width"],
            )
            out_no_data_value = process_kwargs["fillvalue"]
            out_dtype = process_kwargs["dtype"]
            stack = []
        elif process_kwargs["mode"] == "group":
            # We have a bunch of arrays that are already shifted. Stack them.
            stack = [data for data in multi if data is not None]
            if len(stack) == 0:
                return  # instead of returning nodata (because inputs are None)
        elif process_kwargs["mode"] == "warp":
            # There is a single 'source' raster that we are going to shift
            # multiple times into the result. The cellsize is already correct.
            data = multi[0]
            if data is None:
                return
            out_no_data_value = data["no_data_value"]
            source = data["values"]
            out_dtype = source.dtype

            # convert the anchor to pixels (indices inside 'source')
            anchor = process_kwargs["anchor"]
            src_bbox = process_kwargs["src_bbox"]
            size_x, size_y = process_kwargs["cellsize"]
            anchor_px = (
                (anchor[0] - src_bbox[0]) / size_x,
                (anchor[1] - src_bbox[1]) / size_y,
            )

            # compute the output shape
            x1, y1, x2, y2 = process_kwargs["dst_bbox"]
            coordinates = process_kwargs["coordinates"]
            dst_h = round((y2 - y1) / size_y)
            dst_w = round((x2 - x1) / size_x)
            src_d, src_h, src_w = source.shape
            out_shape = (src_d, dst_h, dst_w)

            # determine what indices in 'source' have data
            k, j, i = np.where(get_index(source, out_no_data_value))

            # place the data on each coordinate
            stack = []
            for x, y in coordinates:
                if i.size == 0:  # shortcut: no data at all to place
                    break
                # transform coordinate into pixels (indices in 'values')
                coord_px = (x - x1) / size_x, (y - y1) / size_y
                di = round(coord_px[0] - anchor_px[0])
                dj = round(coord_px[1] - anchor_px[1])
                # because of the y-axis inversion: dj is measured from the
                # other side of the array. if you draw it, you'll arrive at:
                dj = dst_h - src_h - dj

                if di <= -src_w or di >= dst_w or dj <= -src_h or dj >= dst_h:
                    # skip as it would shift completely outside
                    continue
                elif 0 <= di <= (dst_w - src_w) and 0 <= dj <= (dst_h - src_h):
                    # complete place
                    values = np.full(out_shape, out_no_data_value, out_dtype)
                    values[k, j + dj, i + di] = source[k, j, i]
                    stack.append({"values": values, "no_data_value": out_no_data_value})
                else:
                    # partial place
                    i_s = i + di
                    j_s = j + dj
                    m = (i_s >= 0) & (j_s >= 0) & (i_s < dst_w) & (j_s < dst_h)
                    if not m.any():
                        continue
                    values = np.full(out_shape, out_no_data_value, out_dtype)
                    values[k[m], j_s[m], i_s[m]] = source[k[m], j[m], i[m]]
                    stack.append({"values": values, "no_data_value": out_no_data_value})

        # merge the values_stack
        if len(stack) == 0:
            return {
                "values": np.full(out_shape, out_no_data_value, out_dtype),
                "no_data_value": out_no_data_value,
            }
        else:
            return reduce_rasters(stack, process_kwargs["statistic"])
