"""
Module containing raster blocks that aggregate rasters.
"""
from math import ceil, floor, log, sqrt
from collections import defaultdict
from functools import partial

from scipy import ndimage
import numpy as np
import geopandas as gpd

from dask import config
from dask_geomodeling import measurements
from dask_geomodeling import utils
from dask_geomodeling.raster import RasterBlock

from .base import GeometryBlock

__all__ = ["AggregateRaster", "AggregateRasterAboveThreshold"]


class Bucket:
    """
    Track objects in an imaginary grid that may span up to 4 cells.
    """

    def __init__(self):
        self.cells = set()
        self.indices = []

    def __contains__(self, cells):
        """
        Return wether any of the cells defined by indices is already occupied.
        """
        return bool(self.cells & cells)

    def add(self, index, cells):
        """
        Update the set of occupied cells with cells and append index to the
        list of indices.

        Note that this does not fail if cells intersects with already occupied
        cells.
        """
        self.indices.append(index)
        self.cells.update(cells)


def calculate_level_and_cells(bbox):
    """
    Return a tuple (level, cells).

    :param bboxes: list of (xmin, ymin, xmax, ymax) tuples

    The returned cells is a set of indices which represent occupied cells (at
    most 4) in an imaginary sparse grid that has a cellsize defined by the
    integer level.  Level 0 corresponds to the unit cell. Each doubling of the
    cellsize level increase corresponds to a doubling of the cellsize of the
    previous level.
    """
    x1, y1, x2, y2 = bbox
    level = -ceil(log(max(x2 - x1, y2 - y1), 2))

    width = 0.5**level
    height = 0.5**level

    j1 = floor(x1 / width)
    j2 = floor(x2 / width)
    i1 = floor(y1 / height)
    i2 = floor(y2 / height)

    return level, {(i1, j1), (i1, j2), (i2, j1), (i2, j2)}


def bucketize(bboxes):
    """
    Return list of lists with indices into bboxes.

    :param bboxes: list of (xmin, ymin, xmax, ymax) tuples

    Each sublist in the returned list points to a subset of disjoint bboxes.
    Instead of aiming for the smallest amount of subsets possible, this
    approach focuses on speed by avoiding costly intersection operations on all
    bboxes in a bucket.
    """
    bucket_dict = defaultdict(list)

    for index, bbox in enumerate(bboxes):
        level, cells = calculate_level_and_cells(bbox)

        # select bucket by feature size
        bucket_list = bucket_dict[level]

        for bucket in bucket_list:
            # see if it fits in this bucket
            if cells in bucket:
                continue
            # a suitable bucket has been found, break out of the for loop
            break
        else:
            # no break, no suitable bucket found, assign and append a new one
            bucket = Bucket()
            bucket_list.append(bucket)

        # add the item to the bucket
        bucket.add(index=index, cells=cells)

    return [
        bucket.indices for bucket_list in bucket_dict.values() for bucket in bucket_list
    ]


def aggregate_polygons(
    geometries,
    values,
    no_data_value,
    agg_bbox,
    agg_srs,
    threshold_values,
    statistic,
    percentile,
):
    """Compute aggregates for given geometries.

    Geometries are rasterized using gdal_rasterize (without the ALL_TOUCHED option).

    Args:
      geometries (GeoSeries): The geometries to aggregate the raster in.
      values (ndarray): A three-dimensional raster ``(t, y, x)`` to aggregate.
      no_data_value (number): The value in ``values`` that denotes missing data.
      agg_bbox (tuple of 4 numbers): The bbox of ``values``.
      agg_srs (str): The projection of ``values``.
      threshold_values (ndarray, optional): A threshold value per geometry.
      statistic (str): A key in ``AggregateRaster.STATISTICS``.
      percentile (float, optional): Required if ``statistic == "percentile"``

    Returns:
      - ndarray of dtype float32 and with shape ``(t, len(geometries))``
      - list of geometry indexes that didn't cover any cell
    """
    # Select the aggregation function
    agg_func = AggregateRaster.STATISTICS[statistic]["func"]
    if statistic == "percentile":
        agg_func = partial(agg_func, qval=percentile)

    # Append NaN to the threshold values for later usage
    if threshold_values is not None:
        threshold_values = np.concatenate(
            [threshold_values, np.array([np.nan], dtype=threshold_values.dtype)]
        )
    depth, height, width = values.shape
    geometries_no_cells = set()

    agg = np.full((depth, len(geometries)), np.nan, dtype="f4")
    for select in bucketize(geometries.bounds.values):
        rasterize_result = utils.rasterize_geoseries(
            geometries.iloc[select],
            agg_bbox,
            agg_srs,
            height,
            width,
            values=np.asarray(select, dtype=np.int32),  # GDAL needs int32
        )
        labels = rasterize_result["values"][0]
        unique_labels = set(np.unique(labels[labels != rasterize_result["no_data_value"]]).tolist())
        geometries_no_cells |= (set(select) - unique_labels)
        if not unique_labels:
            continue

        # if there is a threshold, generate a raster with thresholds
        if threshold_values is not None:
            # mode="clip" ensures that unlabeled cells use the appended NaN
            thresholds = np.take(threshold_values, labels, mode="clip")
        else:
            thresholds = None

        for frame_no, frame in enumerate(values):
            # limit statistics to active pixels
            active = frame != no_data_value
            # if there is a threshold, mask the frame
            if threshold_values is not None:
                valid = ~np.isnan(thresholds)  # to suppress warnings
                active[~valid] = False  # no threshold -> no aggregation
                active[valid] &= frame[valid] >= thresholds[valid]

            # if there is no single active value: do not aggregate
            if not active.any():
                continue

            # select features that actually have data
            # (min, max, median, and percentile cannot handle it otherwise)
            active_labels = labels[active]
            select_and_active = list(set(np.unique(active_labels)) & set(select))

            if not select_and_active:
                continue

            agg[frame_no][select_and_active] = agg_func(
                1 if statistic == "count" else frame[active],
                labels=active_labels,
                index=select_and_active,
            )
    return agg, list(geometries_no_cells)


def aggregate_points(
    points,
    values,
    no_data_value,
    agg_bbox,
    threshold_values,
    statistic,
):
    """Compute aggregates for given point geometries.

    The values of the raster is taken at the point coordinates.

    Args:
      points (GeoSeries): The geometries (points) to aggregate the raster in.
      values (ndarray): A three-dimensional raster ``(t, y, x)`` to aggregate.
      no_data_value (number): The value in ``values`` that denotes missing data.
      agg_bbox (tuple of 4 numbers): The bbox of ``values``.
      threshold_values (ndarray): A threshold value per geometry.
      statistic (str): A key in ``AggregateRaster.STATISTICS``.

    Returns:
      ndarray of dtype float32 and with shape ``(t, len(n_geometries))``
    """
    # Transform the points to indices
    _, height, width = values.shape
    gt = utils.GeoTransform.from_bbox(agg_bbox, height, width)
    i_y, i_x = gt.get_indices(np.array([points.x.values, points.y.values]).T)
    point_values = values[:, np.clip(i_y, 0, height - 1), np.clip(i_x, 0, width - 1)]

    # if there is a threshold, mask the point values
    active = point_values != no_data_value
    if threshold_values is not None:
        threshold_values = threshold_values[np.newaxis, :]  # broadcast over t
        valid = ~np.isnan(threshold_values)  # to suppress warnings
        active[~valid] = False  # no threshold -> no aggregation
        active[valid] &= point_values[valid] >= threshold_values[valid]

    # Convert nodata to NaN
    agg = point_values.astype("f4")
    agg[~active] = np.nan

    # For all statistics the aggregated value equals the pixel value
    # (if there is only one pixel), with the exception of "count":
    if statistic == "count":
        agg[active] = 1.0

    return agg


class AggregateRaster(GeometryBlock):
    """
    Compute statistics of a raster for each geometry in a geometry source.

    A statistic is computed in a specific projection and with a specified raster
    cell size. If ``projection`` or ``pixel_size`` are not given, these default to
    the native projection of the provided raster source. The following cells are
    selected to perform the statistic (e.g. mean) on:

    - Polygons: all raster cells whose center is inside the polygon
    - Points: the raster cell (singular) that contains the point
    - Linestrings: Bresenham's line algorithm is used

    If this assignment leads to the situation that a geometry covers no raster
    cells (for instance with a polygon much smaller than the raster cell size),
    the geometry is reduced to a point by taking its centroid.

    Should the combination of the requested pixel_size and the extent of the
    source geometry cause the required raster size to exceed ``max_pixels``,
    the ``pixel_size`` can be adjusted automatically if ``auto_pixel_size`` is
    set to ``True``, else (the default) a RuntimeError is raised.

    Please note that for any field operation on the result of this block
    a GetSeriesBlock should be used to retrieve data from the added column. The
    name of the added column is determined by the ``column_name`` parameter.

    Args:
      source (GeometryBlock): The geometry source for which the statistics are
        determined.
      raster (RasterBlock): The raster source that is sampled.
      statistic (str): The type of statistical analysis that should be
        performed. The options are: ``{"sum", "count", "min", "max", "mean",
        "median", "p<percentile>"}``. Percentiles are provided for example as
        follows: ``"p50"``. Default ``"sum"``.
      projection (str, optional): Projection to perform the aggregation in, for
        example ``"EPSG:28992"``. Defaults to the native projection of the
        supplied raster.
      pixel_size (float, optional): The raster cell size used in the
        aggregation. Defaults to the cell size of the supplied raster.
      max_pixels (int, optional): The maximum number of pixels (cells) in the
        aggregation. Defaults to the ``geomodeling.raster-limit`` setting.
      column_name (str, optional): The name of the column where the result
        should be placed. Defaults to ``"agg"``.
      auto_pixel_size (boolean): Determines whether the pixel size is adjusted
        automatically when ``"max_pixels"`` is exceeded. Default False.

    Returns:
      GeometryBlock with aggregation results in an added column

    The global raster-limit setting can be adapted as follows:
      >>> from dask import config
      >>> config.set({"geomodeling.raster-limit": 10 ** 9})
    """

    # extensive (opposite: intensive) means: additive, proportional to size
    STATISTICS = {
        "sum": {"func": ndimage.sum, "extensive": True},
        "count": {"func": ndimage.sum, "extensive": True},
        "min": {"func": ndimage.minimum, "extensive": False},
        "max": {"func": ndimage.maximum, "extensive": False},
        "mean": {"func": ndimage.mean, "extensive": False},
        "median": {"func": ndimage.median, "extensive": False},
        "percentile": {"func": measurements.percentile, "extensive": False},
    }

    def __init__(
        self,
        source,
        raster,
        statistic="sum",
        projection=None,
        pixel_size=None,
        max_pixels=None,
        column_name="agg",
        auto_pixel_size=False,
        *args
    ):
        if not isinstance(source, GeometryBlock):
            raise TypeError("'{}' object is not allowed".format(type(source)))
        if not isinstance(raster, RasterBlock):
            raise TypeError("'{}' object is not allowed".format(type(raster)))
        if not isinstance(statistic, str):
            raise TypeError("'{}' object is not allowed".format(type(statistic)))
        statistic, percentile = utils.parse_percentile_statistic(statistic.lower())
        if percentile:
            statistic = "p{0}".format(percentile)
        elif statistic not in self.STATISTICS or statistic == "percentile":
            raise ValueError("Unknown statistic '{}'".format(statistic))

        if projection is None:
            projection = raster.projection
        if not isinstance(projection, str):
            raise TypeError("'{}' object is not allowed".format(type(projection)))
        if pixel_size is None:
            # get the pixel_size from the raster geo_transform
            geo_transform = raster.geo_transform
            if geo_transform is None:
                raise ValueError(
                    "Cannot get the pixel_size from the source "
                    "raster. Please provide a pixel_size."
                )
            pixel_size = min(abs(float(geo_transform[1])), abs(float(geo_transform[5])))
        else:
            pixel_size = abs(float(pixel_size))
        if pixel_size == 0.0:
            raise ValueError("Pixel size cannot be 0")
        if max_pixels is not None:
            max_pixels = int(max_pixels)
        if not isinstance(auto_pixel_size, bool):
            raise TypeError("'{}' object is not allowed".format(type(auto_pixel_size)))

        super(AggregateRaster, self).__init__(
            source,
            raster,
            statistic,
            projection,
            pixel_size,
            max_pixels,
            column_name,
            auto_pixel_size,
            *args
        )

    @property
    def source(self):
        return self.args[0]

    @property
    def raster(self):
        return self.args[1]

    @property
    def statistic(self):
        return self.args[2]

    @property
    def projection(self):
        return self.args[3]

    @property
    def pixel_size(self):
        return self.args[4]

    @property
    def max_pixels(self):
        return self.args[5]

    @property
    def column_name(self):
        return self.args[6]

    @property
    def auto_pixel_size(self):
        return self.args[7]

    @property
    def columns(self):
        return self.source.columns | {self.column_name}

    def get_sources_and_requests(self, **request):
        if request.get("mode") == "extent":
            return [(self.source, request), (None, None), ({"mode": "extent"}, None)]

        req_srs = request["projection"]
        agg_srs = self.projection

        # acquire the extent of the geometry data
        extent_request = {**request, "mode": "extent"}
        extent = self.source.get_data(**extent_request)["extent"]

        if extent is None:
            # make sources_and_request so that we get an empty result
            return [
                (None, None),
                (None, None),
                ({"empty": True, "projection": req_srs}, None),
            ]

        # transform the extent into the projection in which we aggregate
        x1, y1, x2, y2 = utils.Extent(extent, req_srs).transformed(agg_srs).bbox

        # estimate the amount of required pixels
        required_pixels = int(((x2 - x1) * (y2 - y1)) / (self.pixel_size**2))

        # in case this request is too large, we adapt pixel size
        max_pixels = self.max_pixels
        if max_pixels is None:
            max_pixels = config.get("geomodeling.raster-limit")
        pixel_size = self.pixel_size

        if required_pixels > max_pixels and self.auto_pixel_size:
            # adapt with integer multiples of pixel_size
            pixel_size *= ceil(sqrt(required_pixels / max_pixels))
        elif required_pixels > max_pixels:
            raise RuntimeError(
                "The required raster size for the aggregation exceeded "
                "the maximum ({} > {})".format(required_pixels, max_pixels)
            )

        # snap the extent to (0, 0) to prevent subpixel shifts
        x1 = floor(x1 / pixel_size) * pixel_size
        y1 = floor(y1 / pixel_size) * pixel_size
        x2 = ceil(x2 / pixel_size) * pixel_size
        y2 = ceil(y2 / pixel_size) * pixel_size

        # compute the width and height
        width = max(int((x2 - x1) / pixel_size), 1)
        height = max(int((y2 - y1) / pixel_size), 1)

        # change point-like requests in real point requests
        # (reducing possible edge effects)
        if width == 1 and height == 1:
            raster_req_bbox = ((x1 + x2) / 2, (y1 + y2) / 2) * 2
        else:
            raster_req_bbox = (x1, y1, x2, y2)

        raster_request = {
            "mode": "vals",
            "projection": agg_srs,
            "start": request.get("start"),
            "stop": request.get("stop"),
            "aggregation": None,  # TODO
            "bbox": raster_req_bbox,
            "width": width,
            "height": height,
        }

        # only propagate if provided
        if "time_resolution" in request:
            raster_request["time_resolution"] = request["time_resolution"]

        process_kwargs = {
            "mode": request.get("mode", "intersects"),
            "pixel_size": self.pixel_size,
            "agg_srs": agg_srs,
            "req_srs": req_srs,
            "actual_pixel_size": pixel_size,
            "statistic": self.statistic,
            "result_column": self.column_name,
            "agg_bbox": (x1, y1, x2, y2),
        }

        return [
            (self.source, request),
            (self.raster, raster_request),
            (process_kwargs, None),
        ]

    @staticmethod
    def process(geom_data, raster_data, process_kwargs):
        if process_kwargs.get("empty"):
            return {
                "features": gpd.GeoDataFrame([]),
                "projection": process_kwargs["projection"],
            }
        elif process_kwargs["mode"] == "extent":
            return geom_data

        features = geom_data["features"]
        if len(features) == 0:
            return geom_data

        result = features.copy()

        # transform the features into the aggregation projection
        req_srs = process_kwargs["req_srs"]
        agg_srs = process_kwargs["agg_srs"]

        agg_geometries = utils.geoseries_transform(
            features["geometry"], req_srs, agg_srs
        )

        statistic, percentile = utils.parse_percentile_statistic(process_kwargs["statistic"])
        extensive = AggregateRaster.STATISTICS[statistic]["extensive"]
        result_column = process_kwargs["result_column"]

        # this is only there for the AggregateRasterAboveThreshold
        threshold_name = process_kwargs.get("threshold_name")
        if threshold_name:
            threshold_values = features[threshold_name].values.astype("f4")
        else:
            threshold_values = None

        # investigate the raster data
        agg_bbox = process_kwargs["agg_bbox"]
        if raster_data is None:
            values = no_data_value = None
        else:
            values = raster_data["values"]
            no_data_value = raster_data["no_data_value"]
        if values is None or np.all(values == no_data_value):  # skip the rest
            result[result_column] = 0 if extensive else np.nan
            return {"features": result, "projection": req_srs}

        pixel_size = process_kwargs["pixel_size"]
        actual_pixel_size = process_kwargs["actual_pixel_size"]

        agg, geometries_no_cells = aggregate_polygons(
            agg_geometries,
            values,
            no_data_value,
            agg_bbox,
            agg_srs,
            threshold_values,
            statistic,
            percentile,
        )

        if geometries_no_cells:
            # For points and polygons that don't touch a cell center,
            # use the centroid.
            agg[:, geometries_no_cells] = aggregate_points(
                agg_geometries.iloc[geometries_no_cells].centroid,
                values,
                no_data_value,
                agg_bbox,
                None if threshold_values is None else threshold_values[geometries_no_cells],
                statistic,
            )

        if extensive:  # sum and count
            agg[~np.isfinite(agg)] = 0
            # extensive aggregations have to be scaled
            if actual_pixel_size != pixel_size:
                agg *= (actual_pixel_size / pixel_size) ** 2
        else:
            agg[~np.isfinite(agg)] = np.nan  # replaces inf by nan

        if values.shape[0] == 1:
            result[result_column] = agg[0]
        else:
            # store an array in a dataframe cell: set each cell with [np.array]
            result[result_column] = [[x] for x in agg.T]

        return {"features": result, "projection": req_srs}


class AggregateRasterAboveThreshold(AggregateRaster):
    """
    Compute statistics of a per-feature masked raster for each geometry in a
    geometry source.

    Per feature, a threshold can be supplied to mask the raster with. Only
    values that exceed the threshold of a specific feature are included for
    the statistical value of that feature.

    See :class:``dask_geomodeling.geometry.aggregate.AggregateRaster`` for
    further information.

    Args:
      *args: See :class:``dask_geomodeling.geometry.aggregate.AggregateRaster``
      threshold_name (str): The column that holds the thresholds.

    Returns:
      GeometryBlock with aggregation results in an added column
    """

    def __init__(
        self,
        source,
        raster,
        statistic="sum",
        projection=None,
        pixel_size=None,
        max_pixels=None,
        column_name="agg",
        auto_pixel_size=False,
        threshold_name=None,
    ):
        if not isinstance(threshold_name, str):
            raise TypeError("'{}' object is not allowed".format(type(threshold_name)))
        if threshold_name not in source.columns:
            raise KeyError("Column '{}' is not available".format(threshold_name))
        super().__init__(
            source,
            raster,
            statistic,
            projection,
            pixel_size,
            max_pixels,
            column_name,
            auto_pixel_size,
            threshold_name,
        )

    @property
    def threshold_name(self):
        return self.args[8]

    def get_sources_and_requests(self, **request):
        src_and_req = super().get_sources_and_requests(**request)
        process_kwargs = src_and_req[2][0]
        process_kwargs["threshold_name"] = self.threshold_name
        return src_and_req
