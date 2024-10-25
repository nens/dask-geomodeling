"""
Module containing raster sources.
"""
from dataclasses import dataclass
from osgeo import gdal, gdal_array, osr
from shapely import Point
import numpy as np

from datetime import datetime, timedelta, timezone

from dask_geomodeling import utils

from .base import RasterBlock

__all__ = ["MemorySource", "RasterFileSource"]


@dataclass
class RasterData:
    array: np.ndarray
    projection: str
    geo_transform: tuple
    no_data_value: float
    metadata: dict

    def as_gdal_dataset(self):
        dataset = gdal_array.OpenArray(self.array)
        dataset.SetGeoTransform(self.geo_transform)
        dataset.SetProjection(self.projection)
        for i in range(dataset.RasterCount):
            band = dataset.GetRasterBand(i + 1)
            band.SetNoDataValue(self.no_data_value)
            if self.metadata is not None:
                band.SetMetadataItem("metadata", self.metadata[i])
        return dataset


class RasterSourceBase(RasterBlock):

    @staticmethod
    def process(process_kwargs):
        mode = process_kwargs["mode"]

        # handle empty requests
        if mode == "empty_vals":
            return
        elif mode == "empty_time":
            return {"time": []}
        elif mode == "empty_meta":
            return {"meta": []}

        bands = process_kwargs["bands"]
        length = bands[1] - bands[0]

        # handle time requests
        if mode == "time":
            start = process_kwargs["start"]
            delta = process_kwargs["delta"]
            return {"time": [start + i * delta for i in range(length)]}

        # this is where the code paths for the MemorySource and RasterFileSource come
        # together
        raster_data = process_kwargs.get("raster_data")
        if raster_data is None:
            # coming from a RasterFileSource block
            url = process_kwargs["url"]
            path = utils.safe_abspath(url)
            source = gdal.Open(path)
        else:
            # coming from a MemorySource block
            source = raster_data.as_gdal_dataset()

        # handle meta requests
        if mode == "meta":
            return {
                "meta": [
                    source.GetRasterBand(i + 1).GetMetadataItem("metadata")
                    for i in range(bands[0], bands[1])
                ]
            }

        # handle 'vals' requests
        dtype = process_kwargs["dtype"]
        bbox = process_kwargs["bbox"]
        width = process_kwargs["width"]
        height = process_kwargs["height"]
        target_projection = process_kwargs["projection"]
        target_no_data_value = process_kwargs["fillvalue"].item()

        # return an empty array if 0-sized data was requested
        if width == 0 or height == 0:
            return np.empty((length, height, width), dtype=dtype)

        # handle point request
        if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
            source_x, source_y = utils.shapely_transform(
                geometry=Point(bbox[0], bbox[1]),
                src_srs=target_projection,
                dst_srs=source.GetProjection(),
            ).coords[0]
            source_geo_transform = utils.GeoTransform(source.GetGeoTransform())
            source_i, source_j = source_geo_transform.get_indices(
                ((source_x, source_y))
            )

            # create the result
            shape = source.RasterCount, 1, 1
            result = np.full(shape, target_no_data_value, dtype=dtype)
            if (
                0 <= source_i < source.RasterYSize
                and 0 <= source_j < source.RasterXSize
            ):
                # read data into the result
                source.ReadAsArray(int(source_j), int(source_i), 1, 1, result)

            result = result[bands[0] : bands[1]]
            return {"values": result, "no_data_value": target_no_data_value}

        target_geo_transform = utils.GeoTransform.from_bbox(
            bbox=bbox,
            width=width,
            height=height,
        )

        # transform the complete raster
        shape = source.RasterCount, height, width
        result = np.full(shape, target_no_data_value, dtype=dtype)
        kwargs = {
            "geo_transform": target_geo_transform,
            "no_data_value": target_no_data_value,
            "projection": osr.GetUserInputAsWKT(target_projection),
        }
        with utils.Dataset(result, **kwargs) as target:
            gdal.ReprojectImage(
                source,
                target,
                source.GetProjection(),
                target.GetProjection(),
                gdal.GRA_NearestNeighbour,
                0.0,  # Dummy WarpMemoryLimit, same as default.
                0.125,  # Max error in pixels. Without passing this,
                # it's 0.0, which is very slow. 0.125 is the
                # default of gdalwarp on the command line.
            )

        result = result[bands[0] : bands[1]]
        # fill nan values if they popped up
        result[~np.isfinite(result)] = target_no_data_value
        return {"values": result, "no_data_value": target_no_data_value}


def utc_from_ms_timestamp(timestamp):
    """Returns naive UTC datetime from ms timestamp"""
    return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).replace(tzinfo=None)


class MemorySource(RasterSourceBase):
    """A raster source that interfaces data from memory.

    Nodata values are supported, but when upsampling the data, these are
    assumed to be 0 biasing data edges towards 0.

    The raster pixel with its topleft corner at [x, y] will define ranges
    [x, x + dx) and (y - dy, y]. Here [dx, dy] denotes the (unsigned) pixel
    size. The topleft corner and top and left edges belong to a pixel.

    :param data: the pixel values
        this value will be transformed in a 3D array (t, y, x)
    :param no_data_value: the pixel value that designates 'no data'
    :param projection: the projection of the given pixel values
    :param pixel_size: the size of one pixel (in units given by projection)
        if x and y pixel sizes differ, provide them in (x, y) order
    :param pixel_origin: the location (x, y) of pixel with index (0, 0)
    :param time_first: the timestamp of the first frame in data (in
        milliseconds since 1-1-1970)
    :param time_delta: the difference between two consecutive frames (in ms)
    :param metadata: a list of metadata corresponding to the input frames

    :type data: number or ndarray
    :type no_data_value: number
    :type projection: str
    :type pixel_size: float or length-2 iterable of floats
    :type pixel_origin: length-2 iterable of floats
    :type time_first: integer or naive datetime
    :type time_delta: integer or timedelta or NoneType
    :type metadata: list or NoneType
    """

    def __init__(
        self,
        data,
        no_data_value,
        projection,
        pixel_size,
        pixel_origin,
        time_first=0,
        time_delta=None,
        metadata=None,
    ):
        data = np.asarray(data)
        if data.dtype == "i8":
            # not possible in currently supported gdal version
            data = data.astype("i4")
        if data.ndim == 2:
            data = data[np.newaxis]
        if data.ndim != 3:
            raise ValueError("data should be two- or three-dimensional.")
        no_data_value = data.dtype.type(no_data_value)
        projection = utils.get_epsg_or_wkt(projection)
        if not hasattr(pixel_size, "__iter__"):
            pixel_size = [pixel_size] * 2
        else:
            pixel_size = list(pixel_size)
            if len(pixel_size) != 2:
                raise ValueError("pixel_size should have length 2")
        pixel_size = [float(x) for x in pixel_size]
        pixel_origin = [float(x) for x in pixel_origin]
        if len(pixel_origin) != 2:
            raise ValueError("pixel_origin should have length 2")
        if isinstance(time_first, datetime):
            time_first = utils.dt_to_ms(time_first)
        else:
            time_first = int(time_first)
        if isinstance(time_delta, timedelta):
            time_delta = int(time_delta.total_seconds() * 1000)
        elif time_delta is None:
            if data.shape[0] > 1:
                raise ValueError("time_delta is required for temporal data")
        else:
            time_delta = int(time_delta)
        if metadata is not None:
            metadata = list(metadata)
            if len(metadata) != data.shape[0]:
                raise ValueError("Metadata length should match data length")
        super().__init__(
            data,
            no_data_value,
            projection,
            pixel_size,
            pixel_origin,
            time_first,
            time_delta,
            metadata,
        )

    @property
    def data(self):
        return self.args[0]

    @property
    def no_data_value(self):
        return self.args[1]

    @property
    def projection(self):
        return self.args[2]

    @property
    def pixel_size(self):
        return self.args[3]

    @property
    def pixel_origin(self):
        return self.args[4]

    @property
    def time_first(self):
        return self.args[5]

    @property
    def time_delta(self):
        return self.args[6]

    @property
    def metadata(self):
        return self.args[7]

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def fillvalue(self):
        return self.no_data_value

    @property
    def geo_transform(self):
        p, q = self.pixel_origin
        a, d = self.pixel_size
        return utils.GeoTransform((p, a, 0, q, 0, -d))

    def _get_extent(self):
        if not self.data.size:
            return
        bbox = self.geo_transform.get_bbox((0, 0), self.data.shape[1:])
        return utils.Extent(bbox, self.projection)

    @property
    def extent(self):
        extent = self._get_extent()
        if extent is None:
            return
        return extent.transformed("EPSG:4326").bbox

    @property
    def geometry(self):
        extent = self._get_extent()
        if extent is None:
            return
        return extent.as_geometry()

    def __len__(self):
        return self.data.shape[0]

    @property
    def period(self):
        if len(self) == 0:
            return
        elif len(self) == 1:
            return (utc_from_ms_timestamp(self.time_first),) * 2
        else:
            first = utc_from_ms_timestamp(self.time_first)
            last = first + (len(self) - 1) * self.timedelta
            return first, last

    @property
    def timedelta(self):
        if self.time_delta is None:
            return None
        else:
            return timedelta(milliseconds=self.time_delta)

    @property
    def temporal(self):
        return self.time_delta is not None

    def get_sources_and_requests(self, **request):
        mode = request["mode"]

        if (
            mode == "vals"
            and request.get("projection").upper() != self.projection.upper()
        ):
            raise RuntimeError("This source block cannot reproject data")
        elif mode == "meta" and self.metadata is None:
            return [({"mode": "empty_meta"}, None)]

        # interpret start and stop request parameters
        start, stop, band1, band2 = utils.snap_start_stop(
            request.get("start"),
            request.get("stop"),
            utc_from_ms_timestamp(self.time_first),
            self.timedelta,
            len(self),
        )
        if start is None:
            return [({"mode": "empty_" + request["mode"]}, None)]
        bands = band1, band2 + 1

        # create a dataset
        raster_data = RasterData(
            array=self.data,
            metadata=self.metadata,
            geo_transform=self.geo_transform,
            no_data_value=float(self.no_data_value),
            projection=osr.GetUserInputAsWKT(self.projection),
        )

        # set the process kwargs depending on the mode
        if mode == "vals":
            process_kwargs = {
                "mode": "vals",
                "raster_data": raster_data,
                "bbox": request["bbox"],
                "width": request["width"],
                "height": request["height"],
                "projection": request["projection"],
                "bands": bands,
                "dtype": self.dtype,
                "fillvalue": self.fillvalue,
            }
        elif mode == "meta":
            # metadata can't be None at this point
            process_kwargs = {
                "mode": "meta",
                "raster_data": raster_data,
                "bands": bands,
            }
        elif mode == "time":
            process_kwargs = {
                "mode": "time",
                "start": start,
                "delta": self.timedelta or timedelta(0),
                "bands": bands,
            }
        else:
            raise RuntimeError("Unknown mode '{}'".format(mode))
        return [(process_kwargs, None)]


class RasterFileSource(RasterSourceBase):
    """A raster source that interfaces data from a file path.

    The value at raster cell with its topleft corner at [x, y] is assumed to
    define a value for ranges [x, x + dx) and (y - dy, y]. Here [dx, dy]
    denotes the (unsigned) pixel size. The topleft corner and top and
    left edges belong to a pixel.

    :param url: the path to the file. File paths have to be contained inside
      the current root setting. Relative paths are interpreted relative to this
      setting (but internally stored as absolute paths).
    :param time_first: the timestamp of the first frame in data (in
        milliseconds since 1-1-1970), defaults to 1-1-1970
    :param time_delta: the difference between two consecutive frames (in ms),
        defaults to 5 minutes

    :type url: str
    :type time_first: integer or datetime
    :type time_delta: integer or timedelta

    The global root path can be adapted as follows:
      >>> from dask import config
      >>> config.set({"geomodeling.root": "/my/data/path"})

    Note that this object keeps a file handle open. If you need to close the
    file handle, call block.close_dataset (or dereference the whole object).
    """

    def __init__(self, url, time_first=0, time_delta=300000):
        url = utils.safe_file_url(url)
        if isinstance(time_first, datetime):
            time_first = utils.dt_to_ms(time_first)
        else:
            time_first = int(time_first)
        if isinstance(time_delta, timedelta):
            time_delta = int(time_delta.total_seconds() * 1000)
        else:
            time_delta = int(time_delta)
        super().__init__(url, time_first, time_delta)

    @property
    def url(self):
        return self.args[0]

    @property
    def time_first(self):
        return self.args[1]

    @property
    def time_delta(self):
        return self.args[2]

    @property
    def gdal_dataset(self):
        try:
            return self._gdal_dataset
        except AttributeError:
            path = utils.safe_abspath(self.url)
            self._gdal_dataset = gdal.Open(path)
            return self._gdal_dataset

    def close_dataset(self):
        if hasattr(self, "_gdal_dataset"):
            self._gdal_dataset = None

    @property
    def projection(self):
        return utils.get_epsg_or_wkt(self.gdal_dataset.GetProjection())

    @property
    def dtype(self):
        first_band = self.gdal_dataset.GetRasterBand(1)
        gdal_data_type = first_band.DataType
        numpy_type = gdal_array.GDALTypeCodeToNumericTypeCode(gdal_data_type)
        return np.dtype(numpy_type)

    @property
    def fillvalue(self):
        first_band = self.gdal_dataset.GetRasterBand(1)
        return self.dtype.type((first_band.GetNoDataValue()))

    @property
    def geo_transform(self):
        return utils.GeoTransform(self.gdal_dataset.GetGeoTransform())

    def _get_extent(self):
        bbox = self.geo_transform.get_bbox(
            (0, 0), (self.gdal_dataset.RasterYSize, self.gdal_dataset.RasterXSize)
        )
        return utils.Extent(bbox, self.projection)

    @property
    def extent(self):
        extent_epsg4326 = self._get_extent().transformed("EPSG:4326")
        return extent_epsg4326.bbox

    @property
    def geometry(self):
        return self._get_extent().as_geometry()

    def __len__(self):
        return self.gdal_dataset.RasterCount

    @property
    def period(self):
        if len(self) == 0:
            return
        elif len(self) == 1:
            return (utc_from_ms_timestamp(self.time_first)) * 2
        else:
            first = utc_from_ms_timestamp(self.time_first)
            last = first + (len(self) - 1) * self.timedelta
            return first, last

    @property
    def timedelta(self):
        if len(self) <= 1:
            return None
        return timedelta(milliseconds=self.time_delta)

    @property
    def temporal(self):
        return len(self) > 1

    def get_sources_and_requests(self, **request):
        mode = request["mode"]

        # interpret start and stop request parameters
        start, stop, band1, band2 = utils.snap_start_stop(
            request.get("start"),
            request.get("stop"),
            utc_from_ms_timestamp(self.time_first),
            self.timedelta,
            len(self),
        )
        if start is None:
            return [({"mode": "empty_" + request["mode"]}, None)]
        bands = band1, band2 + 1

        # set the process kwargs depending on the mode
        if mode == "vals":
            process_kwargs = {
                "mode": "vals",
                "url": self.url,
                "bbox": request["bbox"],
                "width": request["width"],
                "height": request["height"],
                "projection": request["projection"],
                "bands": bands,
                "dtype": self.dtype,
                "fillvalue": self.fillvalue,
            }
        elif mode == "meta":
            # metadata can't be None at this point
            process_kwargs = {
                "mode": "meta",
                "url": self.url,
                "bands": bands,
            }
        elif mode == "time":
            process_kwargs = {
                "mode": "time",
                "start": start,
                "delta": self.timedelta or timedelta(0),
                "bands": bands,
            }
        else:
            raise RuntimeError("Unknown mode '{}'".format(mode))
        return [(process_kwargs, None)]
