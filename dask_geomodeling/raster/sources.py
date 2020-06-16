"""
Module containing raster sources.
"""
import numpy as np

from osgeo import gdal, gdal_array

from datetime import datetime, timedelta

from dask_geomodeling import utils

from .base import RasterBlock

__all__ = ["MemorySource", "RasterFileSource"]


class MemorySource(RasterBlock):
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
        data = np.atleast_3d(data)
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
        return utils.Extent(bbox, utils.get_sr(self.projection))

    @property
    def extent(self):
        extent = self._get_extent()
        if extent is None:
            return
        return extent.transformed(utils.EPSG4326).bbox

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
            return (datetime.utcfromtimestamp(self.time_first / 1000),) * 2
        else:
            first = datetime.utcfromtimestamp(self.time_first / 1000)
            last = first + (len(self) - 1) * self.timedelta
            return first, last

    @property
    def timedelta(self):
        if len(self) <= 1:
            return None
        return timedelta(milliseconds=self.time_delta)

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
        start, stop, first_i, last_i = utils.snap_start_stop(
            request.get("start"),
            request.get("stop"),
            datetime.utcfromtimestamp(self.time_first / 1000),
            self.timedelta,
            len(self),
        )
        if start is None:
            return [({"mode": "empty_" + request["mode"]}, None)]

        # set the process kwargs depending on the mode
        if mode == "vals":
            process_kwargs = {
                "mode": "vals",
                "data": self.data[first_i : last_i + 1],
                "no_data_value": self.no_data_value,
                "bbox": request["bbox"],
                "width": request["width"],
                "height": request["height"],
                "geo_transform": self.geo_transform,
            }
        elif mode == "meta":
            # metadata can't be None at this point
            process_kwargs = {
                "mode": "meta",
                "metadata": self.metadata[first_i : last_i + 1],
            }
        elif mode == "time":
            process_kwargs = {
                "mode": "time",
                "start": start,
                "delta": self.timedelta or timedelta(0),
                "length": last_i - first_i + 1,
            }
        else:
            raise RuntimeError("Unknown mode '{}'".format(mode))
        return [(process_kwargs, None)]

    @staticmethod
    def process(process_kwargs):
        mode = process_kwargs["mode"]

        # handle empty results
        if mode == "empty_vals":
            return
        elif mode == "empty_time":
            return {"time": []}
        elif mode == "empty_meta":
            return {"meta": []}

        # handle time requests
        if mode == "time":
            start = process_kwargs["start"]
            length = process_kwargs["length"]
            delta = process_kwargs["delta"]
            return {"time": [start + i * delta for i in range(length)]}
        elif mode == "meta":
            return {"meta": process_kwargs["metadata"]}

        # handle 'vals' requests
        data = process_kwargs["data"]
        no_data_value = process_kwargs["no_data_value"]
        bbox = process_kwargs["bbox"]
        width = process_kwargs["width"]
        height = process_kwargs["height"]
        gt = utils.GeoTransform(process_kwargs["geo_transform"])

        # return an empty array if 0-sized data was requested
        if width == 0 or height == 0:
            return np.empty((data.shape[0], height, width), dtype=data.dtype)

        # transform the requested bounding box to indices into the array
        shape = data.shape
        ranges, padding = gt.get_array_ranges(bbox, shape)
        result = data[:, slice(*ranges[0]), slice(*ranges[1])]

        # pad the data to the shape given by the index
        if padding is not None:
            padding = ((0, 0),) + padding  # for the time axis
            result = np.pad(result, padding, "constant", constant_values=no_data_value)

        # zoom to the desired height and width
        result = utils.zoom_raster(result, no_data_value, height, width)

        # fill nan values if they popped up
        result[~np.isfinite(result)] = no_data_value
        return {"values": result, "no_data_value": no_data_value}


class RasterFileSource(RasterBlock):
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
        return gdal_array.GDALTypeCodeToNumericTypeCode(first_band.DataType)

    @property
    def fillvalue(self):
        first_band = self.gdal_dataset.GetRasterBand(1)
        return self.dtype(first_band.GetNoDataValue())

    @property
    def geo_transform(self):
        return utils.GeoTransform(self.gdal_dataset.GetGeoTransform())

    def _get_extent(self):
        bbox = self.geo_transform.get_bbox(
            (0, 0), (self.gdal_dataset.RasterYSize, self.gdal_dataset.RasterXSize)
        )
        return utils.Extent(bbox, utils.get_sr(self.projection))

    @property
    def extent(self):
        extent_epsg4326 = self._get_extent().transformed(utils.EPSG4326)
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
            return (datetime.utcfromtimestamp(self.time_first / 1000),) * 2
        else:
            first = datetime.utcfromtimestamp(self.time_first / 1000)
            last = first + (len(self) - 1) * self.timedelta
            return first, last

    @property
    def timedelta(self):
        if len(self) <= 1:
            return None
        return timedelta(milliseconds=self.time_delta)

    def get_sources_and_requests(self, **request):
        mode = request["mode"]

        if (
            mode == "vals"
            and request.get("projection").upper() != self.projection.upper()
        ):
            raise RuntimeError("This source block cannot reproject data")

        # interpret start and stop request parameters
        start, stop, first_i, last_i = utils.snap_start_stop(
            request.get("start"),
            request.get("stop"),
            datetime.utcfromtimestamp(self.time_first / 1000),
            self.timedelta,
            len(self),
        )
        if start is None:
            return [({"mode": "empty_" + request["mode"]}, None)]

        # set the process kwargs depending on the mode
        if mode == "vals":
            process_kwargs = {
                "mode": "vals",
                "url": self.url,
                "bbox": request["bbox"],
                "width": request["width"],
                "height": request["height"],
                "first_band": last_i,
                "last_band": first_i,
                "dtype": self.dtype,
                "fillvalue": self.fillvalue,
            }
        elif mode == "meta":
            # metadata can't be None at this point
            process_kwargs = {
                "mode": "meta",
                "url": self.url,
                "first_band": last_i,
                "last_band": first_i,
            }
        elif mode == "time":
            process_kwargs = {
                "mode": "time",
                "start": start,
                "delta": self.timedelta or timedelta(0),
                "length": last_i - first_i + 1,
            }
        else:
            raise RuntimeError("Unknown mode '{}'".format(mode))
        return [(process_kwargs, None)]

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

        # handle time requests
        if mode == "time":
            start = process_kwargs["start"]
            length = process_kwargs["length"]
            delta = process_kwargs["delta"]
            return {"time": [start + i * delta for i in range(length)]}

        # open the dataset
        url = process_kwargs["url"]
        path = utils.safe_abspath(url)
        dataset = gdal.Open(path)
        first_band = process_kwargs["first_band"]
        last_band = process_kwargs["last_band"]

        # handle meta requests
        if mode == "meta":
            return {
                "meta": [
                    dataset.GetRasterBand(i + 1).GetMetadata_Dict()
                    for i in range(first_band, last_band + 1)
                ]
            }

        # handle 'vals' requests
        dtype = process_kwargs["dtype"]
        no_data_value = process_kwargs["fillvalue"]
        bbox = process_kwargs["bbox"]
        width = process_kwargs["width"]
        height = process_kwargs["height"]
        length = last_band - first_band + 1

        # return an empty array if 0-sized data was requested
        if width == 0 or height == 0:
            return np.empty((length, height, width), dtype=dtype)

        # transform the requested bounding box to indices into the array
        shape = dataset.RasterCount, dataset.RasterYSize, dataset.RasterXSize
        gt = utils.GeoTransform(dataset.GetGeoTransform())
        ranges, padding = gt.get_array_ranges(bbox, shape)
        read_shape = [rng[1] - rng[0] for rng in ranges]

        # return nodata immediately for empty
        if any([x <= 0 for x in read_shape]):
            result = np.full(
                shape=(length, height, width), fill_value=no_data_value, dtype=dtype
            )
            return {"values": result, "no_data_value": no_data_value}

        # read arrays from file
        result = np.empty([length] + read_shape, dtype=dtype)
        for k in range(length):
            band = dataset.GetRasterBand(first_band + k + 1)
            result[k] = band.ReadAsArray(
                int(ranges[1][0]),
                int(ranges[0][0]),
                int(read_shape[1]),
                int(read_shape[0]),
            )

        # pad the data to the shape given by the index
        if padding is not None:
            padding = ((0, 0),) + padding  # for the time axis
            result = np.pad(result, padding, "constant", constant_values=no_data_value)

        # zoom to the desired height and width
        result = utils.zoom_raster(result, no_data_value, height, width)

        # fill nan values if they popped up
        result[~np.isfinite(result)] = no_data_value
        return {"values": result, "no_data_value": no_data_value}
