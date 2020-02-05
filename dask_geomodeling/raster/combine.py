"""
Module containing raster blocks that combine rasters.
"""
import itertools
from datetime import timedelta as Timedelta
import numpy as np

from dask_geomodeling.utils import get_dtype_max, get_index, GeoTransform

from .base import RasterBlock

__all__ = ["Group"]


def filter_none(lst):
    return [x for x in lst if x is not None]


class BaseCombine(RasterBlock):
    """ Base block that combines rasters into a larger one.

    The ancestor stores are kept in ``self.args``. Attributes are greedy:
    ``period`` is the union of the ancestor periods, and ``extent`` the union
    of the ancestor extents. The ``timedelta`` is propagated only if the
    ancestor stores have equal ``timedelta`` and if they are aligned. Rasters
    without data are ignored.
    """

    def __init__(self, *args):
        for arg in args:
            if not isinstance(arg, RasterBlock):
                raise TypeError("'{}' object is not allowed".format(type(arg)))
        super(BaseCombine, self).__init__(*args)

    @staticmethod
    def get_aligned_timedelta(sources):
        """ Checks provided sources and returns the timedelta when all sources
        are aligned. Stores without data are ignored. """
        timedeltas = []
        periods = []
        for arg in sources:
            timedelta, period = arg.timedelta, arg.period
            if period is not None and timedelta is not None:
                timedeltas.append(timedelta)
                periods.append(period)

        if len(timedeltas) == 0:
            return None
        elif len(timedeltas) == 1:
            return timedeltas[0]

        # multiple timedeltas: return None if not equal
        if not timedeltas[1:] == timedeltas[:-1]:
            return None
        else:
            # the periods must be spaced an integer times timedelta apart
            timedelta_sec = timedeltas[0].total_seconds()
            first, _ = periods[0]
            for a, _ in periods[1:]:
                if (first - a).total_seconds() % timedelta_sec != 0:
                    return None
            return timedeltas[0]

    @property
    def timedelta(self):
        """ The period between timesteps in case of equidistant time. """
        return self.get_aligned_timedelta(self.args)

    @property
    def period(self):
        """ Return the combined period datetime tuple. """
        periods = filter_none([x.period for x in self.args])
        if len(periods) == 0:
            return None
        elif len(periods) == 1:
            return periods[0]

        # multiple periods: return the joined period
        return min([p[0] for p in periods]), max([p[1] for p in periods])

    @property
    def extent(self):
        """ Boundingbox of combined contents in WGS84 projection. """
        extents = filter_none([x.extent for x in self.args])
        if len(extents) == 0:
            return None
        elif len(extents) == 1:
            return extents[0]

        # multiple extents: return the joined box
        x1 = min([e[0] for e in extents])
        y1 = min([e[1] for e in extents])
        x2 = max([e[2] for e in extents])
        y2 = max([e[3] for e in extents])
        return x1, y1, x2, y2

    @property
    def dtype(self):
        return np.result_type(*self.args)

    @property
    def fillvalue(self):
        return get_dtype_max(self.dtype)

    @property
    def geometry(self):
        """Combined geometry in the projection of the first store geometry. """
        geometries = filter_none([x.geometry for x in self.args])
        if len(geometries) == 0:
            return
        elif len(geometries) == 1:
            return geometries[0]
        result = geometries[0]
        sr = result.GetSpatialReference()
        for geometry in geometries[1:]:
            if not geometry.GetSpatialReference().IsSame(sr):
                geometry = geometry.Clone()
                geometry.TransformTo(sr)
            result = result.Union(geometry)
        return result

    @property
    def projection(self):
        """Projection of the data if they match, else None"""
        projection = self.args[0].projection
        if projection is None:
            return
        for arg in self.args[1:]:
            if projection != arg.projection:
                return
        return projection

    @property
    def geo_transform(self):
        geo_transform = self.args[0].geo_transform
        if geo_transform is None:
            return
        geo_transform = GeoTransform(geo_transform)
        for arg in self.args[1:]:
            other = arg.geo_transform
            if other is None or not geo_transform.aligns_with(other):
                return
        return geo_transform


class Group(BaseCombine):
    """
    Combine multiple rasters into a single one.

    Operation to combine multiple rasters into one along all three axes (x, y
    and temporal). To only fill 'no data' values of input rasters that have the
    same temporal resolution ``dask_geomodeling.raster.elemwise.FillNoData``
    is preferred.

    Values at equal timesteps in the contributing rasters are considered
    starting with the leftmost input raster. Therefore, values from rasters
    that are more 'to the right' are shown in the result. 'no data' values are
    transparent and will show data of rasters more 'to the left'.

    Args:
      *args (list of RasterBlocks): list of rasters to be combined.

    Returns:
      RasterBlock that combines all input rasters
    """

    def get_stores(self, start, stop):
        """ Return all relevant stores for given start and stop. """
        # check stores and select those stores that contain data
        stores = [s for s in self.args if s.period is not None]

        # there could be no store with data at all
        if not stores:
            return stores

        # convenience
        starts, stops = zip(*(s.period for s in stores))

        # pick latest store(s) in time
        if start is None:
            last = max(stops)
            return [s for b, s in zip(stops, stores) if b == last]

        if stop is None:
            # return any stores that contain start
            zipped = zip(starts, stops, stores)
            result = [s for a, b, s in zipped if a <= start <= b]
            if result:
                return result

            # no store contained start, return closest stores
            closest = min(starts + stops, key=lambda d: abs(d - start))
            zipped = zip(stops + starts, stores + stores)
            return [s for d, s in zipped if d == closest]

        # start and stop given, return all relevant stores
        zipped = zip(starts, stops, stores)
        return [s for a, b, s in zipped if not (stop < a or start > b)]

    def get_sources_and_requests(self, **request):
        start = request.get("start", None)
        stop = request.get("stop", None)
        mode = request["mode"]

        sources = self.get_stores(start, stop)

        # just pass on the source and request if we only have one (or none)
        if len(sources) <= 1:
            requests = [(s, request) for s in sources]
            return [(dict(combine_mode="simple"), None)] + requests

        # plan for merging
        timedelta = self.get_aligned_timedelta(sources)
        mixed_time = timedelta is None or start is None or stop is None

        if mixed_time:  # merge by time
            requests = []
            time_requests = []
            for source in sources:
                # add the stores and requests:
                requests.append((source, request))

                # in case we need the time information, add time requests
                if mode != "time":
                    time_request = dict(mode="time", start=start, stop=stop)
                    time_requests.append((source, time_request))

            process_kwargs = dict(
                combine_mode="by_time", mode=mode, start=start, stop=stop
            )

            # note that time_requests is empty if mode is 'time'
            requests = requests + time_requests
        else:  # merge by bands
            td_sec = timedelta.total_seconds()
            period = self.period
            origin = sources[0].period[0]  # any will do; they are aligned

            if start <= period[0]:
                start = period[0]
            else:
                # ceil start to the closest integer timedelta
                start_delta = (origin - start).total_seconds() % td_sec
                start += Timedelta(seconds=start_delta)

            if stop >= period[1]:
                stop = period[1]
            else:
                # floor stop to the closest integer timedelta
                stop_delta = (stop - origin).total_seconds() % td_sec
                stop -= Timedelta(seconds=stop_delta)

            if mode == "time":
                return [
                    (
                        dict(
                            combine_mode="by_bands",
                            mode=mode,
                            start=start,
                            stop=stop,
                            timedelta=timedelta,
                        ),
                        None,
                    )
                ]

            requests = []
            bands = []
            for source in sources:
                # compute 'bands': the index ranges into the result array
                this_start = max(start, source.period[0])
                this_stop = min(stop, source.period[1])
                first_i = int((this_start - start).total_seconds() // td_sec)
                last_i = int((this_stop - start).total_seconds() // td_sec)
                bands.append((first_i, last_i + 1))

                this_request = request.copy()
                this_request.update(start=this_start, stop=this_stop)
                requests.append((source, this_request))

            process_kwargs = dict(combine_mode="by_bands", mode=mode, bands=bands)

        # in case of a 'vals' request, keep track of the dtype
        if mode == "vals":
            process_kwargs["dtype"] = self.dtype

        return [(process_kwargs, None)] + requests

    @staticmethod
    def _unique_times(multi):
        times = filter_none([data.get("time", None) for data in multi])
        return sorted(set(itertools.chain(*times)))

    @staticmethod
    def _nearest_index(time, start):
        if start is None:
            # last band
            return len(time) - 1
        else:
            # nearest band
            return min(enumerate(time), key=lambda d: abs(d[1] - start))[0]

    @staticmethod
    def _merge_vals_by_time(multi, times, kwargs):
        """ Merge chunks using indices. """
        # determine the unique times and assign result bands
        sorted_times = Group._unique_times(times)
        bands = dict((y, x) for x, y in enumerate(sorted_times))
        fillvalue = get_dtype_max(kwargs["dtype"])

        # initialize values array
        shape = (len(sorted_times),) + multi[0]["values"].shape[1:]
        values = np.full(shape, fillvalue, dtype=kwargs["dtype"])

        # populate values array
        for data, time in zip(multi, times):
            # source_index is the index into the source array
            for source_index, datetime in enumerate(time["time"]):
                source_band = data["values"][source_index]
                # determine data index
                index = get_index(
                    values=source_band, no_data_value=data["no_data_value"]
                )
                # find corresponding target band by source datetime
                target_band = values[bands[datetime]]
                # paste source into target provided there is data
                target_band[index] = source_band[index]

        # check if single band result required
        start, stop = kwargs["start"], kwargs["stop"]
        if stop is None and len(sorted_times) > 1:
            index = Group._nearest_index(sorted_times, start)
            values = values[index : index + 1]

        return {"values": values, "no_data_value": fillvalue}

    @staticmethod
    def _merge_meta_by_time(multi, times, kwargs):
        # determine the unique times and assign result bands
        sorted_times = Group._unique_times(times)
        bands = dict((y, x) for x, y in enumerate(sorted_times))
        meta_result = [None] * len(sorted_times)

        # populate result array
        for data, time in zip(multi, times):
            # source_index is the index into the source array
            for source_index, datetime in enumerate(time["time"]):
                source_band = data["meta"][source_index]
                # find corresponding target band by source datetime
                target_band = bands[datetime]
                # paste source into target provided there is data
                meta_result[target_band] = source_band

        # check if single band result required
        start, stop = kwargs["start"], kwargs["stop"]
        if stop is None and len(sorted_times) > 1:
            index = Group._nearest_index(sorted_times, start)
            meta_result = meta_result[index : index + 1]

        return {"meta": meta_result}

    @staticmethod
    def _merge_vals_by_bands(multi, bands, dtype):
        """ Merge chunks using slices. """
        # analyze band structure
        starts, stops = zip(*bands)
        fillvalue = get_dtype_max(dtype)

        # initialize values array
        shape = (max(stops),) + multi[0]["values"].shape[1:]
        values = np.full(shape, fillvalue, dtype=dtype)

        # populate values array
        for data, (a, b) in zip(multi, bands):
            # index is where the source has data
            index = get_index(
                values=data["values"], no_data_value=data["no_data_value"]
            )
            values[a:b][index] = data["values"][index]

        return {"values": values, "no_data_value": fillvalue}

    @staticmethod
    def _merge_meta_by_bands(multi, bands):
        """ Merge metadata by bands. """
        # analyze band structure
        starts, stops = zip(*bands)

        # initialize metadata list
        meta_result = [""] * max(stops)

        # populate metadata list
        for data, (a, b) in zip(multi, bands):
            for i, meta in zip(range(a, b), data["meta"]):
                if meta:
                    meta_result[i] = meta

        return {"meta": meta_result}

    @staticmethod
    def process(process_kwargs, *args):
        # plan for merging
        combine_mode = process_kwargs["combine_mode"]
        mode = process_kwargs.get("mode", None)

        if combine_mode == "simple":
            # simple mode, return the result right away
            if len(args) == 0:
                return None
            else:
                return args[0]
        elif combine_mode == "by_time" and mode == "time":
            sorted_times = Group._unique_times(args)

            # check if single band result required
            start, stop = process_kwargs["start"], process_kwargs["stop"]
            if stop is None and len(sorted_times) > 1:
                index = Group._nearest_index(sorted_times, start)
                sorted_times = sorted_times[index : index + 1]
            return {"time": sorted_times}
        elif combine_mode == "by_time" and mode in ["meta", "vals"]:
            # split the data and time results, skipping None
            n = int(len(args) // 2)

            # assume that we have None at the same positions
            multi, times = filter_none(args[:n]), filter_none(args[n:])
            if len(multi) == 0:
                return None

            if mode == "vals":
                return Group._merge_vals_by_time(multi, times, process_kwargs)
            elif mode == "meta":
                return Group._merge_meta_by_time(multi, times, process_kwargs)
        elif combine_mode == "by_bands" and mode == "time":
            # start and stop are aligned so we can compute the times here
            start = process_kwargs["start"]
            stop = process_kwargs["stop"]
            delta = process_kwargs["timedelta"]
            length = (stop - start).total_seconds() // delta.total_seconds()
            length = int(length) + 1  # the result includes the last frame
            return {"time": [start + i * delta for i in range(int(length))]}
        elif combine_mode == "by_bands" and mode in ["meta", "vals"]:
            # list the data and bands results, skipping None
            multi = []
            bands = []
            for data, _bands in zip(args, process_kwargs["bands"]):
                if data is None:
                    continue
                multi.append(data)
                bands.append(_bands)

            if len(multi) == 0:
                return None

            if mode == "vals":
                dtype = process_kwargs["dtype"]
                return Group._merge_vals_by_bands(multi, bands, dtype)
            elif mode == "meta":
                return Group._merge_meta_by_bands(multi, bands)
        else:
            raise ValueError("Unknown combine_mode / mode combination")
