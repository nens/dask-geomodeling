import glob
import os
import sys
import shutil
import logging
from contextlib import contextmanager

import json
import fiona
import geopandas
import tempfile
from dask.config import config
from dask.base import tokenize
from dask_geomodeling import utils

from .base import BaseSingle
from .parallelize import GeometryTiler

__all__ = ["GeometryFileSink", "to_file"]

logger = logging.getLogger(__name__)


def _to_json(value):
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value)
        except TypeError:
            return "<unable to export>"  # cannot output this value
    else:
        return value


class GeometryFileSink(BaseSingle):
    """Write geometry data to files in a specified directory

    Use GeometryFileSink.merge_files to merge tiles into one large file.

    Args:
      source (GeometryBlock): The block the data is coming from
      url (str): The target directory to put the files in. If relative, it is
        taken relative to the geomodeling.root setting.
      extension (str): The file extension (defines the format), one of
        ``{"shp", "gpkg", "geojson", "gml"}``. On some platforms, these options
        might be limited. For an accurate list, see
        ``GeometryFileSink.supported_extensions``.
      fields (dict): A mapping that relates column names to output file field
        names field names like ``{<output file field name>: <column name>}``.

    Relevant settings can be adapted as follows:
      >>> from dask import config
      >>> config.set({"geomodeling.root": '/my/output/data/path'})
    """

    supported_extensions = {
        k: v
        for k, v in (
            ("shp", "ESRI Shapefile"),
            ("gpkg", "GPKG"),
            ("geojson", "GeoJSON"),
            ("gml", "GML"),
        )
        # take only drivers which Fiona reports as writable
        if "w" in fiona.supported_drivers.get(v, "")
        # skip GPKG and GML on Win32 platform as it yields a segfault
        and not (sys.platform == "win32" and k in ("gpkg", "gml"))
    }

    def __init__(self, source, url, extension="shp", fields=None):
        safe_url = utils.safe_file_url(url)
        if not isinstance(extension, str):
            raise TypeError("'{}' object is not allowed".format(type(extension)))
        if len(extension) > 0 and extension[0] == ".":
            extension = extension[1:]  # chop off the dot
        if extension not in self.supported_extensions:
            raise ValueError("Format '{}' is unsupported".format(extension))
        if fields is None:
            fields = {x: x for x in source.columns if x != "geometry"}
        elif not isinstance(fields, dict):
            raise TypeError("'{}' object is not allowed".format(type(fields)))
        else:
            missing = set(fields.values()) - source.columns
            if missing:
                raise ValueError("Columns {} are not available".format(missing))
        super().__init__(source, safe_url, extension, fields)

    @property
    def url(self):
        return self.args[1]

    @property
    def extension(self):
        return self.args[2]

    @property
    def fields(self):
        return self.args[3]

    @property
    def columns(self):
        return {"saved"}

    def get_sources_and_requests(self, **request):
        process_kwargs = {
            "url": self.url,
            "fields": self.fields,
            "extension": self.extension,
            "hash": tokenize(request)[:7],
        }
        return [(self.source, request), (process_kwargs, None)]

    @staticmethod
    def process(data, process_kwargs):
        if "features" not in data or len(data["features"]) == 0:
            return data  # do nothing for non-feature or empty requests

        features = data["features"].copy()
        projection = data["projection"]
        path = utils.safe_abspath(process_kwargs["url"])
        fields = process_kwargs["fields"]
        extension = process_kwargs["extension"]
        driver = GeometryFileSink.supported_extensions[extension]

        # generate the directory if necessary
        os.makedirs(path, exist_ok=True)

        # the target file path is a deterministic hash of the request
        filename = ".".join([process_kwargs["hash"], extension])

        # add the index to the columns if necessary
        index_name = features.index.name
        if index_name in fields.values() and index_name not in features.columns:
            features[index_name] = features.index

        # copy the dataframe
        features = features[["geometry"] + list(fields.values())]

        # rename the columns
        features.columns = ["geometry"] + list(fields.keys())

        # serialize nested fields (lists or dicts)
        for col in fields.keys():
            series = features[col]
            if series.dtype == object or (
                str(series.dtype) == "category"
                and series.cat.categories.dtype == object
            ):
                features[col] = series.map(_to_json)

        # convert categoricals
        for col in fields.keys():
            series = features[col]
            if str(series.dtype) == "category":
                features[col] = series.astype(series.cat.categories.dtype)

        # GeoJSON needs reprojection to EPSG:4326
        if driver == "GeoJSON" and projection.upper() != "EPSG:4326":
            features = utils.geodataframe_transform(features, projection, "EPSG:4326")

        # generate the file
        features.to_file(os.path.join(path, filename), driver=driver)

        result = geopandas.GeoDataFrame(index=features.index)
        result["saved"] = True
        return {"features": result, "projection": projection}

    @staticmethod
    def merge_files(path, target, remove_source=False):
        """Merge files (the output of this Block) into one single file.

        Optionally removes the source files.
        """
        path = utils.safe_abspath(path)
        target = utils.safe_abspath(target)

        if os.path.exists(target):
            raise IOError("Target '{}' already exists".format(target))

        target_base, ext = os.path.splitext(target)
        source_paths = glob.glob(os.path.join(path, "*" + ext))
        if len(source_paths) == 0:
            raise IOError(
                "No source files found with matching extension '{}'".format(ext)
            )
        elif len(source_paths) == 1:
            # shortcut for single file. we need to copy/move all base_name.*
            # files (e.g. shapefiles have multiple files)
            source_base = os.path.splitext(source_paths[0])[0]
            move_or_copy = shutil.move if remove_source else shutil.copy
            for file_path in glob.glob(source_base + ".*"):
                move_or_copy(file_path, target_base + os.path.splitext(file_path)[1])
            return

        with utils.fiona_env():
            # first detect the driver etc
            with fiona.collection(source_paths[0], "r") as source:
                kwargs = {
                    "driver": source.driver,
                    "crs": source.crs,
                    "schema": source.schema,
                }
                if source.encoding:
                    kwargs["encoding"] = source.encoding

            with fiona.collection(target, "w", **kwargs) as out:
                for source_path in source_paths:
                    with fiona.collection(source_path, "r") as source:
                        out.writerecords(v for k, v in source.items())
                    if remove_source:
                        os.remove(source_path)

            if remove_source:
                try:
                    os.rmdir(path)
                except IOError:  # directory not empty: do nothing
                    pass


@contextmanager
def DryRunTempDir(*args, **kwargs):
    yield "/tmp/dummy"


def to_file(source, url, fields=None, tile_size=None, dry_run=False, **request):
    """Utility function to export data from a GeometryBlock to a file on disk.

    You need to specify the target file path as well as the extent geometry
    you want to save. Feature properties can be saved by providing a field
    mapping to the ``fields`` argument.

    To stay within memory constraints or to parallelize an operation, the
    ``tile_size`` argument can be provided.

    Args:
      source (GeometryBlock): the block the data is coming from
      url (str): The target file path. The extension determines the format. For
        supported formats, consult GeometryFileSink.supported_extensions.
      fields (dict): a mapping that relates column names to output file field
        names field names, ``{<output file field name>: <column name>, ...}``.
      tile_size (int): Optionally use this for large exports to stay within
        memory constraints. The export is split in tiles of given size (units
        are determined by the projection). Finally the tiles are merged.
      dry_run (bool): Do nothing, only validate the arguments.
      geometry (shapely Geometry): Limit exported objects to objects whose
        centroid intersects with this geometry.
      projection (str): The projection as a WKT string or EPSG code.
        Sets the projection of the geometry argument, the target
        projection of the data, and the tiling projection.
      mode (str): one of ``{"intersects", "centroid"}``, default "centroid"
      start (datetime): start date as UTC datetime
      stop (datetime): stop date as UTC datetime
      **request: see GeometryBlock request specification

    Relevant settings can be adapted as follows:
      >>> from dask import config
      >>> config.set({"geomodeling.root": '/my/output/data/path'})
      >>> config.set({"temporary_directory": '/my/alternative/tmp/dir'})
    """
    if "mode" not in request:
        request["mode"] = "centroid"

    path = utils.safe_abspath(url)
    extension = os.path.splitext(path)[1]

    TmpDir = DryRunTempDir if dry_run else tempfile.TemporaryDirectory
    with TmpDir(dir=config.get("temporary_directory", None)) as tmpdir:
        sink = GeometryFileSink(source, tmpdir, extension=extension, fields=fields)

        # wrap the sink in a GeometryTiler
        if tile_size is not None:
            sink = GeometryTiler(sink, tile_size, request["projection"])

        if dry_run:
            return

        # export the dataset to the tmpdir (full dataset or multiple tiles)
        sink.get_data(**request)

        # copy the file / gather the tiles to the target location
        GeometryFileSink.merge_files(tmpdir, path)
