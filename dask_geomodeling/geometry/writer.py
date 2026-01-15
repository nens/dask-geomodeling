import os
import logging
from contextlib import contextmanager

import json
import pyogrio
from dask_geomodeling import utils

from .parallelize import GeometryTiler

__all__ = ["GeometryFileWriter", "to_file"]

logger = logging.getLogger(__name__)


def _to_json(value):
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value)
        except TypeError:
            return "<unable to export>"  # cannot output this value
    else:
        return value


class GeometryFileWriter:
    """Helper class to write a geometry file

    Args:
      url (str): The target location to output the file in. If relative, it is
        taken relative to the geomodeling.root setting.
      fields (dict, optional): A mapping that relates column names to output file field
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
        # take only drivers which pyogrio reports as writable
        if "w" in pyogrio.list_drivers().get(v, "")
    }

    def __init__(self, url, fields=None):
        self.url = utils.safe_file_url(url)
        self.fields = fields

    def write(self, features):
        if features is None or len(features) == 0:
            return None

        path = utils.safe_abspath(self.url)
        if os.path.exists(path):
            raise FileExistsError("Target '{}' already exists".format(path))

        extension = os.path.splitext(path)[1][1:].lower()
        driver = self.supported_extensions[extension]

        if self.fields:
            fields = {x: x for x in features.columns if x != "geometry"}
        else:
            fields = self.fields

        # add the index to the columns if necessary
        index_name = features.index.name
        if index_name in fields.values() and index_name not in features.columns:
            features[index_name] = features.index
            write_index = False
        else:
            write_index = True

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

        # generate the file
        features.to_file(
            path,
            driver=driver,
            index=write_index,
            crs="EPSG:4326" if driver == "GeoJSON" else None,
            engine="pyogrio",
        )


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

    target_path = utils.safe_abspath(url)
    if os.path.exists(target_path):
        raise FileExistsError("Target '{}' already exists".format(target_path))
    target_dir = os.path.dirname(target_path)
    if not os.path.exists(target_dir):
        raise FileNotFoundError(
            "Target directory '{}' does not exist".format(target_dir)
        )

    # wrap the sink in a GeometryTiler (this will concat all features into 1 big geodataframe)
    if tile_size is not None:
        computation = GeometryTiler(source, tile_size, request["projection"])
    else:
        computation = source

    writer = GeometryFileWriter(target_path, fields=fields)

    if dry_run:
        return

    # compute and write the result
    writer.write(computation.get_data(**request)["features"])