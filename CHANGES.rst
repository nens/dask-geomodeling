Changelog of dask-geomodeling
===================================================

2.3.5 (unreleased)
------------------

- Nothing changed yet.


2.3.4 (2021-02-08)
------------------

- Added a default setting "raster-limit-timesteps".


2.3.3 (2020-12-11)
------------------

- Fix field_operations.Classify if used with int-typed labels. A NaN value in
  the result resulted in a Categorical output dtype. To fix this, Classify
  now returns floats also when input labels are integers. (#82)


2.3.2 (2020-11-19)
------------------

- Make Dilate arguments JSON serializable. (#81)


2.3.1 (2020-11-10)
------------------

- Never return Categorical dtypes in the field_operations.Classify and
  ClassifyFromColumns. This leads to pandas incompatibilities with later
  operations (round, subtract, where, mask).

- Never return Categorical dtypes from ParseTextColumn. (#79)

- field_operations.Where and field_operations.Mask now also allow non-boolean
  conditionals. This was already the case for Where on pandas == 0.19.*. (#78)


2.3.0 (2020-10-09)
------------------

- Added Exp, Log and Log10 RasterBlocks.

- Added "std" and "var" statistics to TemporalAggregate.


2.2.12 (2020-09-29)
-------------------

- Fixed point requests for RasterizeWKT.

- Allow empty coordinate list in Place.


2.2.11 (2020-09-01)
-------------------

- Make transformation exceptions more comprehensible.

- Check for matching time resolutions in raster.Clip.

- Added 'product' to raster.reduction STATISTICS.

2.2.10 (2020-07-29)
-------------------

- Fix point requests in raster.Smooth.

- GDAL 3 compatibility fixes.


2.2.9 (2020-06-23)
------------------

- Implemented `RasterTiler`.

- Let raster.Mask accomodate int values larger than uint8.


2.2.8 (2020-06-12)
------------------

- Accept categorical values in GeometryFileSink / to_file.

- Fixed incompatibilities with geopandas >=0.7.

- GeoJSON output is always converted to EPSG:4326 and doesn't have "crs" field.

- Implemented raster.reduction.reduce_rasters.

- Added the 'statistics' argument to raster.spatial.Place to deal with
  overlapping features. 

- Allow point requests in raster.spatial.Place.

- Clarifications about raster cell validity ranges in MemorySource and
  RasterFileSource.


2.2.7 (2020-04-30)
------------------

- Accept list and dict values in GeometryFileSink / to_file.

- Fix bug in ParseTextColumn that added columns in duplicate when outputting
  into the input column.


2.2.6 (2020-04-28)
------------------

- Fixed bug in `FillNoData` block.

- Fixed bug in `AggregateRasterAboveThreshold` (introduced in #37) (#44).


2.2.4 (2020-03-25)
------------------

- Allow up to 1E-7 in the GeoTransform 'tilt' terms to account for possible
  float32 imprecision.

- Handle Nones in geometry.field_operations.Classify and ClassifyFromColumns.

- Validate if labels are unique in geometry.field_operations.Classify and
  ClassifyFromColumns.
  
  - Added raster.spatial.Place.


2.2.3 (2020-02-28)
------------------

-  Fix AggregateRaster: it now returns NaN for no data pixels (#37)


2.2.2 (2020-02-13)
------------------

- Added GeometryWKTSource.

- Updated all docstrings.

- Renamed the 'location' parameter of raster.misc.Step to 'value'.


2.2.1 (2020-02-04)
------------------

- Suppressed "invalid value encountered in greater than" warning in
  ClassifyFromColumns.

- Compatibility fixes for pandas 1.0.0.

- Implemented raster.RasterizeWKT


2.2.0 (2019-12-20)
------------------

- utils.get_crs now leaves EPSG codes instead of converting them to their Proj4
  representation.

- Implemented GeometryFileSink that writes ESRI Shapefile, GeoJSON, GML, and
  geopackage.

- Added a .to_file() method to all GeometryBlocks.

- Added dry_run parameter (for validation) to .to_file().

- Start using google docstring convention.

- Several minor doc fixes.

- Fix setting of the .crs property in the GeometryFileSource.

- Fixed serialization of raster.Classify.


2.1.1 (2019-12-06)
------------------

- Fix empty response of TemporalAggregate and Cumulative.

- Fix elementwise raster blocks in case of empty datasets.


2.1.0 (2019-11-15)
------------------

- Added RasterFileSource.close_dataset to close the GDAL file handle.

- Run unittests on windows.

- Adapt safe_abspath and safe_file_url functions: they now automatically
  interpret the geomodeling.root config instead of the 'start' kwarg.

- Added a geomodeling.strict-file-paths that defaults to False. This changes
  the default behaviour of all blocks that handle file paths: by default, the
  path is not required to be in geomodeling.root.

- Added installation instructions for windows.

- Improved the ipyleaflet plugin so that it can deal with multiple notebook
  servers on the same machine. The parameter 'hostname' was replaced by 'url'.


2.0.4 (2019-11-01)
------------------

- Fixed propagation of the 'extent' and 'geometry' attributes through the
  raster.Clip. Both now return the intersection of the store and mask rasters.

- The MemorySource and elementwise blocks now return None for 'extent' and
  'geometry' if they are empty.

- Preserve functionality of the geometry.Difference block with geopandas 0.6.
  When taking the difference of a geometry with a missing geometry (A - None),
  geopandas < 0.6 returned A as result, while >= 0.6 returns None as result.

- Added default values for RasterFileSource's time parameters.

- Implemented the 'columns' attribute for GeometryFileSource.

- Fixed the projection attribute of elementwise raster blocks in case one of
  the arguments is a number and not a Block instance.

- Implemented the geo_transform attribute of elementwise raster blocks.

- Added an ipyleaflet plugin for visualizing RasterBlocks in jupyter notebook.

- Changed the default geomodeling.root setting to the current working directory


2.0.3 (2019-10-08)
------------------

- Added documentation.

- Fixed MemorySource incase of a request outside of the data boundary.

- Fixed multiple bugs in Reclassify and added some tests. The 'from' dtype can
  now be boolean or integer, and the 'to' dtype integer or float. The returned
  dtype is now decided by numpy (int64 or float64).


2.0.2 (2019-09-04)
------------------

- Clean up the .check() method for RasterBlocks.

- Added a Travisfile testing with against versions since 2017 on Linux and OSX.

- Took some python 3.5 compatibility measures.

- Added fix in ParseText block for pandas 0.23.

- Changed underscores in config to dashes for dask 0.18 compatibility.

- Constrained dask to >= 0.18, numpy to >= 1.12, pandas to >= 0.19,
  geopandas to >= 0.4, scipy to >= 0.19.

- Removed the explicit (py)gdal dependency.


2.0.1 (2019-08-30)
------------------

- Renamed the package to dask-geomodeling.

- Integrated the settings with dask.config.

- Added BSD 3-Clause license.


2.0.0 (2019-08-27)
------------------

- Remove raster-store dependency.

- Removed RasterStoreSource, ThreediResultSource, Result, Interpolate,
  DeprecatedInterpolate, GeoInterface, and GroupTemporal geoblocks.

- Removed all django blocks GeoDjangoSource, AddDjangoFields, GeoDjangoSink.

- Simplified tokenization of Block objects.

- Implemented construct_multiple to construct multiple blocks at once.

- Implemented MemorySource and GeoTIFFSource as new raster sources.

- Add `Cumulative` geoblock for performing temporal cumulatives.


1.2.13 (2019-08-20)
-------------------

- Add `TemporalAggregate` geoblock for performing temporal aggregates on
  raster data.

- Fix raster math geoblocks to not have byte-sized integers 'wrap around'
  when they are added. All integer-types are now at least int32 and all float
  types at least float32.


1.2.12 (2019-07-30)
-------------------

- Made GeoDjangoSource backwards compatible with existing graph definitions.

- Fix Interpolate wrapper.


1.2.11 (2019-07-19)
-------------------

- Added new parameter `filters` to GeoDjangoSource.


1.2.10 (2019-07-05)
-------------------

- Classify block return single series with dtype of `labels`
  if `labels` are floats or integers.


1.2.9 (2019-06-29)
------------------

- Fix bug introduced in tokenization fix.


1.2.8 (2019-06-29)
------------------

- Skip tokenization if a block was already tokenized.


1.2.7 (2019-06-28)
------------------

- Implemented AggregateRasterAboveThreshold.


1.2.6 (2019-06-27)
------------------

- Fix in `ParseTextColumn` for empty column `description`.

- Fix empty dataset case in ClassifyFromColumns.


1.2.5 (2019-06-26)
------------------

- Skip (costly) call to tokenize() when constructing without validation. If a
  graph was supplied that was generated by geoblocks, the token should be
  present in the name. If the name has incorrect format, a warning is emitted
  and tokenize() is called after all.

- Deal with empty datasets in ClassifyFromColumns.


1.2.4 (2019-06-21)
------------------

- Updated ParseTextColumn: allow spaces in values.


1.2.3 (2019-06-21)
------------------

- Rasterize geoblock has a limit of 10000 geometries.

- Implemented Choose geoblock for Series.

- Added the block key in the exception message when construction failed.

- Added caching to get_compute_graph to speedup graph generation.

- Improved the documentation.


1.2.2 (2019-06-13)
------------------

- Fix tokenization of a geoblock when constructing with validate=False.

- The raster requests generated in AggregateRaster have their bbox now snapped
  to (0, 0) for better reproducibility.


1.2.1 (2019-06-12)
------------------

- Fix bug in geoblocks.geometry.constructive.Buffer that was introduced in 1.2.


1.2 (2019-06-12)
----------------

- Extend geometry.field_operations.Classify for classification outside of
  the bins. For example, you can now supply 2 bins and 3 labels.

- Implemented geometry.field_operations.ClassifyFromColumns that takes its bins
  from columns in a GeometryBlock, so that classification can differ per
  feature.

- Extend geometry.base.SetSeriesBlock to setting constant values.

- Implemented geometry.field_operations.Interp.

- Implemented geometry.text.ParseTextColumn that parses a text column into
  multiple value columns.

- AddDjangoFields converts columns to Categorical dtype automatically if the
  data is of 'object' dtype (e.g. strings). This makes the memory footprint of
  large text fields much smaller.

- Make validation of a graph optional when constructing.

- Use dask.get in construct and compute as to not doubly construct/compute.

- Fix bug in geoblocks.geometry.constructive.Buffer that changed the compute
  graph inplace, prohibiting 2 computations of the same graph.


1.1 (2019-06-03)
----------------

- GeoDjangoSink returns a dataframe with the 'saved' column indicating whether
  the save succeeded. IntegrityErrors result in saved=False.

- Added projection argument to `GeometryTiler`. The GeometryTiler only accepts
  requests that have a projection equal to the tiling projection.

- Raise a RuntimeError if the amount of returned geometries by GeoDjangoSource
  exceeds the GEOMETRY_LIMIT setting.

- Added `auto_pixel_size`  argument to geometry.AggregateRaster. If this
  is False, the process raises a RuntimeError when the required raster exceeds
  the `max_size` argument.

- If `max_size` in the geometry.AggregateRaster is None, it defaults to
  the global RASTER_LIMIT setting.

- Remove the index_field_name argument in GeoDjangoSource, instead obtain it
  automatically from model._meta.pk.name. The index can be added as a normal
  column by including it in 'fields'.

- Change the default behaviour of 'fields' in GeoDjangoSource: if not given, no
  extra fields are included. Also start and end field names are not included.

- Added the 'columns' attribute to all geometry blocks except for
  the GeometryFileSource.

- Added tests for SetSeriesBlock and GetSeriesBlock.

- Added check that column exist in GetSeriesBlock, AddDjangoFields and
  GeoDjangoSink.

- Implemented Round geoblock for Series.

- Fixed AggregateRaster when aggregating in a different projection than the
  request projection.

- Allow GeometryTiler to tile in a different projection than the request
  geometry is using.


1.0 (2019-05-09)
----------------

- Improved GeoDjangoSink docstring + fixed bug.

- Bug fix in GeoInterface for handling `inf` values.

- Added `Area` Geoblock for area calculation in Geometry blocks.

- Added MergeGeometryBlocks for `merge` operation between GeoDataFrames.

- Added `GeometryBlock.__getitem__ `and `GeometryBlock.set`, getting single
  columns from and setting multiple columns to a GeometryBlock. Corresponding
  geoblocks are geometry.GetSeriesBlock and geometry.SetSeriesBlock.

- Added basic operations for `add`,`sub`,`mul`,`div`,`truediv`,`floordiv`,
  `mod`, `eq`,`neq`,`ge`,`gt`,`le`,`lt`, `and`, `or`, `xor` and `not`
  operation in SeriesBlocks.

- Documented the request and response protocol for GeometryBlock.

- Added a tokenizer for shapely geometries, so that GeometryBlock request
  hashes are deterministic.

- Added a tokenizer for datetime and timedelta objects.

- Added geopandas dependency.

- Removed GeoJSONSource and implemented GeometryFileSource. This new reader has
  no simplify and intersect functions.

- Implemented geometry.set_operations.Intersection.

- Implemented geometry.constructive.Simplify.

- Adjusted the MockGeometry test class.

- Reimplemented utils.rasterize_geoseries and fixed raster.Rasterize.

- Reimplemented geometry.AggregateRaster.

- Fixed time requests for 3Di Result geoblocks that are outside the range of
  the dataset

- Implemented geometry.GeoDjangoSource.

- Implemented geometry.GeoDjangoSink.

- Added support for overlapping geometries when aggregating.

- Increased performance of GeoSeries coordinate transformations.

- Fixed inconsistent naming of the extent-type geometry response.

- Consistently return an empty geodataframe in case there are no geometries.

- Implemented geometry.Difference.

- Implemented geometry.Classify.

- Implemented percentile statistic for geometry.AggregateRaster.

- Implemented geometry.GeometryTiler.

- Explicitly set the result column name for AggregateRaster (default: 'agg').

- Implemented count statistic for geometry.AggregateRaster.

- Implemented geometry.AddDjangoFields.

- Added temporal filtering for Django geometry sources.

- Allow boolean masks in raster.Clip.

- Implemented raster.IsData.

- Implemented geometry.Where and geometry.Mask.

- Extended raster.Rasterize to rasterize float, int and bool properties.

- Fixed bug in Rasterize that set 'min_size' wrong.


0.6 (2019-01-18)
----------------

- Coerce the geo_transform to a list of floats in the raster.Interpolate,
  preventing TypeErrors in case it consists of decimal.Decimal objects.


0.5 (2019-01-14)
----------------

- Adapted path URLs to absolute paths in RasterStoreSource, GeoJSONSource, and
  ThreediResultSource. They still accept paths relative to the one stored in
  settings.


0.4 (2019-01-11)
----------------

- The `'store_resolution'` result field of `GeoInterface` now returns the
  resolution as integer (in milliseconds) and not as datetime.timedelta.

- Added metadata fields to Optimizer geoblocks.

- Propagate the union of the geometries in a Group (and Optimizer) block.

- Propagate the intersection of the geometries in elementwise blocks.

- Implement the projection metadata field for all blocks.

- Fixed the Shift geoblock by storing the time shift in milliseconds instead of
  a datetime.timedelta, which is not JSON-serializable.


0.3 (2018-12-12)
----------------

- Added geoblocks.raster.Classify.

- Let the raster.Interpolate block accept the (deprecated) `layout` kwarg.


0.2 (2018-11-20)
----------------

- Renamed ThreediResultSource `path` property to `hdf5_path` and fixed it.


0.1 (2018-11-19)
----------------

- Initial project structure created.

- Copied graphs.py, tokenize.py, wrappers.py, results.py, interfaces.py,
  and relevant tests and factories from raster-store.

- Wrappers are renamed into 'geoblocks', which are al subclasses of `Block`. The
  wrappers were restructured into submodules core, raster, geometry, and interfaces.

- The new geoblocks.Block baseclass now provides the infrastructure for
  a) describing a relational block graph and b) generating compute graphs from a
  request for usage in parallelized computations.

- Each element in a relational block graph or compute graph is hashed using the
  `tokenize` module from `dask` which is able to generate unique and deterministic
  tokens (hashes).

- Blocks are saved to a new json format (version 2).

- Every block supports the attributes `period`, `timedelta`, `extent`,
  `dtype`, `fillvalue`, `geometry`, and `geo_transform`.

- The `check` method is implemented on every block and refreshes the
  primitives (`stores.Store` / `results.Grid`).

- `geoblocks.raster.sources.RasterStoreSource` should now be wrapped around a
  `raster_store.stores.Store` in order to include it as a datasource inside a graph.

- Reformatted the code using black code formatter.

- Implemented `GroupTemporal` as replacement for multi-store Lizard objects.

- Adapted `GeoInterface` to mimic now deprecated lizard_nxt.raster.Raster.

- Fixed issue with ciso8601 2.*

- Bumped raster-store dependency to 4.0.0.
