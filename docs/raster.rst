Raster Blocks
=============

RasterBlocks are the main component of raster operations. Most raster
operations take one or more RasterBlocks as input and produce a single
RasterBlock as output.

Raster-type blocks contain rasters with data in three dimensions. Besides the
x- and y-axes they also have a temporal axis.

Internally, dask-geomodeling stores the raster data as
`NumPy <https://numpy.org/>`_ arrays.

API Specification
-----------------

.. automodule:: dask_geomodeling.raster.base
   :members: RasterBlock


:mod:`dask_geomodeling.raster.combine`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: dask_geomodeling.raster.combine
   :members:
   :exclude-members: get_sources_and_requests, process, get_stores


:mod:`dask_geomodeling.raster.elemwise`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: dask_geomodeling.raster.elemwise
   :members:
   :exclude-members: get_sources_and_requests, process


:mod:`dask_geomodeling.raster.misc`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: dask_geomodeling.raster.misc
   :members:
   :exclude-members: get_sources_and_requests, process, extent, geometry


:mod:`dask_geomodeling.raster.sources`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: dask_geomodeling.raster.sources
   :members:
   :exclude-members: get_sources_and_requests, process


:mod:`dask_geomodeling.raster.spatial`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: dask_geomodeling.raster.spatial
   :members:
   :exclude-members: get_sources_and_requests, process


:mod:`dask_geomodeling.raster.temporal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: dask_geomodeling.raster.temporal
   :members:
   :exclude-members: TemporalSum, get_sources_and_requests, process
