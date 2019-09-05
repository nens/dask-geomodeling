Raster Blocks
=============

Raster-type blocks contain rasters with a time axis. Internally, the raster
data is stored as `NumPy <https://numpy.org/>`_ arrays.

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
   :exclude-members: get_sources_and_requests, process


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
