.. dask-geomodeling documentation master file, created by
   sphinx-quickstart on Thu Sep  5 10:36:42 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dask-geomodeling's documentation!
============================================

Dask-geomodeling is a collection of classes that are to be stacked together to
create configurations for on-the-fly operations on geographical maps. By
generating `Dask <https://dask.pydata.org/>`_ compute graphs, these operation
may be parallelized and (intermediate) results may be cached.

Multiple Block instances together make a view. Each Block has the ``get_data``
method that fetches the data in one go, as well as a ``get_compute_graph``
method that creates a graph to compute the data later.

Blocks are used for the on-the-fly modification of raster- and vectordata,
respectively through the baseclasses :meth:`~dask_geomodeling.raster.base.RasterBlock` and
:meth:`~dask_geomodeling.geometry.base.GeometryBlock`. Derived classes support
operations such has grouping
basic math, shifting time, smoothing, reclassification, geometry operations,
zonal statistics, and property field operations.

About
-----

This package was developed by Nelen & Schuurmans and is used commercially
under the name Geoblocks. Please consult the `Lizard <https://www.lizard.net/>`_
website for more information about this product.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   views
   blocks
   raster
   geometry


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
