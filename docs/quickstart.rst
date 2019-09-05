Quickstart
==========

Constructing a view
-------------------

A dask-geomodeling view can be constructed by creating a Block instance:

.. code:: python

   from dask_geomodeling.raster import RasterFileSource
   source = RasterFileSource('/path/to/geotiff')


The view can now be used to obtain data from the specified file. More
complex views can be created by nesting block instances:

.. code:: python

   from dask_geomodeling.raster import Smooth
   smoothed = Smooth(source, 5)
   smoothed_plus_two = smoothed + 2


Obtaining data from a view
--------------------------

To obtain data from a view directly, use the `get_data` method:

.. code:: python

   request = {
       "mode": "vals",
       "bbox": (138000, 480000, 139000, 481000),
       "projection": "epsg:28992",
       "width": 256,
       "height": 256
   }
   data = add.get_data(**request)


Which field to include in the request and what data to expect depends on the
type of the block used. In this example, we used a RasterBlock. The request
and response specifications are listed in the documentation of the specific
block type.

Delayed evaluation
------------------

Dask-geomodeling revolves around *lazy data evaluation*. Each Block first
evaluates what needs to be done for certain request, storing that in a
*compute graph*. This graph can then be evaluated to obtain the data. The data
is evaluated with dask, and the specification of the compute graph also comes
from dask. For more information about how a graph works, consult the dask
docs_:

.. _docs: http://docs.dask.org/en/latest/custom-graphs.html

We use the previous example to demonstrate how this works:

.. code:: python

   import dask
   request = {
       "mode": "vals",
       "bbox": (138000, 480000, 139000, 481000),
       "projection": "epsg:28992",
       "width": 256,
       "height": 256
   }
   graph, name = add.get_compute_graph(**request)
   data = dask.get(graph, [name])

Here, we first generate a compute graph using dask-geomodeling, then evaluate
the graph using dask. The power of this two-step procedure is twofold:

1. Dask supports threaded, multiprocessing, and distributed schedulers. Consult
   the dask documentation_ to try these out.
2. The `name` is a unique identifier of this computation: this can
   easily be used in caching methods.

.. _docs: https://docs.dask.org/en/latest/scheduling.html
