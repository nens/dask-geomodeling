dask-geomodeling
==========================================

Dask-geomodeling is a collection of classes that are to be stacked together to
create configurations for on-the-fly operations on geographical maps. By
generating `Dask <https://dask.pydata.org/>`_ compute graphs, these operation
may be parallelized and (intermediate) results may be cached.

Multiple Block instances together make a view. Each Block has the ``get_data``
method that fetches the data in one go, as well as a ``get_compute_graph``
method that creates a graph to compute the data later.

Constructing a view
-------------------

A dask-geomodeling view can be constructed by creating a Block instance:

.. code:: python

   from dask_geomodeling.raster import RasterFileSource
   source = RasterFileSource('/path/to/geotiff')


The view can now be used to obtain data from the specified file. More
complex views can be created by nesting block instances:

.. code:: python

   from dask_geomodeling.raster import Add, Multiply
   add = Add(source, 2.4)
   mult = Multiply(source, view_add)


Obtaining data from a view
--------------------------

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
   compute_graph, compute_token = add.get_compute_graph(**request)
   data = dask.get(compute_graph, compute_token)

Here, we first generate a compute graph using dask-geomodeling, then evaluate
the graph using dask. The power of this two-step procedure is twofold:

1. Dask supports multi-threading, multi-processing, and cluster processing.
2. The compute_token is a unique identifier of this computation: this can
   easily be used in caching methods.


The Block class
----------------

To write a new geoblock class, we need to write the following:

1. the ``__init__`` that validates the arguments when constructing the block
2. the ``get_sources_and_requests`` that processes the request
3. the ``process`` that processes the data
4. a number of attribute properties such as ``extent`` and ``period``

About the 2-step data processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``get_sources_and_requests`` method of any block is called recursively from
``get_compute_graph`` and feeds the request from the block to its sources. It
does so by returning a list of (source, request) tuples. During the data evaluation
each of these 2-tuples will be converted to a single data object which is
supplied to the ``process`` function.

An example in words. We ask the ``add`` block from the previous example to do the
following:

- give me a 256x256 raster at location (138000, 480000)

The ``get_sources_and_requests`` would respond with the following:

- I need a256x256 raster at location (138000, 480000) from
  ``RasterFileSource('/path/to/geotiff')``
- I need the number 2.4

The ``get_compute_graph`` method works recursively, so it also calls the
``get_sources_and_requests`` of the ``RasterStoreSource``. The result is a
dask compute graph.

When this compute graph is evaluated, the ``process`` method of the ``add``
geoblock will ultimately receive two arguments:

- the 256x256 raster from ``RasterFileSource('/path/to/geotiff')``
- the number 2.4

And the process method produces the end result.

Implementation example
~~~~~~~~~~~~~~~~~~~~~~

As an example, we use a simplified Dilate geoblock, which adds a buffer of 1
pixel around all pixels of given value:

.. code:: python

    class Dilate(RasterBlock):
        def __init__(self, store, values):
            if not isinstance(store, RasterBlock):
                raise TypeError("'{}' object is not allowed".format(type(store)))
            values = np.asarray(values, dtype=store.dtype)
            super(Dilate, self).__init__(store, values)

        @property
        def store(self):
            return self.args[0]

        @property
        def values(self):
            return self.args[1]

        def get_sources_and_requests(self, **request):
            new_request = expand_request_pixels(request, radius=1)
            return [(self.store, new_request), (self.values, None)]

        @staticmethod
        def process(data, values=None):
            if data is None or values is None or 'values' not in data:
                return data
            original = data['values']
            dilated = original.copy()
            for value in values:
                dilated[ndimage.binary_dilation(original == value)] = value
            dilated = dilated[:, 1:-1, 1:-1]
            return {'values': dilated, 'no_data_value': data['no_data_value']}

        @property
        def extent(self):
            return self.store.extent

        @property
        def period(self):
            return self.store.period


In this example, we see all the essentials of a geoblock implementation.

- The ``__init__`` checks the types of the provided arguments and calls the
  ``super().__init__`` that further initializes the geoblock.

- The ``get_sources_and_requests`` expands the request with 1 pixel, so that
  dilation will have no edge effects. It returns two (source, request) tuples.

- The ``process`` (static)method takes the amount arguments that
  ``get_sources_and_requests`` produces. It does the actual work and returns
  a data response.

- Some attributes like ``extent`` and ``period`` need manual specification, as
  they might change through the geoblock.

- The class derives from ``RasterBlock``, which sets the type of geoblock, and
  through that its request/response schema and its required attributes.


Block types
-----------

A block type sets three things:

1. the response schema: e.g. "RasterBlock.process returns a dictionary with
   a numpy array and a no data value"

2. the request schema: e.g. "RasterBlock.get_sources_and_requests expects a
   dictionary with the fields 'mode', 'bbox', 'projection', 'height', 'width'"

3. the attributes to be implemented on each geoblock

This is not enforced at the code level, it is up to the developer to stick to
this specification. The specification is written down in the type baseclass
"RasterBlock", "GeometryBlock", etc.

Local setup (for development)
-----------------------------

These instructions assume that ``git``, ``python3``, ``pip``, and
``virtualenv`` are installed on your host machine.

First make sure you have the GDAL libraries installed. On Ubuntu::

    $ sudo apt install libgdal-dev

Take note the GDAL version::

    $ apt show libgdal-dev

Create and activate a virtualenv::

    $ virtualenv --python=python3 .venv
    $ source .venv/bin/activate

Install PyGDAL with the correct version (example assumes GDAL 2.2.3)::

    $ pip install pygdal==2.2.3.*

Install dask-geomodeling::

    $ pip install -e .[test]

Run the tests::

    $ pytest

Or optionally, with coverage and code style checking::

    $ pytest --cov=dask_geomodeling --black
