Blocks
======

The Block class
----------------

To write a new ``Block`` subclass, we need to write the following:

1. the ``__init__`` that validates the arguments when constructing the block
2. the ``get_sources_and_requests`` that processes the request
3. the ``process`` that processes the data
4. a number of attributes such as ``extent`` and ``period``

About the 2-step data processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``get_sources_and_requests`` method of any block is called recursively from
``get_compute_graph`` and feeds the request from the block to its sources. It
does so by returning a list of (source, request) tuples. During the data
evaluation each of these 2-tuples will be converted to a single data object
which is supplied to the ``process`` function.

First, an example in words. We construct a View
``add = RasterFileSource('path/to/geotiff') + 2.4`` and ask it the following:

- give me a 256x256 raster at location (138000, 480000)

We do that by calling ``get_data``, which calls ``get_compute_graph``, which
calls ``get_sources_and_requests`` on each block instance recursively.

First ``add.get_sources_and_requests`` would respond with the following:

- I will need a 256x256 raster at location (138000, 480000) from
  ``RasterFileSource('/path/to/geotiff')``
- I will need 2.4

Then, on recursion, the ``RasterFileSource.get_sources_and_requests`` would
respond:

- I will give you the 256x256 raster at location (138000, 480000)

These small subtasks get summarized in a compute graph, which is returned by
``get_compute_graph``. Then ``get_data`` feeds that compute graph to dask.

Dask will evaluate this graph by calling the ``process`` methods on each block:

1. A raster is loaded using ``RasterFileSource.process``
2. This, together with the number 2.4, is given to ``Add.process``
3. The resulting raster is presented to the user.


Implementation example
~~~~~~~~~~~~~~~~~~~~~~

As an example, we use a simplified Dilate block, which adds a buffer of 1
pixel around all pixels of given value:

.. code:: python

    class Dilate(RasterBlock):
        def __init__(self, source, value):
            assert isinstance(source, RasterBlock):
            value = float(value)
            super().__init__(source, value)

        @property
        def source(self):
            return self.args[0]

        @property
        def value(self):
            return self.args[1]

        def get_sources_and_requests(self, **request):
            new_request = expand_request_pixels(request, radius=1)
            return [(self.store, new_request), (self.value, None)]

        @staticmethod
        def process(data, values=None):
            # handle empty data cases
            if data is None or values is None or 'values' not in data:
                return data
            # perform the dilation
            original = data['values']
            dilated = original.copy()
            dilated[ndimage.binary_dilation(original == value)] = value
            dilated = dilated[:, 1:-1, 1:-1]
            return {'values': dilated, 'no_data_value': data['no_data_value']}

        @property
        def extent(self):
            return self.source.extent

        @property
        def period(self):
            return self.source.period


In this example, we see all the essentials of a Block implementation.

- The ``__init__`` checks the types of the provided arguments and calls the
  ``super().__init__`` that further initializes the block.

- The ``get_sources_and_requests`` expands the request with 1 pixel, so that
  dilation will have no edge effects. It returns two (source, request) tuples.

- The ``process`` (static)method takes the amount arguments equal to the length
  of the list that ``get_sources_and_requests`` produces. It does the actual
  work and returns a data response.

- Some attributes like ``extent`` and ``period`` need manual specification, as
  they might change through the block.

- The class derives from ``RasterBlock``, which sets the type of block, and
  through that its request/response schema and its required attributes.


Block types specification
-------------------------

A block type sets three things:

1. the response schema: e.g. "RasterBlock.process returns a dictionary with
   a numpy array and a no data value"

2. the request schema: e.g. "RasterBlock.get_sources_and_requests expects a
   dictionary with the fields 'mode', 'bbox', 'projection', 'height', 'width'"

3. the attributes to be implemented on each block

This is not enforced at the code level, it is up to the developer to stick to
this specification. The specification is written down in the type baseclass
:meth:`~dask_geomodeling.raster.base.RasterBlock` or
:meth:`~dask_geomodeling.geometry.base.GeometryBlock`.

API specification
-----------------

.. automodule:: dask_geomodeling.core.graphs
   :members: Block, construct, compute
