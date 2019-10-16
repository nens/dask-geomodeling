Views
=====

A View is a combination of one or more Blocks. For instance:

.. code-block:: python

    from dask_geomodeling.raster import RasterFileSource, Group
    source_1 = RasterFileSource("path/to/some/tiff")
    source_2 = RasterFileSource("path/to/another/tiff")
    view = Group(source_1, source_1)

View serialization
------------------

A View consists of several Block instances that reference each other. To
serialize this we use Dask's graph_ format. This graph format replaces the
nested structure by a flat dictionary with internal references

.. _graph: http://docs.dask.org/en/latest/custom-graphs.html

.. warning::

    A serialized view looks much alike a compute graph. Don't get confused!

An example using the above view definition:

.. code-block:: python

    serialized_view = view.serialize()

.. code-block:: json

    {
      "version": 2,
      "graph": {
        "RasterFileSource_300b56b278d49bea13eff68f8cf52f90": [
          "dask_geomodeling.raster.sources.RasterFileSource",
          "file:///path/to/some/tiff"
        ],
        "RasterFileSource_9dc43c9bd98e2f9069e2b0c879d76cb1": [
          "dask_geomodeling.raster.sources.RasterFileSource",
          "file:///path/to/another/tiff"
        ],
        "Group_0d0a99fe65bffe87dd045627c27bcbbb": [
          "dask_geomodeling.raster.combine.Group",
          "RasterFileSource_300b56b278d49bea13eff68f8cf52f90",
          "RasterFileSource_9dc43c9bd98e2f9069e2b0c879d76cb1"
        ]
      },
      "name": "Group_0d0a99fe65bffe87dd045627c27bcbbb"
    }


The above "view graph" contains all three operations that
we defined together with their arguments. The names are automatically generated
and contain a hash which is useful to uniquely determine the block. Also, we
see the "name", that points to the endpoint block.

To deserialize the view:

.. code-block:: python

    from dask_geomodeling.core import Block
    view = Block.deserialize(serialized_view)

The methods ``Block.to_json`` and ``Block.from_json``, or ``Block.get_graph``
and ``dask_geomodeling.construct`` serve the same purpose, but they in/output
different object types.
