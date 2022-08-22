dask-geomodeling
==========================================

.. image:: https://readthedocs.org/projects/dask-geomodeling/badge/?version=latest
     :target: https://dask-geomodeling.readthedocs.io/en/latest/?badge=latest

.. image:: https://github.com/nens/dask-geomodeling/actions/workflows/test.yml/badge.svg
    :target: https://github.com/nens/dask-geomodeling/actions/workflows/test.yml

.. image:: https://badge.fury.io/py/dask-geomodeling.svg
    :target: https://badge.fury.io/py/dask-geomodeling

.. image:: https://anaconda.org/conda-forge/dask-geomodeling/badges/version.svg
    :target: https://anaconda.org/conda-forge/dask-geomodeling

Dask-geomodeling is a collection of classes that are to be stacked together to
create configurations for on-the-fly operations on geographical maps. By
generating `Dask <https://dask.pydata.org/>`_ compute graphs, these operation
may be parallelized and (intermediate) results may be cached.

Multiple Block instances together make a view. Each Block has the ``get_data``
method that fetches the data in one go, as well as a ``get_compute_graph``
method that creates a graph to compute the data later.

`Read the docs <https://dask-geomodeling.readthedocs.org/>`_ for further information.
