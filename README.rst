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

## Development on this project itself

The standard command to setup your project using [`uv`](https://docs.astral.sh/uv/) is:

    uv sync --dev       # 'uv.lock' is git-ignored, because this is a library

This project uses GDAL, which requires additional system packages to be available on your
system. On Ubuntu, you can install them with:

    sudo apt-get install gdal-bin libgdal-dev

Then, installation of the GDAL python module is done as follows:

    source .venv/bin/activate
    pip install GDAL[numpy]==$(gdal-config --version) --no-build-isolation

Then run the tests:

    uv run pytest

The tests are also run automatically [on "github actions"](https://github.com/nens/dask-geomodeling/actions) for "master" and for pull requests. So don't just make a branch, but turn it into a pull request right away. On your pull request page, you also automatically get the feedback from the automated tests.


## Release

Make sure you have [zest.releaser](https://zestreleaser.readthedocs.io/en/latest/) installed.

    uv run fullrelease

When you created a tag, it will be uploaded automatically [to pypi](https://pypi.org/project/rana-process-sdk/) by a Github Action.

This project is also available on [conda-forge](https://anaconda.org/conda-forge/dask-geomodeling).
After a release on PyPI is made, a Pull Request is automatically created on https://github.com/conda-forge/dask-geomodeling-feedstock. Merge this
to publish the new version on conda-forge.
