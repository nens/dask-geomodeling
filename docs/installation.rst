Installation
============

Local setup (using conda)
-------------------------

1. `Install anaconda <https://docs.anaconda.com/anaconda/install/>`_
2. Run ``conda install dask-geomodeling -c conda-forge``


Local setup with system Python (Ubuntu)
---------------------------------------

These instructions make use of the system-wide Python 3 interpreter.

    $ sudo apt install python3-pip python3-gdal

Install dask-geomodeling::

    $ pip install --user dask-geomodeling[test]

Run the tests::

    $ pytest


Local setup for development (Ubuntu)
------------------------------------

These instructions assume that ``git``, ``python3``, ``pip``, and
``virtualenv`` are installed on your host machine.

Clone the dask-geomodeling repository::

    $ git clone https://github.com/nens/dask-geomodeling

Make sure you have the GDAL libraries installed. On Ubuntu::

    $ sudo apt install libgdal-dev

Take note of the GDAL version::

    $ apt show libgdal-dev

Create and activate a virtualenv::

    $ cd dask-geomodeling
    $ virtualenv --python=python3 .venv
    $ source .venv/bin/activate

Install PyGDAL with the correct version (example assumes GDAL 2.2.3)::

    (.venv) $ pip install pygdal==2.2.3.*

Install dask-geomodeling::

    (.venv) $ pip install -e .[test]

Run the tests::

    (.venv) $ pytest
