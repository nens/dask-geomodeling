Installation
============

Requirements
------------

- python >= 3.5
- GDAL (with BigTIFF support)
- numpy
- scipy
- dask[delayed]
- pandas
- geopandas
- ipyleaflet, matplotlib, pillow (for the ipyleaflet plugin)

Windows
-------

Installation on windows is tricky due to incompatibilities between dependencies
of dask-geomodeling (especially python, GDAL, and scipy). We use the following
steps to consistently create a conda environment in which you can work with
dask-geomodeling:

1. `Install anaconda / miniconda <https://docs.anaconda.com/anaconda/install/>`_
2. Start the `Anaconda Prompt` via the start menu
3. `conda config --add channels conda-forge`
4. `conda config --set channel_priority strict`
5. `conda update conda`
6. `conda create --name geomodeling python=3.8 gdal=3.0.2 scipy=1.3.2 pandas=0.25.3 dask-geomodeling jupyter ipyleaflet matplotlib pillow

.. note::

   Installation of dask-geomodeling on windows has a number of pitfalls related
   to anaconda and GDAL. The above recipe is given to get you started with
   dask-geomodeling quickly. If you need other python or GDAL versions: while
   dask-geomodeling itself is compatible with most current versions, you may
   may have a hard time getting it to work. If you're reading this, good luck
   out there.


Using the ipyleaflet plugin
---------------------------

dask-geomodeling comes with a ipyleaflet plugin for `Jupyter<https://jupyter.org/>`_
so that you can show your generated views on a mapviewer. If you want to use
it, install some additional dependencies::

    $ conda install jupyter ipyleaflet matplotlib pillow

And start your notebook server with the plugin::

    $ jupyter notebook --NotebookApp.nbserver_extensions="{'dask_geomodeling.ipyleaflet_plugin':True}"

Alternatively, you can add this extension to your
`Jupyter configuration<https://jupyter-notebook.readthedocs.io/en/stable/config_overview.html>`_


Advanced: local setup with system Python (Ubuntu)
-------------------------------------------------

These instructions make use of the system-wide Python 3 interpreter.

    $ sudo apt install python3-pip python3-gdal

Install dask-geomodeling::

    $ pip install --user dask-geomodeling[test]

Run the tests::

    $ pytest


Advanced: local setup for development (Ubuntu)
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
