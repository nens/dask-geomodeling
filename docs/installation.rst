Installation
============

Requirements
------------

- python >= 3.5
- GDAL 2.* (with BigTIFF support)
- numpy
- scipy
- dask[delayed]
- pandas
- geopandas
- ipyleaflet, matplotlib, pillow (for the ipyleaflet plugin)

Anaconda (all platforms)
------------------------

1. `Install anaconda / miniconda <https://docs.anaconda.com/anaconda/install/>`_
2. Start the `Anaconda Prompt` via the start menu
3. ``conda config --add channels conda-forge``
4. ``conda update conda``
5. ``conda install python=3.6 gdal=2.4.1 scipy=1.3.1 dask-geomodeling ipyleaflet matplotlib pillow``

.. note::

   The version pins of python, gdal and scipy are related to issues specific
   to Windows. On other platforms you may leave them out. 
   If you need other python or GDAL versions
   on windows: while `dask-geomodeling` itself is compatible with all current
   versions, you may may have a hard time getting it to work via Anaconda and
   it will probably be easier using the pip route listed below.


Windows (pip)
-------------

The following recipe is still a work in progress:

1. `Install Python 3.* (stable) <https://www.python.org/downloads/windows/>`_
2. `Install GDAL 2.* (MSVC 2015) <http://www.gisinternals.com/release.php>`_
3. Add the GDAL installation path to your PATH variable
4. Start the command prompt
5. ``pip install gdal==2.* dask-geomodeling ipyleaflet matplotlib pillow``
6. (optionally) ``pip install ipyleaflet matplotlib pillow``

.. note::

   You might need to setup your C++ compiler according to
   `this <https://wiki.python.org/moin/WindowsCompilers>`_

On the ipyleaflet plugin
------------------------

dask-geomodeling comes with a ipyleaflet plugin for `Jupyter <https://jupyter.org/>`_
so that you can show your generated views on a mapviewer. If you want to use
it, install some additional dependencies::

    $ conda [or pip] install jupyter ipyleaflet matplotlib pillow

And start your notebook server with the plugin::

    $ jupyter notebook --NotebookApp.nbserver_extensions="{'dask_geomodeling.ipyleaflet_plugin':True}"

Alternatively, you can add this extension to your
`Jupyter configuration <https://jupyter-notebook.readthedocs.io/en/stable/config_overview.html>`_


Advanced: local setup with system Python (Ubuntu)
-------------------------------------------------

These instructions make use of the system-wide Python 3 interpreter::

    $ sudo apt install python3-pip python3-gdal

Install dask-geomodeling::

    $ pip install --user dask-geomodeling[test,cityhash]

Run the tests::

    $ pytest


Advanced: local setup for development (Ubuntu)
----------------------------------------------

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

    (.venv) $ pip install -e .[test,cityhash]

Run the tests::

    (.venv) $ pytest
