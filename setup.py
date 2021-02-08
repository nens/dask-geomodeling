from setuptools import setup
import os

version = '2.3.5.dev0'

long_description = "\n\n".join([open("README.rst").read(), open("CHANGES.rst").read()])

install_requires = (
    [
        "dask[delayed]>=0.18",
        "pandas>=0.19",
        "geopandas>=0.4",
        "pytz",
        "numpy>=1.12",
        "scipy>=0.19",
    ],
)

# emulate "--no-deps" on the readthedocs build (there is no way to specify this
# behaviour in the .readthedocs.yml)
if os.environ.get('READTHEDOCS') == 'True':
    install_requires = []


tests_require = ["pytest"]

setup(
    name="dask-geomodeling",
    version=version,
    description="On-the-fly operations on geographical maps.",
    long_description=long_description,
    # Get strings from http://www.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    keywords=["dask"],
    author="Casper van der Wel",
    author_email="casper.vanderwel@nelen-schuurmans.nl",
    url="https://github.com/nens/dask-geomodeling",
    license="BSD 3-Clause License",
    packages=[
        "dask_geomodeling",
        "dask_geomodeling.core",
        "dask_geomodeling.geometry",
        "dask_geomodeling.raster",
    ],
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    tests_require=tests_require,
    python_requires='>=3.5',
    extras_require={"test": tests_require, "cityhash": ["cityhash"]},
    entry_points={"console_scripts": []},
)
