from setuptools import setup

version = '2.0.1.dev0'

long_description = '\n\n'.join(
    [open('README.rst').read(), open('CHANGES.rst').read()]
)

install_requires = (
    [
        'cityhash',  # optional, but speeds up hashing a lot
        'dask[delayed]',
        'geopandas',
        'pygdal',
        'pytz',
        'numpy>=1.11',
        'scipy',
    ],
)

tests_require = ['nose', 'coverage', 'mock']

setup(
    name='dask-geomodeling',
    version=version,
    description="On-the-fly operations on geographical maps.",
    long_description=long_description,
    # Get strings from http://www.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[],
    keywords=[],
    author='Casper van der Wel',
    author_email='casper.vanderwel@nelen-schuurmans.nl',
    url='',
    license='closed source',
    packages=[
        'dask_geomodeling',
        'dask_geomodeling.core',
        'dask_geomodeling.geometry',
        'dask_geomodeling.raster'
    ],
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={'test': tests_require},
    entry_points={'console_scripts': []},
)
