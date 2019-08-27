# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os


class Settings(dict):
    """Settings dictionary that first checks environment variables."""

    def __getitem__(self, key):
        result = os.environ.get('GEOBLOCKS_{}'.format(key.upper()))
        if result:
            return result
        else:
            return super(Settings, self).__getitem__(key)


defaults = Settings({
    'FILE_ROOT': '/',
    'RASTER_LIMIT': 12 * (1024 ** 3),  # about 100 MB of float64
    'GEOMETRY_LIMIT': 10000,  # maximum number of geometries in one dataframe
})

settings = Settings(defaults.copy())
