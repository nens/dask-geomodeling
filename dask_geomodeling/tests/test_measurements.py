import unittest

from dask_geomodeling import measurements


class TestMeasurements(unittest.TestCase):
    def test_nolabel(self):
        expected = 1.9
        actual = measurements.percentile([0, 1, 2], 95)
        self.assertEqual(expected, actual)

    def test_noindex(self):
        expected = 1.9
        actual = measurements.percentile([0, 1, 2, 3], 95, labels=[1, 1, 1, 0])
        self.assertEqual(expected, actual)

    def test_scalar(self):
        expected = 1.9
        actual = measurements.percentile([0, 1, 2, 3], 95, labels=[1, 1, 1, 0], index=1)
        self.assertEqual(expected, actual)

    def test_remap(self):
        expected = [1.9]
        actual = measurements.percentile(
            [0, 1, 2, 3], 95, labels=[5, 5, 5, 0], index=[5]
        )
        self.assertEqual(expected, actual)

    def test_noremap(self):
        expected = [1.9, 9.5]
        actual = measurements.percentile(
            [0, 1, 2, 3, 0, 2, 4, 6, 8, 10],
            95,
            labels=[1, 1, 1, 0, 2, 2, 2, 2, 2, 2],
            index=[1, 2],
        )
        self.assertEqual(expected, actual)
