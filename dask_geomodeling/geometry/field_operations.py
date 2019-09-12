"""
Module containing geometry block operations that act on non-geometry fields
"""
import numpy as np
import pandas as pd
import operator
from .base import GeometryBlock, SeriesBlock, BaseSingleSeries

__all__ = [
    "Classify",
    "ClassifyFromColumns",
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "FloorDivide",
    "Power",
    "Modulo",
    "Equal",
    "NotEqual",
    "Greater",
    "GreaterEqual",
    "Less",
    "LessEqual",
    "And",
    "Or",
    "Xor",
    "Invert",
    "Where",
    "Mask",
    "Round",
]


class Classify(BaseSingleSeries):
    """Classify a continuous-valued property into binned categories

    :param source: source data to classify
    :param bins: a 1-dimensional and monotonic list of bins.
      How values outside of the bins are classified, depends on the length of
      the labels. If len(labels) = len(bins) - 1, then values outside of the
      bins are classified to NaN. If len(labels) = len(bins) + 1, then values
      outside of the bins are classified to the first and last elements of the
      labels list.
    :param labels: the labels for the returned bins
    :param right: whether the intervals include the right or the left bin edge

    :type source: SeriesBlock
    :type bins: list
    :type labels: list
    :type right: boolean

    See also:
      https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.cut.html
    """

    def __init__(self, source, bins, labels, right=True):
        if not isinstance(bins, list):
            raise TypeError("'{}' object is not allowed".format(type(bins)))
        if not isinstance(labels, list):
            raise TypeError("'{}' object is not allowed".format(type(labels)))
        if not isinstance(right, bool):
            raise TypeError("'{}' object is not allowed".format(type(right)))
        bins_arr = np.asarray(bins)
        if bins_arr.ndim != 1:
            raise TypeError("'bins' must be one-dimensional")
        if (np.diff(bins) < 0).any():
            raise ValueError("'bins' must increase monotonically.")
        if len(labels) not in (len(bins) - 1, len(bins) + 1):
            raise ValueError(
                "Expected {} or {} labels, got {}".format(
                    len(bins) - 1, len(bins) + 1, len(labels)
                )
            )
        super().__init__(source, bins, labels, right)

    @property
    def bins(self):
        return self.args[1]

    @property
    def labels(self):
        return self.args[2]

    @property
    def right(self):
        return self.args[3]

    @staticmethod
    def process(series, bins, labels, right):
        open_bounds = len(labels) == len(bins) + 1
        if open_bounds:
            bins = np.concatenate([[-np.inf], bins, [np.inf]])
        result = pd.cut(series, bins, right, labels)
        labels_dtype = pd.Series(labels).dtype
        if labels_dtype.name != "object":
            result = pd.Series(result, dtype=labels_dtype)
        if open_bounds:
            # patch the result, we actually want to classify np.inf
            if right:
                result[series == -np.inf] = labels[0]
            else:
                result[series == np.inf] = labels[-1]
        return result


class ClassifyFromColumns(SeriesBlock):
    """Classify a continuous-valued property based on bins located in different
    columns.

    :param source: geometry source to classify
    :param value_column: the column name that contains values to classify
    :param bin_columns: column names in which the bins are stored.
      The bins values need to be sorted in increasing order.
    :param labels: specifies the labels for the returned bins
    :param right: whether the intervals include the right or the left bin edge
      Default True.

    :type source: GeometryBlock
    :type value_column: string
    :type bin_columns: list
    :type labels: list
    :type right: boolean

    See also:
      :class:`dask_geomodeling.geometry.field_operations.Classify`
    """

    def __init__(self, source, value_column, bin_columns, labels, right=True):
        if not isinstance(source, GeometryBlock):
            raise TypeError("'{}' object is not allowed".format(type(source)))
        if not isinstance(value_column, str):
            raise TypeError("'{}' object is not allowed".format(type(value_column)))
        if not isinstance(bin_columns, list):
            raise TypeError("'{}' object is not allowed".format(type(bin_columns)))
        if not isinstance(labels, list):
            raise TypeError("'{}' object is not allowed".format(type(labels)))
        if not isinstance(right, bool):
            raise TypeError("'{}' object is not allowed".format(type(right)))
        missing_columns = (set(bin_columns) | {value_column}) - source.columns
        if missing_columns:
            raise KeyError("Columns '{}' are not present".format(missing_columns))
        if len(labels) not in (len(bin_columns) - 1, len(bin_columns) + 1):
            raise ValueError(
                "Expected {} or {} labels, got {}".format(
                    len(bin_columns) - 1, len(bin_columns) + 1, len(labels)
                )
            )
        super().__init__(source, value_column, bin_columns, labels, right)

    @property
    def source(self):
        return self.args[0]

    @property
    def value_column(self):
        return self.args[1]

    @property
    def bin_columns(self):
        return self.args[2]

    @property
    def labels(self):
        return self.args[3]

    @property
    def right(self):
        return self.args[4]

    @staticmethod
    def process(data, value_column, bin_columns, labels, right):
        if "features" not in data or len(data["features"]) == 0:
            return pd.Series([])
        features = data["features"]
        values = features[value_column].values
        bins = features[bin_columns].values
        n_bins = len(bin_columns)

        # Check in which bin every value is. because bins may be different for
        # each value, searchsorted is not an option. We assume that bins are
        # sorted in increasing order. Checking that would be costly.
        if right:
            indices = np.sum(values[:, np.newaxis] > bins, axis=1)
        else:
            indices = np.sum(values[:, np.newaxis] >= bins, axis=1)

        # If we have e.g. 2 labels and 3 bins, the outside intervals are closed
        # any index that is 0 or 3 should become -1 (unclassified).
        if len(labels) == n_bins + 1:  # open bounds
            indices[np.isnan(values)] = -1  # else NaN gets classified
        else:  # closed bounds
            indices[indices == n_bins] = 0
            indices -= 1

        # The output of pd.cut is a categorical Series.
        labeled_data = pd.Categorical.from_codes(indices, labels, ordered=True)
        return pd.Series(labeled_data, index=features.index)


class BaseFieldOperation(BaseSingleSeries):
    """Base block for basic operations between series from a GeometryBlock"""

    def __init__(self, source, other):
        if not isinstance(other, (SeriesBlock, int, float, bool)):
            raise TypeError("'{}' object is not allowed".format(type(other)))
        super().__init__(source, other)

    @property
    def other(self):
        return self.args[1]

    @staticmethod
    def process(source, other):
        raise NotImplementedError()  # implement this in subclasses


class Add(BaseFieldOperation):
    """
    Addition of series and other, element-wise.

    :type source: SeriesBlock
    :type other: SeriesBlock, float

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.add.html
    """

    process = staticmethod(operator.add)


class Subtract(BaseFieldOperation):
    """
    Subtraction of series and other, element-wise.

    :type source: SeriesBlock
    :type other: SeriesBlock, float

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.subtract.html
    """

    process = staticmethod(operator.sub)


class Multiply(BaseFieldOperation):
    """
    Multiplication of series and other, element-wise.

    :type source: SeriesBlock
    :type other: SeriesBlock, float

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.multiply.html
    """

    process = staticmethod(operator.mul)


class Divide(BaseFieldOperation):
    """
    Floating division of series and other, element-wise.

    :type source: SeriesBlock
    :type other: SeriesBlock, float

    Putting source in the divisor is not possible: please use the Power
    for that instead.

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.divide.html
    """

    process = staticmethod(operator.truediv)


class FloorDivide(BaseFieldOperation):
    """
    Integer (floor) division of series and other, element-wise.

    :type source: SeriesBlock
    :type other: SeriesBlock, float

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.floordiv.html
    """

    process = staticmethod(operator.floordiv)


class Power(BaseFieldOperation):
    """
    Power (exponent) of series and other, element-wise.

    :type source: SeriesBlock
    :type other: SeriesBlock, float

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.pow.html
    """

    def __init__(self, source, other):
        # the float(other) will raise a TypeError if necessary
        super().__init__(source, float(other))

    process = staticmethod(operator.pow)


class Modulo(BaseFieldOperation):
    """
    Modulo of series and other, element-wise.

    :type source: SeriesBlock
    :type other: SeriesBlock, float

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.mod.html
    """

    process = staticmethod(operator.mod)


class Equal(BaseFieldOperation):
    """
    Equal to of series and other, element-wise.

    :type source: SeriesBlock
    :type other: SeriesBlock, float

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.eq.html
    """

    process = staticmethod(operator.eq)


class NotEqual(BaseFieldOperation):
    """
    Not equal to of series and other, element-wise.

    :type source: SeriesBlock
    :type other: SeriesBlock, float

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ne.html
    """

    process = staticmethod(operator.ne)


class Greater(BaseFieldOperation):
    """
    Greater than of series and other, element-wise.

    :type source: SeriesBlock
    :type other: SeriesBlock, float

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.gt.html
    """

    process = staticmethod(operator.gt)


class GreaterEqual(BaseFieldOperation):
    """
    Greater than or equal to of series and other, element-wise.

    :type source: SeriesBlock
    :type other: SeriesBlock, float

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ge.html
    """

    process = staticmethod(operator.ge)


class Less(BaseFieldOperation):
    """
    Less than of series and other, element-wise.

    :type source: SeriesBlock
    :type other: SeriesBlock, float

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.lt.html
    """

    process = staticmethod(operator.lt)


class LessEqual(BaseFieldOperation):
    """
    Less than or equal to of series and other, element-wise.

    :type source: SeriesBlock
    :type other: SeriesBlock, float

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.le.html
    """

    process = staticmethod(operator.le)


class BaseLogicOperation(BaseFieldOperation):
    """Base block for logic operations between columns in a GeometryBlock"""

    def __init__(self, source, other):
        if not isinstance(other, SeriesBlock):
            raise TypeError("'{}' object is not allowed".format(type(other)))
        super().__init__(source, other)


class And(BaseLogicOperation):
    """
    Logical AND between series and other.

    :type source: SeriesBlock
    :type other: SeriesBlock, float
    """

    process = staticmethod(operator.and_)


class Or(BaseLogicOperation):
    """
    Logical OR between series and other.

    :type source: SeriesBlock
    :type other: SeriesBlock, float
    """

    process = staticmethod(operator.or_)


class Xor(BaseLogicOperation):
    """
    Logical XOR between series and other.

    :type source: SeriesBlock
    :type other: SeriesBlock, float
    """

    process = staticmethod(operator.xor)


class Invert(BaseSingleSeries):
    """
    Logical NOT operation on a series.

    :type source: SeriesBlock
    """

    process = staticmethod(operator.inv)


class Where(BaseSingleSeries):
    """Replace values where the condition is False.

    :param source: source data
    :param cond: condition that determines whether to keep values from source
    :param other: entries where cond is False are replaced with the
      corresponding value from other.

    :type source: SeriesBlock
    :type cond: SeriesBlock
    :type other: SeriesBlock, scalar

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.where.html
    """

    def __init__(self, source, cond, other):
        if not isinstance(cond, SeriesBlock):
            raise TypeError("'{}' object is not allowed".format(type(cond)))
        super().__init__(source, cond, other)

    @property
    def cond(self):
        return self.args[1]

    @property
    def other(self):
        return self.args[2]

    @staticmethod
    def process(source, cond, other):
        return source.where(cond, other)


class Mask(BaseSingleSeries):
    """Replace values where the condition is True.

    :param source: source data
    :param cond: condition that determines whether to mask values from source
    :param other: entries where cond is True are replaced with the
      corresponding value from other.

    :type source: SeriesBlock
    :type cond: SeriesBlock
    :type other: SeriesBlock, scalar

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.mask.html
    """

    def __init__(self, source, cond, other):
        if not isinstance(cond, SeriesBlock):
            raise TypeError("'{}' object is not allowed".format(type(cond)))
        super().__init__(source, cond, other)

    @property
    def cond(self):
        return self.args[1]

    @property
    def other(self):
        return self.args[2]

    @staticmethod
    def process(source, cond, other):
        return source.mask(cond, other)


class Round(BaseSingleSeries):
    """Round each value in a SeriesBlock to the given number of decimals

    :param source: source data
    :param decimals: number of decimal places to round to (default: 0).
      If decimals is negative, it specifies the number of positions to the left
      of the decimal point.

    :type source: SeriesBlock
    :type decimals: int

    See also:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.round.html
    """

    def __init__(self, source, decimals=0):
        if not isinstance(decimals, int):
            raise TypeError("'{}' object is not allowed".format(type(decimals)))
        super().__init__(source, decimals)

    process = staticmethod(np.around)


class Interp(BaseSingleSeries):
    """One-dimensional linear interpolation.

    Compute the one-dimensional piecewise linear interpolant to a function with
    given discrete data points (xp, fp).

    :param source: source data (x-coordinates) to interpolate
    :param xp: the x-coordinates of the data points, must be increasing.
    :param fp: the y-coordinates of the data points, same length as ``xp``.
    :param left: value to return for ``x < xp[0]``, default is ``fp[0]``.
    :param right: value to return for ``x > xp[-1]``, default is ``fp[-1]``.

    :type source: SeriesBlock
    :type xp: list
    :type fp: list
    :type left: float
    :type right: float

    See also:
      https://docs.scipy.org/doc/numpy/reference/generated/numpy.interp.html
    """

    def __init__(self, source, xp, fp, left=None, right=None):
        xp = [float(x) for x in xp]
        fp = [float(x) for x in fp]
        if left is not None:
            left = float(left)
        if right is not None:
            right = float(right)
        if np.any(np.diff(xp) < 0):
            raise ValueError("xp must be monotonically increasing")
        super().__init__(source, xp, fp, left, right)

    @staticmethod
    def process(data, xp, fp, left, right):
        result = np.interp(data, xp, fp, left, right)
        return pd.Series(result, index=data.index)


class Choose(BaseSingleSeries):
    """Construct a SeriesBlock from an index series and a multiple series to
    choose from.

    :param source: series having integers from 0 to n - 1 with n being the
      number of choices. Values outside this range will result in NaN.
    :param choices: multiple series containing the values to choose

    :type source: SeriesBlock
    :type choices: list of SeriesBlock

    See also:
      https://docs.scipy.org/doc/numpy/reference/generated/numpy.choose.html
    """

    def __init__(self, source, *choices):
        if not len(choices) >= 2:
            raise ValueError("The number of choices must be greater than one.")
        if not all([isinstance(choice, SeriesBlock) for choice in choices]):
            raise TypeError("All choices must be SeriesBlock objects")
        super().__init__(source, *choices)

    @property
    def choices(self):
        return self.args[1:]

    @staticmethod
    def process(source, *choices):
        result = pd.Series(np.nan, index=source.index)
        for i, choice in enumerate(choices):
            mask = source == i
            result[mask] = choice[source.index[mask]]
        return result
