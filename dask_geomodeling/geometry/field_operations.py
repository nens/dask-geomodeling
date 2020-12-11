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
    """
    Classify a value column into different bins

    For example: every value below 3 becomes "A", every value between 3 and 5
    becomes "B", and every value above 5 becomes "C".

    The provided SeriesBlock will be classified according to the given
    classification parameters. These parameters consist of two lists, one with
    the edges of the classification bins (i.e. ``[3, 5]``) and one with the
    desired class output (i.e. ``["A", "B", "C"]``). The input data is then
    compared to the classification bins. In this example a value 1 is below 3
    so it gets class ``"A"``. A value 4 is between 3 and 5 so it gets label
    ``"B"``.

    How values outside of the bins are classified depends on the length of the
    labels list. If the length of the labels equals the length of the binedges
    plus 1 (the above example), then values outside of the bins are classified
    to the first and last elements of the labels list. If the length of the
    labels equals the length of the bins minus 1, then values outside of the
    bins are classified to 'no data'.

    Args:
      source (SeriesBlock): The (numeric) data which should be classified.
      bins (list): The edges of the classification intervals
        (i.e. ``[3, 5]``).
      labels (list): The classification returned if a value falls in a specific
        bin (i.e. ``["A", "B", "C"]``). The length of this list is either one
        larger or one less than the length of the ``bins`` argument. Labels
        should be unique. If labels are numeric, they are always
        converted to float to be able to deal with NaN values.
      right (boolean, optional): Determines what side of the intervals are
        closed. Defaults to True (the right side of the bin is closed so a
        value assigned to the bin on the left if it is exactly on a bin edge).

    Returns:
      A SeriesBlock with classified values instead of the original numbers.
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
        if len(set(labels)) != len(labels):
            raise ValueError("Labels should be unique")
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
        if series.dtype == object:
            series = series.fillna(value=np.nan)
        result = pd.cut(series, bins, right, labels)

        # Transform from categorical to whatever suits the "labels". The
        # dtype has to be able to accomodate NaN as well.
        result = result.astype(pd.Series(labels + [np.nan]).dtype)

        if open_bounds:
            # patch the result, we actually want to classify np.inf
            if right:
                result[series == -np.inf] = labels[0]
            else:
                result[series == np.inf] = labels[-1]
        return result


class ClassifyFromColumns(SeriesBlock):
    """
    Classify a continuous-valued geometry property based on bins located in
    other columns.

    See :class:``dask_geomodeling.geometry.field_operations.Classify`` for
    further information.

    Args:
      source (GeometryBlock):The GeometryBlock which contains the column which
        should be clasified as well as columns with the bin edges.
      value_column (str): The column with (float) data which should be
        classified.
      bin_columns (list): A list of columns that contain the bins for the
        classification. The order of the columns should be from low to high
        values.
      labels (list): The classification returned if a value falls in a specific
        bin (i.e. ``["A", "B", "C"]``). The length of this list is either one
        larger or one less than the length of the ``bins`` argument. Labels
        should be unique. If labels are numeric, they are always
        converted to float to be able to deal with NaN values.
      right (boolean, optional): Determines what side of the intervals are
        closed. Defaults to True (the right side of the bin is closed so a
        value assigned to the bin on the left if it is exactly on a bin edge).

    Returns:
      A SeriesBlock with classified values instead of the original floats.

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
        if len(set(labels)) != len(labels):
            raise ValueError("Labels should be unique")
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
            return pd.Series([], dtype=float)
        features = data["features"]
        series = features[value_column]
        if series.dtype == object:
            series = series.fillna(value=np.nan)
        values = series.values
        bins = features[bin_columns].values
        n_bins = len(bin_columns)

        # Check in which bin every value is. because bins may be different for
        # each value, searchsorted is not an option. We assume that bins are
        # sorted in increasing order. Checking that would be costly.
        with np.errstate(invalid="ignore"):  # comparison to NaN is OK here
            if right:
                indices = np.sum(values[:, np.newaxis] > bins, axis=1)
            else:
                indices = np.sum(values[:, np.newaxis] >= bins, axis=1)

        # If values was NaN this is now assigned the value 0 (the first bin).
        # Convert to the last label so that we can map it later to NaN
        if len(labels) == n_bins + 1:
            indices[np.isnan(values)] = len(labels)
        else:
            # If we have e.g. 2 labels and 3 bins, the outside intervals are
            # closed. Therefore, indices 0 and 3 do not map to a bin. Index 0
            # also covers the values = NaN situation.
            indices -= 1  # indices become -1, 0, 1, 2
            indices[indices == -1] = len(labels)  # -1 --> 2

        # Convert indices to labels, append labels with with np.nan to cover
        # unclassified data.
        labeled_data = pd.Series(labels + [np.nan]).loc[indices]
        # Set the index to the features index
        labeled_data.index = features.index
        return labeled_data


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
    Element-wise addition of SeriesBlock or number to another SeriesBlock.

    Args:
      source (SeriesBlock): First addition term
      other (SeriesBlock or number): Second addition term

    Returns:
      SeriesBlock
    """

    process = staticmethod(operator.add)


class Subtract(BaseFieldOperation):
    """
    Element-wise subtraction of SeriesBlock or number with another SeriesBlock.

    Note that if you want to subtract a SeriesBlock from a constant value (like
    ``4 - series``, you have to do ``Add(Multiply(series, -1), 4)``.

    Args:
      source (SeriesBlock): First subtraction term
      other (SeriesBlock or number): Second subtraction term

    Returns:
      SeriesBlock
    """

    process = staticmethod(operator.sub)


class Multiply(BaseFieldOperation):
    """
    Element-wise multiplication of SeriesBlock or number with another
    SeriesBlock.

    Args:
      source (SeriesBlock): First multiplication factor
      other (SeriesBlock or number): Second multiplication factor

    Returns:
      SeriesBlock
    """

    process = staticmethod(operator.mul)


class Divide(BaseFieldOperation):
    """
    Element-wise division of SeriesBlock or number with another SeriesBlock.

    Note that if you want to divide a constant value by a SeriesBlock (like
    ``3 / series``, you have to do ``Multiply(3, Power(series, -1))``.

    Args:
      source (SeriesBlock): Numerator
      other (SeriesBlock or number): Denominator

    Returns:
      SeriesBlock
    """

    process = staticmethod(operator.truediv)


class FloorDivide(BaseFieldOperation):
    """
    Element-wise integer division of SeriesBlock or number with another
    SeriesBlock.

    The outcome of the division is converted to the closest integer below (i.e.
    3.4 becomes 3, 3.9 becomes 3 and -3.4 becomes -4)

    Args:
      source (SeriesBlock): Numerator
      other (SeriesBlock or number): Denominator

    Returns:
      SeriesBlock
    """

    process = staticmethod(operator.floordiv)


class Power(BaseFieldOperation):
    """
    Element-wise raise a SeriesBlock to the power of a number or another
    SeriesBlock.

    For example, the inputs ``[2, 4]`` and ``2`` will give output ``[4, 16]``.

    Args:
      source (SeriesBlock): Base
      other (SeriesBlock or number): Exponent

    Returns:
      SeriesBlock
    """

    def __init__(self, source, other):
        # the float(other) will raise a TypeError if necessary
        super().__init__(source, float(other))

    process = staticmethod(operator.pow)


class Modulo(BaseFieldOperation):
    """
    Element-wise modulo (remainder after division) of SeriesBlock or number
    with another SeriesBlock.

    Example: if the input is ``[31, 5.3, -4]`` and the modulus is ``3``, the
    outcome would be ``[1, 2.3, 2]``. The outcome is always postive and less
    than the modulus.

    Args:
      source (SeriesBlock): Number
      other (SeriesBlock or number): Modulus

    Returns:
      SeriesBlock
    """

    process = staticmethod(operator.mod)


class Equal(BaseFieldOperation):
    """
    Determine whether a SeriesBlock and a second SeriesBlock or a constant
    value are equal.

    Note that 'no data' does not equal 'no data'.

    Args:
      source (SeriesBlock): First comparison term
      other (SeriesBlock or number): Second comparison term

    Returns:
      SeriesBlock with boolean values
    """

    process = staticmethod(operator.eq)


class NotEqual(BaseFieldOperation):
    """
    Determine whether a SeriesBlock and a second SeriesBlock or a constant
    value are not equal.

    Note that 'no data' does not equal 'no data'.

    Args:
      source (SeriesBlock): First comparison term
      other (SeriesBlock or number): Second comparison term

    Returns:
      SeriesBlock with boolean values
    """

    process = staticmethod(operator.ne)


class Greater(BaseFieldOperation):
    """
    Determine for each value in a SeriesBlock whether it is greater than a
    comparison value from a SeriesBlock or constant.

    Args:
      source (SeriesBlock): First comparison term
      other (SeriesBlock or number): Second comparison term

    Returns:
      SeriesBlock with boolean values
    """

    process = staticmethod(operator.gt)


class GreaterEqual(BaseFieldOperation):
    """
    Determine for each value in a SeriesBlock whether it is greater than or
    equal to a comparison value from a SeriesBlock or constant.

    Args:
      source (SeriesBlock): First comparison term
      other (SeriesBlock or number): Second comparison term

    Returns:
      SeriesBlock with boolean values
    """

    process = staticmethod(operator.ge)


class Less(BaseFieldOperation):
    """
    Determine for each value in a SeriesBlock whether it is less than a
    comparison value from a SeriesBlock or constant.

    Args:
      source (SeriesBlock): First comparison term
      other (SeriesBlock or number): Second comparison term

    Returns:
      SeriesBlock with boolean values
    """

    process = staticmethod(operator.lt)


class LessEqual(BaseFieldOperation):
    """
    Determine for each value in a SeriesBlock whether it is less than or equal
    to a comparison value from a SeriesBlock or constant.

    Args:
      source (SeriesBlock): First comparison term
      other (SeriesBlock or number): Second comparison term

    Returns:
      SeriesBlock with boolean values
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
    Perform an elementwise logical AND between two SeriesBlocks.

    If a feature has a True value in both SeriesBlocks, True is returned, else
    False is returned.

    Args:
      source (SeriesBlock): First boolean term
      other (SeriesBlock): Second boolean term

    Returns:
      SeriesBlock with boolean values
    """

    process = staticmethod(operator.and_)


class Or(BaseLogicOperation):
    """
    Perform an elementwise logical OR between two SeriesBlocks.

    If a feature has a True value in any of the input SeriesBlocks, True is
    returned, else False is returned.

    Args:
      source (SeriesBlock): First boolean term
      other (SeriesBlock): Second boolean term

    Returns:
      SeriesBlock with boolean values
    """

    process = staticmethod(operator.or_)


class Xor(BaseLogicOperation):
    """
    Perform an elementwise logical exclusive OR between two SeriesBlocks.

    If a feature has a True value in precisely one of the input SeriesBlocks,
    True is returned, else False is returned.

    Args:
      source (SeriesBlock): First boolean term
      other (SeriesBlock): Second boolean term

    Returns:
      SeriesBlock with boolean values
    """

    process = staticmethod(operator.xor)


class Invert(BaseSingleSeries):
    """
    Invert a boolean SeriesBlock (swap True and False)

    Args:
      source (SeriesBlock): SeriesBlock with boolean values.

    Returns:
      SeriesBlock with boolean values
    """

    process = staticmethod(operator.inv)


class Where(BaseSingleSeries):
    """
    Replace values in a SeriesBlock where values in another SeriesBlock are
    False.

    Provide a source SeriesBlock, a conditional SeriesBlock (True/False) and a
    replacement value which can either be a SeriesBlock or a constant value.
    All entries in the source that correspond to a True value in the
    conditional are left unchanged. The values in the source that correspond to
    a False value in the conditional are replaced with the value from 'other'.

    Args:
      source (SeriesBlock): Source SeriesBlock that is going to be updated
      cond (SeriesBlock): Conditional SeriesBlock that determines
        whether features in the source SeriesBlock will be updated. If this
        is not boolean (True/False), then all data values (including 0) are
        interpreted as True. Missing values are always interpeted as False.
      other (SeriesBlock or constant): The value that should be used as a
        replacement for the source SeriesBlock where the conditional
        SeriesBlock is False.

    Returns:
      SeriesBlock with updated values where condition is False.
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
        if cond.dtype != bool:
            cond = ~pd.isnull(cond)
        return source.where(cond, other)


class Mask(BaseSingleSeries):
    """
    Replace values in a SeriesBlock where values in another SeriesBlock are
    True.

    Provide a source SeriesBlock, a conditional SeriesBlock (True/False) and a
    replacement value which can either be a SeriesBlock or a constant value.
    All entries in the source that correspond to a True value in the
    conditional are left unchanged. The values in the source that correspond to
    a True value in the conditional are replaced with the value from 'other'.

    Args:
      source (SeriesBlock): Source SeriesBlock that is going to be updated
      cond (SeriesBlock): Conditional SeriesBlock that determines
        whether features in the source SeriesBlock will be updated. If this
        is not boolean (True/False), then all data values (including 0) are
        interpreted as True. Missing values are always interpeted as False.
      other (SeriesBlock or constant): The value that should be used as a
        replacement for the source SeriesBlock where the conditional
        SeriesBlock is True.

    Returns:
      SeriesBlock with updated values where condition is True.
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
        if cond.dtype != bool:
            cond = ~pd.isnull(cond)
        return source.mask(cond, other)


class Round(BaseSingleSeries):
    """
    Round each value in a SeriesBlock to the given number of decimals

    Args:
      source (SeriesBlock): SeriesBlock with float data that is rounded to the
        provided number of decimals.
      decimals (int, optional): number of decimal places to round to
        (default: 0). If decimals is negative, it specifies the number of
        positions to the left of the decimal point.

    Returns:
      SeriesBlock with rounded values.
    """

    def __init__(self, source, decimals=0):
        if not isinstance(decimals, int):
            raise TypeError("'{}' object is not allowed".format(type(decimals)))
        super().__init__(source, decimals)

    process = staticmethod(np.around)


class Interp(BaseSingleSeries):
    """One-dimensional piecewise linear interpolation.

    Given a list of datapoints corresponding to x and f(x), compute the
    one-dimensional piecewise linear interpolant f(x) for each input x
    from ``source``.

    Args:
      source (SeriesBlock): Source data (x-coordinates) to interpolate
      xp (list): The x-coordinates of the data points, must be increasing.
      fp (list): The y-coordinates of the data points, same length as ``xp``.
      left (number, optional): Value to return when an x-coordinate from
        ``source`` is below than the first ``xp``. Defaults to the first
        ``fp``.
      right (number, optional): Value to return when an x-coordinate from
        ``source`` is greater than the last ``xp``. Defaults to the last
        ``fp``.
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
    """Construct a SeriesBlock by choosing values from multiple SeriesBlocks.

    For example, consider we have created ``Choose(X, A, B, C)``. The four
    input SeriesBlock contain the following values:

    - X: ``[0, 2, 1]``
    - A: ``[1, 2, 3]``
    - B: ``[4, 5, 6]``
    - C: ``[7, 8, 9]``

    The result will be ``[1, 8, 6]``: the first entry has X=0, so A is
    selected whose first entry is ``1``. The second entry has X=2, so B
    is selected, the third will select C, etc. Note that the choice series
    A, B and C can be of any type.

    If any value in ``source`` is out of bounds (below 0 or larger than the
    number of arguments), 'no data' will be filled in.

    Args:
      source (SeriesBlock): SeriesBlock having integers from 0 to n - 1 with n
        being the number of choices.
      *choices (SeriesBlock): Multiple series containing the values to choose

    Returns:
       SeriesBlock with values from ``choices``
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
