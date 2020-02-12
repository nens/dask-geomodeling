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
        larger or one less than the length of the ``bins`` argument.
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
        larger or one less than the length of the ``bins`` argument.
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
        values = features[value_column].values
        bins = features[bin_columns].values
        n_bins = len(bin_columns)

        # Check in which bin every value is. because bins may be different for
        # each value, searchsorted is not an option. We assume that bins are
        # sorted in increasing order. Checking that would be costly.
        with np.errstate(invalid='ignore'):  # comparison to NaN is OK here
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
    Element-wise addition of SeriesBlock or number to another SeriesBlock.

    Args:
      source (SeriesBlock): first addition term
      other (SeriesBlock or number): second addition term
    
    Returns:
      SeriesBlock where the values are summed.
    """

    process = staticmethod(operator.add)


class Subtract(BaseFieldOperation):
    """ 
    Subtract scalar or SeriesBlock from another SeriesBlock.
    
    Supply a SeriesBlock and either a second SeriesBlock or a constant value to 
    be subtracted from its data. 

    Args:
      source (SeriesBlock): The SeriesBlock which is used in the calculation.
      subtraction_value (SeriesBlock or float): Either a second SeriesBlock or a 
        constant value which is subtracted from the first one.
      
    Returns:
      SeriesBlock where the values are subtracted.
    
    Note that if you want to subtract a SeriesBlock from a constant value (like 
    4 - series, you have to do Add(Multiply(series, -1), 4)
    """

    process = staticmethod(operator.sub)


class Multiply(BaseFieldOperation):
    """
    Multiply a SeriesBlock with a constant value or a second SeriesBlock 
    (element-wise)
    
    Provide a SeriesBlock and a second SeriesBlock or a constant value to be 
    multiplied together. 

    Args:
      source (SeriesBlock): The SeriesBlock which is used in the calculation.
      multiplication_value (SeriesBlock or float): Either a second SeriesBlock 
        or a constant value. The multiplication is performed element-wise. 
      
    Returns:
      SeriesBlock where the values are multiplied.
    """

    process = staticmethod(operator.mul)


class Divide(BaseFieldOperation):
    """ 
    Divide a SeriesBlock by a constant value or a second SeriesBlock 
    (element-wise)
    
    Provide a SeriesBlock and a second SeriesBlock or a constant value to 
    divide the first SeriesBlock by.

    Args:
      source (SeriesBlock): The SeriesBlock which is used in the calculation 
        (numerator).
      denominator (SeriesBlock or float): Either a second SeriesBlock or a 
        constant value. The division is performed element-wise

    Returns:
      SeriesBlock where the values are divided.
    """

    process = staticmethod(operator.truediv)


class FloorDivide(BaseFieldOperation):
    """ 
    Divide a SeriesBlock by a second SeriesBlock or constant value 
    (element-wise) and round to the closest integer below (i.e. 3.4 becomes 3, 
    3.9 becomes 3 and -3.4 becomes -4)

    Provide a SeriesBlock and a second SeriesBlock or a constant value to divide
    the first SeriesBlock by. The outcome is rounded to the nearest integer 
    below.

    Args:
      source (SeriesBlock): The seriesblock which is used in the calculation 
        (numerator).
      denominator (SeriesBlock): Either a second SeriesBlock or a constant 
        value. The division is performed element-wise.
      
    Returns:
      SeriesBlock where the values are divided and rounded to the nearest 
      integer below.
    """

    process = staticmethod(operator.floordiv)


class Power(BaseFieldOperation):
    """ 
    Provide a SeriesBlock which is taken to the power of a constant value or a 
    second SeriesBlock. For example input: [2,4] and 2 gives output [4,16]. 
    
    Provide a SeriesBlock and a second SeriesBlock or a constant value. 
    The (first) input SeriesBlock is taken to the power of the second 
    SeriesBlock or the constant value. In case two SeriesBlocks are provided the
    power is computed element-wise.
 
    Args:
      source (SeriesBlock): The SeriesBlock which is used as the base of the 
        power operation.
      exponent (SeriesBlock or float): The value which is used as the 
        exponent/power value.
      
    Returns:
      SeriesBlock with new values.
    
    """

    def __init__(self, source, other):
        # the float(other) will raise a TypeError if necessary
        super().__init__(source, float(other))

    process = staticmethod(operator.pow)


class Modulo(BaseFieldOperation):
    """
    Compute the modulo (remainder after division) of two series or one series 
    and a constant value.
    
    Provide a SeriesBlock and a second SeriesBlock or a constant value. The 
    first SeriesBlock is expressed as a function of the second SeriesBlock or 
    constant value. The second value functions as the modulus of the system. 
    Example: the input value = 5, the second input value (modulus) = 3. The 
    possible values in the new system become 0,1,2. These values are repeted 
    infinitely, i.e. the number sequence becomes 0,1,2,0,1,2,0. Any value 
    exceeding the modulus is thus becoming smaller. Example: if 5 is expressed 
    with a modulus 3 it becomes the 3rd value past the hightest value of the 
    sequence which is in this case 2 (0,1,2,0,1,2). The easiest way to determine
    the outcome is to repeatedly subtract the modulus from the input until the 
    outcome is below the modulus. Example: input 15, modulus 4: 15-4=11, 11-4=7,
    7-4=3, the outcome = 3.

    Args:
      source (SeriesBlock): The SeriesBlock which is converted into its modular 
        representation.
      modulus (SeriesBlock or float): The value which is used as the modulus of 
        the system. If a SeriesBlock is provided the operations take place 
        element-wise.
    
    Returns:
     SeriesBlock with values expressed as function of the modulus. 
    """

    process = staticmethod(operator.mod)


class Equal(BaseFieldOperation):
    """
    Determine whether a SeriesBlock and a second SeriesBlock or a constant 
    value are equal.
    
    Provide a SeriesBlock and a second SeriesBlock or a constant value. If both 
    are equal the operation returns True and if not a False is returned. 

    Args:
      source (SeriesBlock): The input seriesBlock which is compared to the 
        second input value.
      comparison_value (SeriesBlock or constant): The input SeriesBlock or 
        constant which is used to compare the first SeriesBlock to.

    Returns:
      Boolean SeriesBlock with values True or False for each feature. Outcome is 
      determined by whether they are equal to the comparison value or not.
      
    Note that 'no data' does not equal 'no data'
    """

    process = staticmethod(operator.eq)


class NotEqual(BaseFieldOperation):
    """
    Determine whether a SeriesBlock and a second SeriesBlock or a constant 
    value are different.
    
    Provide a SeriesBlock and a second SeriesBlock or a constant value. If both 
    are different the operation returns True and if not a False is returned. 

    Args:
      source (SeriesBlock): The input SeriesBlock which is compared to the 
        second input value.
      comparison_value (SeriesBlock or constant): The input SeriesBlock or 
        constant which is used to compare the first SeriesBlock to.

    Returns:
      Boolean SeriesBlock with values True or False for each feature. Outcome is
      determined by whether they are different from the comparison value or not.
    """

    process = staticmethod(operator.ne)


class Greater(BaseFieldOperation):
    """
    Determine for each value in a SeriesBlock whether it is larger than a 
    comparison value from a SeriesBlock or constant.
    
    Provide a SeriesBlock and a second SeriesBlock or a constant value. For each
    feature in the first input SeriesBlock is determined whether its value 
    exceeds the value in the second SeriesBlock.

    Args:
      source (SeriesBlock): The input SeriesBlock which is compared to the 
        second input value.
      comparison_value (SeriesBlock or constant): The input SeriesBlock or 
        constant which is used to compare the first SeriesBlock to.
      
    Returns:
      Boolean SeriesBlock with values True or False for each feature. Outcome is 
      determined by whether the value exceeds the comparison value or not.
    """

    process = staticmethod(operator.gt)


class GreaterEqual(BaseFieldOperation):
    """
    Determine for each value in a SeriesBlock whether it is larger than or equal
    to a comparison value from a SeriesBlock or constant.
    
    Provide a SeriesBlock and a second SeriesBlock or a constant value. For each
    feature in the first input SeriesBlock is determined whether its value 
    exceeds or equals the value in the second SeriesBlock.

    Args:
      source (SeriesBlock): The input SeriesBlock which is compared to the 
        second input value.
      comparison_value (SeriesBlock or constant): The input SeriesBlock or 
        constant which is used to compare the first SeriesBlock to.
      
    Returns:
      Boolean SeriesBlock with values True or False for each feature. Outcome is 
      determined by whether the value exceeds or equals the comparison value or
      not.
    """

    process = staticmethod(operator.ge)


class Less(BaseFieldOperation):
    """
    Determine for each value in a SeriesBlock whether it is below a comparison 
    value from a SeriesBlock or constant.
    
    Provide a SeriesBlock and a second SeriesBlock or a constant value. For each
    feature in the first input SeriesBlock is determined whether its value falls
    below the value in the second seriesBlock.

    Args:
      source (SeriesBlock): The input SeriesBlock which is compared to the 
        second input value.
      comparison_value (SeriesBlock or constant): The input SeriesBlock or 
        constant which is used to compare the first SeriesBlock to.
      
    Returns:
      Boolean SeriesBlock with values True or False for each feature. Outcome is 
      determined by whether the value falls below the comparison value or not.
    """

    process = staticmethod(operator.lt)


class LessEqual(BaseFieldOperation):
    """
    Determine for each value in a SeriesBlock whether it is below or equal to a 
    comparison value from a SeriesBlock or constant.
    
    Provide a SeriesBlock and a second SeriesBlock or a constant value. For each 
    feature in the first input SeriesBlock is determined whether its value falls 
    below or equals the value in the second SeriesBlock.

    Args:
      source (SeriesBlock): The input SeriesBlock which is compared to the 
        second input value.
      comparison_value (SeriesBlock or constant): The input SeriesBlock or 
        constant which is used to compare the first SeriesBlock to.
      
    Returns:
      Boolean SeriesBlock with values True or False for each feature. Outcome is 
      determined by whether the value falls below or equals the comparison value 
      or not.
    """

    process = staticmethod(operator.le)


class BaseLogicOperation(BaseFieldOperation):
    """Base block for logic operations between columns in a GeometryBlock"""

    def __init__(self, source, other):
        if not isinstance(other, SeriesBlock):
            raise TypeError("'{}' object is not allowed".format(type(other)))
        super().__init__(source, other)


class And(BaseLogicOperation): #WIP: TE ONDUIDELIJK.
    """
    Determine whether a value is True in two provided (boolean) SeriesBlocks. 

    Provide two SeriesBlocks with boolean values (True/False). If a feature has 
    a True value in both SeriesBlocks, True is returned else False is returned. 

    Args:
      source (SeriesBlock): First boolean (True/False) SeriesBlock.
      comparison_value (SeriesBlock): second boolean (True/False) SeriesBlock.
    
    Returns:
      Boolean SeriesBlock with True value where both input blocks were True. All 
      other values become false.
    """

    process = staticmethod(operator.and_)


class Or(BaseLogicOperation):
    """
    Determine whether at least 1 of 2 provided (boolean) SeriesBlocks is True.
    
    Provide two SeriesBlocks with boolean values (True/False). If one or both 
    features in the SeriesBlocks are True the result will be True. If both are 
    False the outcome will be False.
    
    Args:
      source (SeriesBlock): First boolean (True/False) SeriesBlock
      comparison_value (SeriesBlock): Second boolean (True/False) SeriesBlock
      
    Returns:
      Boolean SeriesBlock with True value where at least one or both of the two 
      SeriesBlocks are True. If both are False, False is returned.
    """

    process = staticmethod(operator.or_)


class Xor(BaseLogicOperation):
    """
    Determine whether one value (and one value only) is True out of two 
    (boolean) SeriesBlocks.
    
    Provide two boolean (True/False) SeriesBlocks. If only one of the features 
    in the blocks is True, True is returned. If either both blocks are True or 
    both blocks are False, False is returned.

    Args:
      source (SeriesBlock): First boolean (True/False) SeriesBlock
      comparison_value (SeriesBlock): Second boolean (True/False) SeriesBlock
      
    Returns:
      Boolean SeriesBlock with True values when only one of the two input 
      SeriesBlocks was True. Else False is returned. 
    """

    process = staticmethod(operator.xor)


class Invert(BaseSingleSeries):
    """
    Invert a boolean SeriesBlock (i.e. True becomes False and vice versa).
    
    Provide a boolean SeriesBlock which is inverted.
    
    Args:
      source (SeriesBlock): SeriesBlock with boolean values.
      
    Returns:
      Inverted, boolean, SeriesBlock.
    """

    process = staticmethod(operator.inv)


class Where(BaseSingleSeries):
    """
    Replace values in a SeriesBlock with either a constant value or values from
    a second SeriesBlock. The values are replaced where the values in a boolean 
    SeriesBlock (True/False) are False.

    Provide a base SeriesBlock, a conditional SeriesBlock (True/False) and a 
    replacement value which can be a SeriesBlock or a constant value. All 
    entries in the base SeriesBlock which correspond to a True value in the 
    conditional SeriesBlock are not changed. The values in the base SeriesBlock
    which correspond to a False value are replaced with the replacement value.
    If the replacement value is a SeriesBlock, the replacement happens element-wise.
    
    Args:
      source (SeriesBlock): Base data which is going to be updated for certain 
        features.
      cond (SeriesBlock): True/False SeriesBlock which determines whether 
        features in the base SeriesBlock should be updated.
      other (SeriesBlock or constant): The value which should be used as a 
        replacement for the base SeriesBlock when the conditional SeriesBlock is 
        False.
      
    Returns:
      SeriesBlock with updated values where condition is false. 
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
    """
    Replace values in a SeriesBlock with either a constant value or values from 
    a second SeriesBlock. The values are replaced where the values in a boolean
    SeriesBlock (True/False) are True.

    Provide a base SeriesBlock, a conditional SeriesBlock (True/False) and a 
    replacement value which can be a SeriesBlock or a constant value. All 
    entries in the base SeriesBlock which correspond to a False value in the 
    conditional SeriesBlock are not changed. The values in the base SeriesBlock 
    which correspond to a True value are replaced with the replacement value. If
    the replacement value is a SeriesBlock, the replacement happens element-wise.
    
    Args:
      source (SeriesBlock): Base data which is going to be updated for certain 
        features.
      cond (SeriesBlock): True/False SeriesBlock which determines whether 
        features in the base SeriesBlock should be updated.
      other (SeriesBlock or constant): The value which should be used as a 
        replacement for the base SeriesBlock when the conditional SeriesBlock is
        True.
      
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
        return source.mask(cond, other)


class Round(BaseSingleSeries):
    """
    Round each value in a SeriesBlock to the given number of decimals
    
    Provide a SeriesBlock with float data. The data is rounded to the provided 
    number of decimals.
    
    Args:
      source (SeriesBlock): SeriesBlock with float data which is rounded to the 
        provided number of decimals.
      decimals (integer, optional): number of decimal places to round to 
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


class Interp(BaseSingleSeries): #WIP: te onduidelijk
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


class Choose(BaseSingleSeries):#WIP: te onduidelijk
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
