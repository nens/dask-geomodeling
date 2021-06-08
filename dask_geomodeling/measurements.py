"""
Module providing a percentile function analogous to the functions provided by
scipy.ndimage.measurements.
"""

import numpy as np


def _safely_castable_to_int(dt):
    """Test whether the numpy data type `dt` can be safely cast to an int."""
    int_size = np.dtype(int).itemsize
    safe = (np.issubdtype(dt, np.signedinteger) and dt.itemsize <= int_size) or (
        np.issubdtype(dt, np.unsignedinteger) and dt.itemsize < int_size
    )
    return safe


def percentile(data, qval, labels=None, index=None):
    """
    Calculate a percentile of the array values over labeled regions.

    Parameters
    ----------
    input : array_like
        Array_like of values. For each region specified by `labels`, the
        percentile value of `input` over the region is computed.
    qval : float
        percentile to compute, which must be between 0 and 100 inclusive.
    labels : array_like, optional
        An array_like of integers marking different regions over which the
        percentile value of `input` is to be computed. `labels` must have the
        same shape as `input`. If `labels` is not specified, the percentile
        over the whole array is returned.
    index : array_like, optional
        A list of region labels that are taken into account for computing the
        percentiles. If index is None, the percentile over all elements where
        `labels` is non-zero is returned.

    Returns
    -------
    percentile : float or list of floats
        List of percentiles of `input` over the regions determined by `labels`
        and whose index is in `index`. If `index` or `labels` are not
        specified, a float is returned: the percentile value of `input` if
        `labels` is None, and the percentile value of elements where `labels`
        is greater than zero if `index` is None.

    Notes
    -----
    The function returns a Python list and not a Numpy array, use
    `np.array` to convert the list to an array.

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.array([[1, 2, 0, 1],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> labels, labels_nb = ndimage.label(a)
    >>> labels
    array([[1, 1, 0, 2],
           [1, 1, 0, 2],
           [0, 0, 0, 2],
           [3, 3, 0, 0]], dtype=int32)
    >>> percentile(a, 75, labels=labels, index=np.arange(1, labels_nb + 1))
    [3.5, 5.5, 7.5]
    >>> percentile(a, 50)
    1.0
    >>> percentile(a, 50, labels=labels)
    3.0
    """
    data = np.asanyarray(data)

    def single_group(vals):
        return np.percentile(vals, qval)

    if labels is None:
        return single_group(data)

    # ensure input and labels match sizes
    data, labels = np.broadcast_arrays(data, labels)

    if index is None:
        mask = labels > 0
        return single_group(data[mask])

    if np.isscalar(index):
        mask = labels == index
        return single_group(data[mask])

    # remap labels to unique integers if necessary, or if the largest
    # label is larger than the number of values.
    if (
        not _safely_castable_to_int(labels.dtype)
        or labels.min() < 0
        or labels.max() > labels.size
    ):
        # remap labels, and indexes
        unique_labels, labels = np.unique(labels, return_inverse=True)
        idxs = np.searchsorted(unique_labels, index)

        # make all of idxs valid
        idxs[idxs >= unique_labels.size] = 0
        found = unique_labels[idxs] == index
    else:
        # labels are an integer type, and there aren't too many.
        idxs = np.asanyarray(index, int).copy()
        found = (idxs >= 0) & (idxs <= labels.max())

    idxs[~found] = labels.max() + 1

    # reorder data and labels, first by labels, then by data
    order = np.lexsort((data.ravel(), labels.ravel()))
    data = data.ravel()[order]
    labels = labels.ravel()[order]

    locs = np.arange(len(labels))
    lo = np.zeros(labels.max() + 2, int)
    lo[labels[::-1]] = locs[::-1]
    hi = np.zeros(labels.max() + 2, int)
    hi[labels] = locs
    lo = lo[idxs]
    hi = hi[idxs]
    # lo is an index to the lowest value in input for each label,
    # hi is an index to the largest value.

    # here starts the part that really diverts from scipy's median finder; the
    # linear interpolation method used corresponds to the default behaviour of
    # np.percentile().
    size = hi - lo + 1  # size of the group
    frac = (size - 1) * (qval / 100)  # fractional index relative to lo
    hi = lo - np.int64(-frac // 1)  # ceiled absolute index to data
    lo = lo + np.int64(frac // 1)  # floored absolute index to data
    part = frac % 1  # fractional part of index
    return (data[lo] + part * (data[hi] - data[lo])).tolist()
