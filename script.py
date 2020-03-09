import dask
from dask.highlevelgraph import HighLevelGraph
from dask.base import tokenize
from dask.array.utils import meta_from_array
import dask.array as da
from dask.array import slicing
from dask.blockwise import Blockwise
import numpy as np
import operator


@dask.delayed
def my_reader(*slices):
    if not slices:
        raise NotImplementedError("These dask methods are incompatible")
    return np.mgrid[slices].astype(np.float32)


def merge_getitem_with_reader(dsk, key, reader_key, index):
    dsk[reader_key] = (my_reader, *index)


def push_getitem_through_blockwise(dsk, key, blockwise, index):
    if any(ind is not None and ind != blockwise.output_indices for _, ind in blockwise.indices):
        return  # can only deal with elementwise or scalar operations

    # add getitem layers
    new_indices = []
    for dependency, ind in blockwise.indices:
        if ind is None:
            new_indices.append((dependency, None))  # scalars
        else:
            new_getitem_name = "getitem-" + tokenize(dependency, index)
            dsk.layers[new_getitem_name] = {
                (new_getitem_name, ) + dep_item[1:]:
                    (operator.getitem, dep_item, index)
                for dep_item in dsk.layers[dependency].keys()
            }
            dsk.dependencies[new_getitem_name] = {dependency}
            new_indices.append((new_getitem_name, ind))

    # adapt blockwise layer  TODO actual name
    new_numblocks = {
        new[0]: blockwise.numblocks[old[0]]
        for old, new in zip(blockwise.indices, new_indices)
        if new[1] is not None
    }
    dsk.layers[key] = Blockwise(
        key,
        blockwise.output_indices,
        blockwise.dsk,
        tuple(new_indices),
        new_numblocks,
        blockwise.concatenate,
        blockwise.new_axes,
    )
    # adapt dependency dict
    dsk.dependencies[key] = set([x[0] for x in new_indices if x[1] is not None])

    # recurse
    for dep, ind in new_indices:
        if ind is not None:
            push_getitem(dsk, dep)


def push_getitem(dsk, key):
    """Pushes a getitem task through its dependency.

    Example:
        (a + 1)[:10]  -->  a[:10] + 1

        layers
          'getitem': {('getitem-<hash>', 0, 0): (getitem, 'add-<hash>', slices)},
          'add': {('add-<hash>', 0, 0): Blockwise<(<a>, <b>)}
        dependencies
          {'add-<hash>': <a>, <b>, 'getitem-<hash>': 'add-<hash>'}

        to

        layers
          'add': {('add-<hash>', 0, 0): Blockwise<(<getitem-<hash1>>, <getitem-<hash2>>)},
          'getitem1': {('getitem-<hash1>', 0, 0): (getitem, '<a>', slices)}
          'getitem2': {('getitem-<hash2>', 0, 0): (getitem, '<b>', slices)}
        dependencies
          {'add-<hash>': {'a-getitem-<hash>', 'b-getitem-<hash>'},
           'a-getitem-<hash>': '<a>',
           'b-getitem-<hash>': '<b>'}
    """
    task = dsk.layers[key]
    if len(task) != 1:
        return  # not sure what to do if there are multiple chunks already
    index = next(iter(task.values()))[2]  # TODO check for slice-with-dask array here
    dependency_key = next(iter(dsk.dependencies[key]))
    dependency = dsk.layers[dependency_key]
    if isinstance(dependency, Blockwise):
        push_getitem_through_blockwise(dsk, key, dependency, index)
    elif dependency_key.startswith("from-value-"):
        if len(dependency) != 1:
            return  # not sure what to do if there are multiple chunks already
        arg = next(iter(dependency.values()))
        if isinstance(arg, str) and arg.startswith("my_reader-"):
            merge_getitem_with_reader(dsk, key, arg, index)


def push_getitems_to_reader(dsk, keys):
    getitems = [x for x in dsk.layers.keys() if x.startswith("getitem-")]
    for key in getitems:
        push_getitem(dsk, key)
    return dsk


a = da.from_delayed(my_reader(), shape=(1000000, 1000000), dtype=np.float32)
b = a + 2
c = b[10:100:5, 10:100:5]  # now we know the extent!


c = dask.optimize([c], optimizations=[push_getitems_to_reader])[0][0]
c.compute()

print(c.__dask_graph__())
