"""
Module containing the core graphs.
"""
import inspect
import sys
import json
import logging

from dask.base import tokenize, normalize_token
from dask.local import get_sync

from shapely.geometry.base import BaseGeometry
from datetime import datetime
from datetime import timedelta

logger = logging.getLogger(__name__)

__all__ = ["construct", "construct_multiple", "compute", "Block", "DummyBlock"]


def _construct_exc_callback(e, dumps):
    """Callback to be used as a 'pack_exception' kwarg in get_sync

    """
    key = inspect.currentframe().f_back.f_locals.get("key")
    e.args = ("{0}: {1}".format(key, str(e)),)
    raise e


def _reconstruct_token(key):
    """Reconstruct a token from a key in a graph ('SomeName_<token>')"""
    if len(key) < 34 or key[-33] != "_":
        return
    token = key[-32:]
    try:
        int(token, 16)
    except ValueError:
        return None
    return token.lower()


def compute(graph, name, *args, **kwargs):
    """Compute a graph ({name: [func, arg1, arg2, ...]}) using dask.get_sync
    """
    return get_sync(graph, [name])[0]


def construct(graph, name, validate=True):
    """Construct a Block with dependent Blocks from a graph and endpoint name.
    """
    return construct_multiple(graph, [name], validate)[0]


def construct_multiple(graph, names, validate=True):
    """Construct multiple Blocks from given graph and endpoint names.
    """
    # deserialize import paths where necessary and cast lists to tuples
    new_graph = {}
    for key, value in graph.items():
        cls = value[0]
        if isinstance(cls, str):
            cls = Block.from_import_path(cls)
        if not issubclass(cls, Block):
            raise TypeError("Cannot construct from object of type '{}'".format(cls))
        args = tuple(value[1:])
        if validate:
            new_graph[key] = (cls,) + args
        else:
            token = _reconstruct_token(key)
            if token is None:
                logger.warning(
                    "Construct received a key with an invalid name ('%s'),"
                    "while validation was turned off",
                    key,
                )
            new_graph[key] = (cls._init_no_validation, token) + args

    return get_sync(new_graph, names, pack_exception=_construct_exc_callback)


class Block(object):
    """ A class that generates dask-like compute graphs for given requests.

    Arguments (args) are always stored in ``self.args``.
    If a request is passed into the Block using the ``get_data`` or
    (the lazy version) ``get_compute_graph`` method, the Block figures out
    what args are actually necessary to evaluate the request, and what
    requests need to be sent to those args. This happens in the method
    ``get_sources_and_requests``.

    After the requests have been evaluated, the data comes back and is passed
    into the ``process`` method.
    """

    JSON_VERSION = 2

    @property
    def token(self):
        """Generates a unique and deterministic representation of this object
        """
        # the token is cached on this Block object
        try:
            return self._cached_token
        except AttributeError:
            pass
        klass_path = self.get_import_path()
        args = [arg.token if isinstance(arg, Block) else arg for arg in self.args]
        self._cached_token = tokenize(klass_path, *args)
        return self._cached_token

    @staticmethod  # must be a static method
    def process(data):
        """
        Overridden to modify data from sources in unlimited ways.

        Default implementation passes single-source unaltered data.
        """
        return data

    def __init__(self, *args):
        """The init should be overridden by subclasses. In each __init__:
         - Explicitly give a call signature and a corresponding docstring
         - Check the types of the provided args, so that exceptions are raised
           during pipeline construction.
         - Call super().__init__().
        """
        self.args = args

    @classmethod
    def _init_no_validation(cls, token, *args):
        """This constructs this block directly from its key and args
        without any validation and tokenization."""
        obj = cls.__new__(cls)
        obj.args = args
        if token:
            obj._cached_token = token
        return obj

    def get_sources_and_requests(self, **request):
        """Adapt the request and/or select the sources to be computed. The
        request is allowed to differ per source.

        This function should return an iterable of (source, request). For
        sources that are no Block instance, the request is ignored.

        Exceptions raised here will be raised before actual computation starts.
        (at .get_compute_graph(request)).
        """
        return ((source, request) for source in self.args)

    """Below are methods that should never be overridden by subclasses"""

    def get_data(self, **request):
        """Directly evaluate the request and return the data."""
        return compute(*self.get_compute_graph(**request))

    def get_compute_graph(self, cached_compute_graph=None, **request):
        """Lazy version of get_data, returns a compute graph dict, that
        can be evaluated with `compute` (or dask's get function).

        The dictionary has keys in the form ``name_token`` and values in the
        form ``tuple(process, *args)``, where ``args`` are the precise
        arguments that need to be passed to ``process``, with the exception
        that args may reference to other keys in the dictionary.
        """
        # generate a token from the specific request passed to the block
        # NB generates a random hash if the request cannot be tokenized
        token = tokenize([self.token, request])
        name = "{}_{}".format(self.__class__.__name__.lower(), token)
        graph = cached_compute_graph or dict()

        if name in graph:
            return graph, name

        args = [self.process]
        for source, req in self.get_sources_and_requests(**request):
            if isinstance(source, Block) and req is not None:
                graph, compute_name = source.get_compute_graph(
                    cached_compute_graph=graph, **req
                )
                args.append(compute_name)
            else:
                args.append(source)

        graph[name] = tuple(args)
        return graph, name

    def get_graph(self, serialize=False):
        """Generate a graph that defines this Block and its dependencies
        in a dictionary.

        The dictionary has keys in the form ``name_token`` and values in the
        form ``tuple(Block class, *args)``, where ``args`` are the precise
        arguments that were used to construct the Block, with the exception
        that args may also reference other keys in the dictionary.

        If serialize == True, the Block classes will be replaced by their
        corresponding import paths.
        """
        if serialize:
            args = [self.get_import_path()]
        else:
            args = [self.__class__]
        graph = dict()
        for arg in self.args:
            if isinstance(arg, Block):
                sub_graph, sub_name = arg.get_graph(serialize=serialize)
                graph.update(sub_graph)
                args.append(sub_name)
            else:
                args.append(arg)
        name = self.name
        graph[name] = args
        return graph, name

    @property
    def name(self):
        return "{}_{}".format(self.__class__.__name__, self.token)

    def __reduce__(self):
        """Serialize the object (pattern: callable, args). Construct is called
        with the arguments (graph, name) to reconstruct this block and its
        dependencies. Validation is skipped by adding False to the args."""
        return construct, self.get_graph() + (False,)

    @classmethod
    def get_import_path(cls):
        """Serialize the Block by returning its import path."""
        name = cls.__name__
        module = cls.__module__

        try:
            __import__(module)
            mod = sys.modules[module]
            klass = getattr(mod, name)
        except (ImportError, KeyError, AttributeError):
            raise Exception(
                "Can't serialize %r: it's not found as %s.%s" % (cls, module, name)
            )
        else:
            if klass is not cls:
                raise Exception(
                    "Can't serialize %r: it's not the same object as %s.%s"
                    % (cls, module, name)
                )

        return "{}.{}".format(module, name)

    @staticmethod
    def from_import_path(path):
        """Deserialize the Block by importing it from given path."""
        module, name = path.rsplit(".", 1)
        __import__(module)
        mod = sys.modules[module]
        klass = getattr(mod, name)
        if issubclass(klass, Block):
            return klass
        else:
            raise TypeError('"{}" is not valid Block.'.format(path))

    @classmethod
    def from_json(cls, val, **kwargs):
        """Construct a graph from a json stream."""
        return cls.deserialize(json.loads(val, **kwargs))

    def to_json(self, **kwargs):
        """Dump the graph to a json stream."""
        return json.dumps(self.serialize(), **kwargs)

    def serialize(self):
        """Serialize this block into a dict containing version, graph and name
        """
        graph, name = self.get_graph(serialize=True)
        return {"version": self.JSON_VERSION, "graph": graph, "name": name}

    @classmethod
    def deserialize(cls, val, validate=False):
        """
        Deserialize this block from a dict containing version, graph and name
        """
        # TODO Compare val['version'] with cls.JSON_VERSION
        return construct(val["graph"], val["name"], validate=validate)

    def __repr__(self):
        name = self.__class__.__name__
        return "{}({})".format(name, ", ".join([repr(x) for x in self.args]))


class DummyBlock(Block):
    """This dummy block pretends that it has the user-supplied name and token

    This is useful for partially evaluating block graphs for computing tokens.
    """

    def __init__(self, name):
        super().__init__(name)

    @property
    def token(self):
        return self.name.split("_")[1]

    @property
    def name(self):
        return self.args[0]


# Dask knows how to hash a couple of object types. If it doesn't know how to,
# it iterates over the MRO of the object until it ends up at 'object'. Then
# it checks for the __dask_tokenize__ slot, and if it does not exist, it
# generates a random hash.

# Tokenize shapely geometry by using its WKB representation.
@normalize_token.register(BaseGeometry)
def normalize_shapely_geometry(geometry):
    return geometry.wkb


# Tokenize datetime and timedeltas using their pickle handle
@normalize_token.register((datetime, timedelta))
def normalize_datetime(value):
    return hash(value)
