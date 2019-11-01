from unittest import mock
import unittest
import pickle
import logging

from datetime import datetime
from datetime import timedelta

import numpy as np
from numpy.testing import assert_equal

from shapely.geometry import box

from dask_geomodeling.core import Block, compute, construct, DummyBlock

from dask.base import tokenize


class MockBlock(Block):
    @staticmethod
    def process(shape, fill_value):
        return dict(
            no_data_value=255, values=np.full(shape, fill_value, dtype=np.uint8)
        )

    def __init__(self, fill_value):
        super(MockBlock, self).__init__(fill_value)

    def get_sources_and_requests(self):
        fill_value, = self.args
        return [(self.shape, None), (fill_value, None)]

    @property
    def shape(self):
        return 2, 3


class Add(Block):
    def __init__(self, a, b):
        super().__init__(a, b)

    @staticmethod
    def process(a, b):
        return dict(no_data_value=255, values=a["values"] + b["values"])


class Mul(Block):
    @staticmethod
    def process(a, b):
        return dict(no_data_value=255, values=a["values"] * b["values"])


class TestBlock(unittest.TestCase):
    def setUp(self):
        self.N = 10
        self.request = dict()
        self.block = MockBlock(1.0)
        self.shape = self.block.shape

    def test_tokenize_dummy(self):
        block = MockBlock(2)
        dummy = DummyBlock(block.name)
        self.assertEqual(block.token, dummy.token)
        self.assertEqual(block.name, dummy.name)

    def test_tokenize_dummy_derived(self):
        block = MockBlock(2)
        dummy = DummyBlock(block.name)

        expected = MockBlock(block)
        actual = MockBlock(dummy)
        self.assertEqual(expected.token, actual.token)
        self.assertEqual(expected.name, actual.name)

    def test_tokenize_float(self):
        """Compare tokens of blocks having different float arguments"""
        hashes = set()
        for n in np.random.random(self.N):
            block1 = MockBlock(n)
            block2 = MockBlock(n)

            # the same n generate the same names (names are deterministic)
            self.assertEqual(block1.name, block2.name)
            hashes.add(block1.name)

        # we should now have N unique names (names are unique)
        self.assertEqual(len(hashes), self.N)

    def test_tokenize_ancestor_difference(self):
        """Compare tokens of blocks having different source blocks"""
        hashes = set()

        for n_source in np.random.random(self.N):
            source = MockBlock(n_source)

            add1 = Add(source, 2.0)
            add2 = Add(source, 2.0)

            # the same n generate the same names (names are deterministic)
            self.assertEqual(add1.name, add2.name)
            hashes.add(add1.name)

        # we should now have N unique names (names are unique)
        self.assertEqual(len(hashes), self.N)

    def test_tokenize_level3_difference(self):
        """Compare tokens of blocks having source blocks that have
        different source blocks."""
        hashes = set()

        for n_source in np.random.random(self.N):
            source = MockBlock(n_source)
            add = Add(source, 2)

            multiply1 = Mul(add, 0.2)
            multiply2 = Mul(add, 0.2)

            # the same n generate the same names (names are deterministic)
            self.assertEqual(multiply1.name, multiply2.name)
            hashes.add(multiply1.name)

        # we should now have N unique names (names are unique)
        self.assertEqual(len(hashes), self.N)

    def test_tokenize_shapely_geometry(self):
        """Compare tokens of different shapely geometries"""
        hashes = set()
        for n in np.random.random(self.N):
            token1 = tokenize(box(0 + n, 0, 10, 10))
            token2 = tokenize(box(0 + n, 0, 10, 10))

            # the same geometries generate the same tokens
            self.assertEqual(token1, token2)
            hashes.add(token1)

        # we should now have N unique tokens
        self.assertEqual(len(hashes), self.N)

    def test_tokenize_datetime(self):
        """Compare tokens of different datetimes"""
        hashes = set()
        for n in np.random.randint(0, 2000000000, self.N):
            token1 = tokenize(datetime.fromtimestamp(n))
            token2 = tokenize(datetime.fromtimestamp(n))

            # the same datetimes generate the same tokens
            self.assertEqual(token1, token2)
            hashes.add(token1)

        # we should now have N unique tokens
        self.assertEqual(len(hashes), self.N)

    def test_tokenize_timedelta(self):
        """Compare tokens of different timedeltas"""
        hashes = set()
        for n in np.random.randint(0, 2000000000, self.N):
            token1 = tokenize(timedelta(microseconds=int(n)))
            token2 = tokenize(timedelta(microseconds=int(n)))

            # the same timedeltas generate the same tokens
            self.assertEqual(token1, token2)
            hashes.add(token1)

        # we should now have N unique tokens
        self.assertEqual(len(hashes), self.N)

    @mock.patch("dask_geomodeling.core.graphs.tokenize")
    def test_cache_token(self, patched_tokenize):
        """A Block's token is saved"""
        block = MockBlock(1)
        _ = block.token
        # tokenized 1 argument
        self.assertEqual(1, patched_tokenize.call_count)
        _ = block.token
        # did not tokenize again
        self.assertEqual(1, patched_tokenize.call_count)

    def test_graph_equal_sources(self):
        add = Add(self.block, self.block)
        graph, _ = add.get_graph()

        self.assertEqual(len(graph), 2)
        self.assertIn(add.name, graph)

    def test_graph_different_sources(self):
        add = Add(self.block, MockBlock(2.0))
        graph, _ = add.get_graph()

        self.assertEqual(len(graph), 3)
        self.assertIn(add.name, graph)

    def test_compute_graph(self):
        add = Add(self.block, self.block)
        graph, name = add.get_compute_graph(**self.request)

        self.assertEqual(len(graph), 2)
        result = compute(graph, name)

        assert_equal(result["values"].shape, self.shape)
        assert_equal(result["values"], 2.0)

    def test_compute_graph_uses_cache(self):
        add = Add(self.block, self.block)

        # patch the get_sources_and_requests only to get its call_count
        with mock.patch.object(
            self.block,
            "get_sources_and_requests",
            side_effect=self.block.get_sources_and_requests,
        ) as patched:
            add.get_compute_graph(**self.request)
            self.assertEqual(patched.call_count, 1)

    def test_compute_direct(self):
        add = Add(self.block, self.block)
        result = add.get_data(**self.request)

        assert_equal(result["values"].shape, self.shape)
        assert_equal(result["values"], 2.0)

    def test_pickle(self):
        pkl = pickle.dumps(self.block)
        view2 = pickle.loads(pkl)

        data1 = self.block.get_data(**self.request)
        data2 = view2.get_data(**self.request)

        assert_equal(data1, data2)

    def test_json(self):
        json_dump = self.block.to_json(indent=2)
        block2 = Block.from_json(json_dump)

        data1 = self.block.get_data(**self.request)
        data2 = block2.get_data(**self.request)

        assert_equal(data1, data2)

    def test_construct(self):
        block = Add(self.block, 2)
        graph, name = block.get_graph(serialize=True)
        result = construct(graph, name)
        self.assertEqual(result.token, block.token)

    def test_construct_no_validation(self):
        block = Add(self.block, 2)
        graph, name = block.get_graph(serialize=True)
        result = construct(graph, name, validate=False)
        self.assertEqual(result.token, block.token)

    def test_construct_raises_with_key_in_exception(self):
        block = Add(self.block, 2)
        graph, name = block.get_graph(serialize=True)
        graph[name] = graph[name][:2]  # chop of one arg, making this invalid
        self.assertRaisesRegex(
            TypeError, "^{}: (.*?)".format(name), construct, graph, name
        )

    def test_construct_invalid_no_validation(self):
        block = Add(self.block, 2)
        graph, name = block.get_graph(serialize=True)
        graph[name] = graph[name][:2]  # chop of one arg, making this invalid
        result = construct(graph, name, validate=False)
        self.assertEqual(len(result.args), 1)
        # token is retrieved from the key
        self.assertEqual(result.token, block.token)

    def test_construct_different_token_no_validation(self):
        # if the graph key is of a valid name format (produced by Block.name),
        # then that one is used, skipping the generation of new token
        block = Add(self.block, 2)
        graph, name = block.get_graph(serialize=True)
        different_name = "name_1aed3ec7419dadffb050a1274e1c8dc9"
        graph[different_name] = graph[name]
        result = construct(graph, different_name, validate=False)
        self.assertEqual(result.token, "1aed3ec7419dadffb050a1274e1c8dc9")

    def test_construct_invalid_token_no_validation(self):
        # if the graph key is of an invalid name format then a new one is
        # generated, but a warning is emitted
        block = Add(self.block, 2)
        graph, name = block.get_graph(serialize=True)
        for invalid_name in ["", "abc", "a_2", "a_xaed3ec7419dadffb050a1274e1c8dc9"]:
            graph[invalid_name] = graph[name]
            with self.assertLogs(level=logging.WARNING):
                result = construct(graph, invalid_name, validate=False)
            self.assertEqual(result.token, block.token)
