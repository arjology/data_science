import pytest
import unittest
import hashlib
import gc

from shortest_route.lib.graph import CityMapperGraph
from shortest_route.lib.elements import Node, Edge


class GraphTest(object):

    @classmethod
    def setUpGraph(cls):
        cls.nodes = []
        cls.edges = []

        cls.g = CityMapperGraph()
        cls.node_true = Node(uid=316319897)
        cls.node_false = Node(uid=0)
        cls.edge_true = Edge(uid="1c0597617b180d1bbe34e89bdd5370d923428868554c7190274535614b806d33",
                             src=Node(uid=316319897),
                             dst=Node(uid=316319936),
                             length=121)
        cls.edge_false = Edge(uid="false", src=Node(uid=0), dst=Node(uid=1), length=0)
        cls.src_node_true, cls.dst_node_true = cls.edge_true.src, cls.edge_true.dst

    def tearDownFiles(cls):
        del cls.nodes, cls.edges, cls.g
        gc.collect()

    def test_add_get_nodes(self):
        cls.g.add_nodes(1084140204)
        cls.g.add_nodes([cls.node_true, cls.src_node_true, cls.dst_node_true])

    def test_add_get_edges(self):
        cls.g.add_edges(cls.edge_true)
