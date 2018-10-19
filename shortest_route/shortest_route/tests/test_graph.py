import pytest
import unittest
import hashlib
import gc

from shortest_route.lib.graph import CityMapperGraph
from shortest_route.lib.elements import Node, Edge


class GraphTestBase(object):

    @classmethod
    def setUpGraph(cls):
        cls.nodes = []
        cls.edges = []

        cls.g = CityMapperGraph()
        cls.node_true = Node(uid_='316319897')
        cls.node_false = Node(uid_=0)

        cls.src_node_true, cls.dst_node_true = Node(uid_='316319897'), Node(uid_='316319936')
        cls.edge_true = Edge(uid_="1c0597617b180d1bbe34e89bdd5370d923428868554c7190274535614b806d33",
                             src=cls.src_node_true,
                             dst=cls.dst_node_true,
                             length=121)
        cls.edge_false = Edge(src=Node(uid_='0'), dst=Node(uid_='1'), uid_="false", length=0)

    @classmethod
    def tearDownGraph(cls):
        cls.g.clear()
        del cls.nodes, cls.edges, cls.g
        gc.collect()


class CityMapperGraphTest(unittest.TestCase, GraphTestBase):

    @classmethod
    def setUpClass(cls):
        cls.setUpGraph()

    def test_add_get_nodes(self):
        self.g.add_node(self.node_true)
        rs0 = list(self.g.find_nodes(self.node_true))
        rs1 = list(self.g.find_nodes(self.node_false))

        assert (len(rs0) == 1
                and isinstance(rs0[0], Node) 
                and rs0[0].uid() == self.node_true.uid()
                and rs0[0].label() == self.node_true.label()
                )
        assert len(rs1)==0

    def test_add_get_edges(self):
        self.g.add_edge(self.edge_true)
        rs0 = list(self.g.find_edges(self.edge_true))
        rs1 = list(self.g.find_edges(self.edge_false))

        assert (len(rs0)==1
                and isinstance(rs0[0], Edge)
                and isinstance(rs0[0].src, Node) and rs0[0].src == self.src_node_true
                and isinstance(rs0[0].src, Node) and rs0[0].dst == self.dst_node_true
                and rs0[0].props.length==121
                )
        assert len(rs1)==0

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls):
        cls.tearDownGraph()
