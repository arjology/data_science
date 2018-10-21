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
        
        cls.node_1 = Node(uid_=1)
        cls.node_2 = Node(uid_=2)
        cls.node_5 = Node(uid_=5)

        cls.edge_12 = Edge(uid_="0to2", src=cls.node_1, dst=cls.node_2, length=1)
        cls.edge_15 = Edge(uid_="0to5", src=cls.node_1, dst=cls.node_5, length=5)
        cls.edge_25 = Edge(uid_="2to5", src=cls.node_2, dst=cls.node_5, length=2)

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

        self.g.add_node(self.node_1)
        self.g.add_node(self.node_2)

        assert self.g.is_empty() == False

        rs0 = list(self.g.find_nodes(self.node_1))
        rs1 = list(self.g.find_nodes(Node(uid_=-1)))

        count = self.g.count()
        assert count == 2

        nodes = self.g.nodes()
        assert nodes == [self.node_1, self.node_2]

        assert (len(rs0) == 1
                and isinstance(rs0[0], Node) 
                and rs0[0].uid() == self.node_1.uid()
                and rs0[0].label() == self.node_1.label()
                )
        assert len(rs1)==0

    def test_add_get_edges(self):
        self.g.add_node(self.node_1)
        self.g.add_node(self.node_2)

        self.g.add_edge(self.edge_12)
        rs0 = list(self.g.find_edges(self.edge_12))
        rs1 = list(self.g.find_edges(Edge(uid_="bad", src=Node(uid_='0'), dst=Node(uid_='1'), length=1)))

        assert (len(rs0)==1
                and isinstance(rs0[0], Edge)
                and isinstance(rs0[0].src, Node) and rs0[0].src == self.node_1
                and isinstance(rs0[0].src, Node) and rs0[0].dst == self.node_2
                and rs0[0].props.length==1
                )
        assert len(rs1)==0

    def test_find_neighbors(self):
        rs0 = list(self.g.neighbours(self.node_1))
        assert len(rs0) == 1
        assert len(rs0[0]) == 2
        assert rs0[0][0].match(self.node_2)
        assert rs0[0][1] == 1

    def test_shortest_path(self):
        """Find the shortest path from A to C in the simple triangle.
        Going directly from A -> C has a cost of 5,
        however going from A -> B has a cost of 1, and B -> C has a cost of 2,
        so the patch A -> B -> C has a lower total cost of 3.

        [A] ---(2)-- [B]
           \          |
            \         |
            (5)      (2)
               \      |
                \     |
                  \   |
                    \[C]  

        """
        self.g.add_node(self.node_5)
        self.g.add_edge(self.edge_15)
        self.g.add_edge(self.edge_25)

        path, distance = self.g.distance(self.node_1, self.node_5)
        assert path == [self.node_1.uid(), self.node_2.uid(), self.node_5.uid()]
        assert distance == 3

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls):
        cls.tearDownGraph()
