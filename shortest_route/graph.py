from typing import Set, Union, List, Generator, Iterator, Iterable, Tuple
from collections import namedtuple
import hashlib

from elements import Node, Edge

class CityMapperGraph(object):
    
    def __init__(self, 
                 nodes: Union[Iterator[Node]]=None, 
                 edges: Union[Iterator[Edge]]=None):
        self.nodes = set(nodes) if nodes else set()
        self.edges = set(edges) if edges else set()
        self.adjacency = {}
        self.inverted_edges = {}
        self._populate_indexes()
        
    def make_edge_id(self, src: Union[str, int, Node], dst: Union[str, int, Node]) -> str:
        if isinstance(src, Node):
            src = src.id
        if isinstance(dst, Node):
            dst = dst.id
        return hashlib.sha256("{}_{}".format(str(src),str(dst)).encode()).hexdigest()
        
    def _populate_indexes(self, edges: Iterable[Edge]=None):
        edges = edges if edges else self.edges
        for e in self.edges:
            # Populate adjacency list for source -> destination
            if e.src in self.adjacency:
                self.adjacency[e.src].update([e.dst])
            else:
                self.adjacency[e.src] = set([e.dst])
            # Populate adjacency list for destination -> source
            if e.dst in self.adjacency:
                self.adjacency[e.dst].update([e.src])
            else:
                self.adjacency[e.dst] = set([e.src])
            if e.id not in self.inverted_edges:
                self.inverted_edges[e.id] = [e.src, e.dst, e.length]
                
    def add_nodes(self, nodes: Union[Node,
                                     str, 
                                     int, 
                                     Iterable[Union[Node, str, int]]]
                 ):
        if isinstance(nodes, (str, int)):
            node = Node(id=int(nodes))
            self.nodes.update([node])
        elif isinstance(nodes, Iterable):
            for node in nodes:
                if isinstance(node, (int, str)):
                    node = Node(id=int(node))
                self.nodes.update([node])
                
    def add_edges(self, edges: Union[Edge, Iterable[Edge]]):
        if isinstance(edges, Edge):
            self.add_nodes([edges.src, edges.dst])
            self.edges.update([edges])
            self._populate_indexes([edges])
        elif isinstance(edges, Iterable):
            for edge in edges:
                self.add_nodes([edge.src, edge.dst])
                self.edges.update([edge])
            self._populate_indexes([edge])

    def find_nodes(self, nodes: Union[str, 
                                     int, 
                                     Iterable[Union[Node, str, int]]]
                  ) -> Generator[bool, None, int]:
        count = 0
        if isinstance(nodes, (str, int)):
            if Node(id=int(nodes)) in self.nodes:
                count += 1
                yield True
        elif isinstance(nodes, Node):
            if nodes in self.nodes:
                count += 1
                yield True
        elif isinstance(nodes, Iterable):
            for node in nodes:
                if isinstance(node, (str, int)):
                    node = Node(id=int(node))
                if node in self.nodes:
                    count += 1
                    yield True
        return count

    def find_edges(self, edges: Union[str,
                                     Tuple[Union[str, Node], Union[str, Node]],
                                     Iterable[Union[Edge, str]]]
                 ) -> Generator[bool, None, int]:
        count = 0
        if isinstance(edges, str):
            if edges in self.inverted_edges:
                count += 1
                yield True
        elif isinstance(edges, Edge):
            if edges.id in self.inverted_edges:
                count += 1
                yield True
        elif isinstance(edges, Tuple):
            edge_id = self.make_edge_id(edges[0], edges[1])
            if edge_id in self.inverted_edges:
                count += 1
                yield True            
        elif isinstance(edges, Iterable):
            for edge in edges:
                if isinstance(edge, Edge):
                    edge = edge.id
                if isinstance(edge, Tuple):
                    edge = self.make_edge_id(edge[0], edge[1])
                if edge in self.inverted_edges:
                    count += 1
                    yield True
        return count