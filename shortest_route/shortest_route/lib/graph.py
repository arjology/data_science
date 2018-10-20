from typing import Set, Union, List, Generator, Iterator, Iterable, Tuple
from collections import namedtuple, deque
import hashlib
from pathlib import Path
import pickle
import math

from shortest_route.lib.elements import Node, NodeId, Edge
from shortest_route.lib.utilities import optmap


class Graph(object):
    """In-memory graph.

    This image graph implementation can be used as an in-memory graph, where graph nodes
    and edges must be inserted into the graph before this data is required. Alternatively,
    this data can be loaded from a binary file to which a previous in-memory graph instance has
    been saved after serialization, e.g., using Python's pickle module. This may be useful for
    testing, but a proper graph database with efficient search indices must be used for production.
    """
    def __init__(self):
        """Initialize graph interface.

        Args:
            nodes: Collection of or individual node
            egdes: Collection of or individual edge
        """
        self._nodes = []
        self._edges = []
        self._adjacency = {}
        self._index = {}
        self._indverted_edges = {}
        self._path = None

    def is_empty(self) -> bool:
        """Whether graph is empty."""
        return not self._nodes

    def count(self) -> int:
        """Total number of nodes."""
        return len(self._nodes)

    def nodes(self) -> Set[Node]:
        """ Get list of graph nodes."""
        return self._nodes

    def edges(self) -> List[Node]:
        """Get list of graph edges."""
        return self._edges

    def clear(self) -> object:
        """Drop nodes and edges."""
        self._nodes = []
        self._edges = []
        self._adjacency = {}
        self._index = {}
        self._indverted_edges = {}
        return self

    def clear_edges(self) -> object:
        """Drop edges."""
        self._edges = {}
        self._indverted_edges = {}
        self._adjacency = {}
        del self._index[Edge.label()]
        return self

    def set_path(self, path: Union[Path, str]) -> 'Graph':
        """Set path of pickle file used to make absolute local file paths relative and vice versa."""
        path = optmap(lambda path: Path(path).absolute(), path)
        if self._path != path:
            self._path = path
        return self

    def get_path(self) -> Union[Path, None]:
        """Get local file path of graph if any."""
        return self._path

    def make_edge_id(self, src: Union[str, int, Node], dst: Union[str, int, Node]) -> str:
        if isinstance(src, Node):
            src = src.uid()
        if isinstance(dst, Node):
            dst = dst.uid()
        return hashlib.sha256("{}_{}".format(str(src), str(dst)).encode()).hexdigest()

    # ----------------------------------------------------------------------------------------------
    # Comparison

    def __eq__(self, other: object) -> bool:
        """Compare graph to another."""
        return (
            self._nodes == other._nodes
            and self._edges == other._edges
            and self._index == other._index
            and self._indverted_edges == other._indverted_edges
            )

    # ----------------------------------------------------------------------------------------------
    # Pickling

    def __getstate__(self):
        """Get object state as dictionary for pickling."""
        return {
            "nodes": self._nodes,
            "edges": self._edges,
            "index": self._index,
            "indverted_edges": self._indverted_edges,
            "adjacency": self._adjacency
            }

    def __setstate__(self, values):
        """Set object state from unpickled dictionary."""
        self.__init__()
        self._nodes = values["nodes"]
        self._edges = values["edges"]
        self._index = values["index"]
        self._indverted_edges = values["indverted_edges"]
        self._adjacency = values["adjacency"]

    def save(self):
        """Save graph."""
        if self._path is not None:
            self.dump(self._path)

    def dumps(self) -> bytes:
        """Serialize graph."""
        return pickle.dumps(self)

    @classmethod
    def loads(cls, data: bytes) -> 'Graph':
        """Deserialize graph."""
        return pickle.loads(data)

    def dump(self, path: Union[Path, str]):
        """Write graph to binary file."""
        path = Path(path).absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "wb") as fobj:
            fobj.write(self.dumps())

    @classmethod
    def load(cls, path: Union[Path, str]) -> 'Graph':
        with open(str(path), "rb") as fobj:
            return cls.loads(fobj.read())

    # ----------------------------------------------------------------------------------------------
    # Upsert node or edge

    def add_node(self, node: Node):
        """Upsert graph node and return inserted shallow copy."""
        assert isinstance(node, Node)
        lbl = node.label()
        uid = node.uid()
        if lbl not in self._index:
            self._index[lbl] = {}
        idx = self._index[lbl].get(uid, len(self._nodes))
        if idx < len(self._nodes):
            graph_node = self._nodes[idx]
            graph_node.update(node)
        else:
            graph_node = node.copy()
            self._nodes.append(graph_node)
            self._index[lbl][uid] = idx
        return graph_node

    def add_edge(self, edge: Edge) -> Edge:
        """Upsert graph edge and return inserted shallow copy."""
        assert isinstance(edge, Edge)
        lbl = edge.label()
        src, dst = edge.uid()
        if lbl not in self._index:
            self._index[lbl] = {src: {}}
        elif src not in self._index[lbl]:
            self._index[lbl][src] = {}
        num = len(self._edges)
        idx = self._index[lbl][src].get(dst, num)
        if idx < num:
            graph_edge = self._edges[idx]
            graph_edge.update(edge)
        else:
            graph_edge = edge.copy()
            self._edges.append(graph_edge)
            self._index[lbl][src][dst] = idx

        # Populate adjacency list for source -> destination
        if src in self._adjacency:
            self._adjacency[src].update([dst])
        else:
            self._adjacency[src] = set([dst])
        # Populate adjacency list for destination -> source
        if dst in self._adjacency:
            self._adjacency[dst].update([src])
        else:
            self._adjacency[dst] = set([src])
        if edge.uid not in self._indverted_edges:
            self._indverted_edges[edge.props.uid_] = edge.uid()

        return graph_edge

    # ----------------------------------------------------------------------------------------------
    # Find vertices or edges

    def find_nodes(self, *nodes: Node) \
            -> Generator[Node, None, int]:
        """Find elements in the graph which share the specified subset of properties with the given example.

        Args:
            nodes: Example nodes with properties of similar nodes to look for.

        Returns:
            Generator of matching nodes whose return value is the total number of results.
        """
        count = 0
        for node in nodes:
            if node is not None:
                assert isinstance(node, Node)
                index = self._index.get(node.label(), {})
                uid = node.uid()
                if uid is None:
                    candidates = [self._nodes[idx] for idx in index.values()]
                else:
                    idx = index.get(uid, -1)
                    candidates = [] if idx < 0 else [self._nodes[idx]]
                for candidate in candidates:
                    if node.match(candidate):
                        yield candidate.copy()
                        count += 1
        return count

    def find_edges(self, *edges: Edge, undirected: bool=True) \
            -> Generator[Edge, None, int]:
        """Find edges in the graph which share the specified subset of properties with the given example.

        Args:
            edges: Example edges with properties of similar elements to look for.
            undirected: Ignore order of 'src' and 'dst' vertices of the edge.

        Returns:
            Generator of matching edges whose return value is the total number of results.
        """
        # unused argument 'select': pylint: disable=W0613
        count = 0
        for edge in edges:
            if edge is not None:
                assert isinstance(edge, Edge)
                index = self._index.get(edge.label(), {})
                src = edge.src.uid()
                dst = edge.dst.uid()
                if src is None and dst is None:
                    candidates = range(len(self._edges))
                else:
                    candidates = set()
                    for a, b in [(src, dst), (dst, src)] if undirected else [(src, dst)]:
                        if a is None:
                            for item in index.values():
                                idx = item.get(b, -1)
                                if idx != -1:
                                    candidates.add(idx)
                        else:
                            candidates.update(index.get(a, {}).values())
                for idx in candidates:
                    candidate = self._edges[idx]
                    if edge.match(candidate):
                        yield candidate.copy()
                        count += 1
        return count

    def neighbours(self, node: Union[Node, NodeId]) -> Generator[Tuple[Node, float], None, int]:
        count = 0
        uid = node.uid() if isinstance(node, Node) else node.uid_ if isinstance(node, NodeId) else None
        found = self._adjacency.get(uid)
        if found:
            for neighbour in found:
                neighbour = Node.from_uid(NodeId(uid_=neighbour))
                edge = Edge(src=node, dst=neighbour)
                cost = next(self.find_edges(edge)).props.length
                yield (neighbour, cost)
                count += 1
        return count


class CityMapperGraph(Graph):
    def __init__(self):
        super().__init__()

    # ----------------------------------------------------------------------------------------------
    # Helper function to load lists of nodes and edges

    def load_nodes_and_edges(self, nodes: List[Node]=None, edges: List[Node]=None):
        for node in nodes:
            self.add_node(node)
        for edge in edges:
            self.add_edge(edge)

    # ----------------------------------------------------------------------------------------------
    # Dijkstra's algorithm for finding the shortest paths between all nodes in the graph

    def distance(self, src: Union[Node, NodeId], dst: Union[Node, NodeId]) \
            -> Tuple[Union[deque, None], Union[float, None]]:
        """Dijkstra's shortest route algorithm

        Args:
            src: Source node
            dst: Destination node

        Returns:
            Total distance as float.
        """

        self.src = Node.from_arg(src)
        self.dst = Node.from_arg(dst)

        if self.src.uid() not in self._index[self.src.label()] \
            or self.dst.uid() not in self._index[self.dst.label()]:
            return (None, None)

        prev_nodes = {
            node.uid(): None for node in self._nodes
        }
        distances = {
            node.uid(): math.inf for node in self._nodes
        }
        distances[self.src.uid()] = 0
        nodes = [n.uid() for n in self._nodes]
        while nodes:
            curr_node = min(nodes, key=lambda node: distances[node])
            if distances[curr_node] == math.inf:
                break
            for neighbour, cost in self.neighbours(NodeId(uid_=curr_node)):
                path_cost = distances[curr_node] + cost
                if path_cost < distances[neighbour.uid()]:
                    distances[neighbour.uid()] = path_cost
                    prev_nodes[neighbour.uid()] = curr_node
            nodes.remove(curr_node)
        total_path, curr_node = deque(), self.dst.uid()
        while prev_nodes[curr_node] is not None:
            total_path.appendleft(curr_node)
            curr_node = prev_nodes[curr_node]
        if total_path:
            total_path.appendleft(curr_node)
        return total_path, distances[self.dst.uid()]
