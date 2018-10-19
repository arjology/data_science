from typing import Set, Union, List, Generator, Iterator, Iterable, Tuple
from collections import namedtuple
import hashlib
from pathlib import Path
import pickle

from shortest_route.lib.elements import Node, Edge
from shortest_route.lib.utilities import optmap


class Graph(object):
    """In-memory graph.

    This image graph implementation can be used as an in-memory graph, where graph nodes
    and edges must be inserted into the graph before this data is required. Alternatively,
    this data can be loaded from a binary file to which a previous in-memory graph instance has
    been saved after serialization, e.g., using Python's pickle module. This may be useful for
    testing, but a proper graph database with efficient search indices must be used for production.
    """
    def __init__(self,
                 nodes: Union[Iterator[Node], Node] = None,
                 edges: Union[Iterator[Edge], Edge] = None):
        """Initialize graph interface.

        Args:
            nodes: Collection of or individual node
            egdes: Collection of or individual edge
        """
        self._nodes = nodes if nodes else []
        self._edges = edges if edges else []
        self._adjacency = {}
        self._index = {}
        self._indverted_edges = {}
        self._path = None

    def is_empty(self) -> bool:
        """Whether graph is empty."""
        return not self._nodes

    def number_of_vertices(self) -> int:
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
            src = src.id
        if isinstance(dst, Node):
            dst = dst.id
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
            "indverted_edges": self._indverted_edges
            }

    def __setstate__(self, values):
        """Set object state from unpickled dictionary."""
        self.__init__(nodes=self._nodes, edges=self._edges)
        self._nodes = values["nodes"]
        self._edges = values["edges"]
        self._index = values["index"]
        self._indverted_edges = values["indverted_edges"]

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
    # Upsert vertex or edge

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
            self._index[uid] = idx
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
        return graph_edge

    def _populate_indexes(self, nodes: Iterator[Node]=None, edges: Iterator[Edge]=None):
        nodes = nodes if nodes else self._nodes if self._nodes else []
        edges = edges if edges else self._edges if self._edges else []
        self.add_node(nodes)
        self.add_edge(edges)
        for edge in edges:
            # Populate adjacency list for source -> destination
            if edge.src.uid() in self._adjacency:
                self._adjacency[edge.src.uid()].update([edge.dst.uid()])
            else:
                self._adjacency[edge.src.uid()] = set([edge.dst.uid()])
            # Populate adjacency list for destination -> source
            if edge.dst.uid() in self._adjacency:
                self._adjacency[edge.dst.uid()].update([edge.src.uid()])
            else:
                self._adjacency[edge.dst.uid()] = set([edge.src.uid()])
            if edge.id not in self._indverted_edges:
                self._indverted_edges[edge.uid] = edge.uid()

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
                    idx = self._index.get(uid, -1)
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


class CityMapperGraph(Graph):
    def __init__(self,
                 nodes: Union[Iterator[Node]]=None,
                 edges: Union[Iterator[Edge]]=None):
        self._edges = edges if edges else []
        self._nodes = nodes if nodes else []
        self._adjacency = {}
        self._index = {}
        self._indverted_edges = {}
        self._path = None
