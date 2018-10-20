from copy import copy as make_copy
from typing import Union, NamedTuple, Tuple

Label = str

NodeId = NamedTuple("NodeId", [("uid_", Union[int, str])])
NodeId.__new__.__defaults__ = (None,) * len(NodeId._fields)

EdgeId = NamedTuple("EdgeId", [("src_id", NodeId), ("dst_id", NodeId)])

GraphElementId = Union[NodeId, EdgeId]

class GraphElement(object):
    Properties = NamedTuple("GraphElementProperties", [])

    def __init__(self, props: NamedTuple=None):
        """Construct graph element with properties."""
        self.props = props or self.Properties()  # pylint: disable=E1102
        paths = {}
        names = self.properties()

    def __str__(self) -> str:
        """Get human-readable string representation."""
        out = "Label: " + self.label()
        out += "\nProps:"
        for name in self.defined():
            out += "\n    - {name}: {value}".format(
                name=name,
                value=repr(self.property_value(name))
            )
        return out

    def uid(self) -> Union[NodeId, EdgeId, None]:
        """Get hashable graph element ID."""
        raise NotImplementedError("Must be implemented by subclass")

    @classmethod
    def from_uid(cls, uid: GraphElementId) -> 'GraphElement':
        """Construct graph element from its ID object."""
        raise NotImplementedError("Must be implemented by subclass")

    @classmethod
    def label(cls) -> Label:
        """Get label of graph element which determines its type / properties."""
        return cls.__name__

    def defined(self) -> Tuple[str]:
        """Get tuple of defined property names, i.e., with value that is not None."""
        return tuple([name for name in self.properties() if getattr(self, name) is not None])

    def property(self, name, value):
        """Set property of graph element."""
        self.props = self.props._replace(**{name: value})
        return self

    def properties(self) -> Tuple[str]:
        """Get tuple of property names."""
        return self.props._fields

    def property_type(self, name: str) -> type:
        """Get type of property."""
        return self.props._field_types[name]

    def property_value(self, name: str) -> object:
        """Get value of property."""
        return getattr(self.props, name)

    def match(self, other: object) -> bool:
        """Test if graph element properties of given element matches those defined by this instance."""
        if type(self) is not type(other):
            return False
        for name in self.defined():
            try:
                value = other.property_value(name)
            except AttributeError:
                value = None
            if value != self.property_value(name):
                return False
        return True

    def update(self, other: 'GraphElement') -> 'GraphElement':
        """Update those properties defined by other graph element.

        Undefined properties of the other graph element are not copied.
        Note that this function makes only a shallow copies of property values.
        """
        assert type(self) is type(other)
        props = {}
        for name in other.defined():
            value = other.property_value(name)
            props[name] = make_copy(value)
        self.props = self.props._replace(**props)
        return self

    def __setattr__(self, name: str, value: object):
        """Set graph element property value."""
        if hasattr(self, "props") and name in self.props._fields:
            self.property(name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str):
        """Get graph element property value."""
        if name == "props":
            raise AttributeError(
                "Expected object to have 'props' attribute;"
                " ensure to call GraphElement.__init__ in subclass __init__"
            )
        if name.startswith("__"):
            return super().__getattr__(name)
        return self.property_value(name)

    def __setitem__(self, name: str, value: object):
        """Set graph element property value."""
        self.property(name, value)

    def __getitem__(self, name: str):
        """Get graph element property value."""
        return self.property_value(name)

    def __eq__(self, other: object) -> bool:
        """Test equality of graph element properties."""
        if not isinstance(other, GraphElement):
            return False
        if self.label() != other.label():
            return False
        return self.props == other.props
    
    def copy(self) -> 'GraphElement':
        """Make shallow copy of graph element properties."""
        return self.update(self)

    def __hash__(self):
        return hash(self.uid())

GraphElementProperties = GraphElement.Properties  # required by pickle module        


class Node(GraphElement):

    Properties = NamedTuple("NodeProperties", [
        ("uid_", int),
        ])
    Properties.__new__.__defaults__ = (None,) * len(Properties._fields)

    def __init__(self, **kwargs):
        """Create node of graph.

        Args:
            kwargs: Property values
        """
        super().__init__(props=self.Properties(**kwargs))

    def uid(self) -> Union[NodeId, None]:
        """Get node ID."""
        if not self.uid_:
            return None
        return self.uid_

    @classmethod
    def from_uid(cls, uid: NodeId) -> 'Node':
        """Construct node from NodeId."""
        return cls(uid_=uid.uid_)

    @classmethod
    def from_arg(cls, node_or_uid: Union['Node', NodeId, None]) -> 'Node':
        """Construct node from Node."""
        if node_or_uid is None:
            return cls()
        if isinstance(node_or_uid, Node):
            return node_or_uid
        if isinstance(node_or_uid, NodeId):
            return cls.from_uid(node_or_uid)
        raise TypeError("Invalid argument type, must be either Node or NodeId")

    def __str__(self) -> str:
        """Get human-readable string representation."""
        return "Node:\n  {sup}".format(
            sup=super().__str__().replace("\n", "\n" + " " * 2)
        )


class Edge(GraphElement):
    Properties = NamedTuple("EdgeProperties", [
        ("uid_", str),
        ("length", Union[int, float]),
    ])
    Properties.__new__.__defaults__ = (None,) * len(Properties._fields)

    def __init__(self, 
                 src: Union[Node, NodeId],
                 dst: Union[Node, NodeId],
                 **kwargs):
        """Construct graph edge.

        Args:
            src: Source node. Must not be ``None``.
            dst: Destination node. Must not be ``None``.
            props: Edge properties.
        """
        self.src = Node.from_arg(src)
        self.dst = Node.from_arg(dst)
        if self.src.uid() is None:
            if self.dst.uid() is not None:
                self.src, self.dst = self.dst, self.src
        elif self.dst.uid() is not None:
            if self.src.uid() > self.dst.uid():
                self.src, self.dst = self.dst, self.src
        assert isinstance(self.src, Node)
        assert isinstance(self.dst, Node)
        super().__init__(props=self.Properties(**kwargs))

    def uid(self) -> Union[EdgeId, None]:
        """Get hashable edge ID."""
        assert self.src is not None
        assert self.dst is not None
        src_id = self.src.uid()
        dst_id = self.dst.uid()
        if src_id is None or dst_id is None:
            return None
        return EdgeId(src_id=src_id, dst_id=dst_id)

    @classmethod
    def from_uid(cls, uid: EdgeId) -> 'Edge':
        """Construct edge from its ID object."""
        assert isinstance(uid, tuple)
        assert len(uid) == 2
        assert isinstance(uid[0], NodeId)
        assert isinstance(uid[1], NodeId)
        return cls(uid[0], uid[1])

    def match(self, other: object) -> bool:
        # """Test if node and properties of given edge matches those defined by this instance."""
        # if not GraphElement.match(self, other):
        #     return False
        if self.src.uid() is not None:
            if self.dst.uid() is None:
                if self.src.uid() in (other.src.uid(), other.dst.uid()):
                    return True
            else:
                for src_id, dst_id in [(other.src.uid(), other.dst.uid()), (other.dst.uid(), other.src.uid())]:
                    if self.src.uid() == src_id and self.dst.uid() == dst_id:
                        return True
        return False

    def length(self) -> Union[float, int, None]:
        """Length (distance) of edge"""
        return self.length

    def __str__(self) -> str:
        """Get human-readable string representation."""
        return "Edge:\n  {sup}\n  From:\n{src}\n  To:\n{dst}".format(
            sup=super().__str__().replace("\n", "\n" + " " * 2),
            src=str(self.src).replace("\n", "\n" + " " * 4),
            dst=str(self.dst).replace("\n", "\n" + " " * 4)
        )
