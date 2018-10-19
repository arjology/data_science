from typing import Union, NamedTuple, Tuple

NodeId = NamedTuple("NodeId", [("uid", int)])
EdgeId = NamedTuple("EdgeId", [("uid", str), ("src_uid", NodeId), ("dst_uid", NodeId), ("length", Union[float, int])])


class GraphElement(object):
    Properties = NamedTuple("GraphElementProperties", [])

    def __init__(self, uid: Union[int, str], props: NamedTuple=None):
        """Construct graph element with properties."""
        self.uid = uid
        self.props = props or self.Properties()  # pylint: disable=E1102
        paths = {}
        names = self.properties()

    def __str__(self) -> str:
        """Get human-readable string representation."""
        out = "\nProps:"
        for name in self.defined():
            out += "\n    - {name}: {value}".format(
                name=name,
                value=repr(self.property_value(name))
            )
        return out

    def uid(self) -> Union[VertexId, None]:
        """Get hashable graph element ID."""
        raise NotImplementedError("Must be implemented by subclass")

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
        return self.props == other.props

    def __hash__(self):
        return hash(self.uid)


class Node(GraphElement):

    Properties = NamedTuple("NodeProperties", [
        ("uid", int)
        ])
    Properties.__new__.__defaults__ = (None,) * len(Properties._fields)

    def __init__(self, uid: int, **kwargs):
        """Create node of graph.

        Args:
            kwargs: Property values
        """
        super().__init__(uid=uid, props=self.Properties(**kwargs))
        self.uid = uid

    def uid(self) -> Union[NodeId, None]:
        """Get geo-image vertex ID."""
        if not self.uid:
            return None
        return NodeId(uid=self.uid)

    def __str__(self) -> str:
        """Get human-readable string representation."""
        return "Node:\n  {sup}".format(
            sup=super().__str__().replace("\n", "\n" + " " * 2)
        )


class Edge(GraphElement):
    Properties = NamedTuple("EdgeProperties", [
        ("uid", str),
        ("src", Node),
        ("dst", Node),
        ("length", Union[int, float])
    ])
    Properties.__new__.__defaults__ = (None,) * len(Properties._fields)

    def __init__(self, uid: str, src: Node, dst: Node, **kwargs):
        """Construct graph edge.

        Args:
            kwargs: Property values
        """
        assert isinstance(src, Node)
        assert isinstance(dst, Node)
        super().__init__(uid=uid, props=self.Properties(**kwargs))
        self.uid = uid
        self.src = src
        self.dst = dst

    def uid(self) -> Union[EdgeId, None]:
        """Get hashable edge ID."""
        assert self.src is not None
        assert self.dst is not None
        src_uid = self.src.uid()
        dst_uid = self.dst.uid()
        if src_id is None or dst_id is None:
            return None
        return EdgeId(uid=self.uid, src_uid=src_uid, dst_uid=dst_uid, length=self.length)

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
