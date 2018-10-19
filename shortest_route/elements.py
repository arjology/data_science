from typing import Union, NamedTuple, Tuple


NodeId = NamedTuple("NodeId", [("id", int)])
EdgeId = NamedTuple("EdgeId", [("id", str), ("src_id", NodeId), ("dst_id", NodeId), ("length", Union[float, int])])

class GraphElement(object):
    Properties = NamedTuple("GraphElementProperties", [])

    def __init__(self, id: Union[int, str], props: NamedTuple=None):
        """Construct graph element with properties."""
        self.id = id
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
        return hash(self.id)

class Node(GraphElement):

    Properties = NamedTuple("NodeProperties", [
        ("id", int)
        ])
    Properties.__new__.__defaults__ = (None,) * len(Properties._fields)

    def __init__(self, id: int, **kwargs):
        """Create node of graph.

        Args:
            kwargs: Property values
        """        
        super().__init__(id=id, props=self.Properties(**kwargs))
        self.id = id
        
    def __str__(self) -> str:
        """Get human-readable string representation."""
        return "Node:\n  {sup}".format(
            sup=super().__str__().replace("\n", "\n" + " " * 2)
        )


class Edge(GraphElement):
    Properties = NamedTuple("EdgeProperties", [
        ("id", str),
        ("src", Node),
        ("dst", Node),
        ("length", Union[int, float])
    ])
    Properties.__new__.__defaults__ = (None,) * len(Properties._fields)

    def __init__(self, id: str, src: Node, dst: Node, **kwargs):
        """Construct graph edge.

        Args:
            kwargs: Property values
        """
        assert isinstance(src, Node)
        assert isinstance(dst, Node)
        super().__init__(id=id, props=self.Properties(**kwargs))
        self.id = id
        self.src = src
        self.dst = dst


    def __str__(self) -> str:
        """Get human-readable string representation."""
        return "Edge:\n  {sup}\n  From:\n{src}\n  To:\n{dst}".format(
            sup=super().__str__().replace("\n", "\n" + " " * 2),
            src=str(self.src).replace("\n", "\n" + " " * 4),
            dst=str(self.dst).replace("\n", "\n" + " " * 4)
        )