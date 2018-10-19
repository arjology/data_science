import hashlib
from lib.graph import CityMapperGraph
from lib.elements import Node, Edge
from graph import CityMapperGraph
from elements import Node, Edge

nodes = []
edges = []

g = CityMapperGraph()
nodeTrue = Node(id=316319897)
nodeFalse = Node(0)
edgeTrue = Edge(id="1c0597617b180d1bbe34e89bdd5370d923428868554c7190274535614b806d33", 
                src=Node(id=316319897), 
                dst=Node(316319936), 
                length=121)
edgeFalse = Edge(id="false", src=Node(id=0), dst=Node(id=1), length=0)
srcNodeTrue, dstNodeTrue = edgeTrue.src, edgeTrue.dst

g.add_nodes(1084140204)
g.add_nodes([nodeTrue, srcNodeTrue, dstNodeTrue])
g.add_edges(edgeTrue)

