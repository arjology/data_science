import argparse
import hashlib

from shortest_route.lib.graph import CityMapperGraph as graph
from shortest_route.lib.elements import Node, Edge

def main():
    arg_parser = argparse.ArgumentParser(description='Set the source and destination nodes.')
    arg_parser.add_argument("-i", "--input_data",
                            dest="input",
                            help="Input data to populate graph with.",
                            type=str,
                            metavar="<input>")
    arg_parser.add_argument("-s", "--source",
                            dest="src",
                            help="Source node ID.",
                            type=str,
                            metavar="<source>")
    arg_parser.add_argument("-d", "--destination",
                            dest="dst",
                            help="Destination node ID.",
                            type=str,
                            metavar="<destination>")
    args = arg_parser.parse_args()

    nodes = []
    edges = []

    with open(args.input, 'r') as f:
        data = f.readlines()
        
    # Number of vertices
    N = int(data[0].strip("\n"))
    for i in range(1, N+1):
        nodes.append(Node(id=int(data[i].strip("\n"))))

    # Number of edges
    E = int(data[N+1])
    for i in range(N+2, E+1):
        src, dst, length = data[i].strip("\n").split(" ")
        e_id = hashlib.sha256("{}_{}".format(src,dst).encode()).hexdigest()
        edges.append(Edge(id=e_id, src=Node(id=int(src)), dst=Node(id=int(dst)), length=int(length)))


if __name__ == '__main__':
    main()
