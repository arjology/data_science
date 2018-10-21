import argparse
import hashlib
import logging

from shortest_route.lib.graph import CityMapperGraph as Graph
from shortest_route.lib.elements import Node, Edge


def main():
    arg_parser = argparse.ArgumentParser(description='Populate the graph and set the source and destination nodes.')
    arg_parser.add_argument("-l", "--logging",
                            dest="logging",
                            help="Set logging level",
                            type=str,
                            choices=["INFO", "DEBUG", "WARN"],
                            default="WARN",
                            metavar="<logging>")
    arg_parser.add_argument("-i", "--input_data",
                            dest="input",
                            help="Input data to populate graph with.",
                            type=str,
                            metavar="<input>")
    arg_parser.add_argument("-s", "--source",
                            dest="src",
                            help="Source node ID.",
                            type=int,
                            metavar="<source>")
    arg_parser.add_argument("-d", "--destination",
                            dest="dst",
                            help="Destination node ID.",
                            type=int,
                            metavar="<destination>")
    args = arg_parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=args.logging)
    logger = logging.getLogger("shortest_route")

    nodes = []
    edges = []

    logger.info("Reading data file [{}]".format(args.input))
    with open(args.input, 'r') as f:
        data = f.readlines()

    # Number of vertices
    N = int(data[0].strip("\n"))
    for i in range(1, N+1):
        nodes.append(Node(uid_=int(data[i].strip("\n"))))

    # Number of edges
    E = int(data[N+1])
    for i in range(N+2, E+1):
        src, dst, length = data[i].strip("\n").split(" ")
        edges.append(Edge(src=Node(uid_=int(src)), dst=Node(uid_=int(dst)), length=int(length)))

    logger.info("Initializing graph with {} nodes and {} edges".format(len(nodes), len(edges)))
    g = Graph()
    g.load_nodes_and_edges(nodes=nodes, edges=edges)

    logger.info("Finding shortest path between [{}] and [{}]".format(args.src, args.dst))
    path, distance = g.distance(src=Node(uid_=args.src), dst=Node(uid_=args.dst))
    logger.info("Shortest path:\n{}".format(path))
    print(distance)

if __name__ == '__main__':
    main()
