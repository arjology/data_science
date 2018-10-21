# Shortest Route
## In-memory Graph with Dijkstra's shortest route algorithm

## Table of Contents
1. [Introduction](#introduction)
2. [Installation and use](#install)
3. [Graph class](#graph)
4. [Shortest path algorithm](#path)

## Introduction <a name="introduction"></a>
This package provides a simple in-memory graph class, where graph nodes
and edges can be inserted into the graph individually or alternatively data
can be loaded from a text file, or from a binary file to which a previous 
in-memory graph instance has been saved after serialization, e.g., using
Python's pickle module.

The structure of an input text file is as follows:
```
<number of nodes>
<OSM id of node>
...
<OSM id of node>
<number of edges>
<from node OSM id> <to node OSM id> <length in meters>
...
<from node OSM id> <to node OSM id> <length in meters>
```

## Installation and use <a name="install"></a>
To install, simply execute `make install` or `make develop` to install locally
(this will create a `.egg-link` in the deployment directory back to this project 
source code directory). There are no external dependencies, only using standard 
Python 3.5+ libraries. You can also run the unit tests with `make test`.

To run the program, use the `run.sh` shell script which will print only the distance:
```
./run -i <input.dat> -s <source_node> -d <destination_node>
```

You can also enable more verbose logging by passing in `-l`/`--logging` and specifying
the logging level (`INFO`, `DEBUG`, `WARN`). This will allow you to see where it is
in the processing stage and also print the full path, not just the total distance.


## Graph class <a name="graph"></a>
Aside from adding edges and vertices, the graph class provides a search capability
for finding nodes or edges; has an inverted adjacency list; and uses the
inverted adjacency to find neighbours of a given node.

## Shortest path algorithm <a name="path"></a>
For a weighted graph such as the OSM data (where weights are distances between nodes,
represented as properties on the edges), Dijkstra's algorithm is used to find
the shortest path between given source and destination nodes. To accomplish this,
it maintains two sets, visitied and unvisited nodes. At each step, a node with the
mininum distance from source node is selected from the unvisited set.

This will work for directed and undirected graphs, but the weights must be
non-negative. As we are dealing with OSM data, it is highly unlikely to have 
negative distances between nodes. 

[Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)