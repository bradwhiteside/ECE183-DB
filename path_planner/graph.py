import numpy as np

# Graph object contains a dict of nodes (V) and a set of edges (E)
# it constructs an adjacency list in the form of a nested dictionary
# example:  graph_dict = { 'A': {'B': 3, 'C': 5},
#                          'B': {'A': 3, 'C': 1},
#                          'C': {'A': 5, 'B': 1},
#                          'D': {'A': 4} }
#           represents 4 nodes A, B, C, D and 3 undirected edges
#           (A, B) of length 3, (A, C) of length 5, and (B, C) of length 1
#           and 1 directed edge (D, A) of length 4

class Graph:
    def __init__(self, nodes=None, edges=None, gd=None):
        self.V = nodes
        self.E = edges
        self.graph_dict = gd

    def construct_graph_dict(self):
        graph_dict = {}

        for v in self.V:
            graph_dict[v] = {}

        for e in self.E:
            if e.node2 not in graph_dict[e.node1]:
                graph_dict[e.node1][e.node2] = e.weight
            if e.node1 not in graph_dict[e.node2]:
                graph_dict[e.node2][e.node1] = e.weight

        self.graph_dict = graph_dict

    def insert_node(self, pos, type=None):
        if self.V is None:
            self.V = set()
        self.V.add(Node(pos, type))

    def insert_edge(self, node1, node2, weight):
        if self.E is None:
            self.E = set()
        self.E.add(Edge(node1, node2, weight))


class Node:
    def __init__(self, pos=None, t=None):
        if type(pos) is list:
            pos = np.array(pos)
        self.pos = pos  # 3D
        # type = 0 means transit, = 1 means pick up, = 2 means drop off
        self.type = t

    def __eq__(self, other):
        return (self.pos == other.pos).all()

    def __hash__(self):
        return hash((self.pos[0], self.pos[1], self.pos[2]))


class Edge:
    def __init__(self, node1, node2, weight):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight

    def __lt__(self, other):
        if (self.node1 == other.node1 and self.node2 == other.node2) or \
                (self.node1 == other.node2 and self.node2 == other.node1):
            return self.weight < other.weight  # compare based on weight
        else:
            raise ValueError("Cannot compare edges that aren't connecting the same nodes")

    def __eq__(self, other):
        if ((self.node1 == other.node1 and self.node2 == other.node2) or
                (self.node1 == other.node2 and self.node2 == other.node1)):
            return self.weight == other.weight
        else:
            return False

    def __hash__(self):
        return hash((self.node1, self.node2, self.weight))
