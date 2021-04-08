import yaml
import graph
from abc import ABC, abstractmethod

class PathFinder(ABC):
    @abstractmethod
    def find_path(self, cur, target, graph):
        pass


class Dijkstra(PathFinder):
    def __init__(self):
        pass

    def find_path(self, cur, target, graph):
        return 0


def load_graph(filename):
    G = graph.Graph()
    with open(filename) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

        nodes = data["Nodes"]
        for v in nodes:
            G.insert_node(v, nodes[v]["pos"], nodes[v]["type"])

        edges = data["Edges"]
        for e1 in edges:
            for e2 in edges[e1]:
                G.insert_edge(e1, e2, edges[e1][e2])

    G.construct_graph_dict()
    return G


def find_path(algorithm, cur_pos, target_pos, graph):
    return algorithm.find_path(cur_pos, target_pos, graph)



