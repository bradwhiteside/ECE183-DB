import yaml
import graph
import numpy as np
from abc import ABC, abstractmethod

class PathFinder(ABC):
    @abstractmethod
    def find_path(self, cur, target, graph):
        pass

# Based off https://likegeeks.com/python-dijkstras-algorithm/
class Dijkstra(PathFinder):
    def __init__(self):
        self.costs = {}
        self.parents = {}

    def init_costs(self, cur, graph):
        for v in graph.V.values():
            if cur.name == v.name:
                self.costs[v.name] = 0
            else:
                self.costs[v.name] = np.inf

    def find_path(self, cur, target, graph):
        self.init_costs(cur, graph)
        adj_list = graph.graph_dict
        nextNode = cur.name

        while nextNode != target.name:
            for neighbor in adj_list[nextNode]:
                if adj_list[nextNode][neighbor] + self.costs[nextNode] < self.costs[neighbor]:
                    self.costs[neighbor] = adj_list[nextNode][neighbor] + self.costs[nextNode]
                    self.parents[neighbor] = nextNode
                del adj_list[neighbor][nextNode]
            del self.costs[nextNode]
            nextNode = min(self.costs, key=self.costs.get)

        node = target.name
        path = []
        while True:
            path.insert(0, node)
            node = self.parents[node]
            if node == cur.name:
                break

        return path


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


def point2node(pt, graph, dist_threshold):
    if type(pt) is list:
        pt = np.array(pt)
        
    for v in graph.V:
        if np.linalg.norm(pt - graph.V[v].pos) <= dist_threshold:
            return graph.V[v]
        
    return None


def node2point(name, graph):
    for v in graph.V:
        if name == v:
            return graph.V[v].pos
    return None


# take in two 3D points, a graph, and an algorithm to use
# round input pts to closest node on graph if within dist_threshold
# output a list of 3D points representing a path between those points
def find_path(algorithm: PathFinder, cur_pos, target_pos, graph, dist_threshold=0):
    cur_node = point2node(cur_pos, graph, dist_threshold)
    target_node = point2node(target_pos, graph, dist_threshold)
    if cur_node is None:
        raise ValueError("Current position does not correspond to any node on the graph")
    if target_node is None:
        raise ValueError("Target position does not correspond to any node on the graph")

    return algorithm.find_path(cur_node, target_node, graph)

# for testing only
if __name__ == '__main__':
    filename = "venues\RoseBowl.yaml"
    G = load_graph(filename)
    start = node2point('C1', G)
    target = node2point('D5', G)
    alg = Dijkstra()
    path = find_path(alg, start, target, G, 1)
    print(path)
