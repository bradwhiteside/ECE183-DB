import yaml
from path_planner.graph import Graph, Node
from path_planner.grid import Grid
import numpy as np
import cv2
from queue import PriorityQueue
from abc import ABC, abstractmethod


class PathFinder(ABC):
    @abstractmethod
    def find_path(self, cur, target, min_alt=10):
        pass


# Based off https://likegeeks.com/python-dijkstras-algorithm/
class Dijkstra(PathFinder):
    def __init__(self, filename):
        self.graph = self.load_graph_from_yaml(filename)
        self.costs = {}
        self.parents = {}

    def init_costs(self, cur, graph):
        for v in graph.V:
            if cur == v:
                self.costs[v] = 0
            else:
                self.costs[v] = np.inf

    def load_graph_from_yaml(self, filename):
        G = Graph()
        with open(filename) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

            nodes = data["Nodes"]
            edges = data["Edges"]
            for v in nodes:
                G.insert_node(nodes[v]["pos"], nodes[v]["type"])

            for e1 in edges:
                for e2 in edges[e1]:
                    node1 = Node(nodes[e1]["pos"], nodes[e1]["type"])
                    node2 = Node(nodes[e2]["pos"], nodes[e2]["type"])
                    G.insert_edge(node1, node2, edges[e1][e2])

        G.construct_graph_dict()
        return G

    def point2node(self, pt, dist_threshold):
        if type(pt) is list:
            pt = np.array(pt)

        min_dist = dist_threshold
        min_node = None
        for v in self.graph.V:
            dist = np.linalg.norm(pt - v.pos)
            if dist <= min_dist:
                min_node = v

        return min_node

    def find_path(self, start, target, min_alt):
        self.init_costs(start, self.graph)
        self.parents = {}
        adj_list = self.graph.graph_dict
        nextNode = start

        while nextNode != target:
            for neighbor in adj_list[nextNode]:
                if adj_list[nextNode][neighbor] + self.costs[nextNode] < self.costs[neighbor]:
                    self.costs[neighbor] = adj_list[nextNode][neighbor] + self.costs[nextNode]
                    self.parents[neighbor] = nextNode
                del adj_list[neighbor][nextNode]
            del self.costs[nextNode]
            nextNode = min(self.costs, key=self.costs.get)

        node = target
        path = []
        while True:
            pt = node.pos
            pt[2] = pt[2] + min_alt
            path.insert(0, pt)
            node = self.parents[node]
            if node == start:
                break

        return path


# Based off https://github.com/alpesis-robotics/drone-planner
class A_star(PathFinder):
    def __init__(self, image_name, DEBUG=False):
        self.image = image_name
        self.DEBUG = DEBUG
        self.grid = Grid(image_name, DEBUG)

    def h1(self, cur, target):
        return np.linalg.norm(cur.pos[:2] - target.pos[:2])

    def h2(selfself, cur, target):
        return np.linalg.norm(cur.pos - target.pos)

    def collinearity_check(self, p1, p2, p3, epsilon=5):
        m = np.concatenate((p1, p2, p3)).reshape(-1, 3)
        if p1[2] <= p2[2] <= p3[2]:
            m[:, 2] = 1
        det = np.linalg.det(m)
        return abs(det) < epsilon

    def prune_path(self, path):
        print("Pruning path")
        print("Path is %d points long" % len(path))
        i = 0
        pruned_path = path
        while i < (len(pruned_path) - 2):
            if self.collinearity_check(pruned_path[i],
                                       pruned_path[i + 1],
                                       pruned_path[i + 2]):
                del pruned_path[i + 1]
            else:
                i += 1
        print("Pruned Path is %d points long" % len(pruned_path))
        return pruned_path

    def find_path(self, start, target, alt=10, h=h1):
        start = self.grid.get_cell(start[0], start[1], alt)
        target = self.grid.get_cell(target[0], target[1], alt)
        queue = PriorityQueue()
        queue.put((0, start))
        visited = set()
        visited.add(start)

        branch = {}
        found = False

        while not queue.empty():
            _, cur = queue.get()
            #print("Current cell = {}".format(cur.pos))
            if cur == start:
                cur_cost = 0
            else:
                cur_cost = branch[cur][0]

            if cur == target:
                print('Found a path.')
                found = True
                break
            else:
                for next in self.grid.get_neighbors(cur, alt):
                    if next in visited:
                        continue

                    branch_cost = cur_cost + next.cost
                    queue_cost = branch_cost + h(self, next, target)
                    branch[next] = (branch_cost, cur)
                    queue.put((queue_cost, next))
                    visited.add(next)

        if not found:
            print('**********************')
            print('Failed to find a path!')
            print('**********************')
            exit(1)

        # retrace steps
        path = []
        path.append(target.pos)
        path_cost = branch[target][0]
        n = target
        while branch[n][1] != start:
            path.append(branch[n][1].pos)
            n = branch[n][1]
        path.append(branch[n][1].pos)

        pruned_path = self.prune_path(path[::-1])
        if self.DEBUG:
            path_image = draw_path(self.image, pruned_path)
            path_image_name = self.image[:-4] + "Path.png"
            cv2.imwrite(path_image_name, path_image)

        return pruned_path, path_cost


def draw_path(image_name, path):
    length = len(path)
    img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    color = (0, 255, 255, 255)  # yellow
    for i in range(length-1):
        img = cv2.line(img, (path[i][0], path[i][1]), (path[i+1][0], path[i+1][1]), color, 4)
    return img


if __name__ == '__main__':
    map_image = "venues\Test\Test.png"
    a = A_star(map_image, True)
    path, cost = a.find_path((19,24), (211, 20))
    print("Path cost: %d" % cost)
    print(path)

    #path_image = draw_path(map_image, path)
    #cv2.imwrite("venues\Test\TestPath.png", path_image)
    exit(0)



