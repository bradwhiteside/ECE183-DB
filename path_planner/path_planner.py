import yaml

if __name__ == '__main__':
    from graph import Graph, Node
    from grid import Grid
else:
    from path_planner.graph import Graph, Node
    from path_planner.grid import Grid
import numpy as np
import cv2
from queue import PriorityQueue
from abc import ABC, abstractmethod


def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


def collinearity(p):
    m = len(p)
    n = len(p[0])
    if m < n:
        return 0
    matrix = np.concatenate(p).reshape(m, n)
    while True:
        i, j = matrix.shape
        if i == j:
            break
        matrix = np.concatenate((matrix, np.ones((m, 1))), axis=1)
    return abs(np.linalg.det(matrix))


def collinearity_check(p1, p2, p3, epsilon=2):
    m = np.concatenate((p1, p2, p3)).reshape(-1, 3)
    if p1[2] <= p2[2] <= p3[2]:
        m[:, 2] = 1
    det = np.linalg.det(m)
    return abs(det) < epsilon


def draw_path(image, path, color=(0, 255, 255, 255), fade=False):
    length = len(path)
    if type(image) is str:
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    thickness = 5 if 5 < image.shape[0] // 128 else image.shape[0] // 128
    for i in range(length - 1):
        if fade:
            color = (color[0], color[1], color[2], int((i / length) * 196) + 64)
        image = cv2.line(image, (int(path[i][0]), int(path[i][1])), (int(path[i + 1][0]), int(path[i + 1][1])), color,
                         thickness)
    return image


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
        min_dist = dist_threshold
        min_node = None
        for v in self.graph.V:
            dist = distance(pt, v.pos)
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
        self.branch = {}

    # Euclidean distance
    def h1(self, cur, target):
        return distance(cur.pos[:2], target.pos[:2])

    # Chebyshev or "Chessboard" distance
    def h2(self, cur, target):
        dx = abs(cur.pos[0] - target.pos[0])
        dy = abs(cur.pos[1] - target.pos[1])
        return dx + dy + min(dx, dy)

    # Encourage straight lines + Euclidean distance
    def h3(self, cur, target, past=10):
        p = [cur.pos]
        n = cur
        for i in range(past):
            if n not in self.branch:
                break
            _, b = self.branch[n]
            p.append(b.pos)
            n = b

        det = collinearity(p)
        return det + self.h1(cur, target)

    def obstacle_check(self, p1, p2, p3):
        old_cost = distance(p1, p2) + distance(p2, p3)
        new_cost = distance(p1, p3)
        direction = (np.array(p3) - np.array(p1)) / distance(p3, p1)
        pt = np.array(p1, dtype=float)
        while not (np.round(pt) == np.array(p3)).all():
            cell = self.grid.get_cell(int(pt[0]), int(pt[1]))
            if cell.alpha < 200:
                return False
            pt += direction
            new_cost += cell.cost

        return new_cost <= old_cost

    def prune_path(self, path, max_dist=32):
        print("Pruning path")
        print("Path is %d points long" % len(path))
        i = 0
        pruned_path = path
        while i < (len(pruned_path) - 2):
            p1 = pruned_path[i]
            p2 = pruned_path[i + 1]
            p3 = pruned_path[i + 2]
            if distance(p1, p3) <= max_dist and (self.obstacle_check(p1, p2, p3) or collinearity_check(p1, p2, p3)):
                del pruned_path[i + 1]
            else:
                i += 1
        print("Pruned Path is %d points long" % len(pruned_path))
        return pruned_path

    def diffuse(self, iter, k=(9, 9), transparent_cost=127):
        diffused = self.grid.grid.copy()

        for i in range(self.grid.w):
            for j in range(self.grid.h):
                if diffused[i][j][3] == 0:
                    diffused[i][j][1] = transparent_cost
                    diffused[i][j][3] = 255

        for _ in range(iter):
            diffused = cv2.blur(diffused, k, cv2.BORDER_TRANSPARENT)
            for i in range(self.grid.w):
                for j in range(self.grid.h):
                    if self.grid.grid[i][j][3] == 0:
                        diffused[i][j][1] = transparent_cost
                    if diffused[i][j][1] < self.grid.grid[i][j][1]:
                        diffused[i][j][1] = self.grid.grid[i][j][1]

        for i in range(self.grid.w):
            for j in range(self.grid.h):
                if self.grid.grid[i][j][3] == 0:
                    diffused[i][j] = np.array([0, 0, 0, 0])

        return diffused

    def find_path(self, start, target, alt=10, h=h3):
        start = self.grid.get_cell(start[0], start[1], alt)
        target = self.grid.get_cell(target[0], target[1], alt)
        queue = PriorityQueue()
        queue.put((0, start))
        visited = set()
        visited.add(start)

        self.branch = {}
        found = False

        while not queue.empty():
            _, cur = queue.get()
            if cur == start:
                cur_cost = 0
            else:
                cur_cost = self.branch[cur][0]

            if cur == target:
                found = True
                break

            for next in self.grid.get_neighbors(cur, alt):
                if next in visited:
                    continue

                branch_cost = cur_cost + (next.cost // 16)
                queue_cost = branch_cost + h(self, next, target)
                self.branch[next] = (branch_cost, cur)
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
        path_cost = self.branch[target][0]
        n = target
        while self.branch[n][1] != start:
            path.append(self.branch[n][1].pos)
            n = self.branch[n][1]
        path.append(self.branch[n][1].pos)

        pruned_path = self.prune_path(path[::-1])
        if self.DEBUG:
            path_image = draw_path(self.image, pruned_path)
            path_image_name = self.image[:-4] + "Path.png"
            cv2.imwrite(path_image_name, path_image)

        return pruned_path, path_cost


if __name__ == '__main__':
    map_image = "venues/Test/Test.png"
    a = A_star(map_image, True)

    diffused = a.diffuse(8, (25, 25), 127)
    cv2.imwrite("venues\Test\TestDiffused.png", diffused)
    a.grid.grid = diffused

    path, cost = a.find_path((19, 24), (211, 20))
    path_image = draw_path(map_image, path)
    diffused_path_image = draw_path(diffused, path)
    cv2.imwrite("venues\Test\TestPath.png", path_image)
    cv2.imwrite("venues\Test\TestDiffusedPath.png", diffused_path_image)
    cv2.imshow("Path", diffused_path_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
