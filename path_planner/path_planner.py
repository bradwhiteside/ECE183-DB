import yaml

if __name__ == '__main__':
    from graph import Graph, Node
    from grid import Grid
else:
    from path_planner.graph import Graph, Node
    from path_planner.grid import Grid
import numpy as np
import scipy.ndimage
import cv2
from queue import PriorityQueue
from abc import ABC, abstractmethod


def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))

def color2cost(color, shift=6, scale=16):
    c = 0 if color == 0 else np.log2(color)
    sig = 1 / (1 + np.exp(-c + shift))
    return scale * sig


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


def collinearity_check(p1, p2, p3, epsilon=7):
    m = np.vstack((p1, p2, p3))
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

def check_cached_path(path):
    try:
        with open(path) as f:
            return np.genfromtxt(path, delimiter=' ', dtype=int)
    except (FileNotFoundError, IOError):
        return None


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
        self.diffuse_params = 0

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

    def cost(self, p1, p2):
        c = distance(p1, p2)
        direction = (np.array(p2) - np.array(p1)) / c
        pt = np.array(p1, dtype=float)
        while not (np.round(pt) == np.array(p2)).all():
            cell = self.grid.get_cell(round(pt[0]), round(pt[1]))
            if cell.alpha < 200:
                return np.inf
            pt += direction
            c += color2cost(cell.cost)
        return c

    def costs(self, P):
        total_cost = 0
        for i in range(len(P) - 1):
            total_cost += self.cost(P[i], P[i+1])

    def prune_path(self, path, path_cost):
        length = len(path)
        print("Path is %d points long" % length)
        pruned_path = np.array(path).reshape((length, 3))

        # From https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
        box_pts = 12
        box = np.ones(box_pts) / box_pts
        for _ in range(3):
            x = pruned_path[:, 0]
            y = pruned_path[:, 1]
            pruned_path[:, 0] = scipy.ndimage.convolve(x, box)
            pruned_path[:, 1] = scipy.ndimage.convolve(y, box)

        # prune collinear pts
        i = 0
        while i < (pruned_path.shape[0] - 2):
            p1 = pruned_path[i]
            p2 = pruned_path[i+1]
            p3 = pruned_path[i+2]
            if collinearity_check(p1, p2, p3):
                pruned_path = np.delete(pruned_path, i + 1, axis=0)
            else:
                i += 1
        print("Pruned Path is %d points long" % len(pruned_path))
        return pruned_path

    def diffuse(self, iter, k=(9, 9), transparent_cost=196):
        self.diffuse_params = (iter, k[0], k[1], transparent_cost)
        diffused = self.grid.grid.copy()
        max_i = self.grid.h
        max_j = self.grid.w

        for i in range(max_i):
            for j in range(max_j):
                if diffused[i][j][3] == 0:
                    diffused[i][j][1] = transparent_cost
                    diffused[i][j][3] = 255

        for _ in range(iter):
            diffused = cv2.blur(diffused, k, cv2.BORDER_TRANSPARENT)
            for i in range(max_i):
                for j in range(max_j):
                    if self.grid.grid[i][j][3] == 0:
                        diffused[i][j][1] = transparent_cost
                    elif diffused[i][j][1] < self.grid.grid[i][j][1]:
                        diffused[i][j][1] = self.grid.grid[i][j][1]

        for i in range(max_i):
            for j in range(max_j):
                if self.grid.grid[i][j][3] == 0:
                    diffused[i][j] = np.array([0, 0, 0, 0])

        return diffused

    def find_path(self, start, target, alt=10, h=h3):
        num = abs(hash((start, target, alt, self.diffuse_params)))
        venue_name = self.image.split('/')[-1][:-4]
        cache_path = "cache/" + venue_name + '/' + str(num) + '.csv'
        r = check_cached_path(cache_path)
        if r is not None:
            return r

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

                # cell_cost = 0 if next.cost < 32 else np.log2(next.cost)
                cell_cost = color2cost(next.cost)
                branch_cost = cur_cost + cell_cost
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
        path_cost = []
        path_cost.append(self.branch[target][0])
        n = target
        while self.branch[n][1] != start:
            c, p = self.branch[n]
            path.append(p.pos)
            path_cost.append(c)
            n = p
        path.append(self.branch[n][1].pos)
        path_cost.append(0)

        pruned_path = self.prune_path(path[::-1], path_cost[::-1])
        if self.DEBUG:
            path_image = draw_path(self.image, pruned_path)
            path_image_name = self.image[:-4] + "Path.png"
            cv2.imwrite(path_image_name, path_image)

        with open(cache_path, 'w') as out:
            np.savetxt(out, pruned_path, delimiter=' ', fmt='%d')
        return pruned_path


if __name__ == '__main__':
    venue = "Test"

    map_image_name = "venues/" + venue + "/" + venue + ".png"
    a = A_star(map_image_name, True)
    # a.grid.find_endpoints(127, 255); exit(0)

    diffused = a.diffuse(12, (15, 15), 196)  # Test
    # diffused = a.diffuse(12, (25, 25), 196)  # Rose Bowl
    # diffused = a.diffuse(12, (15, 15), 196)  # Coachella
    cv2.imwrite(map_image_name[:-4] + "Diffused.png", diffused)
    a.grid.grid = diffused

    start = (19, 24); target = (211, 20)  # Test
    # start = (46, 950); target = (682, 310)  # RoseBowl
    # start = (19, 24); target = (211, 20)  # Coachella
    path = a.find_path(start, target)
    path_image = draw_path(map_image_name, path)
    diffused_path_image = draw_path(diffused, path)
    orig_path_image = draw_path(map_image_name[:-4] + "Orig.png", path)
    cv2.imwrite(map_image_name[:-4] + "DiffusedPath.png", diffused_path_image)
    cv2.imwrite(map_image_name[:-4] + "OrigPath.png", orig_path_image)
    # cv2.imshow("Path", diffused_path_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()