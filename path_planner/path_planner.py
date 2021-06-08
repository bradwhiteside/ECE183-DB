import threading
import yaml
from copy import deepcopy
PATH_PREFIX = './'
if __name__ == '__main__':
    from graph import Graph, Node
    from grid import Grid
else:
    from path_planner.graph import Graph, Node
    from path_planner.grid import Grid
    PATH_PREFIX = './path_planner/'
import numpy as np
import scipy.ndimage
import cv2
from queue import PriorityQueue
from abc import ABC, abstractmethod

DEBUG = True

def debug(*msg):
    if DEBUG:
        print(*msg)

def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


def color2cost(color, shift=6, scale=4):
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


def draw_path(image, path, out_file=None, color=(0, 255, 255, 255), fade=False):
    length = len(path)
    if type(image) is str:
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    thickness = 5 if 5 < image.shape[0] // 128 else image.shape[0] // 128
    for i in range(length - 1):
        if fade:
            color = (color[0], color[1], color[2], int((i / length) * 196) + 64)
        image = cv2.line(image, (int(path[i][0]), int(path[i][1])), (int(path[i + 1][0]), int(path[i + 1][1])), color,
                         thickness)

    if out_file is not None:
        cv2.imwrite(out_file, image)

    return image

def check_cached_path(path):
    try:
        with open(path) as f:
            return np.genfromtxt(path, delimiter=' ', dtype=float)
    except (FileNotFoundError, IOError):
        debug("No cached file: " + path)
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
            pt[2] = (pt[2] // 4) + min_alt
            path.insert(0, pt)
            node = self.parents[node]
            if node == start:
                break

        return path


# Based off https://github.com/alpesis-robotics/drone-planner
class A_star(PathFinder):
    def __init__(self, venue_name, map_image_path):
        self.venue_name = venue_name
        self.map_image_path = map_image_path
        self.grid = Grid(map_image_path, DEBUG)
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
        debug("Path is %d points long" % length)
        pruned_path = np.array(path).reshape((length, 3))

        # From https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
        box_pts = 15
        box = np.ones(box_pts) / box_pts
        for _ in range(5):
            x = pruned_path[:, 0]
            y = pruned_path[:, 1]
            pruned_path[:, 0] = scipy.ndimage.convolve(x, box)
            pruned_path[:, 1] = scipy.ndimage.convolve(y, box)

        # remove duplicate points that may have been introduced by last step
        _, indexes = np.unique(pruned_path, axis=0, return_index=True)
        pruned_path = np.array([pruned_path[index] for index in sorted(indexes)])

        # prune collinear pts
        i = 0
        while i < (pruned_path.shape[0] - 2):
            p1 = pruned_path[i]
            p2 = pruned_path[i+1]
            p3 = pruned_path[i+2]
            if distance(p1, p2) <= 4 / TEST_PARAMS[self.venue_name]["distance_to_pixel_ratio"] and collinearity_check(p1, p2, p3):
                pruned_path = np.delete(pruned_path, i + 1, axis=0)
            else:
                i += 1
        debug("Pruned Path is %d points long" % len(pruned_path))
        return pruned_path

    def diffuse(self, iter, k=(9, 9), transparent_cost=196, USE_CACHE=True):
        if iter == 0:
            self.diffuse_params = (0, 0, 0, 0)
        else:
            self.diffuse_params = (iter, k[0], k[1], transparent_cost)
        num = abs(hash(self.diffuse_params))
        diffused_image_path = self.map_image_path[:-4] + "Diffused" + str(num) + ".png"
        debug("Looking for: ", diffused_image_path)
        diffused = cv2.imread(diffused_image_path, cv2.IMREAD_UNCHANGED)
        if diffused is not None and USE_CACHE:
            self.grid.grid = diffused
            debug("Found ", diffused_image_path)
            return diffused
        debug(diffused_image_path, "not found")

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

        cv2.imwrite(diffused_image_path, diffused)
        self.grid.grid = diffused
        return diffused

    def find_path(self, start, target, alt=10, h=h3, DRAW=True, USE_CACHE=True):
        num = abs(hash((start, target, alt, self.diffuse_params)))
        v = 'Demo' if self.venue_name[0:4] == 'Demo' else self.venue_name
        cache_path = PATH_PREFIX + "cache/" + v + '/' + str(num) + '.csv'
        if USE_CACHE:
            r = check_cached_path(cache_path)
            if r is not None:
                debug("Returning cached path for venue: {} start: {} target: {} alt: {} diffuse_params: {}"
                          .format(self.venue_name, start, target, alt, self.diffuse_params))
                if DRAW:
                    name_base = self.map_image_path[:-4]
                    draw_path(self.map_image_path, r, out_file=name_base + "Path" + str(num) + ".png")
                    # draw_path(name_base + "Orig.png", r, out_file=name_base + "OrigPath" + str(num) + ".png")
                    draw_path(self.grid.grid.copy(), r, out_file=name_base + "DiffusedPath" + str(num) + ".png")
                return r
        debug("No cache file for venue:{} hash: {} start: {} target: {} alt: {} diffuse_params: {}"
                  .format(self.venue_name, str(num), start, target, alt, self.diffuse_params))
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

        if DRAW:
            name_base = self.map_image_path[:-4]
            draw_path(self.map_image_path, pruned_path, out_file=name_base + "Path" + str(num) + ".png")
            # draw_path(name_base + "Orig.png", pruned_path, out_file=name_base + "OrigPath" + str(num) + ".png")
            draw_path(self.grid.grid.copy(), pruned_path, out_file=name_base + "DiffusedPath" + str(num) + ".png")

        ratio = TEST_PARAMS[self.venue_name]["distance_to_pixel_ratio"]
        pruned_path = pruned_path.astype(float)
        for i in range(pruned_path.shape[0]):
            pruned_path[i][0] = pruned_path[i][0] * ratio
            pruned_path[i][1] = pruned_path[i][1] * ratio

        with open(cache_path, 'w') as out:
            debug("Writing {} to cache".format(out))
            np.savetxt(out, pruned_path, delimiter=' ', fmt='%.2f')

        return pruned_path


def get_test_paths(venue, DRAW=True, USE_PATH_CACHE=True, USE_DIFFUSED_CACHE=True):
    global TEST_PARAMS
    debug("USE_PATH_CACHE = {}".format(USE_PATH_CACHE))
    debug("USE_DIFFUSED_CACHE = {}".format(USE_DIFFUSED_CACHE))
    debug("DRAW = {}".format(DRAW))
    if venue not in TEST_PARAMS:
        print("Invalid venue name:", venue)
        exit(1)

    v = 'Demo' if venue[0:4] == 'Demo' else venue
    map_image_path = PATH_PREFIX + "venues/" + v + "/" + venue + ".png"
    a = A_star(venue, map_image_path)
    if len(TEST_PARAMS[venue]["start"]) == 0 or len(TEST_PARAMS[venue]["target"]) == 0:
        a.grid.find_endpoints(50, 255)
        exit(0)

    a.diffuse(*TEST_PARAMS[venue]["diffuse_params"], USE_CACHE=USE_DIFFUSED_CACHE)
    start_pts = TEST_PARAMS[venue]["start"]
    target_pts = TEST_PARAMS[venue]["target"]
    paths = {}

    for s in start_pts:
        for t in target_pts:
            endpts_string = '(' + str(s[0]) + ", " + str(s[1]) + ') --> (' + str(t[0]) + ", " + str(t[1]) + ')'
            debug(venue + ": Finding path for ", endpts_string)
            path = a.find_path(s, t, DRAW=DRAW, USE_CACHE=USE_PATH_CACHE)
            paths[(s, t)] = path
            debug(venue + ": Found path for " + venue + ": " + endpts_string + ": {} pts long".format(len(path)))

    return paths


TEST_PARAMS = {
    "Test": {
        "distance_to_pixel_ratio": 1.0,
        "diffuse_params": [12, (15, 15), 196],
        "start": [(19, 24)],
        "target": [(211, 20)],
    },

    "RoseBowl": {
        "distance_to_pixel_ratio": 0.25,
        "diffuse_params": [12, (25, 25), 196],
        "start": [(128, 296), (1263, 530), (46, 950), (1198, 1062)],
        "target": [(682, 310), (934, 676), (352, 684), (534, 1112)],
    },

    "Coachella": {
        "distance_to_pixel_ratio": 0.75,
        "diffuse_params": [5, (7, 7), 196],
        "start": [(1447, 546), (1515, 547), (1568, 780), (1414, 855), (1413, 908), (1411, 957)],
        "target": [(263, 356), (1086, 701), (201, 790), (186, 1359), (666, 1391)],
    },

    "ElectricForest": {
        "distance_to_pixel_ratio": 0.63,
        "diffuse_params": [5, (7, 7), 128],
        "start": [(923, 432), (669, 509), (457, 543)],
        "target": [(414, 131), (72, 194), (683, 200), (164, 508), (774, 731), (258, 756)]
    },
    "Demo1": {
        "distance_to_pixel_ratio": 1.0,
        "diffuse_params": [9, (31, 31), 196],
        "start": [(55, 46)],
        "target": [(697, 590)]
    }
}

if __name__ == '__main__':
    DEBUG = False
    paths = get_test_paths("Demo1", DRAW=True, USE_PATH_CACHE=False, USE_DIFFUSED_CACHE=True)

    exit(0)
    paths = get_test_paths("ElectricForest", DRAW=True, USE_PATH_CACHE=True, USE_DIFFUSED_CACHE=True)
    paths = get_test_paths("Test", DRAW=True, USE_PATH_CACHE=False, USE_DIFFUSED_CACHE=True)
    paths = get_test_paths("RoseBowl", DRAW=True, USE_PATH_CACHE=False, USE_DIFFUSED_CACHE=True)
    paths = get_test_paths("Coachella", DRAW=True, USE_PATH_CACHE=False, USE_DIFFUSED_CACHE=True)
