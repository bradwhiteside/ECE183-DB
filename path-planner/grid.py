import numpy as np
import cv2


class Grid:
    def __init__(self, image_name, DEBUG=False):
        image_data = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)  # [height, width, [B, G, R, alpha]]
        if image_data.shape[2] != 4:
            raise ValueError("Image must have a transparency channel, like in a .png")
        self.grid = image_data
        self.h = image_data.shape[0]
        self.w = image_data.shape[1]
        self.DEBUG = DEBUG

    def get_cell(self, x, y, alt):
        return Cell(x, y, self.grid[y][x][0] + alt, self.grid[y][x])

    def get_neighbors(self, cell, alt):
        x = cell.pos[0]
        y = cell.pos[1]
        neighbors = []
        for i in [y-1, y, y+1]:
            for j in [x-1, x, x+1]:
                if i < 0 or i >= self.h or j < 0 or j >= self.w:
                    continue
                if self.grid[i][j][3] == 0:
                    continue
                if x == j and y == i:
                    continue
                z = self.grid[i][j][0] + alt
                neighbors.append(Cell(j, i, z, self.grid[i][j]))
        return neighbors

    def find_endpoints(self, min_R, max_R):
        if not self.DEBUG:
            return

        for i in range(self.h):
            for j in range(self.w):
                if min_R <= self.grid[i][j][2] <= max_R:
                    print("Endpoint of value %d located at (%d, %d)" % (self.grid[i][j][2], j, i))


class Cell:
    def __init__(self, x, y, z, grid_point):
        self.pos = np.array([x, y, z])
        self.min_alt = grid_point[0]
        self.cost = grid_point[1]
        self.alpha = grid_point[3]

    def __hash__(self):
        return hash((self.pos[0], self.pos[1], self.pos[2]))

    def __eq__(self, other):
        return (self.pos == other.pos).all()

    def __lt__(self, other):
        return self.cost < other.cost
