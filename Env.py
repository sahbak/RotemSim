import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np
import time

from shapely.geometry import Point, LineString, Polygon
from descartes import PolygonPatch


class Env:
    def __init__(self, border_polygon, obs_array, trg_pos, ax, res):
        self.border_polygon_for_grid = border_polygon
        self.border_polygon = Polygon(border_polygon)
        self.target_pos = trg_pos
        self.target_alive = True
        self.is_target_mang = False
        self.obs_array = list()
        self.obs_vertex_list = []
        self.target_handel = []
        self.los_to_target_handel = []
        self.res = res
        self.grid = Grid(border_polygon, res, ax)
        self.grid.plot_grid()
        for obs in obs_array:
            self.obs_array.append(Polygon(obs))
            for vertex in obs:
                self.obs_vertex_list.append(vertex)
            del self.obs_vertex_list[-1]
        self.ax = ax

    def __init__(self, trg_pos, ax, res):
        self.target_pos = trg_pos
        self.target_alive = True
        self.is_target_mang = False
        self.obs_array = list()
        self.obs_vertex_list = []
        self.target_handel = []
        self.los_to_target_handel = []
        self.res = res
        self.build_barel_maze()
        self.ax = ax
        self.grid = Grid(self.border_polygon_for_grid, res, ax)
        self.grid.plot_grid()

    def plot_target(self):
        self.target_handel = self.ax.plot(self.target_pos[0][0], self.target_pos[0][1], 'ro', markersize=20)

    def plot_border(self):
        border_polygon_patch = PolygonPatch(self.border_polygon, facecolor='white')
        self.ax.add_patch(border_polygon_patch)

    def is_in_border_polygon(self, pos):
        return self.border_polygon.contains(Point(pos[0][0], pos[0][1]))

    def plot_obs_array(self):
        for obs in self.obs_array:
            border_polygon_patch = PolygonPatch(obs, facecolor='orange')
            self.ax.add_patch(border_polygon_patch)

    def is_in_obs(self, pos):
        for obs in self.obs_array:
            if obs.contains(Point(pos[0][0], pos[0][1])):
                return True
        return  False

    def is_step_legal(self, curr_pos, step):
        new_pos = curr_pos + step
        return self.is_in_border_polygon(new_pos) and (self.grid.is_los(curr_pos, new_pos))

    def is_los(self, p1, p2):
        line = LineString([(p1[0][0], p1[0][1]), (p2[0][0], p2[0][1])])
        for obs in self.obs_array:
            if obs.intersection(line):
                return False
        return True

    def is_los_to_trg(self, pos):
        return self.is_los(self.target_pos, pos)

    def is_detect_trg(self, pos):
        if self.target_alive:
            if self.is_los_to_trg(pos):
                return True
        return False

    def plot_homing(self, pos):
        if self.los_to_target_handel.__len__() > 0:
            self.los_to_target_handel[0].remove()
        self.los_to_target_handel = self.ax.plot([pos[0][0], self.target_pos[0][0]], [pos[0][1], self.target_pos[0][1]],'y')
        if np.linalg.norm(pos - self.target_pos) < 30:
            if self.target_alive:
                self.target_handel[0].remove()
                self.target_alive = False

    def build_wall(self, nw, se):
        # The polygon coordinate is from Barel's xls, in cell indices
        return [(nw[0]*self.res, nw[1]*self.res), (nw[0]*self.res, se[1]*self.res), (se[0]*self.res, se[1]*self.res), (se[0]*self.res, nw[1]*self.res), (nw[0]*self.res, nw[1]*self.res)]

    def build_barel_maze(self):

        self.border_polygon_for_grid = [(0*self.res, 0*self.res), (57*self.res, 0*self.res), (57*self.res, 55*self.res), (0*self.res, 55*self.res), (0*self.res, 0*self.res)]
        self.border_polygon = Polygon(self.border_polygon_for_grid)
        obs_list = list()
        # The wall coordinate is from Barel's xls, in cell indices
        obs_list.append(self.build_wall([0, 0], [1, 55]))
        obs_list.append(self.build_wall([1, 0], [31, 1]))
        obs_list.append(self.build_wall([1, 48], [16, 49]))
        obs_list.append(self.build_wall([1, 54], [56, 55]))
        obs_list.append(self.build_wall([6, 12], [11, 13]))
        obs_list.append(self.build_wall([10, 1], [11, 6]))
        obs_list.append(self.build_wall([10, 13], [11, 23]))
        obs_list.append(self.build_wall([12, 28], [13, 43]))
        obs_list.append(self.build_wall([12, 43], [17, 44]))
        obs_list.append(self.build_wall([16, 44], [17, 54]))
        obs_list.append(self.build_wall([17, 1], [18, 21]))
        obs_list.append(self.build_wall([28, 11], [29, 20]))
        obs_list.append(self.build_wall([20, 27], [21, 37]))
        obs_list.append(self.build_wall([21, 27], [31, 28]))
        obs_list.append(self.build_wall([21, 36], [31, 37]))
        obs_list.append(self.build_wall([18, 20], [29, 21]))
        obs_list.append(self.build_wall([28, 1], [29, 6]))
        obs_list.append(self.build_wall([21, 44], [22, 54]))
        obs_list.append(self.build_wall([22, 44], [27, 45]))
        obs_list.append(self.build_wall([31, 27], [32, 37]))
        obs_list.append(self.build_wall([32, 44], [33, 54]))
        obs_list.append(self.build_wall([36, 0], [57, 1]))
        obs_list.append(self.build_wall([39, 1], [40, 21]))
        obs_list.append(self.build_wall([39, 28], [40, 39]))
        obs_list.append(self.build_wall([39, 44], [40, 54]))
        obs_list.append(self.build_wall([40, 28], [49, 29]))
        obs_list.append(self.build_wall([45, 29], [46, 44]))
        obs_list.append(self.build_wall([45, 49], [46, 54]))
        obs_list.append(self.build_wall([46, 9], [56, 10]))
        obs_list.append(self.build_wall([46, 20], [56, 21]))
        obs_list.append(self.build_wall([46, 40], [56, 41]))
        obs_list.append(self.build_wall([56, 1], [57, 55]))
        for obs in obs_list:
            self.obs_array.append(Polygon(obs))
            for vertex in obs:
                self.obs_vertex_list.append(vertex)
            del self.obs_vertex_list[-1]

class Grid:

    def __init__(self, border_polygon, res, ax):
        self.x_lim = [border_polygon[0][0], border_polygon[0][0]]
        self.y_lim = [border_polygon[0][1], border_polygon[0][1]]
        for i in range(1, border_polygon.__len__()):
            if self.x_lim[0] > border_polygon[i][0]:
                self.x_lim[0] = border_polygon[i][0]
            if self.x_lim[1] < border_polygon[i][0]:
                self.x_lim[1] = border_polygon[i][0]
            if self.y_lim[0] > border_polygon[i][1]:
                self.y_lim[0] = border_polygon[i][1]
            if self.y_lim[1] < border_polygon[i][1]:
                self.y_lim[1] = border_polygon[i][1]

        self.res = res
        self.matrix = np.zeros([np.int64(np.ceil((self.x_lim[1]-self.x_lim[0])/self.res)), np.int64(np.ceil((self.y_lim[1]-self.y_lim[0])/self.res))])
        self.ax = ax
        self.tail_handles = list()

    def xy_to_ij(self, x, y):
        i = int(np.floor((x - self.x_lim[0])/self.res))
        j = int(np.floor((y - self.y_lim[0]) / self.res))
        return i, j

    def ij_to_xy(self, i, j):
        x = self.x_lim[0] + i*self.res + self.res/2
        y = self.y_lim[0] + j*self.res + self.res/2
        return x, y

    def plot_ij(self, i, j):
        pol_center = self.ij_to_xy(i, j)
        tail = Polygon([(pol_center[0]-self.res/2, pol_center[1]-self.res/2), (pol_center[0]-self.res/2, pol_center[1]+self.res/2)
                       , (pol_center[0]+self.res/2, pol_center[1]+self.res/2), (pol_center[0]+self.res/2, pol_center[1]-self.res/2)
                       , (pol_center[0]-self.res/2, pol_center[1]-self.res/2)])
        return self.ax.add_patch(PolygonPatch(tail, facecolor='gray'))

    def plot_grid(self):
        for i in range(0, self.matrix.__len__()):
            handles_list = list()
            for j in range(0, self.matrix[i].__len__()):
                handles_list.append(self.plot_ij(i, j))
            self.tail_handles.append(handles_list)

    def change_tail_color_ij(self, i, j, color):
        self.tail_handles[i][j].set_fc(color)

    def change_tail_to_empty(self, i, j):
        self.change_tail_color_ij(i, j, 'k')
        self.matrix[i][j] = 1

    def change_tail_to_wall(self, i, j):
        self.change_tail_color_ij(i, j, 'w')
        self.matrix[i][j] = 2

    def is_los(self, p1, p2):
        n = int(np.ceil(np.linalg.norm(p1-p2)/self.res)*3)
        x = np.linspace(p1[0][0], p2[0][0], num=n, endpoint=True)
        y = np.linspace(p1[0][1], p2[0][1], num=n, endpoint=True)
        for ind in range(0, n):
            i, j = self.xy_to_ij(x[ind], y[ind])
            if self.matrix[i][j] != 1:
                return False
        return True

    def update_with_radial_sensor(self, p, r, env):
        for x in np.linspace(p[0][0]-r, p[0][0]+r, num=r/self.res*2, endpoint=True):
            for y in np.linspace(p[0][1] - r, p[0][1] + r, num=r/self.res*2, endpoint=True):
                if np.linalg.norm([[x, y]] - p) < r:
                    if env.is_los(p, [[x, y]]):
                        i, j = self.xy_to_ij(x, y)
                        if 0 <= i and i< self.matrix.shape[0] and 0 <= j and j< self.matrix.shape[1]:
                            self.change_tail_to_empty(i, j)

    def update_with_radial_sensor2(self, p, r, env):
        base_unit_vector = [[0, 1]]
        for theta in np.linspace(0, 2*np.pi, num=10, endpoint=False):
            unit_vector = [[base_unit_vector[0][0]*np.cos(theta) - base_unit_vector[0][1]*np.sin(theta),
                            base_unit_vector[0][0] * np.sin(theta) + base_unit_vector[0][1] * np.cos(theta)]]
            for dr in np.linspace(0, r, num=r / self.res * 2, endpoint=True):
                p2 = p + [[unit_vector[0][0]*dr, unit_vector[0][1]*dr]]
                i, j = self.xy_to_ij(p2[0][0], p2[0][1])
                if 0 <= i and i <= self.matrix.shape[0] and 0 <= j and j <= self.matrix.shape[1]:
                    if env.is_in_obs(p2):
                        self.change_tail_to_wall(i, j)
                        break
                    else:
                        self.change_tail_to_empty(i, j)

    def non_scaned_list(self, p, r, env):
        non_sc_list = list()
        base_unit_vector = [[0, 1]]
        for theta in np.linspace(0, 2*np.pi, num=10, endpoint=False):
            unit_vector = [[base_unit_vector[0][0]*np.cos(theta) - base_unit_vector[0][1]*np.sin(theta),
                            base_unit_vector[0][0] * np.sin(theta) + base_unit_vector[0][1] * np.cos(theta)]]
            for dr in np.linspace(self.res, r, num=r / self.res * 2, endpoint=True):
                p2 = p + [[unit_vector[0][0]*dr, unit_vector[0][1]*dr]]
                i, j = self.xy_to_ij(p2[0][0], p2[0][1])
                if 0 <= i and i < self.matrix.shape[0] and 0 <= j and j < self.matrix.shape[1]:
                    if env.grid.matrix[i][j] == 2:
                        break
                    else:
                        if self.matrix[i][j] == 0:
                            non_sc_list.append([self.ij_to_xy(i, j)])
        return non_sc_list
