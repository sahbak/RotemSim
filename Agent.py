# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 08:59:56 2018

@author: manorr
"""
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np
import time
from Env import Env


class Agent:

    def __init__(self, AgentID, pos, ax):
        self.ID = AgentID
        self.agent_alive = True
        self.is_homing = False
        self.VelocityFactor = 20
        self.step_noise_size = 50
        self.step_snr = 1
        self.stepSizeLimit = 30
        self.step_factor = 10
        self.next_pos = pos

        self.current_pos = self.next_pos
        self.VisibilityRange = 400.
        self.scanning_range = 200
        self.repulse_range = self.VisibilityRange/10
        self.pull_range = self.VisibilityRange*4/5
        self.goal_orianted_flag_flip_prob = 0.1
        self.goal_orianted_flag = np.random.rand(1) < self.goal_orianted_flag_flip_prob

        self.ax = ax
        self.plot_color = 'ob'
        self.AgetPlotHadel, = ax.plot(self.current_pos[0][0], self.current_pos[0][1], self.plot_color)
        self.edg_to_neighbors_plot_hadels = []


    def caculate_step(self, agentsArr, env):
        self.delete_edges()
        self.PlotAgent(self.ax)
        if env.is_detect_trg(self.current_pos) and not env.is_target_mang:
            self.is_homing = True
            env.is_target_mang = True
        if self.is_homing:
            self.dynam_homing(env)
            env.plot_homing(self.current_pos)
        else:
            neigbours_pos_list = self.Sensing(agentsArr, env)
            neigbours_pos_list = self.neighborhood_reduction(neigbours_pos_list, env)
            self.plot_edges(self.ax, neigbours_pos_list)
            self.Dynam_Search_in_maze(neigbours_pos_list, env)   # This function should be replaced - Rita

    def perform_step(self, env):
        self.current_pos = self.next_pos
        env.grid.update_with_radial_sensor2(self.current_pos, self.scanning_range, env)

    def dynam_homing(self, env):
        dist_to_target = np.linalg.norm(env.target_pos - self.current_pos)
        if dist_to_target > 30:
            heading_direction = (env.target_pos - self.current_pos)/dist_to_target
            self.next_pos = self.current_pos + heading_direction*self.VelocityFactor
        else:
            self.agent_alive = False

    def Dynam_LinearProtocol(self, NeighborsPosList):
        self.current_pos = self.Pos
        step = np.multiply(self.VelocityFactor, np.sum(NeighborsPosList, 0))
        step = np.divide(step, agents.__len__())
        self.Pos = np.sum([self.Pos, step], 0)

    def Dynam_AndoCircles_Formation(self, NeighborsPosList, env):
        flag = False
        break_counter = 0
        step_base = [[0, 0]]
        for NeighborPos in NeighborsPosList:
            step_base = step_base + NeighborPos - NeighborPos / np.linalg.norm(NeighborPos) * self.VisibilityRange / 2

        while not (flag) and break_counter < 20:
            break_counter = break_counter + 1
            step = step_base / np.linalg.norm(step_base) * self.step_factor
            if env.is_step_legal(self.Pos, step):
                flag = True
                for NeighborPos in NeighborsPosList:
                    if self.outOfLimit_Ando(NeighborPos, step):
                        flag = False
                        break
        if break_counter < 20:
            self.next_pos = self.current_pos + step

# This is the important function, that should be rewriten - Rita
    def Dynam_Search_in_maze(self, NeighborsPosList, env):
        flag = False
        break_counter = 0
        rep_att_vec = np.zeros(2)

        # Goal oriented movement
        noise_fac = 1
        if self.goal_orianted_flag:
            optinal_goal_list = env.grid.non_scaned_list(self.current_pos, 5 * self.scanning_range, env)
            if optinal_goal_list.__len__() > 0:
                vec = optinal_goal_list[0] - self.current_pos
                goal_vec = vec / np.linalg.norm(vec)
                rep_att_vec = rep_att_vec + goal_vec
            else:
                noise_fac = 5
        else:
            noise_fac = 2
        # Neighbours oriented movement
        for NeighborPos in NeighborsPosList:
            #rep_att_vec = rep_att_vec - NeighborPos / np.linalg.norm(NeighborPos)
            if np.linalg.norm(NeighborPos) < self.repulse_range:
                rep_att_vec = rep_att_vec - NeighborPos / np.linalg.norm(NeighborPos) * self.VisibilityRange
            if np.linalg.norm(NeighborPos) > self.pull_range*noise_fac:
                rep_att_vec = rep_att_vec + NeighborPos / np.linalg.norm(NeighborPos) * self.VisibilityRange
        if np.linalg.norm(rep_att_vec) > 0:
            rep_att_vec = rep_att_vec/np.linalg.norm(rep_att_vec)
        if np.random.rand() < self.goal_orianted_flag_flip_prob:
            self.goal_orianted_flag = not self.goal_orianted_flag

        if np.linalg.norm(rep_att_vec) > 0:
            rep_att_vec = self.step_noise_size*self.step_snr*rep_att_vec/np.linalg.norm(rep_att_vec)


        while not flag and break_counter < 20:
            break_counter = break_counter + 1
            step = self.step_noise_size * noise_fac * ([0.5, 0.5] - np.random.rand(2)) + rep_att_vec
            if env.is_step_legal(self.current_pos, step):
                flag = True
                for neighbor_pos in NeighborsPosList:
                    if self.outOfLimit_Ando(neighbor_pos, step):
                        flag = False
                        break
                    if not self.is_step_in_corridor(step, neighbor_pos, env):
                        flag = False
                        break

        if break_counter < 20:
            self.next_pos = self.current_pos + step #todo: intetegrate VelocityFactor

    def Dynam_Search_in_maze2(self, NeighborsPosList, env):
        if np.random.rand(1) < self.goal_orianted_flag_flip_prob:
            self.goal_orianted_flag = not self.goal_orianted_flag

        if self.goal_orianted_flag: # Goal oriented movement
            optinal_goal_list = env.grid.non_scaned_list(self.current_pos, 5 * self.scanning_range, env)
            if optinal_goal_list.__len__() > 0:
                rand_ind = int(np.floor(optinal_goal_list.__len__() * np.random.rand(1)))
                vec = optinal_goal_list[rand_ind] - self.current_pos
                goal_vec = vec / np.linalg.norm(vec)
                step = self.step_noise_size * (goal_vec + 0.5 * ([0.5, 0.5] - np.random.rand(2)))
            else:
                step = self.step_noise_size * ([0.5, 0.5] - np.random.rand(2))
        else: #  self.goal_orianted_flag == false  # Neighbours oriented movement
                step = self.step_noise_size * ([0.5, 0.5] - np.random.rand(2))

        flag = False
        break_counter = 0
        while not flag and break_counter < 20:
            step = step * 0.5
            break_counter = break_counter + 1
            if env.is_step_legal(self.current_pos, step):
                flag = True
                for neighbor_pos in NeighborsPosList:
                    if self.outOfLimit_Ando(neighbor_pos, step):
                        flag = False
                        break
                    if not self.is_step_in_corridor(step, neighbor_pos, env):
                        flag = False
                        break

        if break_counter < 20:
            self.next_pos = self.current_pos + step

    def is_step_in_corridor(self, step, neighbor_pos, env):
        neighbor_abs_pos = self.current_pos + neighbor_pos
        if env.is_step_legal(neighbor_abs_pos, step):
            neighbor_abs_pos_potential = neighbor_abs_pos + step
        else:
            neighbor_pos_unit = neighbor_pos / np.linalg.norm(neighbor_pos)
            neighbor_step_potential = step - 2 * np.dot(step[0], neighbor_pos_unit[0]) * neighbor_pos_unit
            neighbor_abs_pos_potential = neighbor_abs_pos + neighbor_step_potential

        return env.grid.is_los(self.current_pos + step, neighbor_abs_pos_potential)

    def Dynam_AndoCircles_disperse(self, NeighborsPosList, env):
        flag = False
        break_counter = 0
        rep_att_vec = np.zeros(2)
        # Goal oriented movement
        noise_fac = 1
        if self.goal_orianted_flag:
            optinal_goal_list = env.grid.non_scaned_list(self.current_pos, 5 * self.scanning_range, env)
            if optinal_goal_list.__len__() > 0:
                vec = optinal_goal_list[0] - self.current_pos
                goal_vec = vec / np.linalg.norm(vec)
                rep_att_vec = rep_att_vec + goal_vec
            else:
                noise_fac = 5
        else:
            noise_fac = 2
        # Neighbours oriented movement
        for NeighborPos in NeighborsPosList:
            # rep_att_vec = rep_att_vec - NeighborPos / np.linalg.norm(NeighborPos)
            if np.linalg.norm(NeighborPos) < self.repulse_range:
                rep_att_vec = rep_att_vec - NeighborPos / np.linalg.norm(NeighborPos) * self.VisibilityRange
            if np.linalg.norm(NeighborPos) > self.pull_range * noise_fac:
                rep_att_vec = rep_att_vec + NeighborPos / np.linalg.norm(NeighborPos) * self.VisibilityRange
        if np.linalg.norm(rep_att_vec) > 0:
            rep_att_vec = rep_att_vec / np.linalg.norm(rep_att_vec)
        if np.random.rand() < self.goal_orianted_flag_flip_prob:
            self.goal_orianted_flag = not self.goal_orianted_flag

        if np.linalg.norm(rep_att_vec) > 0:
            rep_att_vec = self.step_noise_size * self.step_snr * rep_att_vec / np.linalg.norm(rep_att_vec)

        while not flag and break_counter < 20:
            break_counter = break_counter + 1
            step = self.step_noise_size * noise_fac * ([0.5, 0.5] - np.random.rand(2)) + rep_att_vec
            if env.is_step_legal(self.current_pos, step):
                flag = True
                for NeighborPos in NeighborsPosList:
                    if self.outOfLimit_Ando(NeighborPos, step):
                        flag = False
                        break
                    neighbor_abs_pos = self.current_pos + NeighborPos
                    if env.is_step_legal(neighbor_abs_pos, step):
                        neighbor_abs_pos_potential = neighbor_abs_pos + step
                    else:
                        NeighborPos_unit = NeighborPos / np.linalg.norm(NeighborPos)
                        neighbor_step_potential = step - 2 * np.dot(step[0], NeighborPos_unit[0]) * NeighborPos_unit
                        neighbor_abs_pos_potential = neighbor_abs_pos + neighbor_step_potential

                    if not env.grid.is_los(self.current_pos + step, neighbor_abs_pos_potential):
                        flag = False
                        break
                    # if not(self.is_in_los_corridor(NeighborPos, env, step)):
                    #    flag = False
                    #    break
        if break_counter < 20:
            self.next_pos = self.current_pos + step  # todo: intetegrate VelocityFactor
        # else:
        #    step = self.step_noise_size * np.random.rand(2) - [[0.5, 0.5]]
        #    if env.is_step_legal(self.current_pos, step):
        #        self.next_pos = self.current_pos + step

    def Dynam_AndoCircles(self, NeighborsPosList, env):

        self.current_pos = self.Pos
        flag = False
        break_counter = 0
        while not(flag) and break_counter < 20:
            break_counter = break_counter+1
            step = 2 * self.stepSizeLimit * np.random.rand(2) - self.stepSizeLimit
            if env.is_step_legal(self.Pos, step):
                flag = True
                for NeighborPos in NeighborsPosList:
                    if self.outOfLimit_Ando(NeighborPos, step):
                        flag = False
                        break

        if break_counter < 20:
            self.Pos = self.Pos + step

    def Dynam_AndoCircles_and_los(self, NeighborsPosList, env):
        Flag = False
        break_counter = 0
        while not (Flag) and break_counter < 20:
            break_counter = break_counter + 1
            step = 2 * self.stepSizeLimit * np.random.rand(2) - self.stepSizeLimit
            if env.is_step_legal(self.Pos, step):
                Flag = True
                for NeighborPos in NeighborsPosList:
                    if self.outOfLimit_Ando(NeighborPos, step):
                        Flag = False
                        break
        if break_counter < 20:
            self.Pos = self.Pos + step

    def Dynam_los_corridor(self, neighbor_rel_pos, env):
        nieghbor_vecneighbor_unit = neighbor_rel_pos / np.linalg.norm(neighbor_rel_pos)
        left_list = []
        right_list = []
        min_left_cross_prod = self.VisibilityRange/2
        min_right_cross_prod = -self.VisibilityRange / 2
        for vertex in env.obs_vertex_list:
            vec = [vertex[0] - self.Pos[0][0], vertex[1] - self.Pos[0][1]]
            if np.linalg.norm(neighbor_rel_pos/2 - vec) < self.VisibilityRange/2:
                cross_prod = np.cross(nieghbor_vecneighbor_unit, vec)
                if cross_prod > 0:
                    if cross_prod < min_left_cross_prod:
                        min_left_cross_prod = cross_prod
                else:
                    if cross_prod > min_right_cross_prod:
                        min_right_cross_prod = cross_prod
        return min_right_cross_prod, min_left_cross_prod

    def is_in_los_corridor(self, neighbor_rel_pos, env, step):
        min_right_cross_prod, min_left_cross_prod = self.Dynam_los_corridor(neighbor_rel_pos, env)

        nieghbor_vecneighbor = neighbor_rel_pos[0]
        nieghbor_vecneighbor_unit = nieghbor_vecneighbor / np.linalg.norm(nieghbor_vecneighbor)
        cross_prod = np.cross(step, nieghbor_vecneighbor_unit)
        return (min_right_cross_prod < cross_prod) and (cross_prod < min_left_cross_prod)

    #def is_in_los_corridor(self, right_border_dist, left_border_dist, neighbor_rel_pos, rel_terget_pos):
    #    nieghbor_vec = (neighbor_rel_pos[0][0], neighbor_rel_pos[0][1])
    #    nieghbor_vec_unit = nieghbor_vec / np.linalg.norm(nieghbor_vec)
#
    #    cross_prod = np.cross(nieghbor_vec_unit, rel_terget_pos)
    #    if (-1*right_border_dist < cross_prod) and (cross_prod < left_border_dist):
    #        return True
    #    else:
    #        return False
#

    def getKey(self, item):
        return item[0]

    def outOfLimit_Ando(self, neighbor_pos, step):
        avg_pos = np.divide(neighbor_pos, 2)
        deltas_step = step - avg_pos
        return np.linalg.norm(deltas_step) > self.VisibilityRange/2

    def Sensing(self, agents_arr, env_in):
        neighbors_pos = []
        for i in range(0, agents_arr.__len__()):
            diff = agents_arr[i].current_pos - self.current_pos
            if (agents_arr[i].ID != self.ID and np.linalg.norm(diff) < self.VisibilityRange and
                    env_in.grid.is_los(agents_arr[i].current_pos, self.current_pos)):
                neighbors_pos.append(diff)
        return neighbors_pos

    def neighborhood_reduction(self, neighbors_pos, env):
        reduced_neighbors_pos = []
        for i in range(0, neighbors_pos.__len__()):
            flag = True
            for j in range(0, neighbors_pos.__len__()):
                if i != j and ((np.linalg.norm(neighbors_pos[i]) > np.linalg.norm(neighbors_pos[j])) and
                               (np.linalg.norm(neighbors_pos[i]) >
                                np.linalg.norm(neighbors_pos[j] - neighbors_pos[i]))):
                    if env.grid.is_los(self.current_pos + neighbors_pos[i], self.current_pos + neighbors_pos[j]):
                        flag = False
                        break
            if flag:
                reduced_neighbors_pos.append(neighbors_pos[i])
        return reduced_neighbors_pos

    def PlotAgent(self, ax):
        if self.goal_orianted_flag:
            col = 'r'
        else:
            col = 'b'
        self.AgetPlotHadel.set_data(self.current_pos[0][0], self.current_pos[0][1])
        self.AgetPlotHadel._color = col

    def plot_edges(self, ax, neighbors_list):
        for pos in neighbors_list:
            edge = ax.plot([self.current_pos[0][0], self.current_pos[0][0]+pos[0][0]], [self.current_pos[0][1], self.current_pos[0][1]+pos[0][1]])
            self.edg_to_neighbors_plot_hadels.append(edge)

    def delete_edges(self):
        for edge_handel in self.edg_to_neighbors_plot_hadels:
            edge_handel[0].remove()
        self.edg_to_neighbors_plot_hadels.clear()

# **************** main *******************

Fig = plt.figure()
plt.axis([0, 570, 0, 550])
ax = Fig.add_subplot(111)
border_polygon = [(-45, -40), (40, -40), (40, 40), (-40, 40), (-40, -40)]
obs_array = list()
#obs_array.append([(-30, 10), (-30, 20), (130, 20), (130, 10), (-30, 10)])
#obs_array.append([(-30, -10), (-30, -20), (130, -20), (130, -10), (-30, -10)])



obs_array.append([(-40, -10), (-35, -10), (-35, -40), (-40, -40), (-40, -10)])
obs_array.append([(-40, 10), (-35, 10), (-35, 40), (-40, 40), (-40, 10)])
#obs_array.append([(-10, -10), (20, -10), (20, 10), (-10, 10), (-10, -10)])
obs_array.append([(-10, -10), (20, -10), (20, -5), (-10, -5), (-10, -10)])
obs_array.append([(20, -5), (20, 20), (15, 20), (15, -5), (20, -5)])
obs_array.append([(-10, 5), (-10, 20), (-5, 20), (-5, 5), (-10, 5)])
obs_array.append([(-10, 20), (20, 20), (20, 25), (-10, 25), (-10, 20)])

obs_array.append([(-40, -40), (-40, -35), (40, -35), (40, -40), (-40, -40)])
obs_array.append([(-40, 40), (-40, 35), (40, 35), (40, 40), (-40, 40)])
obs_array.append([(40, 40), (35, 40), (35, -40), (40, -40), (40, 40)])
#env = Env(border_polygon, obs_array, [[25, 0]], ax, 5)

env = Env([[420, 340]], ax, 10)

#env.plot_border()
env.plot_obs_array()
env.plot_target()
Fig.show()
Fig.canvas.draw()

n = 10
agents = []
for i in range(0, n):
    agent_pos = [[-10000, -10000]]
    while not(env.is_step_legal(agent_pos, [0, 0])):
        agent_pos = [[345, 54]] + 5 * np.random.rand(1, 2)

    agents.append(Agent(i, agent_pos, ax))

Fig.show()
Fig.canvas.draw()
time.sleep(1)

movie_flag = False
if not movie_flag:
    for t in range(1, 500):
        for i in range(0, n):
            agents[i].caculate_step(agents, env)
        Fig.canvas.draw()
        time.sleep(0.1)
        for i in range(0, n):
            agents[i].perform_step(env)

        for i in range(0, n):
            if not agents[i].agent_alive:
                agents[i].AgetPlotHadel.remove()
                agents.remove(agents[i])
                if env.los_to_target_handel.__len__() > 0:
                    env.los_to_target_handel[0].remove()

                n = n-1
                break
else: # Creating a movie

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=10, metadata=metadata)
    with writer.saving(Fig, "writer_test.mp4", 100):
        for t in range(1, 500):
            for i in range(0, n):
                agents[i].caculate_step(agents, env)
            Fig.canvas.draw()
        #time.sleep(0.1)
            for i in range(0, n):
                agents[i].perform_step(env)

            for i in range(0, n):
                if not agents[i].agent_alive:
                    agents[i].AgetPlotHadel.remove()
                    agents.remove(agents[i])
                    if env.los_to_target_handel.__len__() > 0:
                        env.los_to_target_handel[0].remove()

                    n = n - 1
                    break
            writer.grab_frame()