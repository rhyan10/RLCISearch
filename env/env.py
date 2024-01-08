import torch
import torch.nn.functional as F
import numpy as np
import dill
import schnetpack.transform as trn
import numpy as np
import ase.db
from schnetpack.interfaces import AtomsConverter as SchNetConverter
from ase.optimize import BFGS
from math import sqrt
import random
from ase import Atoms
import os
import sys
import chemcoord as cc
import copy
from ase.visualize import view
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation


def exploding_function(x):
    return np.piecewise(x, [x < 0.05, x >= 0.05], [lambda x: 1, np.exp(1000*(0.05-x))])

def exploding_function_dist(x):
    return np.piecewise(x, [x < 1.5, x >= 1.5], [lambda x: np.exp(1000*(x-1.5)), 1])

class Env():

    def __init__(self, batch_size, device):
        self.device = device
        self.batch_size = batch_size
        self.scaling = 0.5

    def step(self, actions, scaler, mols, o_mols, std_e):
        reward = scaler
        new_mols = []
        theta = actions['theta'].detach().cpu().numpy()
        phi = actions['phi'].detach().cpu().numpy()
        all_mins = []
        for i, mol in enumerate(mols):
            o_pos = copy.deepcopy(mol.positions)
            n_pos = copy.deepcopy(mol.positions)

            curr_theta = theta[i] * self.scaling * reward.detach().cpu().numpy()[i]
            curr_phi = phi[i] * self.scaling * reward.detach().cpu().numpy()[i]
            #print(curr_theta)

            not_parallel_vector = np.array([1, 0, 0])
            vec_01 = (o_pos[0] - o_pos[1]) / np.linalg.norm(o_pos[0] - o_pos[1])

            # Take the cross product to find a vector perpendicular to both v and not_parallel_vector
            u = np.cross(vec_01, not_parallel_vector)
            vec_p0 = u + o_pos[0]
            vec_p1 = u + o_pos[1]
            rotation_012 = Rotation.from_rotvec(curr_theta[2] * vec_01)
            rotation_matrix_012 = rotation_012.as_matrix()
            rotation_013 = Rotation.from_rotvec(curr_theta[3] * vec_01)
            rotation_matrix_013 = rotation_013.as_matrix()
            rotation_014 = Rotation.from_rotvec(curr_theta[4] * vec_01)
            rotation_matrix_014 = rotation_014.as_matrix()
            rotation_015 = Rotation.from_rotvec(curr_theta[5] * vec_01)
            rotation_matrix_015 = rotation_015.as_matrix()
            rotation_502 = Rotation.from_rotvec(curr_phi[2] * vec_p0)
            rotation_matrix_502 = rotation_502.as_matrix()
            rotation_205 = Rotation.from_rotvec(curr_phi[5] * vec_p0)
            rotation_matrix_205 = rotation_205.as_matrix()
            rotation_314 = Rotation.from_rotvec(curr_phi[4] * vec_p1)
            rotation_matrix_314 = rotation_314.as_matrix()
            rotation_413 = Rotation.from_rotvec(curr_phi[5] * vec_p1)
            rotation_matrix_413 = rotation_413.as_matrix()

#            print(rotation_matrix_012)

            m2 = np.matmul(rotation_matrix_012, rotation_matrix_502)
            n_pos[2] = m2.dot(o_pos[2])

            m3 = np.matmul(rotation_matrix_013, rotation_matrix_413)
            n_pos[3] = m3.dot(o_pos[3])

            m4 = np.matmul(rotation_matrix_014, rotation_matrix_314)
            n_pos[4] = m4.dot(o_pos[4])

            m5 = np.matmul(rotation_matrix_015, rotation_matrix_205)
            n_pos[5] = m5.dot(o_pos[5])

            b_dist = np.linalg.norm(n_pos[0] -  n_pos[1])
            b_unitvect = (n_pos[0] -  n_pos[1])/b_dist

            new_dist = (theta[i][0] + theta[i][1] + phi[i][0] + phi[i][1])* 0.1 * self.scaling * 0.5 + b_dist

            if (new_dist < 2.43304128 or new_dist > 5):
                new_dist = 3.52320655

            diff_dist = new_dist - b_dist
            n_pos[1] = n_pos[1] - diff_dist * b_unitvect
            n_pos[3] = n_pos[3] - diff_dist * b_unitvect
            n_pos[4] = n_pos[4] - diff_dist * b_unitvect

            d1 = np.linalg.norm(o_pos[0] - o_pos[2])
            d2 = np.linalg.norm(o_pos[0] - o_pos[5])
            d3 = np.linalg.norm(o_pos[1] - o_pos[3])
            d4 = np.linalg.norm(o_pos[1] - o_pos[4])
            v1 = n_pos[2] - n_pos[0]
            v2 = n_pos[5] - n_pos[0]
            v3 = n_pos[3] - n_pos[1]
            v4 = n_pos[4] - n_pos[1]
            u1 = (v1 / np.linalg.norm(v1))*d1 + n_pos[0]
            u2 = (v2 / np.linalg.norm(v2))*d2 + n_pos[0]
            u3 = (v3 / np.linalg.norm(v3))*d3 + n_pos[1]
            u4 = (v4 / np.linalg.norm(v4))*d4 + n_pos[1]
            n_pos[2] = u1
            n_pos[5] = u2
            n_pos[3] = u3
            n_pos[4] = u4

#            print(n_pos)
            min_distance = np.min(pdist(n_pos))
#            #all_mins.append(min_distance)
#            dist_m = np.array(squareform(pdist(n_pos)))
#            print(curr_theta)
#            for j, d1 in enumerate(dist_m):
#                for k, d2 in enumerate(d1):
#                    if 0 < d2 < 1.5:
#                        energy_scaling = exploding_function(d2)
#                        curr_theta[j] = curr_theta[j] * energy_scaling
#                        curr_phi[j] = curr_phi[j] * energy_scaling
#            print(std_e[i])


            energy_scaling = exploding_function(std_e[i])
            energy_scaling_dist = exploding_function_dist(min_distance)
            #print(energy_scaling)
            #print(energy_scaling_dist)
            if energy_scaling_dist < 1 or energy_scaling < 1:
                new_mols.append(copy.deepcopy(o_mols[i]))
            else:
                mol.positions = n_pos
                new_mols.append(copy.deepcopy(mol))

        return reward, new_mols
