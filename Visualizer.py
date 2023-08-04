import math
import numpy as np
import torch
from copy import deepcopy
from State_batch import State_batch
import matplotlib.pyplot as plt
import itertools
from matplotlib.ticker import FormatStrFormatter


def get_predefined_needs():
    temp_need = [[-10, -5, 0, 5, 10]] * 2
    need_num = len(temp_need[0]) ** 2
    need_batch = torch.zeros((need_num, 2))
    for i, (n1, n2) in enumerate(itertools.product(*temp_need)):
        need_batch[i, :] = torch.tensor([n1, n2])
    return need_batch


def get_reward_plot(ax, r, c, **kwargs):
    ax[r, c].plot(kwargs['reward'], linewidth=1)
    ax[r, c].set_title(kwargs['title'], fontsize=9)
    # ax[r, c].set_box_aspect(aspect=1)
    c += 1
    return ax, r, c


def get_loss_plot(ax, r, c, **kwargs):
    ax[r, c].plot(kwargs['loss'], linewidth=1)
    ax[r, c].set_title(kwargs['title'], fontsize=9)
    # ax[r, c].set_box_aspect(aspect=1)
    c += 1
    return ax, r, c


class Visualizer:
    def __init__(self, utility):
        params = utility.params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = params.HEIGHT
        self.width = params.WIDTH
        self.object_type_num = params.OBJECT_TYPE_NUM
        allactions_np = [np.array([0, 0]), np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1]),
                         np.array([1, 1]), np.array([-1, -1]), np.array([-1, 1]), np.array([1, -1])]
        self.allactions = [torch.from_numpy(x).unsqueeze(0) for x in allactions_np]
        self.action_mask = np.zeros((self.height, self.width, 1, len(self.allactions)))
        self.initialize_action_masks()
        self.episode_num = None
        self.episode_len = params.EPISODE_LEN

    def get_epsilon_plot(self, ax, r, c, steps_done, **kwargs):
        pass

    def initialize_action_masks(self):
        for i in range(self.height):
            for j in range(self.width):
                agent_location = torch.tensor([[i, j]])
                aa = np.ones((agent_location.size(0), len(self.allactions)))
                for ind, location in enumerate(agent_location):
                    if location[0] == 0:
                        aa[ind, 2] = 0
                        aa[ind, 6] = 0
                        aa[ind, 7] = 0
                    if location[0] == self.height - 1:
                        aa[ind, 1] = 0
                        aa[ind, 5] = 0
                        aa[ind, 8] = 0
                    if location[1] == 0:
                        aa[ind, 4] = 0
                        aa[ind, 6] = 0
                        aa[ind, 8] = 0
                    if location[1] == self.width - 1:
                        aa[ind, 3] = 0
                        aa[ind, 5] = 0
                        aa[ind, 7] = 0
                self.action_mask[i, j, :, :] = aa
