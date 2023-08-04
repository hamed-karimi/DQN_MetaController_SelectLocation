# import math
#
# from Visualizer import Visualizer
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from State_batch import State_batch
# from matplotlib.ticker import FormatStrFormatter
#
#
# class ControllerVisualizer(Visualizer):
#     def __init__(self, utility):
#         super().__init__(utility)
#         self.episode_num = utility.params.CONTROLLER_EPISODE_NUM
#         self.row_num = self.height - 1
#         self.col_num = self.width + 2
#         self.scale = self.row_num * self.col_num / (self.width//2)
#         self.asterisk_size = (1/math.log(self.row_num * self.col_num, 500))**(self.width/2)
#         self.episode_chunk_rewards = []
#
#
#     def get_greedy_values_figure(self, controller):
#         fig, ax = plt.subplots(self.row_num, self.col_num, figsize=(15, 10))
#         # mpl.rcParams['ytick.major.pad'] = 1
#         r, c = 0, 0
#         for x in range(self.height):
#             for y in range(self.width):
#                 action_values = torch.zeros((self.height, self.width))
#                 which_action = torch.zeros((self.height, self.width), dtype=torch.int32)
#                 for i in range(self.height):
#                     for j in range(self.width):
#                         env_map = torch.zeros((1, 2, self.height, self.width))
#                         env_map[0, 0, i, j] = 1  # Agent
#                         env_map[0, 1, x, y] = 1  # Object
#                         with torch.no_grad():
#                             state = State_batch(env_map.to(self.device), None)
#                             controller_values = controller.policy_net(state)
#                             action_mask = torch.tensor(self.action_mask[i, j, :, :])
#                             controller_values[torch.logical_not(action_mask.bool())] = -3.40e+38
#                             action_values[i, j], which_action[i, j] = controller_values.max(1)[0], \
#                                                                       controller_values.max(1)[1].detach()
#                 Xs = np.arange(0, self.height, 1)
#                 Ys = np.arange(0, self.width, 1)
#                 arrows_x = np.zeros((self.height, self.width))
#                 arrows_y = np.zeros((self.height, self.width))
#                 for i in range(self.height):
#                     for j in range(self.width):
#                         arrows_y[i, j] = -self.allactions[which_action[i, j].int()][0, 0]
#                         arrows_x[i, j] = self.allactions[which_action[i, j].int()][0, 1]
#
#                 fig_num = x*self.height + y
#                 r = fig_num // self.col_num
#                 c = fig_num % self.col_num
#                 ax[r, c].quiver(Xs, Ys, arrows_x, arrows_y, scale=self.scale, headwidth=1.5*self.asterisk_size)
#                 ax[r, c].set_xticks([])
#                 ax[r, c].set_yticks([])
#                 ax[r, c].invert_yaxis()
#                 ax[r, c].scatter(y, x, marker='*', s=4*self.asterisk_size, facecolor=[1, 0, .2])
#                 ax[r, c].set_box_aspect(aspect=1)
#
#         plt.tight_layout(pad=0.1) #, w_pad=.05, h_pad=.05)
#         fig.subplots_adjust(wspace=0.1, hspace=0.1)
#         return fig, ax, r, c+1
#
#     def get_epsilon_plot(self, ax, r, c, steps_done, **kwargs):
#         ax[r, c].scatter(np.arange(steps_done), kwargs['controller_epsilons'], s=.1)
#         ax[r, c].tick_params(axis='both', which='major', labelsize=5, pad=.5)
#         ax[r, c].set_title('Epsilon', fontsize=8, pad=1)
#         ax[r, c].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#         ax[r, c].set_box_aspect(aspect=1)
#         return ax, r, c + 1
#
#     def get_reward_plot(self, ax, r, c, controller_reward):
#         self.episode_chunk_rewards.append(controller_reward)
#         ax[r, c].plot(self.episode_chunk_rewards)
#         ax[r, c].tick_params(axis='both', which='major', labelsize=5, pad=.5)
#         ax[r, c].set_title('Episode reward', fontsize=8, pad=1)
#         ax[r, c].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#         ax[r, c].set_box_aspect(aspect=1)
#         return ax, r, c + 1