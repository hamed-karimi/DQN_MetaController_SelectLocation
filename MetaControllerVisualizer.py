import itertools
import math
from Visualizer import Visualizer
import numpy as np
import torch
import matplotlib.pyplot as plt
from State_batch import State_batch
from copy import deepcopy
from matplotlib.ticker import FormatStrFormatter


def get_predefined_needs(num_object):
    temp_need = [[-10, -5, 0, 5, 10]] * num_object
    need_num = len(temp_need[0]) ** num_object
    need_batch = torch.zeros((need_num, num_object))
    ns = np.zeros((1, num_object))
    for i, ns in enumerate(itertools.product(*temp_need)):
        need_batch[i, :] = torch.tensor(ns)
    return need_batch


class MetaControllerVisualizer(Visualizer):
    def __init__(self, utility):
        super().__init__(utility)
        self.episode_num = utility.params.META_CONTROLLER_EPISODE_NUM
        allactions_np = [np.array([0, 0]), np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1]),
                         np.array([1, 1]), np.array([-1, -1]), np.array([-1, 1]), np.array([1, -1])]
        self.allactions = [torch.from_numpy(x).unsqueeze(0) for x in allactions_np]
        # self.action_mask = np.zeros((self.height, self.width, 1, len(self.allactions)))
        self.initialize_action_masks()
        self.needs = get_predefined_needs(self.object_type_num)
        self.color_options = [[1, 0, .2], [0, .8, .2], [0, 0, 0]]
        self.goal_shape_options = ['*', 's', 'P', 'o', 'D', 'X']
        self.objects_color_name = ['red', 'green', 'black']  # 2: stay
        self.row_num = 5
        self.col_num = 6

    def get_figure_title(self, need):
        title = '$n_{0}: {1:.2f}'.format('{' + self.objects_color_name[0] + '}', need[0])
        for i in range(1, self.object_type_num):
            title += ", n_{0}: {1:.2f}$".format('{' + self.objects_color_name[i] + '}', need[i])
        return title

    def get_agent_goal_map(self, env_map, goal_location):
        agent_goal_map = torch.zeros_like(env_map[:, 1:, :, :])
        agent_goal_map[0, 0, :, :] = env_map[0, 0, :, :]
        agent_goal_map[0, 1, goal_location[0, 0], goal_location[0, 1]] = 1
        return agent_goal_map

    def get_object_shape_dictionary(self, environment):
        shape_map = dict()
        for obj_type in range(self.object_type_num):
            for at_obj in range(environment.each_type_object_num[obj_type]):
                key = tuple(environment.object_locations[obj_type, at_obj].tolist())
                shape_map[key] = self.goal_shape_options[at_obj]
        return shape_map

    def get_goal_directed_actions(self, environment, meta_controller, controller):
        # which_action = torch.zeros((self.height, self.width), dtype=torch.int16)
        which_goal = np.empty((self.height, self.width), dtype=str)
        row_num = 5
        col_num = 5
        fig, ax = plt.subplots(row_num, col_num, figsize=(15, 12))

        for fig_num, need in enumerate(self.needs):
            r = fig_num // col_num
            c = fig_num % col_num
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            ax[r, c].invert_yaxis()
            for i in range(self.height):
                for j in range(self.width):

                    shape_map = self.get_object_shape_dictionary(environment=environment)
                    env_map = torch.zeros((1, 1 + self.object_type_num, self.height, self.width))  # +1 for agent layer
                    env_map[0, 0, i, j] = 1
                    env_map[0, 1:, :, :] = deepcopy(environment.env_map[0, 1:, :, :])
                    shape_map[(i, j)] = '.'  # Staying
                    with torch.no_grad():
                        output_values = meta_controller.policy_net(env_map.to(self.device),
                                                                   need.unsqueeze(0).to(self.device)).clone()  # 1 * 3
                        object_mask = env_map.sum(dim=1)
                        output_values[object_mask == 0] = -math.inf
                        goal_location = torch.where(torch.eq(output_values, output_values.max()))
                        goal_location = torch.as_tensor(goal_location[1:])
                        which_goal[i, j] = shape_map[tuple(goal_location.tolist())]
                        goal_type = torch.where(env_map[0, :, goal_location[0], goal_location[1]])[0].min()
                        goal_type = 2 if goal_type == 0 else goal_type - 1
                        selected_goal_shape = shape_map[tuple(goal_location.tolist())]
                        size = 10 if selected_goal_shape == '.' else 50
                        ax[r, c].scatter(j, i,
                                         marker=selected_goal_shape,
                                         s=size,
                                         alpha=0.4,
                                         facecolor=self.color_options[goal_type])

            ax[r, c].set_title(self.get_figure_title(need), fontsize=10)

            for obj_type in range(self.object_type_num):
                for obj in range(environment.object_locations.shape[1]):
                    if environment.object_locations[obj_type, obj, 0] == -1:
                        break
                    ax[r, c].scatter(environment.object_locations[obj_type, obj, 1],
                                     environment.object_locations[obj_type, obj, 0],
                                     marker=self.goal_shape_options[obj],
                                     s=200,
                                     edgecolor=self.color_options[obj_type],
                                     facecolor='none')
            ax[r, c].tick_params(length=0)
            ax[r, c].set(adjustable='box')
        plt.tight_layout(pad=0.1, w_pad=6, h_pad=1)
        return fig, ax

    def add_needs_plot(self, ax, agent_needs, global_index, r, c):
        ax[r, c].set_prop_cycle('color', self.color_options)
        ax[r, c].plot(agent_needs[:global_index, :], linewidth=.1)
        ax[r, c].tick_params(axis='both', which='major', labelsize=9)
        ax[r, c].set_title('Needs', fontsize=9)
        return ax, r, c + 1

    def get_epsilon_plot(self, ax, r, c, steps_done, **kwargs):
        ax[r, c].scatter(np.arange(steps_done), kwargs['meta_controller_epsilon'], s=.1)
        ax[r, c].tick_params(axis='both', which='major', labelsize=9)
        ax[r, c].set_title('Meta Controller Epsilon', fontsize=9)
        ax[r, c].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        # ax[r, c].set_box_aspect(aspect=1)
        return ax, r, c + 1

    def policynet_values(self, object_locations, object_layers, meta_controller):
        num_object = object_locations.shape[0]
        row_num = 5
        col_num = 5
        fig, ax = plt.subplots(row_num, col_num, figsize=(15, 12))
        for fig_num, need in enumerate(self.needs):
            goals_values_text = []
            for i in range(self.height):
                row_table_texts = []
                for j in range(self.width):
                    env_map = torch.zeros((1, 1 + num_object, self.height, self.width))  # +1 for agent layer
                    env_map[0, 0, i, j] = 1
                    env_map[0, 1:, :, :] = deepcopy(object_layers)
                    with torch.no_grad():
                        state = State_batch(env_map.to(self.device), need.unsqueeze(0).to(self.device))
                        goals_values = meta_controller.policy_net(state).clone()  # 1 * 3
                        row_table_texts.append(
                            '\n'.join([str(round(goals_values[0, v].item(), 2)) for v in range(num_object + 1)]))
                goals_values_text.append(row_table_texts)

            r = fig_num // col_num
            c = fig_num % col_num

            values_table = ax[r, c].table(cellText=goals_values_text,
                                          rowLabels=np.arange(self.height),
                                          colLabels=np.arange(self.width),
                                          # colWidths=1,
                                          loc='center')
            values_table.auto_set_font_size(False)
            values_table.set_fontsize(8)
            values_table.scale(1, 2)
            ax[r, c].set_title(self.get_figure_title(need), fontsize=10)
            ax[r, c].axis('off')

        plt.tight_layout(pad=0.1, w_pad=1, h_pad=1)
        return fig, ax

    def add_needs_difference_hist(self, ax, agent_needs, needs_range, global_index, r, c):
        ax[r, c].set_prop_cycle('color', self.color_options)
        ax[r, c].hist(agent_needs[:global_index, 0] - agent_needs[:global_index, 1],
                      bins=np.linspace(needs_range[0] - needs_range[1], needs_range[1] - needs_range[0], 49),
                      linewidth=.1)
        ax[r, c].tick_params(axis='both', which='major', labelsize=9)
        ax[r, c].set_title('Needs', fontsize=9)
        return ax, r, c + 1
