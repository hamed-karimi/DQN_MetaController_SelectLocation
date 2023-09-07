import numpy as np
import torch
from torch.nn import ReLU, Sigmoid
import random
from copy import deepcopy
from itertools import product


class Agent:
    def __init__(self, h, w, n, lambda_need, prob_init_needs_equal, predefined_location, preassigned_needs,
                 rho_function='ReLU',epsilon_function='Linear'):  # n: number of needs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = h
        self.width = w
        self.location = self.initial_location(predefined_location)
        self.num_need = n
        self.initial_range_of_need = [-12, 20]
        self.range_of_need = [-12, 12]
        self.prob_init_needs_equal = prob_init_needs_equal
        self.need = self.set_need(preassigned_needs)
        self.steps_done = 0
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.lambda_need = lambda_need  # How much the need increases after each action
        self.lambda_satisfaction = 3
        self.lambda_cost = 1
        self.no_reward_threshold = -10
        self.relu = ReLU()
        total_need_functions = {'ReLU': self.relu, 'PolyReLU': self.poly_relu}
        self.rho_function = total_need_functions[rho_function]
        # self.total_need = self.get_total_need()
        possible_h_w = [list(range(h)), list(range(w))]
        self.epsilon_function = epsilon_function
        self.all_locations = torch.from_numpy(np.array([element for element in product(*possible_h_w)]))

    def poly_relu(self, x, p=2):
        return self.relu(x) ** p

    def set_need(self, preassigned_needs=None):
        if any(preassigned_needs):
            need = torch.tensor(preassigned_needs)
        else:
            p = random.uniform(0, 1)
            if p <= self.prob_init_needs_equal:
                need = torch.rand((1, self.num_need))
                need[0, 1:] = need[0, 0]
            else:
                need = torch.rand((1, self.num_need))
            need = (self.initial_range_of_need[1] - self.initial_range_of_need[0]) * need + self.initial_range_of_need[0]
        return need

    def initial_location(self, predefined_location): # predefined_location is a list
        if len(predefined_location[0]) > 0:
            return torch.tensor(predefined_location)
        return torch.from_numpy(np.asarray((np.random.randint(self.height), np.random.randint(self.width)))).unsqueeze(0)

    def update_need_after_step(self, time_past):
        for i in range(self.num_need):
            self.need[0, i] += (self.lambda_need * time_past)

    def reward_function(self, need):
        x = need.clone()
        pos = self.relu(x)
        # neg = (torch.pow(1.1, x)-1) * 7
        neg = x
        neg[x > 0] = 0
        neg[x < self.no_reward_threshold] = self.no_reward_threshold #(pow(1.1, self.no_reward_threshold) - 1) * 7
        return pos + neg
        # positive_part = torch.minimum(reward, self.relu(self.need))
        # negative_part = sig_lin(self.need) - sig_lin(self.need - reward)
        # return sig_lin(positive_part) + abs(negative_part)

    def update_need_after_reward(self, reward):
        adjusted_reward = self.reward_function(self.need) - self.reward_function(self.need - reward)
        self.need = self.need - adjusted_reward
        # self.need = self.need - reward
        # for i in range(self.num_need):
        #     self.need[0, i] = max(self.need[0, i], -10)

    def get_positive_needs(self):
        total_need = self.rho_function(self.need)  # .sum().squeeze()
        return total_need

    def take_action(self, environment, action_id):
        # Consider this for a change: the needs cost is NOT the carried need,
        # and is the need you end up with after the action

        selected_action = environment.allactions[action_id].squeeze()  # to device
        self.location[0, :] += selected_action
        at_cost = environment.get_cost(action_id)
        # time_passed = 1. if at_cost < 1.4 else at_cost
        time_passed = 1
        # carried_total_need = self.get_total_need()
        moving_cost = self.lambda_cost * at_cost
        # needs_cost = time_passed * carried_total_need

        environment.update_agent_location_on_map(self)
        self.update_need_after_step(time_passed)
        last_total_need = self.get_positive_needs().sum().squeeze()

        f, _ = environment.get_reward()
        self.update_need_after_reward(f)
        at_needs = self.get_positive_needs()
        needs_cost = at_needs
        at_total_need = at_needs.sum().squeeze()
        satisfaction = self.relu(last_total_need - at_total_need) * self.lambda_satisfaction
        # total_cost = (-1) * at_cost * at_total_need - at_cost
        # rho = (-1) * total_cost + satisfaction * self.lambda_satisfaction
        # return rho.unsqueeze(0), satisfaction
        return satisfaction, moving_cost, needs_cost.sum().squeeze()
