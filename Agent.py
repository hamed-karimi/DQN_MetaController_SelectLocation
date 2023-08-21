import numpy as np
import torch
from torch.nn import ReLU
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
        self.initial_range_of_need = [-12, 12]
        self.range_of_need = [-12, 12]
        self.prob_init_needs_equal = prob_init_needs_equal
        self.need = self.set_need(preassigned_needs)
        self.steps_done = 0
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.lambda_need = lambda_need  # How much the need increases after each action
        self.lambda_satisfaction = 3
        self.relu = ReLU()
        total_need_functions = {'ReLU': self.relu, 'PolyReLU': self.poly_relu}
        self.rho_function = total_need_functions[rho_function]
        self.total_need = self.get_total_need()
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

    def update_need_after_reward(self, reward):
        self.need = self.need - reward
        for i in range(self.num_need):
            self.need[0, i] = max(self.need[0, i], -12)

    def get_total_need(self):
        total_need = self.rho_function(self.need).sum().squeeze()
        return total_need

    def take_action(self, environment, action_id):
        selected_action = environment.allactions[action_id].squeeze()  # to device
        self.location[0, :] += selected_action
        at_cost = environment.get_cost(action_id)
        time_past = 1. if at_cost < 1.4 else at_cost
        self.update_need_after_step(time_past)
        last_total_need = self.get_total_need()
        total_cost = time_past * last_total_need - at_cost
        environment.update_agent_location_on_map(self)

        f, _ = environment.get_reward()
        self.update_need_after_reward(f)
        at_total_need = self.get_total_need()
        satisfaction = self.relu(last_total_need - at_total_need)
        # total_cost = (-1) * at_cost * at_total_need - at_cost
        rho = (-1) * total_cost + satisfaction * self.lambda_satisfaction
        self.total_need = deepcopy(at_total_need)
        return rho.unsqueeze(0), satisfaction
