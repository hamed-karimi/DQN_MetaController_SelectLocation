import random
from torch import optim
import torch
from State_batch import State_batch
from DQN import hDQN, weights_init_orthogonal
from ReplayMemory import ReplayMemory
from collections import namedtuple, deque
from torch import nn
import numpy as np
import math


class MetaControllerMemory(ReplayMemory):
    def __init__(self, capacity):
        super().__init__(capacity=capacity)

    def get_transition(self, *args):
        Transition = namedtuple('Transition',
                                ('initial_map', 'initial_need', 'goal_map', 'reward', 'done', 'final_map',
                                 'final_need'))
        return Transition(*args)

    # def update_top_n_experiences(self, n, episode_satisfactions):
    #     temp_experiences = []
    #     top_rewards = deque([], maxlen=n)
    #     for i in range(n-1, -1, -1):
    #         top = self.memory.pop()
    #         top_rewards.append(episode_satisfactions[i:].sum().unsqueeze(dim=0))
    #         temp_experiences.append(top)
    #
    #     for exp in temp_experiences.__reversed__():
    #         self.push_experience(exp.initial_map, exp.initial_need, exp.goal_index, top_rewards.pop() + exp.reward,
    #                              exp.done, exp.final_map, exp.final_need)


class MetaController:

    def __init__(self, batch_size, num_objects, gamma, episode_num, episode_len, memory_capacity,
                 rewarded_action_selection_ratio):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = hDQN().to(self.device)
        self.policy_net.apply(weights_init_orthogonal)
        self.target_net = hDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = MetaControllerMemory(memory_capacity)
        self.rewarded_action_selection_ratio = rewarded_action_selection_ratio
        self.object_type_num = num_objects
        self.steps_done = 0
        self.EPS_START = 0.95
        self.EPS_END = 0.05
        self.episode_num = episode_num
        self.episode_len = episode_len
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.batch_size_mul = 3
        self.epsilon_list = []
        # self.selected_goal = np.ones((self.episode_num, 2)) * -1

    def get_nonlinear_epsilon(self, episode):
        x = math.log(episode + 1, self.episode_num)
        epsilon = -x ** 40 + 1
        return epsilon

    def get_linear_epsilon(self, episode):
        epsilon = self.EPS_START - (episode / self.episode_num) * \
                  (self.EPS_START - self.EPS_END)
        return epsilon

    def get_goal_map(self, environment, agent, episode):
        # epsilon = self.get_nonlinear_epsilon(episode)
        # height = environment.env_map.shape[2]
        # width = environment.env_map.shape[3]
        goal_map = torch.zeros_like(environment.env_map[:, 0, :, :])
        epsilon = self.get_linear_epsilon(episode)
        self.epsilon_list.append(epsilon)
        e = random.random()
        all_object_locations = torch.stack(torch.where(environment.env_map[0, 1:, :, :]), dim=1)
        if e < epsilon:  # random (goal or stay)
            # goal_index = torch.randint(low=0, high=all_object_locations.shape[0], size=(1,))
            # goal_type = all_object_locations[goal_index, 0]
            # goal_location = all_object_locations[goal_index, 1:]
            goal_location = torch.randint(low=0, high=environment.env_map.shape[2], size=(2,))
        else:
            with torch.no_grad():
                env_map = environment.env_map.clone().to(self.device)
                need = agent.need.to(self.device)
                output_values = self.policy_net(env_map, need)
                # object_mask = environment.env_map.sum(dim=1) # Either the agent or an object exists
                object_mask = torch.ones_like(output_values)
                output_values[object_mask == 0] = -math.inf
                goal_location = torch.where(torch.eq(output_values, output_values.max()))
                goal_location = torch.as_tensor([ll[0] for ll in goal_location][1:])
                # goal_type = torch.where(environment.env_map[0, :, goal_location[0, 0], goal_location[0, 1]])[0]
                # goal_type -= 1  # first layer is agent layer but goal index starts from 0

        goal_map[0, goal_location[0], goal_location[1]] = 1
        self.steps_done += 1
        return goal_map, goal_location  #, goal_type

    def save_experience(self, initial_map, initial_need, goal_map, acquired_reward, done, final_map, final_need):
        self.memory.push_experience(initial_map, initial_need, goal_map, acquired_reward, done, final_map, final_need)
        # relu = nn.ReLU()
        # sigmoid = nn.Sigmoid()
        # memory_prob = relu(acquired_reward) + 1  # This should be changed to sigmoid
        # memory_prob = .5 if acquired_reward == 0 else 1/abs(acquired_reward)
        memory_prob = 1
        self.memory.push_selection_ratio(selection_ratio=memory_prob)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize(self):
        if self.memory.__len__() < self.BATCH_SIZE * self.batch_size_mul:
            return float('nan')
        transition_sample = self.memory.sample(self.BATCH_SIZE)
        batch = self.memory.get_transition(*zip(*transition_sample))

        initial_map_batch = torch.cat([batch.initial_map[i] for i in range(len(batch.initial_map))]).to(self.device)
        initial_need_batch = torch.cat([batch.initial_need[i] for i in range(len(batch.initial_need))]).to(self.device)
        goal_map_batch = torch.cat(batch.goal_map).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        final_map_batch = torch.cat([batch.final_map[i] for i in range(len(batch.final_map))]).to(self.device)
        final_need_batch = torch.cat([batch.final_need[i] for i in range(len(batch.final_need))]).to(self.device)
        # final_map_object_mask_batch = final_map_batch.sum(dim=1)


        policynet_goal_values_of_initial_state = self.policy_net(initial_map_batch,
                                                                 initial_need_batch).to(self.device)
        targetnet_goal_values_of_final_state = self.target_net(final_map_batch,
                                                               final_need_batch).to(self.device)

        final_map_object_mask_batch = torch.ones_like(targetnet_goal_values_of_final_state)
        targetnet_goal_values_of_final_state[final_map_object_mask_batch == 0] = -math.inf

        targetnet_max_goal_value = torch.amax(targetnet_goal_values_of_final_state,
                                              dim=(1, 2)).detach().float()
        goal_values_of_selected_goals = policynet_goal_values_of_initial_state[goal_map_batch == 1]
        expected_goal_values = targetnet_max_goal_value * self.GAMMA + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(goal_values_of_selected_goals, expected_goal_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss
