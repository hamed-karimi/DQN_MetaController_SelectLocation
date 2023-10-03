import random
from torch import optim
import torch
from torch.optim.lr_scheduler import MultiplicativeLR
from DQN import hDQN, weights_init_orthogonal
from ReplayMemory import ReplayMemory
from collections import namedtuple
from torch import nn
import math


class MetaControllerMemory(ReplayMemory):
    def __init__(self, capacity, first_steps_sample_ratio):
        super().__init__(capacity=capacity,
                         first_steps_sample_ratio=first_steps_sample_ratio)

    def get_transition(self, *args):
        Transition = namedtuple('Transition',
                                ('initial_map', 'initial_need', 'goal_map', 'reward', 'n_steps', 'dt', 'final_map',
                                 'final_need'))
        return Transition(*args)


class MetaController:

    def __init__(self, params, pre_trained_weights_path=''):
        self.env_height = params.HEIGHT
        self.env_width = params.WIDTH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = hDQN(params).to(self.device)
        if pre_trained_weights_path != "":
            self.policy_net.load_state_dict(torch.load(pre_trained_weights_path,
                                                       map_location=self.device))
        else:
            self.policy_net.apply(weights_init_orthogonal)
        self.target_net = hDQN(params).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = MetaControllerMemory(params.META_CONTROLLER_MEMORY_CAPACITY,
                                           params.FIRST_STEP_SAMPLING_RATIO)

        self.object_type_num = params.OBJECT_TYPE_NUM
        self.steps_done = 0
        self.EPS_START = 0.95
        self.EPS_END = 0.05
        self.episode_num = params.META_CONTROLLER_EPISODE_NUM
        self.episode_len = params.EPISODE_LEN
        self.target_net_update = params.META_CONTROLLER_TARGET_UPDATE
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=params.INIT_LEARNING_RATE)
        self.lr_scheduler = MultiplicativeLR(self.optimizer,
                                             lambda epoch: 1 / (1 + params.LEARNING_RATE_DECAY * epoch),
                                             last_epoch=-1, verbose=False)
        self.BATCH_SIZE = params.META_CONTROLLER_BATCH_SIZE
        self.gammas = [0.]
        self.gamma_episodes = [0]
        self.gamma_delay_episodes = [0]
        self.gamma_max_delay = params.GAMMA_CASCADE_DELAY
        self.gamma_reached_max_episode = []
        self.gamma_cascade = params.GAMMA_CASCADE
        self.max_gamma = params.MAX_GAMMA
        self.max_step_num = params.MAX_STEP_NUM
        self.min_gamma = 0
        # self.GAMMA = 0 if self.gamma_cascade else self.max_gamma
        self.batch_size_mul = 3
        self.epsilon_list = []

    def gamma_function(self, episode):
        m = 2
        ratio = m / self.target_net_update
        gamma = min(1 / (1 + math.exp(-episode * ratio + math.exp(2.3))),
                    self.max_gamma)
        return gamma

    def update_gammas(self):
        if self.gamma_cascade:
            if len(self.gammas) <= self.max_step_num:
                for g in range(len(self.gammas)):
                    self.gammas[g] = self.gamma_function(self.gamma_episodes[g])
                    self.gamma_episodes[g] += 1
            if self.gammas[-1] == self.max_gamma and len(self.gammas) < self.max_step_num and self.gamma_delay_episodes[
                -1] < self.gamma_max_delay:
                self.gamma_delay_episodes[-1] += 1

            if self.gammas[-1] == self.max_gamma and len(self.gammas) < self.max_step_num and self.gamma_delay_episodes[
                -1] == self.gamma_max_delay:
                self.gammas.append(self.min_gamma)
                self.gamma_episodes.append(0)
                self.gamma_delay_episodes.append(0)

    def get_nonlinear_epsilon(self, episode):
        x = math.log(episode + 1, self.episode_num)
        epsilon = -x ** 40 + 1
        return epsilon

    def get_linear_epsilon(self, episode):
        # return 1-self.gammas[-1]
        epsilon = self.EPS_START - (episode / self.episode_num) * \
                  (self.EPS_START - self.EPS_END)
        return epsilon

    def get_goal_map(self, environment, agent, episode, epsilon=None):
        # epsilon = self.get_nonlinear_epsilon(episode)
        goal_map = torch.zeros_like(environment.env_map[:, 0, :, :])
        if epsilon is None:
            epsilon = self.get_linear_epsilon(episode)

        self.epsilon_list.append(epsilon)
        e = random.random()
        all_object_locations = torch.stack(torch.where(environment.env_map[0, 1:, :, :]), dim=1)
        if e < epsilon:  # random (goal or stay)
            stay_prob = .3
            if random.random() <= stay_prob:  # Stay
                goal_location = environment.agent_location.squeeze()
            else:  # Object
                goal_index = torch.randint(low=0, high=all_object_locations.shape[0], size=())
                goal_location = all_object_locations[goal_index, 1:]

            # goal_location = torch.randint(low=0, high=environment.env_map.shape[2], size=(2,))
        else:
            self.policy_net.eval()
            with torch.no_grad():
                env_map = environment.env_map.clone().to(self.device)
                need = agent.need.to(self.device)
                output_values = self.policy_net(env_map, need)
                object_mask = environment.env_map.sum(dim=1)  # Either the agent or an object exists
                # object_mask = torch.ones_like(output_values)
                output_values[object_mask < 1] = -math.inf
                goal_location = torch.where(torch.eq(output_values, output_values.max()))
                goal_location = torch.as_tensor([ll[0] for ll in goal_location][1:])
                # goal_type = torch.where(environment.env_map[0, :, goal_location[0, 0], goal_location[0, 1]])[0]
                # goal_type -= 1  # first layer is agent layer but goal index starts from 0
        goal_map[0, goal_location[0], goal_location[1]] = 1
        self.steps_done += 1
        return goal_map, goal_location  # , goal_type

    def save_experience(self, initial_map, initial_need, goal_map, acquired_reward, n_steps, dt, final_map, final_need):
        self.memory.push_experience(initial_map, initial_need, goal_map, acquired_reward, n_steps, dt, final_map,
                                    final_need)
        memory_prob = 1
        self.memory.push_selection_ratio(selection_ratio=memory_prob)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize(self, episode):
        if self.memory.__len__() < self.BATCH_SIZE * self.batch_size_mul:
            return float('nan')
        transition_sample = self.memory.sample(self.BATCH_SIZE)
        batch = self.memory.get_transition(*zip(*transition_sample))
        self.policy_net.train()

        initial_map_batch = torch.cat([batch.initial_map[i] for i in range(len(batch.initial_map))]).to(self.device)
        initial_need_batch = torch.cat([batch.initial_need[i] for i in range(len(batch.initial_need))]).to(self.device)
        goal_map_batch = torch.cat(batch.goal_map).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        n_steps_batch = torch.cat(batch.n_steps).to(self.device)
        final_map_batch = torch.cat([batch.final_map[i] for i in range(len(batch.final_map))]).to(self.device)
        final_need_batch = torch.cat([batch.final_need[i] for i in range(len(batch.final_need))]).to(self.device)
        final_map_object_mask_batch = final_map_batch.sum(dim=1)

        policynet_goal_values_of_initial_state = self.policy_net(initial_map_batch,
                                                                 initial_need_batch).to(self.device)
        targetnet_goal_values_of_final_state = self.target_net(final_map_batch,
                                                               final_need_batch).to(self.device)

        targetnet_goal_values_of_final_state[final_map_object_mask_batch < 1] = -math.inf

        targetnet_max_goal_value = torch.amax(targetnet_goal_values_of_final_state,
                                              dim=(1, 2)).detach().float()
        goal_values_of_selected_goals = policynet_goal_values_of_initial_state[goal_map_batch == 1]

        steps_discounts = torch.zeros(reward_batch.shape,
                                      device=self.device)
        steps_discounts[:, :len(self.gammas)] = torch.as_tensor(self.gammas, device=self.device)
        steps_discounts = torch.cat([torch.ones(steps_discounts.shape[0], 1, device=self.device),
                                     steps_discounts], dim=1)  # step reward is not discounted

        cum_steps_discounts = torch.cumprod(steps_discounts, dim=1)[:, :-1]  # step reward is not discounted, so the
                                                                             # num of gammas for discounting rewards is
                                                                             # one less than for Q
        discounted_reward = (reward_batch * cum_steps_discounts).sum(dim=1)

        q_gammas = torch.cumprod(steps_discounts[:, 1:], dim=1).gather(dim=1,
                                                                       index=n_steps_batch.unsqueeze(dim=1).long()-1)

        expected_goal_values = targetnet_max_goal_value * q_gammas.squeeze() + discounted_reward

        criterion = nn.SmoothL1Loss()
        loss = criterion(goal_values_of_selected_goals, expected_goal_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.update_gammas()
        return loss
