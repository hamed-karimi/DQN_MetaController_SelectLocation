import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

import Utilities


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


# meta controller network
class hDQN(nn.Module):
    def __init__(self):
        utilities = Utilities.Utilities()
        self.params = utilities.params
        super(hDQN, self).__init__()
        env_layer_num = self.params.OBJECT_TYPE_NUM + 1  # +1 for agent layer
        kernel_size = 4
        self.conv1 = nn.Conv2d(in_channels=env_layer_num,
                               out_channels=self.params.DQN_CONV1_OUT_CHANNEL,
                               kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(in_channels=self.params.DQN_CONV1_OUT_CHANNEL,
                               out_channels=self.params.DQN_CONV1_OUT_CHANNEL,
                               kernel_size=kernel_size-1)
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size-2,
                                     stride=1)
        self.conv3 = nn.Conv1d(in_channels=self.params.DQN_CONV1_OUT_CHANNEL,
                               out_channels=self.params.DQN_CONV1_OUT_CHANNEL,
                               kernel_size=4+self.params.OBJECT_TYPE_NUM)

        self.fc1 = nn.Linear(in_features=self.params.DQN_CONV1_OUT_CHANNEL, #params.DQN_CONV1_OUT_CHANNEL + params.OBJECT_TYPE_NUM,
                             out_features=64)
        # self.fc2 = nn.Linear(in_features=64,
        #                      out_features=64)  # +2 for needs
        # self.fc3 = nn.Linear(16, 8)
        # self.fc4 = nn.Linear(8, 3)

    def forward(self, env_map, agent_need):
        batch_size = env_map.shape[0]

        y = F.relu(self.conv1(env_map))
        y = F.relu(self.conv2(y))
        y = F.relu(self.max_pool(y))
        y = y.flatten(start_dim=2, end_dim=-1)
        need_map = torch.tile(agent_need.unsqueeze(dim=1),
                              dims=(1, y.shape[1], 1))
        y = torch.concat([y, need_map], dim=-1)
        y = F.relu(self.conv3(y))
        y = y.flatten(start_dim=1, end_dim=-1)
        y = self.fc1(y)
        y = y.reshape(batch_size,
                      self.params.HEIGHT,
                      self.params.WIDTH)
        return y
