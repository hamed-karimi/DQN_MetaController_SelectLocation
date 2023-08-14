import math
from collections import deque
import random
import numpy as np
import torch


class ReplayMemory():
    def __init__(self, capacity):
        self.early_memory = deque([], maxlen=capacity//2)
        self.later_memory = deque([], maxlen=capacity//2)

        self.early_memory_weights = deque([], maxlen=capacity//2)
        self.later_memory_weights = deque([], maxlen=capacity // 2)
        self.first_steps_sample_ratio = .1
        self.memory_pointer = {'early': lambda _: (self.early_memory, self.early_memory_weights),
                               'later': lambda _: (self.later_memory, self.later_memory_weights)}

    def get_transition(self, *args):
        pass

    def push_experience(self, *args):
        if len(self.early_memory) < self.early_memory.maxlen:
            self.early_memory.append(self.get_transition(*args))
        else:
            self.later_memory.append(self.get_transition(*args))

    def push_selection_ratio(self, **kwargs):
        if len(self.early_memory) < self.early_memory.maxlen:
            self.early_memory_weights.append(kwargs['selection_ratio'])
        else:
            self.later_memory_weights.append(kwargs['selection_ratio'])

    def sample(self, size):
        first_steps_sample_size = math.floor(size * self.first_steps_sample_ratio)
        other_steps_sample_size = size - first_steps_sample_size

        first_sample_indices = torch.multinomial(torch.ones(math.floor(len(self.early_memory)*0.1), dtype=torch.float),
                                                 num_samples=first_steps_sample_size,
                                                 replacement=False)
        first_samples = [self.early_memory[i] for i in first_sample_indices]

        weights = torch.cat([torch.ones_like(torch.as_tensor(self.early_memory_weights, dtype=torch.float)),
                             torch.ones_like(torch.as_tensor(self.later_memory_weights, dtype=torch.float))])

        weights[first_sample_indices] = 0
        sample_indices = torch.multinomial(weights,
                                           num_samples=other_steps_sample_size,
                                           replacement=False)
        early_memory_indices = sample_indices[sample_indices < len(self.early_memory)]
        later_memory_indices = sample_indices[sample_indices >= len(self.early_memory)] - len(self.early_memory)

        early_samples = [self.early_memory[i] for i in early_memory_indices]
        later_samples = [self.later_memory[i] for i in later_memory_indices]

        return first_samples+early_samples+later_samples

    def __len__(self):
        return len(self.early_memory) + len(self.later_memory)
