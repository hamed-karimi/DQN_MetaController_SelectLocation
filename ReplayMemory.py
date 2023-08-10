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
        sample_indices = torch.multinomial(torch.as_tensor(self.early_memory_weights+self.later_memory_weights).float(),
                                           num_samples=size,
                                           replacement=False)
        early_memory_indices = sample_indices[sample_indices < len(self.early_memory)]
        later_memory_indices = sample_indices[sample_indices >= len(self.early_memory)] - len(self.early_memory)

        early_samples = [self.early_memory[i] for i in early_memory_indices]
        later_samples = [self.later_memory[i] for i in later_memory_indices]

        return early_samples+later_samples

    def __len__(self):
        return len(self.early_memory) + len(self.later_memory)
