from collections import deque
import random


class ReplayMemory():
    def __init__(self, capacity=300000):
        self.memory = deque([], maxlen=capacity)
        self.weights = deque([], maxlen=capacity)

    def get_transition(self, *args):
        pass

    def push_experience(self, *args):
        self.memory.append(self.get_transition(*args))

    def push_selection_ratio(self, **kwargs):
        self.weights.append(kwargs['selection_ratio'])

    def sample(self, size):
        return random.choices(self.memory, weights=self.weights, k=size)

    def __len__(self):
        return len(self.memory)
