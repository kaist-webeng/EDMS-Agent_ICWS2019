import numpy as np
import random


# Basic experience memory that randomly sampling experiences for DQN
class ExperienceMemory:
    def __init__(self, size):
        self.memory = []
        self.size = size

    def add(self, observation, action, reward, next_observation):
        while self.is_full():
            # Random pop-up
            # self.memory.pop(random.randrange(0, len(self.memory)))
            # FIFO
            self.memory.pop(0)
        self.memory.append([observation, action, reward, next_observation])

    def sample(self, batch_size):
        if 0 < len(self.memory) < batch_size:
            return self.memory
        return random.sample(self.memory, batch_size)

    def is_full(self):
        return len(self.memory) == self.size

    def is_empty(self):
        return len(self.memory) == 0
