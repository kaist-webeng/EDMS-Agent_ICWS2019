import numpy as np
import random
import operator
from abc import abstractmethod


class ExperienceMemory:
    @abstractmethod
    def add(self, observation, action, reward, next_observation, done):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass

    @abstractmethod
    def is_full(self):
        pass

    @abstractmethod
    def is_empty(self):
        pass


# Basic experience memory that randomly sampling experiences for DQN
class BasicExperienceMemory(ExperienceMemory):
    def __init__(self, size):
        self.memory = []
        self.size = size

    def add(self, observation, action, reward, next_observation, done):
        while self.is_full():
            # Random pop-up
            # self.memory.pop(random.randrange(0, len(self.memory)))
            # FIFO
            self.memory.pop(0)
        self.memory.append({
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done
        })

    def sample(self, batch_size):
        if 0 < len(self.memory) < batch_size:
            return self.memory
        return random.sample(self.memory, batch_size)

    def is_full(self):
        return len(self.memory) == self.size

    def is_empty(self):
        return len(self.memory) == 0


# Balancing experience memory that balancing reward distribution
class BalancingExperienceMemory(ExperienceMemory):
    def __init__(self, size):
        self.memory = []
        self.size = size
        self.count = {}

    def add(self, observation, action, reward, next_observation, done):
        while self.is_full():
            """ pop a memory instance from largest reword set, balancing reward distribution """
            target_reward = max(self.count.items(), key=operator.itemgetter(1))[0]
            for m in self.memory:
                # FIFO
                if m["reward"].get_overall_score() == target_reward:
                    self.count[target_reward] -= 1
                    self.memory.remove(m)
                    break

        # TODO only works for discrete value of reward
        reward_value = reward.get_overall_score()
        if reward_value not in self.count:
            self.count[reward_value] = 1
        else:
            self.count[reward_value] += 1

        self.memory.append({
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done
        })

    def sample(self, batch_size):
        if 0 < len(self.memory) < batch_size:
            return self.memory
        return random.sample(self.memory, batch_size)

    def is_full(self):
        return len(self.memory) == self.size

    def is_empty(self):
        return len(self.memory) == 0
