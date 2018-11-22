import datetime
import random
from abc import abstractmethod


class Agent:
    def __init__(self, env, num_episode, num_step):
        self.env = env

        # Training configuration
        self.num_episode = num_episode
        self.num_step = num_step

        # Date of now, for logging
        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    @abstractmethod
    def selection(self, user, services):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass


class RandomSelectionAgent(Agent):
    """ RandomSelectionAgent: the baseline agent that selects services randomly """
    def selection(self, user, services):
        return random.choice(services)


class NearestSelectionAgent(Agent):
    """ ClosestSelectionAgent: the baseline agent that selects the nearest service"""
    def selection(self, user, services):
        services.sort(key=lambda service: user.distance(service.device))
        return services[0]
