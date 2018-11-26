import datetime
import random
from abc import abstractmethod
from models.environment import Environment


class Agent:
    def __init__(self, env, num_episode, num_step):
        assert isinstance(env, Environment)
        self.env = env

        # Training configuration
        self.num_episode = num_episode
        self.num_step = num_step

        # Date of now, for logging
        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    @abstractmethod
    def selection(self, user, services):
        return None

    @abstractmethod
    def train(self):
        pass

    def test(self):
        print("Test phase")
        for i_episode in range(self.num_episode):
            print("Episode %d" % i_episode)

            score = 0.
            observation = self.env.reset()

            """ since service selection is non-episodic task, restrict maximum step rather than observe done-signal """
            for i_step in range(self.num_step):
                """ select action """
                action = self.selection(observation["user"], observation["services"])
                """ perform the selected action on the environment """
                observation, reward, done = self.env.step(action)
                """ add reward to total score """
                score += reward

            print("Episode %d ends with  score %r" % (i_episode, score))


class RandomSelectionAgent(Agent):
    """ RandomSelectionAgent: the baseline agent that selects services randomly """
    def selection(self, user, services):
        return random.choice(services)


class NearestSelectionAgent(Agent):
    """ ClosestSelectionAgent: the baseline agent that selects the nearest service"""
    def selection(self, user, services):
        services.sort(key=lambda service: user.distance(service.device))
        return services[0]


class DQNSelectionAgent(Agent):
    def __init__(self, env, num_episode, num_step):
        Agent.__init__(self, env, num_episode, num_step)

    def selection(self, user, services):
        pass

    def train(self):
        pass
