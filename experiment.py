from abc import abstractmethod

from models.environment import SingleUserSingleServicePartialObservable3DEnvironment
from models.observation import EuclideanObservation
from agent import RandomSelectionAgent


class Experiment:
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def run(self):
        pass


class EffectDrivenVisualServiceSelectionExperiment(Experiment):
    """
        EffectDrivenVisualServiceSelectionExperiment
        
        Experiment done on the SingleUserSingleServicePartialObservable3DEnvironment
    """
    def __init__(self, num_device, width, height, depth, max_speed, observation_range, num_episode, num_step):
        observation = EuclideanObservation(observation_range=observation_range)
        self.env = SingleUserSingleServicePartialObservable3DEnvironment(service_type='visual',
                                                                         num_device=num_device,
                                                                         width=width,
                                                                         height=height,
                                                                         depth=depth,
                                                                         observation=observation,
                                                                         max_speed=max_speed)
        self.agent = RandomSelectionAgent(self.env, num_episode, num_step)

    def reset(self):
        self.env.reset()

    def run(self):
        self.agent.train()
        self.agent.test()

