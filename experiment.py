from abc import abstractmethod

from environment import SingleUserSingleServicePartialObservable3DEnvironment
from models.observation import EuclideanObservation


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
    def __init__(self, num_device, width, height, depth, max_speed, observation_range):
        observation = EuclideanObservation(observation_range=observation_range)
        env = SingleUserSingleServicePartialObservable3DEnvironment(service_type='visual',
                                                                    num_device=num_device,
                                                                    width=width,
                                                                    height=height,
                                                                    depth=depth,
                                                                    observation=observation,
                                                                    max_speed=max_speed)
