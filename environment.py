from abc import abstractmethod
from entity import User, Device, Service


class Environment:
    """ Environment: abstract class of IoT environments for required methods """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def solvable(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def update_state(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_observation(self):
        pass

    @abstractmethod
    def render(self):
        pass

