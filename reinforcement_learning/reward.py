from abc import abstractmethod

from models.effectiveness import Effectiveness


class Reward:
    """ Reward: abstract class for reward signal models """
    @abstractmethod
    def measure(self, user, service, context=None):
        pass


class HandoverPenaltyReward(Reward):
    """ HandoverPenaltyReward: giving penalty when handover, otherwise effectiveness """
    def __init__(self, effectiveness):
        assert isinstance(effectiveness, Effectiveness)
        self.effectiveness = effectiveness

    def measure(self, user, service, context=None):
        """ Handover """
        if not (service.in_use and service.user == user):
            return -1
        return self.effectiveness.measure(user, service, context)
