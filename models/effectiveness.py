import numpy as np
from abc import abstractmethod


class Effectiveness:
    """ Effectiveness: abstract class for defining service effectiveness models """
    @abstractmethod
    def measure(self, user, service, context=None):
        pass


class DistanceEffectiveness(Effectiveness):
    """ DistanceEffectiveness: effectiveness model simply measures distance between the user and the service """
    def measure(self, user, service, context=None):
        return 1/user.distance(service.device)


class VisualEffectiveness(Effectiveness):
    """ VisualEffectiveness: effectiveness model for visual services """
    def measure(self, user, service, context=None):
        """ Handover """
        if not (service.in_use and service.user == user):
            return -1

        """ Visual angle """
        visual_angle = np.degrees(2 * np.arctan(1 / (2 * user.distance(service.device))))
        if visual_angle > 5 or visual_angle < 1/60:
            return 0.0

        return 1/user.distance(service.device)
