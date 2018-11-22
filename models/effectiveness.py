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
        return user.distance(service.device)
