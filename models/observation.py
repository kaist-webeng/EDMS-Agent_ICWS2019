from abc import abstractmethod


class Observation:
    """ Observation: abstract class for Observation models """
    @abstractmethod
    def get_observation(self, user, services):
        pass

    @abstractmethod
    def __str__(self):
        pass


class EuclideanObservation(Observation):
    def __init__(self, observation_range):
        self.observation_range = observation_range

    def get_observation(self, user, services):
        """ get_observation: return the Euclidean-distance-based partial observation on the environment """
        service_observation = [service for service in services
                               if user.distance(service.device) <= self.observation_range]
        """ return objects, rather than matrix: agent will transform the observation into matrix """
        return {
            "user": user,
            "services": service_observation
        }

    def __str__(self):
        return "EuclideanObservation({range})".format(range=self.observation_range)


class FullObservation(Observation):
    def get_observation(self, user, services):
        return {
            "user": user,
            "services": services
        }

    def __str__(self):
        return "FullObservation"
