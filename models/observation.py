from abc import abstractmethod


class Observation:
    """ Observation: abstract class for Observation models """
    @abstractmethod
    def get_observation(self, user, services):
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


class FullObservation(Observation):
    def get_observation(self, user, services):
        return {
            "user": user,
            "services": services
        }
