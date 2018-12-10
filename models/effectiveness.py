import numpy as np
from abc import abstractmethod

from .mobility import Vector


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
        """ 
            6/6 vision is defined as: at 6 m distance, human can recognize 5 arc-min letter.
            so size of the minimum letter is: 2 * 6 * tan(5 / 120) = 0.00873 m  
        """
        visual_angle = np.degrees(2 * np.arctan(0.00873 / (2 * user.distance(service.device))))
        """
            "the size of a letter on the Snellen chart of Landolt C chart is a visual angle of 5 arc minutes"
            https://en.wikipedia.org/wiki/Visual_acuity 
        """
        if visual_angle < 5/60:
            return 0

        """ User position """
        """"
            User should be in front of the display
        """
        relative_coordinate = user.coordinate - service.device.coordinate
        if relative_coordinate.scalar_projection(service.device.orientation.face.get_vector_part()) < 0:
            return -1

        """ Orientation """
        """
            face of the visual display should be opposite of the user's face
        """
        user_face = user.infer_orientation()
        device_face = service.device.orientation.face.get_vector_part()
        cosine_face_angle = user_face.dot(device_face) / (user_face.size() * device_face.size())
        if cosine_face_angle < 0:
            # angle between user sight and device face is larger than 60 degree
            return 0

        """
            head of the visual display should be close to the user's head
        """
        user_head = Vector(0, 0, 1)
        device_head = service.device.orientation.head.get_vector_part()
        cosine_head_angle = user_head.dot(device_head) / (user_head.size() * device_head.size())
        if cosine_head_angle < 0:
            # angle between user head and device head is larger than 60 degree
            return 0

        return 1
