import numpy as np
from abc import abstractmethod

from models.math import Vector


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
        """ Visual angle """
        """ 
            6/6 vision is defined as: at 6 m distance, human can recognize 5 arc-min letter.
            so size of the minimum letter is: 2 * 6 * tan(5 / 120) = 0.00873 m  
        """
        # TODO currently set text_size as 108 pixels
        # actual text size shown on display, assuming 1080p resolution
        text_size = service.device.size * 108 / 1080
        visual_angle = np.degrees(2 * np.arctan(text_size / (2 * user.distance(service.device))))
        """
            "the size of a letter on the Snellen chart of Landolt C chart is a visual angle of 5 arc minutes"
            https://en.wikipedia.org/wiki/Visual_acuity 
        """
        if visual_angle < 5/60:
            return 0

        """ FoV """
        """"
            Device should be inside of user's FoV
        """
        # TODO dev
        user_face = user.infer_orientation()
        relative_coordinate = service.device.coordinate - user.coordinate
        cosine_relative_location_angle = relative_coordinate.get_cosine_angle(user_face)
        if cosine_relative_location_angle < 0:
            return 0

        """ Orientation """
        """
            face of the visual display should be opposite of the user's face
        """
        device_face = service.device.orientation.face.get_vector_part()
        cosine_face_angle = user_face.get_cosine_angle(device_face)
        if cosine_face_angle > 0:
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
            # return 0
            pass  # skip orientation check

        return 1
