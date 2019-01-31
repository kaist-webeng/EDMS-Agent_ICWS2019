import numpy as np
import random

from models.math import Quaternion


class Rotation(Quaternion):
    """ Rotation: class of rotation quaternion, receives axis and angle, then construct unit rotation vector """
    def __init__(self, theta, i, j, k):
        assert -2 * np.pi <= theta <= 2 * np.pi  # theta is radian

        """ Rotation should be a unit vector """
        denominator = np.square(i) + np.square(j) + np.square(k)
        Quaternion.__init__(
            self,
            w=np.cos(theta / 2),
            i=np.sign(i) * np.sqrt(np.square(i) / denominator) * np.sin(theta / 2),
            j=np.sign(j) * np.sqrt(np.square(j) / denominator) * np.sin(theta / 2),
            k=np.sign(k) * np.sqrt(np.square(k) / denominator) * np.sin(theta / 2)
        )

    def rotate(self, quaternion):
        return self * quaternion * self.get_conjugate()


class Orientation:
    """
        Orientation: class that represents orientation of a physical entity in a 3-dimensional space
        to un-ambiguously state orientation and head of a body,
        orientation receives a Quaternion and rotate vectors (1, 0, 0) and (0, 0, 1) according to the Quaternion,
        where each vector is face and head, respectively
    """
    def __init__(self, theta, i, j, k):
        rotation = Rotation(theta, i, j, k)

        default_face = Quaternion(0, 1, 0, 0)  # x-axis direction
        default_head = Quaternion(0, 0, 0, 1)  # z-axis direction

        self.face = rotation.rotate(default_face)
        self.head = rotation.rotate(default_head)

    def __str__(self):
        return "Orientation face:{face} | head:{head}".format(face=self.face, head=self.head)

    def vectorize(self):
        return self.face.get_vector_part().vectorize() + self.head.get_vector_part().vectorize()


def generate_random_orientation():
    return Orientation(random.uniform(-2, 2)*np.pi, random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))


def generate_random_vertical_orientation():
    """ Rotating axis is z-axis, so Orientation head always (0, 0, 1) """
    return Orientation(random.uniform(-2, 2)*np.pi, 0, 0, 1)


def generate_random_half_line_orientation(width, height, depth, x, y, z):
    """ orientation that faces half-line of the environment """
    orientation = generate_random_vertical_orientation()
    while (height/2 - y) * orientation.face.get_vector_part().y < 0:
        orientation = generate_random_vertical_orientation()
    return orientation
