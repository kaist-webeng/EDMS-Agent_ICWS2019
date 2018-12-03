import numpy as np
import random

from .mobility import Vector


class Quaternion:
    """ Quaternion: class of 4-dimensional quaternion for Orientation and Rotation """
    def __init__(self, w, i, j, k):
        self.w = w
        self.i = i
        self.j = j
        self.k = k

    def get(self):
        return self.w, self.i, self.j, self.k

    def update(self, w, i, j, k):
        self.w = w
        self.i = i
        self.j = j
        self.k = k

    def vectorize(self):
        return [self.w, self.i, self.j, self.k]

    def get_vector_part(self):
        return Vector(self.i, self.j, self.k)

    def get_scalar_part(self):
        return self.w

    def get_conjugate(self):
        return Quaternion(self.w, -self.i, -self.j, -self.k)

    def is_unit(self):
        return np.square(self.w) + np.square(self.i) + np.square(self.j) + np.square(self.k) == 1.

    def __str__(self):
        return "(W:{w}, I:{i}, J:{j}, K:{k})".format(w=self.w, i=self.i, j=self.j, k=self.k)

    def __mul__(self, other):
        assert isinstance(other, Quaternion)
        return Quaternion(
            w=self.w*other.w - self.i*other.i - self.j*other.j - self.k*other.k,
            i=self.w*other.i + self.i*other.w + self.j*other.k - self.k*other.j,
            j=self.w*other.j - self.i*other.k + self.j*other.w + self.k*other.i,
            k=self.w*other.k + self.i*other.j - self.j*other.i + self.k*other.w
        )

    def __rmul__(self, other):
        assert isinstance(other, Quaternion)
        return Quaternion(
            w=other.w*self.w - other.i*self.i - other.j*self.j - other.k*self.k,
            i=other.w*self.i + other.i*self.w + other.j*self.k - other.k*self.j,
            j=other.w*self.j - other.i*self.k + other.j*self.w + other.k*self.i,
            k=other.w*self.k + other.i*self.j - other.j*self.i + other.k*self.w
        )


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

    def vectorize(self):
        return self.face.get_vector_part().vectorize() + self.head.get_vector_part().vectorize()


def generate_random_orientation():
    return Orientation(random.uniform(-2, 2)*np.pi, random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
