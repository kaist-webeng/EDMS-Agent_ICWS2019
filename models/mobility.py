import random
import numpy as np
from abc import abstractmethod
from utils import clamp


class Vector:
    """ Vector: class of 3-dimensional vector for Coordinate and Direction """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def get(self):
        return self.x, self.y, self.z

    def update(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def vectorize(self):
        return [self.x, self.y, self.z]

    def __str__(self):
        return "(X:{x}, Y:{y}, Z:{z})".format(x=self.x, y=self.y, z=self.z)


class Coordinate(Vector):
    """ Coordinate: class that represents coordinate of a physical entity in a 3-dimensional space """
    def distance(self, other):
        """ distance: calculate Euclidean distance between coordinates"""
        assert isinstance(other, Coordinate)
        return np.sqrt(np.square(self.x - other.x) + np.square(self.y - other.y) + np.square(self.z - other.z))


def generate_random_coordinate(width, height, depth):
    return Coordinate(x=random.random() * width, y=random.random() * height, z=random.random() * depth)


class Direction(Vector):
    """ Direction: class that represents direction of a physical entity in a 3-dimensional space """
    def __init__(self, x, y, z):
        """ Direction should be a unit vector """
        denominator = np.square(x) + np.square(y) + np.square(z)
        unit_x = np.sqrt(np.square(x)/denominator)
        sign_x = 1 if x > 0 else -1
        unit_y = np.sqrt(np.square(y)/denominator)
        sign_y = 1 if y > 0 else -1
        unit_z = np.sqrt(np.square(z)/denominator)
        sign_z = 1 if z > 0 else -1
        Vector.__init__(self, sign_x*unit_x, sign_y*unit_y, sign_z*unit_z)


def generate_random_direction():
    return Direction(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))


class Mobility:
    """ Mobility: class that represents mobility of a physical entity """
    def __init__(self, direction, speed):
        """ direction: direction of the mobility, Vector class """
        assert isinstance(direction, Vector)
        self.direction = direction
        self.speed = speed

    @abstractmethod
    def update(self, coordinate):
        """ update: receives current coordinate and returns new """
        return coordinate

    def vectorize(self):
        return self.direction.vectorize() + [self.speed]


class RectangularDirectedMobility(Mobility):
    """ RectangularDirectedMobility: mobility that has fixed direction and speed, restricted in a rectangular area"""
    def __init__(self, width, height, depth, direction, speed):
        Mobility.__init__(self, direction, speed)
        self.width = width
        self.height = height
        self.depth = depth

    def update(self, coordinate):
        new_x = clamp(coordinate.x + self.direction.x * self.speed, 0, self.width)
        new_y = clamp(coordinate.y + self.direction.y * self.speed, 0, self.height)
        new_z = clamp(coordinate.z + self.direction.z * self.speed, 0, self.depth)

        if new_x == 0 or new_x == self.width or new_y == 0 or new_y == self.height or new_z == 0 or new_z == self.depth:
            """ if new direction is on the boundary of the area, reset direction randomly """
            self.direction = generate_random_direction()

        coordinate.update(new_x, new_y, new_z)


def generate_random_rectangular_directed_mobility(width, height, depth, max_speed):
    return RectangularDirectedMobility(width, height, depth,
                                       generate_random_direction(),
                                       random.random() * max_speed)


class StaticMobility(Mobility):
    def __init__(self):
        Mobility.__init__(self, Vector(0., 0., 0.), 0.)

    def update(self, coordinate):
        return coordinate


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
