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

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def size(self):
        return np.sqrt(np.square(self.x) + np.square(self.y) + np.square(self.z))

    def projection(self, target):
        assert isinstance(target, Vector)
        return (self.dot(target) / target.size()) * (target / target.size())

    def scalar_projection(self, target):
        assert isinstance(target, Vector)
        return self.dot(target) / target.size()

    def __str__(self):
        return "(X:{x}, Y:{y}, Z:{z})".format(x=self.x, y=self.y, z=self.z)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __radd__(self, other):
        return Vector(other.x + self.x, other.y + self.y, other.z + self.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __rsub__(self, other):
        return Vector(other.x - self.x, other.y - self.y, other.z - self.z)

    def __mul__(self, other):
        return Vector(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return Vector(other * self.x, other * self.y, other * self.z)

    def __truediv__(self, other):
        return Vector(self.x / other, self.y / other, self.z / other)

    def __eq__(self, other):
        return isinstance(other, Vector) and self.x == other.x and self.y == other.y and self.z == other.z


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
        return (self.speed * self.direction).vectorize()


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


def generate_random_direction_random_speed_mobility(width, height, depth, max_speed):
    return RectangularDirectedMobility(width, height, depth,
                                       generate_random_direction(),
                                       random.random() * max_speed)


def generate_random_direction_specific_speed_mobility(width, height, depth, speed):
    return RectangularDirectedMobility(width, height, depth,
                                       generate_random_direction(),
                                       speed)


class StaticMobility(Mobility):
    def __init__(self):
        Mobility.__init__(self, Vector(0., 0., 0.), 0.)

    def update(self, coordinate):
        return coordinate
