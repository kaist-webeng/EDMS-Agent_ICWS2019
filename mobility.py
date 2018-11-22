import random
import math
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

    def __str__(self):
        return "(X:{x}, Y:{y}), Z:{z}".format(x=self.x, y=self.y, z=self.z)


class Coordinate(Vector):
    """ Coordinate: class that represents coordinate of a physical entity in a 3-dimensional space """
    def distance(self, other):
        """ distance: calculate Euclidean distance between coordinates"""
        assert type(other) == Coordinate
        return np.sqrt(np.square(self.x - other.x) + np.square(self.y - other.y) + np.square(self.z - other.z))


def generate_random_coordinate(width, height, depth):
    return Coordinate(x=random.random() * width, y=random.random() * height, z=random.random() * depth)


class Direction(Vector):
    """ Direction: class that represents direction of a physical entity in a 3-dimensional space """
    def __init__(self, x, y, z):
        """ Direction should be a unit vector """
        denominator = np.square(x) + np.square(y) + np.square(z)
        unit_x = np.sqrt(np.square(x)/denominator)
        unit_y = np.sqrt(np.square(y)/denominator)
        unit_z = np.sqrt(np.square(z)/denominator)
        assert np.square(unit_x) + np.square(unit_y) + np.square(unit_z) == 1
        Vector.__init__(self, unit_x, unit_y, unit_z)


def generate_random_direction():
    return Direction(random.random(), random.random(), random.random())


class Mobility:
    """ Mobility: class that represents mobility of a physical entity """
    pass

    @abstractmethod
    def update(self, coordinate):
        """ update: receives current coordinate and returns new """
        return coordinate


class RectangularDirectedMobility(Mobility):
    """ RectangularDirectedMobility: mobility that has fixed direction and speed, restricted in a rectangular area"""
    def __init__(self, width, height, depth, direction, speed):
        self.width = width
        self.height = height
        self.depth = depth

        """ direction: direction of the mobility, Direction class """
        assert type(direction) == Direction
        self.direction = direction
        self.speed = speed

    def update(self, coordinate):
        coordinate.update(clamp(coordinate.x, 0, self.width),
                          clamp(coordinate.y, 0, self.height),
                          clamp(coordinate.z, 0, self.depth))
