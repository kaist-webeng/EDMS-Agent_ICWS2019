import numpy as np
from abc import abstractmethod
from utils import *


class Coordinate:
    """ Coordinate: class that represents coordinate of a physical entity """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other):
        assert type(other) == Coordinate
        return np.sqrt(np.square(self.x - other.x) + np.square(self.y - other.y) + np.square(self.z - other.z))

    def get(self):
        return self.x, self.y, self.z

    def update(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return "(X:{x}, Y:{y}), Z:{z}".format(x=self.x, y=self.y, z=self.z)


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

        self.direction = direction
        self.speed = speed

    def update(self, coordinate):
        coordinate.update(clamp(coordinate.x, 0, self.width),
                          clamp(coordinate.y, 0, self.height),
                          clamp(coordinate.z, 0, self.depth))
