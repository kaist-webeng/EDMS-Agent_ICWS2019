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
    def __init__(self, min_x, max_x, min_y, max_y, min_z, max_z, direction, speed):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.min_z = min_z
        self.max_z = max_z

        self.direction = direction
        self.speed = speed

    def update(self, coordinate):
        coordinate.update(clamp(coordinate.x, self.min_x, self.max_x),
                          clamp(coordinate.y, self.min_y, self.max_y),
                          clamp(coordinate.z, self.min_z, self.max_z))
