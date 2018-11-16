import numpy as np


class Coordinate:
    """ Coordinate: class that represents coordinate of physical entity """
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
