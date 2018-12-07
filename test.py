import numpy as np
import random
from abc import abstractmethod

from models.orientation import Quaternion, Rotation
from models.mobility import Vector


class Test:
    @abstractmethod
    def run(self):
        pass


class QuaternionRotationTest(Test):
    def run(self):
        face = Quaternion(0, 1, 0, 0)
        head = Quaternion(0, 0, 0, 1)

        rotation = Rotation(np.pi/2, 0, 1, 0)

        print(rotation)
        print(rotation.rotate(face))
        print(rotation.rotate(head))


class VectorOperationTest(Test):
    def run(self):
        """ Multiplication within an integer """
        v1 = Vector(random.random(), random.random(), random.random())
        m = random.randint(1, 5)
        print(v1, m, v1 * m, m * v1, v1*m == m*v1)

        """ Division within a float """
        print(v1, m, v1 / m)

        """ Addition """
        v2 = Vector(random.random(), random.random(), random.random())
        print(v1, v2, v1 + v2, v2 + v1)

        """ Subtraction """
        print(v1, v2, v1 - v2, v2 - v1)

        """ Projection """
        projection = v1.projection(v2)
        print(projection.x, v2.x, projection.x / v2.x)
        print(projection.y, v2.y, projection.y / v2.y)
        print(projection.z, v2.z, projection.z / v2.z)


if __name__ == '__main__':
    test = VectorOperationTest()
    test.run()
