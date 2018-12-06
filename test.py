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


class VectorMultiplicationTest(Test):
    def run(self):
        v = Vector(random.random(), random.random(), random.random())
        m = random.randint(1, 5)
        print(v, m, v * m, m * v, v*m == m*v)


if __name__ == '__main__':
    test = VectorMultiplicationTest()
    test.run()
