import numpy as np
from abc import abstractmethod

from models.mobility import Quaternion, Rotation


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


if __name__ == '__main__':
    test = QuaternionRotationTest()
    test.run()
