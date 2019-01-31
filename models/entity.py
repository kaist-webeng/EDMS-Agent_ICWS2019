import numpy as np
import random
from abc import abstractmethod

from models.mobility import Coordinate, Mobility, StaticMobility
from models.orientation import Orientation


class Service:
    """ Service: a basic class that represents service instances """
    def __init__(self, name, service_type, device=None):
        self.name = name
        self.type = service_type

        """ Type of associated device should be Device """
        # TODO currently, service - device is one-to-one matching
        assert isinstance(device, Device)
        self.device = device

        """ Flag: whether the service is in use or not """
        self.in_use = False
        self.user = None

    def acquire(self, user):
        """ acquire: user acquires the service to use """
        assert isinstance(user, User)
        self.in_use = True
        self.user = user

    def release(self):
        """ release: user releases the service """
        self.in_use = False
        self.user = None

    def vectorize(self):
        # TODO vector representation of services: multi-user situation
        if self.in_use:
            return self.device.vectorize() + [1]
        else:
            return self.device.vectorize() + [0]

    def __str__(self):
        return "Service name {name}, type {type}, device {device}".format(name=self.name,
                                                                          type=self.type,
                                                                          device=self.device)


class Body:
    """ Body: physical body class, mainly deals with coordinate and mobility """
    def __init__(self, coordinate, mobility):
        """ mobility of the body """
        assert isinstance(mobility, Mobility)
        self.mobility = mobility

        """ coordinate of the body """
        assert isinstance(coordinate, Coordinate)
        self.coordinate = coordinate

    def get_coordinate(self):
        return self.coordinate.get()

    def distance(self, other):
        assert isinstance(other, Body)
        return self.coordinate.distance(other.coordinate)

    def move(self):
        self.mobility.update(self.coordinate)

    @abstractmethod
    def vectorize(self):
        return []


class Device(Body):
    """ Device: a basic class that represents devices """

    def __init__(self, name, device_type, coordinate, mobility, orientation):
        Body.__init__(self, coordinate, mobility)
        self.name = name
        self.type = device_type

        assert isinstance(orientation, Orientation)
        self.orientation = orientation

        # TODO size of device: move range configuration to outside of the code
        # device size 50cm ~ 5m
        self.size = 0.5 * random.random() * 1

    def __str__(self):
        return "Device {name}, type {type} at {coordinate}, {orientation}".format(name=self.name,
                                                                                  type=self.type,
                                                                                  coordinate=self.coordinate,
                                                                                  orientation=self.orientation)

    def vectorize(self):
        return self.coordinate.vectorize() + self.mobility.vectorize() + self.orientation.vectorize()


class User(Body):
    """ User: a basic class that represents users """
    def __init__(self, uid, coordinate, mobility):
        Body.__init__(self, coordinate, mobility)
        self.uid = uid
        self.service = None

    def __str__(self):
        return "User {uid} at {coordinate} orientation {face}".format(uid=self.uid,
                                                                      coordinate=self.coordinate,
                                                                      face=self.infer_orientation())

    def utilize(self, service):
        assert isinstance(service, Service)
        self.service = service

    def infer_orientation(self):
        """ infer orientation of user head from mobility """
        # TODO currently, simply returns direction of user mobility since usually user heads its direction
        return self.mobility.direction

    def vectorize(self):
        return self.coordinate.vectorize() + self.mobility.vectorize()
