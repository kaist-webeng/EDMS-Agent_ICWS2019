from .mobility import *


class Service:
    """ Service: a basic class that represents service instances """
    def __init__(self, name, service_type, device=None):
        self.name = name
        self.type = service_type

        """ Type of associated device should be Device """
        assert not device or type(device) == Device
        self.device = device

        """ Flag: whether the service is in use or not """
        self.in_use = False
        self.user = None

    def acquire(self, user):
        """ acquire: user acquires the service to use """
        assert user and type(user) == User
        self.in_use = True
        self.user = user

    def release(self):
        """ release: user releases the service """
        self.in_use = False
        self.user = None


class Body:
    """ Body: physical body class, mainly deals with coordinate and mobility """
    def __init__(self, coordinate, mobility):
        """ mobility of the body """
        assert type(mobility) == Mobility
        self.mobility = mobility

        """ coordinate of the body """
        assert type(coordinate) == Coordinate
        self.coordinate = coordinate

    def get_coordinate(self):
        return self.coordinate.get()

    def move(self):
        self.mobility.update(self.coordinate)


class Device(Body):
    """ Device: a basic class that represents devices """
    def __init__(self, name, device_type, coordinate, mobility):
        Body.__init__(self, coordinate, mobility)
        self.name = name
        self.type = device_type


class User(Body):
    """ User: a basic class that represents users """
    def __init__(self, uid, coordinate, mobility):
        Body.__init__(self, coordinate, mobility)
        self.uid = uid
