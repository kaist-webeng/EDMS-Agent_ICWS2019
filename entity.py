class Service:
    """ Service: a class that represents service instances """
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
        assert user and type(user) == User
        self.in_use = True
        self.user = user

    def release(self):
        self.in_use = False
        self.user = None


class Device:
    pass


class User:
    pass
