from abc import abstractmethod

from models.entity import User, Device, Service
from models.mobility import generate_random_coordinate, generate_random_rectangular_directed_mobility
from models.observation import EuclideanObservation


class Environment:
    """ Environment: abstract class of IoT environments for required methods """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def update_state(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_observation(self):
        pass

    @abstractmethod
    def render(self):
        pass


class SingleUserSingleServicePartialObservable3DEnvironment(Environment):
    """
        SingleUserSingleServicePartialObservable3DEnvironment
        
        Environment settings
            - single user
            - single service selected
            - Partial observation based on Euclidean-distance
            - 3-dimensional space
    """
    def __init__(self, service_type, num_device, width, height, depth, observation, max_speed):
        """ service_type: type of service to simulate """
        self.service_type = service_type
        """ num_device: number of devices """
        self.num_device = num_device
        """ width: x-axis size of the environment """
        self.width = width
        """ height: y-axis size of the environment """
        self.height = height
        """ depth: z-axis size of the environment """
        self.depth = depth
        """ observation_range: observation range of the user, for distance-based partial observation """
        assert type(observation) == EuclideanObservation
        self.observation = observation
        """ max_speed: maximum speed that a mobile object can have """
        self.max_speed = max_speed

        self.user = None
        self.devices = []
        self.services = []

        self.reset()

    def reset(self):
        self.devices = []
        self.services = []

        """ Reset user """
        self.user = User(uid=0,
                         coordinate=generate_random_coordinate(self.width, self.height, self.depth),
                         mobility=generate_random_rectangular_directed_mobility(self.width,
                                                                                self.height,
                                                                                self.depth,
                                                                                self.max_speed))
        """ Reset devices and services in the environment """
        for i in range(self.num_device):
            # TODO currently, service is a simple encapsulation of device functionality, so device_type == service_type
            new_device = Device(name=i,
                                device_type=self.service_type,
                                coordinate=generate_random_coordinate(self.width, self.height, self.depth),
                                mobility=generate_random_rectangular_directed_mobility(self.width,
                                                                                       self.height,
                                                                                       self.depth,
                                                                                       self.max_speed))
            new_service = Service(name=i,
                                  service_type=self.service_type,
                                  device=new_device)
            self.devices.append(new_device)
            self.services.append(new_service)

    def get_state(self):
        """ get_state: return the full state of the environment """
        return {
            "user": self.user,
            "services": self.services
        }

    def update_state(self):
        self.user.move()
        for device in self.devices:
            device.move()

    def get_observation(self):
        return self.observation.get_observation(self.user, self.services)

    def step(self, action):
        reward = 0.
        done = False

        # TODO reward calculation
        # TODO state update according to the given action
        self.update_state()

        return self.get_observation(), reward, done

    def render(self):
        pass
