from abc import abstractmethod

from models.entity import User, Device, Service
from models.mobility import generate_random_coordinate, generate_random_direction_random_speed_mobility, StaticMobility, generate_random_direction_specific_speed_mobility
from models.orientation import generate_random_orientation, generate_vertical_orientation
from models.observation import Observation
from models.effectiveness import Effectiveness, DistanceEffectiveness, VisualEffectiveness


class Environment:
    """ Environment: abstract class of IoT environments for required methods """
    def __init__(self, service_type, num_device, width, height, depth, effectiveness):
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

        """ effectiveness: effectiveness model """
        assert isinstance(effectiveness, Effectiveness)
        self.effectiveness = effectiveness

        self.user = None
        self.devices = []
        self.services = []

        self.reset()

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
    def render(self):
        pass

    @abstractmethod
    def get_observation(self):
        pass

    @abstractmethod
    def get_observation_size(self):
        pass

    @abstractmethod
    def get_action_size(self):
        pass


class SingleUserSingleServicePartialObservableEnvironment(Environment):
    """
        SingleUserSingleServicePartialObservable3DEnvironment
        
        Environment settings
            - single user
            - single service selected
            - Partial observation based on Euclidean-distance
            - 3-dimensional space
    """
    def __init__(self, service_type, num_device, width, height, depth, max_speed, observation, effectiveness):
        """ max_speed: maximum speed that a mobile object can have """
        self.max_speed = max_speed

        """ observation: observation model, for distance-based partial observation """
        assert isinstance(observation, Observation)
        self.observation = observation

        Environment.__init__(self, service_type, num_device, width, height, depth, effectiveness)

    def reset(self):
        self.devices = []
        self.services = []

        """ Reset user """
        self.user = User(uid=0,
                         coordinate=generate_random_coordinate(self.width, self.height, self.depth),
                         mobility=generate_random_direction_specific_speed_mobility(self.width,
                                                                                    self.height,
                                                                                    self.depth,
                                                                                    self.max_speed))
        """ Reset devices and services in the environment """
        for i in range(self.num_device):
            # TODO currently, service is a simple encapsulation of device functionality, so device_type == service_type
            new_device = Device(name=i,
                                device_type=self.service_type,
                                coordinate=generate_random_coordinate(self.width, self.height, self.depth),
                                mobility=StaticMobility(),
                                # mobility=generate_random_rectangular_directed_mobility(self.width,
                                #                                                        self.height,
                                #                                                        self.depth,
                                #                                                        self.max_speed),
                                # orientation=generate_random_orientation())
                                orientation=generate_vertical_orientation())
            new_service = Service(name=i,
                                  service_type=self.service_type,
                                  device=new_device)
            self.devices.append(new_device)
            self.services.append(new_service)

        while not self.get_observation()["services"]:
            """ update until at least one service discovered """
            self.update_state()

        return self.get_observation()

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

    def step(self, action):
        """ receives selection result as a service instance """
        assert isinstance(action, Service)

        done = False
        reward = self.effectiveness.measure(self.user, action)

        # Release service and acquire new
        if self.user.service:
            self.user.service.release()
        self.user.utilize(action)
        action.acquire(self.user)

        # TODO reward calculation
        # TODO state update according to the given action

        self.update_state()
        while not self.get_observation()["services"]:
            """ update until at least one service discovered """
            self.update_state()

        return self.get_observation(), reward, done

    def render(self):
        print(self.user)
        for service in self.services:
            print(service)

    def get_observation(self):
        """ return observation in both dictionary format """
        return self.observation.get_observation(self.user, self.services)

    def get_observation_vector(self):
        return {
            "user": self.user.vectorize(),
            "services": [service.vectorize() for service in self.services]
        }

    def get_observation_size(self):
        return len(self.user.vectorize())

    def get_action_size(self):
        return len(self.services[0].vectorize())
