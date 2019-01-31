from abc import abstractmethod

import matplotlib
import matplotlib.pyplot as plt

from models.entity import User, Device, Service
from models.mobility import *
from models.orientation import generate_random_orientation, generate_random_vertical_orientation, generate_random_half_line_orientation
from models.observation import Observation
from models.effectiveness import Effectiveness, DistanceEffectiveness, VisualEffectiveness
from reinforcement_learning.reward import RewardFunction


class Environment:
    """ Environment: abstract class of IoT environments for required methods """
    def __init__(self, service_type, num_device, width, height, depth, reward_function):
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

        """ reward: reward model """
        assert isinstance(reward_function, RewardFunction)
        self.reward_function = reward_function

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
    def __init__(self, service_type, num_device, width, height, depth, max_speed, observation, reward_function):
        """ max_speed: maximum speed that a mobile object can have """
        self.max_speed = max_speed

        """ observation: observation model, for distance-based partial observation """
        assert isinstance(observation, Observation)
        self.observation = observation

        Environment.__init__(self, service_type, num_device, width, height, depth, reward_function)

    def reset(self):
        self.devices = []
        self.services = []

        """ Reset user """
        self.user = User(uid=0,
                         # Start from edge of the environment
                         coordinate=generate_custom_coordinate(self.width, self.height, self.depth,
                                                               x=10, y=self.height/2,
                                                               # Common height of a human
                                                               z=1.7),
                         # Go across the environment
                         mobility=generate_custom_mobility(self.width,
                                                           self.height,
                                                           self.depth,
                                                           generate_custom_direction(1, 0, 0),
                                                           self.max_speed))

        """ Reset devices and services in the environment """
        for i in range(self.num_device):
            # TODO currently, service is a simple encapsulation of device functionality, so device_type == service_type
            coordinate = generate_random_coordinate(self.width, self.height, self.depth)
            orientation = generate_random_half_line_orientation(self.width, self.height, self.depth,
                                                                coordinate.x, coordinate.y, coordinate.z)
            new_device = Device(name=i,
                                device_type=self.service_type,
                                coordinate=coordinate,
                                mobility=StaticMobility(),
                                # mobility=generate_random_rectangular_directed_mobility(self.width,
                                #                                                        self.height,
                                #                                                        self.depth,
                                #                                                        self.max_speed),
                                # orientation=generate_random_orientation())
                                orientation=orientation)
            new_service = Service(name=i,
                                  service_type=self.service_type,
                                  device=new_device)
            self.devices.append(new_device)
            self.services.append(new_service)

        if not self.get_observation()["services"]:
            """ reset until at least one service discovered """
            return self.reset()

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
        reward = self.reward_function.measure(self.user, action)

        # Release service and acquire new
        if self.user.service:
            self.user.service.release()
        self.user.utilize(action)
        action.acquire(self.user)

        # TODO reward calculation
        # TODO state update according to the given action

        self.update_state()
        if not self.get_observation()["services"]:
            """ if no service is discovered, done is True """
            done = True

        return self.get_observation(), reward, done

    def render(self):
        fig = plt.figure()

        # locations of user and devices
        plt.scatter(x=[device.coordinate.x for device in self.devices] + [self.user.coordinate.x],
                    y=[device.coordinate.y for device in self.devices] + [self.user.coordinate.y],
                    c=["blue" for _ in range(self.num_device)] + ["red"])

        # orientations of user and devices
        head_width = 0.05
        head_length = 0.05
        for device in self.devices:
            plt.arrow(x=device.coordinate.x, y=device.coordinate.y,
                      dx=device.orientation.face.i, dy=device.orientation.face.j,
                      head_width=head_width, head_length=head_length)
        plt.arrow(x=self.user.coordinate.x, y=self.user.coordinate.y,
                  dx=self.user.infer_orientation().x, dy=self.user.infer_orientation().y,
                  head_width=head_width, head_length=head_length)

        # TODO observation range

        plt.show()

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
