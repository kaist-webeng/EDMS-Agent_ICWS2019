import tensorflow as tf
import numpy as np
import json
import os
import errno

from models.observation import Observation
from reinforcement_learning.reward import RewardFunction


class Configuration:
    def __init__(self, num_device, width, height, depth, device_size_min, device_size_max,
                 max_speed,
                 observation,
                 reward_function,
                 num_episode, num_step,
                 memory_size, batch_size, learning_rate, discount_factor,
                 eps_init, eps_final,
                 agent):
        # Environment
        self.num_device = num_device
        self.width = width
        self.height = height
        self.depth = depth
        self.device_size_min = device_size_min
        self.device_size_max = device_size_max

        # Dynamics
        self.max_speed = max_speed

        # Observation
        self.observation = observation

        # Reward
        self.reward_function = reward_function

        # Experiment
        self.num_episode = num_episode
        self.num_step = num_step

        # Learning
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        # Epsilon-greedy policy
        self.eps_init = eps_init
        self.eps_final = eps_final
        # set decaying rate according to the number of episodes: to make epsilon reaches eps_final at the end
        self.eps_decay = np.power(eps_final/eps_init, 1 / self.num_episode)

        self.agent = agent

    def save(self, name, date):
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Observation) or isinstance(obj, RewardFunction):
                    return str(obj)
                return json.JSONEncoder.default(self, obj)
        file_path = "{path}/{name}/{date}/configuration.txt".format(path=tf.flags.FLAGS.summary_path,
                                                                    name=name,
                                                                    date=date)
        if not os.path.exists(os.path.dirname(file_path)):
            try:
                os.makedirs(os.path.dirname(file_path))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        with open(file_path, 'w') as f:
            f.write(json.dumps(self.__dict__, indent=4, cls=CustomEncoder))
            f.close()
