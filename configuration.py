import tensorflow as tf
import json
import os
import errno

from models.observation import Observation
from reinforcement_learning.reward import RewardFunction


class Configuration:
    def __init__(self, num_device, width, height, depth, max_speed, observation, reward_function,
                 num_episode, num_step, memory_size,
                 batch_size, learning_rate, discount_factor, agent):
        # Environment
        self.num_device = num_device
        self.width = width
        self.height = height
        self.depth = depth

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
        self.agent = agent

    def save(self, name, phase, date):
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Observation) or isinstance(obj, RewardFunction):
                    return str(obj)
                return json.JSONEncoder.default(self, obj)
        file_path = "{path}/{name}/{phase}/{date}/configuration.txt".format(path=tf.flags.FLAGS.summary_path,
                                                                            name=name,
                                                                            phase=phase,
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
