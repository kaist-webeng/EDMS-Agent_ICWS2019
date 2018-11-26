import tensorflow as tf
from abc import abstractmethod

from models.effectiveness import DistanceEffectiveness
from models.environment import SingleUserSingleServicePartialObservable3DEnvironment
from models.observation import EuclideanObservation
from reinforcement_learning.agent import RandomSelectionAgent, DRRNSelectionAgent


class Experiment:
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def run(self):
        pass


class EffectDrivenVisualServiceSelectionExperiment(Experiment):
    """
        EffectDrivenVisualServiceSelectionExperiment
        
        Experiment done on the SingleUserSingleServicePartialObservable3DEnvironment
    """
    def __init__(self, num_device, width, height, depth, max_speed, observation_range, num_episode, num_step, memory_size, batch_size):
        observation = EuclideanObservation(observation_range=observation_range)
        effectiveness = DistanceEffectiveness()
        self.env = SingleUserSingleServicePartialObservable3DEnvironment(service_type='visual',
                                                                         num_device=num_device,
                                                                         width=width,
                                                                         height=height,
                                                                         depth=depth,
                                                                         max_speed=max_speed,
                                                                         observation=observation,
                                                                         effectiveness=effectiveness)
        self.num_episode = num_episode
        self.num_step = num_step
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.agent = DRRNSelectionAgent(self.env, self.num_episode, self.num_step,
                                        learning_rate=0.0001,
                                        discount_factor=1,
                                        memory_size=self.memory_size,
                                        batch_size=self.batch_size)

    def reset(self):
        self.env.reset()

    def run(self):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            self.agent.train(sess)
            self.agent.test(sess)

