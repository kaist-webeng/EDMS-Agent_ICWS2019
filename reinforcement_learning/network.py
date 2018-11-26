import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from abc import abstractmethod


class Network:
    def __init__(self, name, learning_rate):
        self.name = "Network/{name}".format(name=name)
        self.learning_rate = learning_rate


class DRRN(Network):
    """
        DRRN network for variable size of action space
        He, Ji, et al. "Deep reinforcement learning with an action space defined by natural language." (2016).
    """
    # TODO currently, batch normalization is not available
    def __init__(self, name, learning_rate, observation_size, action_size):
        Network.__init__(self, "DRRN/{name}".format(name=name), learning_rate)
        self.observation_size = observation_size
        self.action_size = action_size

        """ State/observation """
        self.observation = tf.placeholder(shape=[self.observation_size], dtype=tf.float32,
                                          name="observation")
        self.action = tf.placeholder(shape=[self.action_size], dtype=tf.float32,
                                     name="action")

        self.input = tf.concat(
            [self.observation,
             self.action],
            axis=0
        )

        self.Q = layers.fully_connected(inputs=self.input,
                                        num_outputs=1,
                                        activation_fn=None,
                                        biases_initializer=None)

        """ Training """
        self.target = tf.placeholder(shape=[], dtype=tf.float32)
        self.loss = self.target - self.Q
        self.trainer = tf.train.AdamOptimizer(self.learning_rate)
        self.update_model = self.trainer.minimize(self.loss)

    def sample(self, sess, observation, action):
        """ calculate Q value of given observation and action """
        return sess.run(self.Q, feed_dict={
            self.observation: observation,
            self.action: action
        })

    def update(self, sess, observation, action, target):
        """ update network according to given observation, action, and target value """
        return sess.run(
            [self.loss, self.update_model],
            feed_dict={
                self.observation: observation,
                self.action: action,
                self.target: target
            }
        )
