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
    def __init__(self, name, learning_rate, discount_factor, observation_size, action_size):
        Network.__init__(self, "DRRN/{name}".format(name=name), learning_rate)
        self.observation_size = observation_size
        self.action_size = action_size
        self.discount_factor = discount_factor

        """ State/observation """
        self.user_observation = tf.placeholder(shape=[self.observation_size], dtype=tf.float32,
                                               name="observation")

        self.action_observation = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32,
                                                 name="action")
        num_actions = tf.shape(self.action_observation)[0]

        """ hidden layers """
        self.user_tile = tf.tile(
            tf.reshape(self.user_observation, [1, self.observation_size]),
            [num_actions, 1])

        self.user_hidden_layer_output = layers.fully_connected(inputs=self.user_tile,
                                                               num_outputs=128)
        self.action_hidden_layer_output = layers.fully_connected(inputs=self.action_observation,
                                                                 num_outputs=128)

        """ combine observations """
        self.input = tf.concat(
            [self.user_hidden_layer_output,
             self.action_hidden_layer_output],
            axis=1
        )

        self.Q = layers.fully_connected(inputs=self.input,
                                        num_outputs=1,
                                        activation_fn=None,
                                        biases_initializer=None)

        """ Training """
        self.target = tf.placeholder(shape=[], dtype=tf.float32)
        self.action = tf.placeholder(shape=[], dtype=tf.int32)

        """ only considers output that selected """
        self.responsible_output = tf.gather(self.Q, self.action)
        self.loss = self.target - self.responsible_output
        self.trainer = tf.train.AdamOptimizer(self.learning_rate)
        self.update_model = self.trainer.minimize(self.loss)

    def sample(self, sess, observation, actions):
        """ calculate Q value of given observation and action """
        return sess.run(self.Q, feed_dict={
            self.user_observation: observation,
            self.action_observation: actions
        })

    def update(self, sess, observation, actions, action, reward, next_observation, next_actions):
        """ update network according to given observation, action, reward and next_observation value """
        target = reward + self.discount_factor * np.max(self.sample(sess, next_observation, next_actions))
        return sess.run(
            (self.loss, self.update_model),
            feed_dict={
                self.user_observation: observation,
                self.action_observation: actions,
                self.action: action,
                self.target: target
            }
        )
