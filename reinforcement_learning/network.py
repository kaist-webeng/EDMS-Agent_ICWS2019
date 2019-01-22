import tensorflow as tf
import numpy as np
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
        with tf.variable_scope("Observation"):
            self.observation = tf.placeholder(shape=[self.observation_size], dtype=tf.float32,
                                              name="observation")

            self.action_set = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32,
                                             name="action")
            num_actions = tf.shape(self.action_set)[0]

            self.observation_tile = tf.tile(
                tf.reshape(self.observation, [1, self.observation_size]),
                [num_actions, 1])

        """ hidden layers """
        with tf.variable_scope("Hidden"):
            self.user_hidden_layer_1 = tf.layers.dense(inputs=self.observation_tile,
                                                       activation=tf.nn.tanh,
                                                       units=256)
            self.user_hidden_layer_output = tf.layers.dense(inputs=self.user_hidden_layer_1,
                                                            activation=tf.nn.tanh,
                                                            units=256)

            self.action_hidden_layer_1 = tf.layers.dense(inputs=self.action_set,
                                                         activation=tf.nn.tanh,
                                                         units=256)
            self.action_hidden_layer_output = tf.layers.dense(inputs=self.action_hidden_layer_1,
                                                              activation=tf.nn.tanh,
                                                              units=256)

        """ combine observation and action """
        self.combine = tf.concat(
            [self.user_hidden_layer_output,
             self.action_hidden_layer_output],
            axis=1
        )

        self.Q = tf.layers.dense(inputs=self.combine,
                                 units=1,
                                 activation=None,
                                 bias_initializer=None)

        with tf.variable_scope("Training"):
            """ Training """
            self.target = tf.placeholder(shape=[], dtype=tf.float32, name="target")
            self.action = tf.placeholder(shape=[], dtype=tf.int32, name="action")

            """ only considers output that selected """
            self.action_one_hot = tf.one_hot(self.action, num_actions, dtype=tf.float32)
            self.responsible_Q = tf.reduce_sum(tf.multiply(self.Q, self.action_one_hot))
            self.loss = tf.square(self.target - self.responsible_Q)
            self.trainer = tf.train.AdamOptimizer(self.learning_rate)
            self.update_model = self.trainer.minimize(self.loss)

    def sample(self, sess, observation, actions):
        """ calculate Q value of given observation and action """
        return sess.run(self.Q, feed_dict={
            self.observation: observation,
            self.action_set: actions
        })

    def update(self, sess, observation, actions, action, reward, next_observation, next_actions, done):
        """ update network according to given observation, action, reward and next_observation value """
        if done or not next_actions:
            """ if done, just add reward """
            target = reward
        else:
            """ else, bootstrapping next Q value """
            target = reward + self.discount_factor * np.max(self.sample(sess, next_observation, next_actions))
        return sess.run(
            (self.loss, self.update_model),
            feed_dict={
                self.observation: observation,
                self.action_set: actions,
                self.action: action,
                self.target: target
            }
        )
