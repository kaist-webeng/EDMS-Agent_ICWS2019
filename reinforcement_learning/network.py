import tensorflow as tf
import numpy as np
from abc import abstractmethod


class Network:
    def __init__(self, name, learning_rate):
        self.name = "Network/{name}".format(name=name)
        self.learning_rate = learning_rate

        self.target_network = None
        self.variables = None
        self.target_variables = None

        self.scope = name

    # Set target network to update from this network
    def set_target_network(self, target_network):
        assert type(self) == type(target_network)
        self.target_network = target_network

        # Get variables of itself
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        # Get variables of target network
        self.target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_network.scope)

    def update_target_network(self, sess, tau):
        assert self.target_network and len(self.variables) == len(self.target_variables)

        for idx, var in enumerate(self.variables):
            sess.run(self.target_variables[idx].assign((tau * var.value()) +
                                                       ((1 - tau) * self.target_variables[idx].value())))

    def copy_from_target_network(self, sess):
        assert self.target_network and len(self.variables) == len(self.target_variables)

        for idx, var in enumerate(self.target_variables):
            sess.run(self.variables[idx].assign(var.value()))

    @abstractmethod
    def sample(self, sess, observation, actions):
        pass

    @abstractmethod
    def bootstrap(self, sess, next_observation, next_actions):
        pass

    @abstractmethod
    def update(self, sess, observation, actions, action, reward, next_observation, next_actions, done):
        pass


class EDSSNetworkDQN(Network):
    """
        EDSSNetworkDQN network for variable size of action space

        Refer DRRN
        He, Ji, et al. "Deep reinforcement learning with an action space defined by natural language." (2016).
    """
    def __init__(self, name, learning_rate, discount_factor, observation_size, action_size):
        Network.__init__(self, "{name}".format(name=name), learning_rate)
        self.observation_size = observation_size
        self.action_size = action_size
        self.discount_factor = discount_factor

        with tf.variable_scope(self.name):
            """ State/observation """
            with tf.variable_scope("Observation"):
                self.observation = tf.placeholder(shape=[self.observation_size], dtype=tf.float32,
                                                  name="observation")

                self.action_set = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32,
                                                 name="action")
                num_actions = tf.shape(self.action_set)[0]

                observation_tile = tf.tile(
                    tf.reshape(self.observation, [1, self.observation_size]),
                    [num_actions, 1])

                relative_coordinate = tf.subtract(tf.slice(self.action_set, [0, 0], [num_actions, 3]),
                                                  tf.slice(observation_tile, [0, 0], [num_actions, 3]))

            """ combine values """
            with tf.variable_scope("Combine"):
                """ concatenate observation and action """
                combine = tf.concat(
                    [tf.slice(observation_tile, [0, 3], [num_actions, 3]),
                     tf.slice(self.action_set, [0, 3], [num_actions, 5]),
                     relative_coordinate],
                    axis=1
                )

                combine_hidden_layer_1 = tf.layers.dense(inputs=combine,
                                                         activation=tf.nn.leaky_relu, units=128)
                combine_hidden_layer_2 = tf.layers.dense(inputs=combine_hidden_layer_1,
                                                         activation=tf.nn.leaky_relu, units=128)
                combine_hidden_layer_output = tf.layers.dense(inputs=combine_hidden_layer_2,
                                                              activation=tf.nn.leaky_relu, units=128)

            self.Q = tf.layers.dense(inputs=combine_hidden_layer_output,
                                     activation=None,
                                     units=1)

            with tf.variable_scope("Training"):
                """ Training """
                self.target_Q = tf.placeholder(shape=[], dtype=tf.float32, name="target_Q")
                self.action = tf.placeholder(shape=[], dtype=tf.int32, name="action")

                """ only considers output that selected """
                self.action_one_hot = tf.one_hot(self.action, num_actions, dtype=tf.float32)
                self.responsible_Q = tf.reduce_sum(tf.multiply(self.Q, self.action_one_hot))
                self.loss = tf.square(self.target_Q - self.responsible_Q)
                self.trainer = tf.train.AdamOptimizer(self.learning_rate)
                self.update_model = self.trainer.minimize(self.loss)

    def sample(self, sess, observation, actions):
        """ calculate Q value of given observation and action """
        return sess.run(self.Q, feed_dict={
            self.observation: observation,
            self.action_set: actions
        })

    def bootstrap(self, sess, next_observation, next_actions):
        """ bootstrapping Q value of given next_observation and next_actions"""
        if self.target_network:
            """ [DDQN] if target network exist, bootstrapping from target network """
            return self.target_network.sample(sess, next_observation, next_actions)
        return self.sample(sess, next_observation, next_actions)

    def update(self, sess, observation, actions, action, reward, next_observation, next_actions, done):
        """ update network according to given observation, action, reward and next_observation value """
        if done or not next_actions:
            """ if done, just add reward """
            target_Q = reward
        else:
            """ else, bootstrapping next Q value """
            target_Q = reward + self.discount_factor * np.max(self.bootstrap(sess, next_observation, next_actions))

        return sess.run(
            (self.loss, self.update_model),
            feed_dict={
                self.observation: observation,
                self.action_set: actions,
                self.action: action,
                self.target_Q: target_Q
            }
        )
