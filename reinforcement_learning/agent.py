import tensorflow as tf
import numpy as np
import datetime
import random
import time
from abc import abstractmethod

from reinforcement_learning.network import EDSS
from reinforcement_learning.experience_memory import BasicExperienceMemory, BalancingExperienceMemory
from models.environment import Environment
from utils import variable_summaries


class Agent:
    def __init__(self, name, env, date, num_episode, num_step):
        self.name = name

        assert isinstance(env, Environment)
        self.env = env

        # Training configuration
        self.num_episode = num_episode
        self.num_step = num_step

        # Date of now, for logging
        self.date = date

        with tf.variable_scope("Summary"):
            """ Summary """
            self.loss_list = tf.placeholder(shape=[None], dtype=tf.float32, name="LossList")
            variable_summaries(self.loss_list, "Loss")
            self.execution_time_list = tf.placeholder(shape=[None], dtype=tf.float32, name="ExecutionTImeList")
            variable_summaries(self.execution_time_list, "ExecutionTime")

        self.summary = tf.summary.merge_all()

    @abstractmethod
    def selection(self, sess, user, services):
        """ return selected service object and its index """
        return None, 0

    def train(self, sess):
        pass

    def test(self, sess):
        print("Test phase")

        writer = tf.summary.FileWriter('./summary/{name}/{date}/test'.format(name=self.name, date=self.date),
                                       sess.graph)

        for i_episode in range(self.num_episode):
            print("Episode %d" % i_episode)
            random.seed(i_episode)

            reward_list = []
            execution_time_list = []
            observation = self.env.reset()

            """ since service selection is non-episodic task, restrict maximum step rather than observe done-signal """
            for i_step in range(self.num_step):
                start_time = time.time()
                """ select action """
                action, _ = self.selection(sess, observation["user"], observation["services"])
                execution_time_list.append(time.time() - start_time)
                """ perform the selected action on the environment """
                observation, reward, done = self.env.step(action)
                """ add reward to total score """
                reward_list.append(reward)

                if done:
                    break

            self.summarize_episode(sess, writer, i_episode, [], reward_list, execution_time_list)
            print("Episode {i} ends with average score {reward}".format(i=i_episode,
                                                                        reward=np.mean(reward_list)))

    def summarize_episode(self, sess, writer, i_episode, loss_list, reward_list, execution_time_list):
        feed_dict = self.env.reward_function.get_summary_feed_dict(reward_list)
        feed_dict[self.loss_list] = loss_list
        feed_dict[self.execution_time_list] = execution_time_list
        writer.add_summary(
            sess.run(self.summary, feed_dict=feed_dict),
            i_episode
        )


class RandomSelectionAgent(Agent):
    """ RandomSelectionAgent: a baseline agent that selects services randomly """
    def selection(self, sess, user, services):
        index = random.choice(range(len(services)))
        return services[index], index


class NearestSelectionAgent(Agent):
    """ ClosestSelectionAgent: a baseline agent that selects the nearest service"""
    def selection(self, sess, user, services):
        minimum = 1000000
        index = -1
        for i in range(len(services)):
            if user.distance(services[i].device) < minimum:
                index = i
                minimum = user.distance(services[i].device)
        return services[index], index


class NoHandoverSelectionAgent(Agent):
    """ NoHandoverSelectionAgent: a baseline agent that minimizes the number of handovers """
    def selection(self, sess, user, services):
        for i in range(len(services)):
            if services[i].in_use and services[i].user == user:
                return services[i], i
        """ if no service is currently in-use, select randomly """
        index = random.choice(range(len(services)))
        return services[index], index


class GreedySelectionAgent(Agent):
    """ GreedySelectionAgent: a baseline agent that selects best one currently, in terms of effectiveness """
    def selection(self, sess, user, services):
        maximum = -1000000
        index = -1
        for i in range(len(services)):
            reward = self.env.reward_function.measure(user, services[i])
            if reward.effectiveness > maximum:
                index = i
                maximum = reward.effectiveness
        return services[index], index


class EDSSAgent(Agent):
    def __init__(self, name, env, date, num_episode, num_step,
                 memory_size, batch_size, learning_rate, discount_factor,
                 eps_init, eps_final, eps_decay):
        Agent.__init__(self, name, env, date, num_episode, num_step)


        self.main_network = EDSS(name="main",
                                 learning_rate=learning_rate,
                                 discount_factor=discount_factor,
                                 observation_size=self.env.get_observation_size(),
                                 action_size=self.env.get_action_size())

        #self.main_network.set_target_network(self.target_network)

        """ Experience memory setting """
        self.memory = BasicExperienceMemory(memory_size)
        self.batch_size = batch_size

        """ Epsilon greedy setting """
        self.eps_init = eps_init
        self.eps_final = eps_final
        self.eps_decay = eps_decay

    def selection(self, sess, user, services):
        Q_set = self.main_network.sample(sess, user.vectorize(), [service.vectorize() for service in services])
        selection = np.argmax(Q_set)
        return services[selection], selection

    def train(self, sess):
        print("Train phase")

        writer = tf.summary.FileWriter('{path}/{name}/{date}/train'.format(path=tf.flags.FLAGS.summary_path,
                                                                           name=self.name, date=self.date),
                                       sess.graph)

        """ Epsilon greedy configuration """
        eps = self.eps_init

        stop_training_threshold = 1

        for i_episode in range(self.num_episode):
            print("Episode %d" % i_episode)

            reward_list = []
            loss_list = []
            execution_time_list = []
            observation = self.env.reset()

            for i_step in range(self.num_step):
                start_time = time.time()
                """ select action: epsilon-greedy """
                if random.random() <= eps:
                    action_index = random.choice(range(len(observation["services"])))
                    action = observation["services"][action_index]
                else:
                    action, action_index = self.selection(sess, observation["user"], observation["services"])
                execution_time_list.append(time.time() - start_time)

                """ perform the selected action on the environment """
                next_observation, reward, done = self.env.step(action)
                reward_list.append(reward)

                self.memory.add(observation, action_index, reward, next_observation, done)

                if self.memory.is_full():
                    """ training the network """
                    loss_list += self.learn(sess)

                if done:
                    break

                """ set observation to next state """
                observation = next_observation
                
            """ epsilon decaying for each episode """
            if eps > self.eps_final:
                eps = self.eps_decay * eps
            else:
                eps = self.eps_final

            self.summarize_episode(sess, writer, i_episode, loss_list, reward_list, execution_time_list)
            print("Episode {i} ends with average score {reward}, loss {loss}".format(i=i_episode,
                                                                                     reward=np.mean(reward_list),
                                                                                     loss=np.mean(loss_list)))

            if isinstance(self.memory, BalancingExperienceMemory):
                print("Reward distribution: %s" % self.memory.count)

            if loss_list and np.max(loss_list) < stop_training_threshold:
                print("Stop training")
                break

    def learn(self, sess):
        batch = self.memory.sample(self.batch_size)
        loss_list = []

        for memory in batch:
            loss, _ = self.main_network.update(sess=sess,
                                               observation=memory["observation"]["user"].vectorize(),
                                               actions=[service.vectorize() for service in memory["observation"]["services"]],
                                               action=memory["action"],
                                               reward=memory["reward"].get_overall_score(),
                                               next_observation=memory["next_observation"]["user"].vectorize(),
                                               next_actions=[service.vectorize() for service in memory["next_observation"]["services"]],
                                               done=memory["done"])
            loss_list.append(loss)
        return loss_list
