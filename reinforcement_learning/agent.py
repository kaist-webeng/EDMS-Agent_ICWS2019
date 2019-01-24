import tensorflow as tf
import numpy as np
import datetime
import random
import time
from abc import abstractmethod

from reinforcement_learning.network import DRRN
from reinforcement_learning.experience_memory import ExperienceMemory
from models.environment import Environment
from utils import variable_summaries


class Agent:
    def __init__(self, name, env, num_episode, num_step):
        self.name = name

        assert isinstance(env, Environment)
        self.env = env

        # Training configuration
        self.num_episode = num_episode
        self.num_step = num_step

        # Date of now, for logging
        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

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

        writer = tf.summary.FileWriter('./summary/{name}/test/{date}'.format(name=self.name, date=self.date),
                                       sess.graph)

        for i_episode in range(self.num_episode):
            print("Episode %d" % i_episode)

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
    """ GreedySelectionAgent: a baseline agent that selects best one currently """
    def selection(self, sess, user, services):
        maximum = -1000000
        index = -1
        for i in range(len(services)):
            if self.env.reward_function.measure(user, services[i]) > maximum:
                index = i
                maximum = self.env.reward_function.measure(user, services[i])
        return services[index], index


class DRRNSelectionAgent(Agent):
    def __init__(self, name, env, num_episode, num_step, learning_rate, discount_factor, memory_size, batch_size):
        Agent.__init__(self, name, env, num_episode, num_step)

        self.network = DRRN(name="DRRN",
                            learning_rate=learning_rate,
                            discount_factor=discount_factor,
                            observation_size=self.env.get_observation_size(),
                            action_size=self.env.get_action_size())

        """ Experience memory setting """
        self.memory = ExperienceMemory(memory_size)
        self.batch_size = batch_size

    def selection(self, sess, user, services):
        Q_set = self.network.sample(sess, user.vectorize(), [service.vectorize() for service in services])
        selection = np.argmax(Q_set)
        return services[selection], selection

    def train(self, sess):
        print("Train phase")

        writer = tf.summary.FileWriter('./summary/{name}/train/{date}'.format(name=self.name, date=self.date),
                                       sess.graph)

        """ Epsilon greedy configuration """
        eps = 1.0
        eps_final = 1e-2
        # set decaying rate according to the number of episodes: to make epsilon reaches eps_final at the end
        eps_decay = np.power(eps_final, 1 / self.num_episode)

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
            if eps > eps_final:
                eps = eps_decay * eps

            self.summarize_episode(sess, writer, i_episode, loss_list, reward_list, execution_time_list)
            print("Episode {i} ends with average score {reward}, loss {loss}".format(i=i_episode,
                                                                                     reward=np.mean(reward_list),
                                                                                     loss=np.mean(loss_list)))
            if loss_list and np.max(loss_list) < stop_training_threshold:
                print("Stop training")
                break

    def learn(self, sess):
        batch = self.memory.sample(self.batch_size)
        loss_list = []

        for memory in batch:
            loss, _ = self.network.update(sess=sess,
                                          observation=memory["observation"]["user"].vectorize(),
                                          actions=[service.vectorize() for service in memory["observation"]["services"]],
                                          action=memory["action"],
                                          reward=memory["reward"].get_overall_score(),
                                          next_observation=memory["next_observation"]["user"].vectorize(),
                                          next_actions=[service.vectorize() for service in memory["next_observation"]["services"]],
                                          done=memory["done"])
            loss_list.append(loss)
        return loss_list
