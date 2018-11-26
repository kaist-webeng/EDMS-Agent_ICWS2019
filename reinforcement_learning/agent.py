import tensorflow as tf
import numpy as np
import datetime
import random
from abc import abstractmethod

from reinforcement_learning.network import DRRN
from reinforcement_learning.experience_memory import ExperienceMemory
from models.environment import Environment
from utils import variable_summaries


class Agent:
    def __init__(self, env, num_episode, num_step):
        assert isinstance(env, Environment)
        self.env = env

        # Training configuration
        self.num_episode = num_episode
        self.num_step = num_step

        # Date of now, for logging
        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    @abstractmethod
    def selection(self, sess, user, services):
        """ return selected service object and its index """
        return None, 0

    @abstractmethod
    def train(self, sess):
        pass

    def test(self, sess):
        print("Test phase")
        for i_episode in range(self.num_episode):
            print("Episode %d" % i_episode)

            score = 0.
            observation = self.env.reset()

            """ since service selection is non-episodic task, restrict maximum step rather than observe done-signal """
            for i_step in range(self.num_step):
                """ select action """
                action, _ = self.selection(sess, observation["user"], observation["services"])
                """ perform the selected action on the environment """
                observation, reward, done = self.env.step(action)
                """ add reward to total score """
                score += reward

            print("Episode %d ends with average score %r" % (i_episode, score / self.num_step))


class RandomSelectionAgent(Agent):
    """ RandomSelectionAgent: the baseline agent that selects services randomly """
    def selection(self, sess, user, services):
        index = random.choice(range(len(services)))
        return services[index], index


class NearestSelectionAgent(Agent):
    """ ClosestSelectionAgent: the baseline agent that selects the nearest service"""
    def selection(self, sess, user, services):
        services.sort(key=lambda service: user.distance(service.device))
        return services[0], 0


class DRRNSelectionAgent(Agent):
    def __init__(self, env, num_episode, num_step, learning_rate, discount_factor, memory_size, batch_size):
        Agent.__init__(self, env, num_episode, num_step)

        self.network = DRRN(name="test",
                            learning_rate=learning_rate,
                            discount_factor=discount_factor,
                            observation_size=self.env.get_observation_size(),
                            action_size=self.env.get_action_size())

        """ Experience memory setting """
        self.memory = ExperienceMemory(memory_size)
        self.batch_size = batch_size

        """ Summary """
        self.loss_list = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="LossList")
        variable_summaries(self.loss_list, "Loss")
        self.reward_list = tf.placeholder(shape=[None], dtype=tf.float32, name="RewardList")
        variable_summaries(self.reward_list, "Reward")
        self.summary = tf.summary.merge_all()

    def selection(self, sess, user, services):
        Q_set = self.network.sample(sess, user.vectorize(), [service.vectorize() for service in services])
        selection = np.argmax(Q_set)
        return services[selection], selection

    def train(self, sess):
        print("Train phase")

        writer = tf.summary.FileWriter('./summary/train/' + self.date, sess.graph)

        for i_episode in range(self.num_episode):
            print("Episode %d" % i_episode)

            reward_list = []
            loss_list = []
            observation = self.env.reset()

            for i_step in range(self.num_step):
                """ select action """
                action, action_index = self.selection(sess, observation["user"], observation["services"])
                """ perform the selected action on the environment """
                next_observation, reward, done = self.env.step(action)
                reward_list.append(reward)

                if observation["services"] and next_observation["services"]:
                    """ record only when services are discovered currently and next state """
                    self.memory.add(observation, action_index, reward, next_observation)

                if self.memory.is_full():
                    """ training the network """
                    loss_list += self.learn(sess)

                """ set observation to next state """
                observation = next_observation

            self.summarize_episode(sess, writer, i_episode, loss_list, reward_list)
            print(reward_list)
            print("Episode %d ends with average score %r" % (i_episode, np.mean(reward_list)))

    def learn(self, sess):
        batch = self.memory.sample(self.batch_size)
        loss_list = []

        for memory in batch:
            loss, _ = self.network.update(sess=sess,
                                          observation=memory["observation"]["user"].vectorize(),
                                          actions=[service.vectorize() for service in memory["observation"]["services"]],
                                          action=memory["action"],
                                          reward=memory["reward"],
                                          next_observation=memory["next_observation"]["user"].vectorize(),
                                          next_actions=[service.vectorize() for service in memory["next_observation"]["services"]])
            loss_list.append(loss)
        return loss_list

    def summarize_episode(self, sess, writer, i_episode, loss_list, reward_list):
        writer.add_summary(
            sess.run(self.summary,
                     feed_dict={
                         self.loss_list: loss_list,
                         self.reward_list: reward_list
                     }),
            i_episode
        )
