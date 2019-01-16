from models.effectiveness import VisualEffectiveness
from models.environment import SingleUserSingleServicePartialObservableEnvironment
from models.observation import EuclideanObservation, FullObservation
from reinforcement_learning.agent import *
from reinforcement_learning.reward import HandoverPenaltyRewardFunction


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
    def __init__(self, num_device, width, height, depth, max_speed, observation_range, num_episode, num_step, memory_size, batch_size, learning_rate, discount_factor, agent):
        # observation = EuclideanObservation(observation_range=observation_range)
        observation = FullObservation()
        effectiveness = VisualEffectiveness()
        reward_function = HandoverPenaltyRewardFunction(effectiveness=effectiveness)
        self.env = SingleUserSingleServicePartialObservableEnvironment(service_type='visual',
                                                                       num_device=num_device,
                                                                       width=width,
                                                                       height=height,
                                                                       depth=depth,
                                                                       max_speed=max_speed,
                                                                       observation=observation,
                                                                       reward_function=reward_function)
        self.num_episode = num_episode
        self.num_step = num_step
        self.memory_size = memory_size
        self.batch_size = batch_size

        """ In the code, only one agent should be constructed, Otherwise, error occurs in summary """
        if agent == "random":
            self.agent = RandomSelectionAgent("Random", self.env, self.num_episode, self.num_step)
        if agent == "nearest":
            self.agent = NearestSelectionAgent("Nearest", self.env, self.num_episode, self.num_step)
        if agent == "nohandover":
            self.agent = NoHandoverSelectionAgent("NoHandover", self.env, self.num_episode, self.num_step)
        if agent == "greedy":
            self.agent = GreedySelectionAgent("Greedy", self.env, self.num_episode, self.num_step)
        if agent == "DRRN":
            self.agent = DRRNSelectionAgent("DRRN", self.env, self.num_episode, self.num_step,
                                            learning_rate=learning_rate,
                                            discount_factor=discount_factor,
                                            memory_size=self.memory_size,
                                            batch_size=self.batch_size)

    def reset(self):
        self.env.reset()

    def run(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            self.agent.train(sess)
            self.agent.test(sess)

