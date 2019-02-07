from models.environment import SingleUserSingleServicePartialObservableEnvironment
from reinforcement_learning.agent import *


class Experiment:
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def run(self):
        pass


class EffectDrivenServiceSelectionExperiment(Experiment):
    """
        EffectDrivenServiceSelectionExperiment
        
        Experiment done on the SingleUserSingleServicePartialObservable3DEnvironment
    """
    def __init__(self, configuration):
        self.configuration = configuration

        self.env = SingleUserSingleServicePartialObservableEnvironment(service_type='visual',
                                                                       num_device=configuration.num_device,
                                                                       width=configuration.width,
                                                                       height=configuration.height,
                                                                       depth=configuration.depth,
                                                                       max_speed=configuration.max_speed,
                                                                       observation=configuration.observation,
                                                                       reward_function=configuration.reward_function)
        self.num_episode = configuration.num_episode
        self.num_step = configuration.num_step
        self.memory_size = configuration.memory_size
        self.batch_size = configuration.batch_size

        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        """ In the code, only one agent should be constructed, Otherwise, error occurs in summary """
        if configuration.agent == "random":
            self.agent = RandomSelectionAgent("Random", self.env, self.date, self.num_episode, self.num_step)
        if configuration.agent == "nearest":
            self.agent = NearestSelectionAgent("Nearest", self.env, self.date, self.num_episode, self.num_step)
        if configuration.agent == "nohandover":
            self.agent = NoHandoverSelectionAgent("NoHandover", self.env, self.date, self.num_episode, self.num_step)
        if configuration.agent == "greedy":
            self.agent = GreedySelectionAgent("Greedy", self.env, self.date, self.num_episode, self.num_step)
        if configuration.agent == "EDSS":
            self.agent = EDSSAgent("EDSS", self.env, self.date, self.num_episode, self.num_step,
                                   learning_rate=configuration.learning_rate,
                                   discount_factor=configuration.discount_factor,
                                   memory_size=self.memory_size,
                                   batch_size=self.batch_size)

    def reset(self):
        self.env.reset()

    def run(self):
        self.configuration.save(name=self.configuration.agent,
                                date=self.date)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            self.agent.train(sess)
            self.agent.test(sess)

