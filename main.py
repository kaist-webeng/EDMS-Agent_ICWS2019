import tensorflow as tf

from configuration import Configuration
from experiment import EffectDrivenServiceSelectionExperiment
from models.observation import EuclideanObservation, FullObservation
from models.effectiveness import VisualEffectiveness
from reinforcement_learning.reward import HandoverPenaltyRewardFunction


flags = tf.flags
FLAGS = tf.flags.FLAGS

flags.DEFINE_string('summary_path', "./summary", 'Path of summary files')


def main():
    conf = Configuration(num_device=100,
                         width=200,
                         height=10,
                         depth=3,
                         device_size_min=0.5,
                         device_size_max=2,
                         max_speed=2,
                         observation=EuclideanObservation(observation_range=10),
                         reward_function=HandoverPenaltyRewardFunction(effectiveness=VisualEffectiveness()),
                         num_episode=10000,
                         num_step=100,
                         memory_size=1000,
                         batch_size=100,
                         learning_rate=0.000001,
                         discount_factor=.95,
                         agent="EDSS")
                         eps_init=1.0,
                         eps_final=1e-1,

    """ unit of distance is Meter """
    experiment = EffectDrivenServiceSelectionExperiment(configuration=conf)
    experiment.run()


if __name__ == '__main__':
    main()
