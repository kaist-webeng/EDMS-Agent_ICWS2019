import tensorflow as tf
import argparse

from configuration import Configuration
from experiment import EffectDrivenServiceSelectionExperiment
from models.observation import EuclideanObservation, FullObservation
from models.effectiveness import VisualEffectiveness
from reinforcement_learning.reward import HandoverPenaltyRewardFunction


flags = tf.flags
FLAGS = tf.flags.FLAGS

flags.DEFINE_string('summary_path', "./summary", 'Path of summary files')

parser = argparse.ArgumentParser()
parser.add_argument("agent", help="name of agent to simulate")
args = parser.parse_args()


def main():
    conf = Configuration(num_device=100,
                         width=200,
                         height=10,
                         depth=3,
                         device_size_min=0.5,
                         device_size_max=2,
                         max_speed=1,
                         observation=EuclideanObservation(observation_range=10),
                         #observation=FullObservation(),
                         reward_function=HandoverPenaltyRewardFunction(effectiveness=VisualEffectiveness(
                             text_size_pixel=27,
                             resolution=1080,
                             visual_angle_min=5/60,
                             FoV_angle_max=105,
                             face_angle_max=90
                         )),
                         num_episode=1000,
                         num_step=100,
                         memory_size=1000,
                         batch_size=100,
                         learning_rate=1e-9,
                         discount_factor=.95,
                         eps_init=1.0,
                         eps_final=1e-2,
                         agent=args.agent)

    """ unit of distance is Meter """
    experiment = EffectDrivenServiceSelectionExperiment(configuration=conf)
    experiment.run()


if __name__ == '__main__':
    main()
