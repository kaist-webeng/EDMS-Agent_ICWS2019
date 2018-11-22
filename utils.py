import random
from mobility import *


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def generate_random_coordinate(width, height, depth):
    return Coordinate(x=random.random() * width, y=random.random() * height, z=random.random() * depth)
