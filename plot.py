import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from models.entity import User, Service, Device
from models.mobility import Direction, Coordinate, RectangularDirectedMobility, StaticMobility
from models.orientation import generate_random_vertical_orientation
from models.effectiveness import Effectiveness, VisualEffectiveness


def plot_effectiveness(width, height, user_speed, effectiveness):
    """
        plot_effectiveness: plot effectiveness over rectangular environment

        orientation of devices are random
        devices are static
        ignore z-axis coordinate
        ignore handover

        :param width: width of the environment
        :param height: height of the environment\
        :param user_speed: speed of the user
        :param effectiveness: effectiveness model
        :return:
    """
    def calculate_effectiveness(user_instance, x, y):
        device_mobility = StaticMobility()
        device_coordinate = Coordinate(x, y, 0)
        device_orientation = generate_random_vertical_orientation()
        device = Device(name="", device_type="Visual",
                        coordinate=device_coordinate, mobility=device_mobility, orientation=device_orientation)

        service = Service(name="", service_type="Visual", device=device)
        service.in_use = True
        service.user = user

        return effectiveness.measure(user_instance, service)

    assert isinstance(effectiveness, Effectiveness)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = np.arange(0, width, .1)
    Y = np.arange(0, height, .1)

    user_coordinate = Coordinate(width/2, height/2, 0)
    user_mobility = RectangularDirectedMobility(width, height, depth=3, direction=Direction(1, 0, 0), speed=user_speed)
    user = User(uid=0, coordinate=user_coordinate, mobility=user_mobility)

    Z = np.array([[calculate_effectiveness(user, x, y) for x in X] for y in Y])

    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(X, Y, Z)

    plt.show()


plot_effectiveness(10, 10, 1, VisualEffectiveness())
