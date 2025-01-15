import numpy as np

from .environment import Environment


def pendulum():
    m = 1.0
    l = 1.0
    g = 10.0

    # Dynamics that are continuous
    A = np.matrix([[0.0, 1.0], [g / l, 0.0]])
    B = np.matrix([[0.0], [1.0 / (m * l**2.0)]])

    # intial state space
    s_min = np.array([[-0.35], [-0.35]])
    s_max = np.array([[0.35], [0.35]])

    # reward function
    Q = np.matrix([[1.0, 0.0], [0.0, 1.0]])
    R = np.matrix([[0.005]])

    # safety constraint
    x_min = np.array([[-0.5], [-0.5]])
    x_max = np.array([[0.5], [0.5]])
    u_min = np.array([[-15.0]])
    u_max = np.array([[15.0]])

    env = Environment(
        A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, continuous=True
    )

    return env
