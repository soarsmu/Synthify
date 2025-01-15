import numpy as np

from .environment import Environment


def cartpole():
    A = np.matrix([[0, 1, 0, 0], [0, 0, 0.716, 0], [0, 0, 0, 1], [0, 0, 15.76, 0]])
    B = np.matrix([[0], [0.9755], [0], [1.46]])

    # intial state space
    s_min = np.array([[-0.05], [-0.1], [-0.05], [-0.05]])
    s_max = np.array([[0.05], [0.1], [0.05], [0.05]])

    Q = np.matrix("1 0 0 0; 0 1 0 0 ; 0 0 1 0; 0 0 0 1")
    R = np.matrix(".0005")

    x_min = np.array([[-0.3], [-0.5], [-0.3], [-0.5]])
    x_max = np.array([[0.3], [0.5], [0.3], [0.5]])
    u_min = np.array([[-15.0]])
    u_max = np.array([[15.0]])
    env = Environment(
        A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, continuous=True
    )

    return env
