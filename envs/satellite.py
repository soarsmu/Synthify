import numpy as np
from .environment import Environment


def satellite():
    A = np.matrix([[2, -1], [1, 0]])

    B = np.matrix([[2], [0]])

    # intial state space
    s_min = np.array([[-1.0], [-1.0]])
    s_max = np.array([[1.0], [1.0]])

    Q = np.matrix("1 0 ; 0 1")
    R = np.matrix(".0005")

    x_min = np.array([[-1.5], [-1.5]])
    x_max = np.array([[1.5], [1.5]])
    u_min = np.array([[-10.0]])
    u_max = np.array([[10.0]])

    env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R)

    return env
