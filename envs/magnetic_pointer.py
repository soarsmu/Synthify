import numpy as np

from .environment import Environment


def magnetic_pointer():
    A = np.matrix([[2.6629, -1.1644, 0.66598], [2, 0, 0], [0, 0.5, 0]])

    B = np.matrix([[0.25], [0], [0]])

    # intial state space
    s_min = np.array([[-1.0], [-1.0], [-1.0]])
    s_max = np.array([[1.0], [1.0], [1.0]])

    Q = np.matrix("1 0 0 ; 0 1 0; 0 0 1")
    R = np.matrix("1")

    x_min = np.array([[-3.5], [-3.5], [-3.5]])
    x_max = np.array([[3.5], [3.5], [3.5]])
    u_min = np.array([[-15.0]])
    u_max = np.array([[15.0]])

    env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R)

    return env
