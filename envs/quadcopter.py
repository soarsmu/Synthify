import numpy as np

from .environment import Environment

def quadcopter():
    A = np.matrix([[1,1], [0,1]])
    B = np.matrix([[0],[1]])

    #intial state space
    s_min = np.array([[-0.5],[-0.5]])
    s_max = np.array([[ 0.5],[ 0.5]])

    # LQR quadratic cost per state
    Q = np.matrix("1 0; 0 0")
    R = np.matrix("1.0")

    x_min = np.array([[-1.],[-1.]])
    x_max = np.array([[ 1.],[ 1.]])
    u_min = np.array([[-15.]])
    u_max = np.array([[ 15.]])

    env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R)

    return env