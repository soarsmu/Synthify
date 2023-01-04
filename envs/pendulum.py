import numpy as np

from .environment import Environment

def pendulum():
    m = 1.
    l = 1.
    g = 10.

    #Dynamics that are continuous
    A = np.matrix([
        [ 0., 1.],
        [g/l, 0.]
        ])
    B = np.matrix([
        [          0.],
        [1./(m*l**2.)]
        ])


    #intial state space
    s_min = np.array([[-0.35],[-0.35]])
    s_max = np.array([[ 0.35],[ 0.35]])

    #reward function
    Q = np.matrix([[1., 0.],[0., 1.]])
    R = np.matrix([[.005]])

    #safety constraint
    x_min = np.array([[-0.5],[-0.5]])
    x_max = np.array([[ 0.5],[ 0.5]])
    u_min = np.array([[-15.]])
    u_max = np.array([[ 15.]])

    env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, continuous=True)

    return env
