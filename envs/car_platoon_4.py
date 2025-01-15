import numpy as np

from .environment import Environment


def car_platoon_4():
    A = np.matrix(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0.1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0.1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0.1],
            [0, 0, 0, 0, 0, 0, 1],
        ]
    )
    B = np.matrix(
        [
            [0.1, 0, 0, 0],
            [0, 0, 0, 0],
            [0.1, -0.1, 0, 0],
            [0, 0, 0, 0],
            [0, 0.1, -0.1, 0],
            [0, 0, 0, 0],
            [0, 0, 0.1, -0.1],
        ]
    )

    # intial state space
    s_min = np.array([[19.9], [0.9], [-0.1], [0.9], [-0.1], [0.9], [-0.1]])
    s_max = np.array([[20.1], [1.1], [0.1], [1.1], [0.1], [1.1], [0.1]])

    Q = np.matrix(
        "1 0 0 0 0 0 0; 0 1 0 0 0 0 0; 0 0 1 0 0 0 0; 0 0 0 1 0 0 0; 0 0 0 0 1 0 0; 0 0 0 0 0 1 0; 0 0 0 0 0 0 1"
    )
    R = np.matrix(".0005 0 0 0; 0 .0005 0 0; 0 0 .0005 0; 0 0 0 .0005")

    x_min = np.array([[18], [0.5], [-0.35], [0.5], [-1], [0.5], [-1]])
    x_max = np.array([[22], [1.5], [0.35], [1.5], [1], [1.5], [1]])
    u_min = np.array([[-10.0], [-10.0], [-10.0], [-10.0]])
    u_max = np.array([[10.0], [10.0], [10.0], [10.0]])

    # Coordination transformation
    origin = np.array([[20], [1], [0], [1], [0], [1], [0]])
    s_min -= origin
    s_max -= origin
    x_min -= origin
    x_max -= origin

    env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R)

    return env
