import numpy as np

from .environment import PolySysEnvironment


def biology():
    # 10-dimension and 1-input system and 1-disturbance system
    ds = 3
    us = 2

    # Dynamics that are defined as a continuous function!
    def f(x, u):
        # random disturbance
        # d = random.uniform(0, 20)
        delta = np.zeros((ds, 1), float)
        delta[0, 0] = -0.01 * x[0, 0] - x[1, 0] * (x[0, 0] + 4.5) + u[0, 0]
        delta[1, 0] = -0.025 * x[1, 0] + 0.000013 * x[2, 0]
        delta[2, 0] = -0.093 * (x[2, 0] + 15) + (1 / 12) * u[1, 0]
        return delta

    def K_to_str(K):
        # Control policy K to text
        nvars = len(K[0])
        X = []
        for i in range(nvars):
            X.append("x[" + str(i + 1) + "]")

        ks = []
        for i in range(len(K)):
            strstr = ""
            for k in range(len(X)):
                if strstr is "":
                    strstr = str(K[i, k]) + "*" + X[k]
                else:
                    strstr = strstr + "+" + str(K[i, k]) + "*" + X[k]
            ks.append(strstr)
        return ks

    # Closed loop system dynamics to text
    def f_to_str(K):
        kstr = K_to_str(K)
        f = []
        f.append("-0.01*x[1] - x[2]*(x[1]+4.5) + {}".format(kstr[0]))
        f.append("-0.025*x[2] + 0.000013*x[3]")
        f.append("-0.093*(x[3] + 15) + (1/12)*{}".format(kstr[1]))
        return f

    h = 0.01

    # amount of Gaussian noise in dynamics
    eq_err = 1e-2

    # intial state space
    s_min = np.array([[-2], [-0], [-0.1]])
    s_max = np.array([[2], [0], [0.1]])

    Q = np.zeros((ds, ds), float)
    R = np.zeros((us, us), float)
    np.fill_diagonal(Q, 1)
    np.fill_diagonal(R, 1)

    # user defined unsafety condition
    def unsafe_eval(x):
        if x[0, 0] >= 5:
            return True
        return False

    def unsafe_string():
        return ["x[1] - 5"]

    def rewardf(x, Q, u, R):
        reward = 0
        reward += -np.dot(x.T, Q.dot(x)) - np.dot(u.T, R.dot(u))
        if unsafe_eval(x):
            reward -= 100
        return reward

    def testf(x, u):
        if unsafe_eval(x):
            return -1
        return 0

    u_min = np.array([[-50.0], [-50]])
    u_max = np.array([[50.0], [50]])

    env = PolySysEnvironment(
        f,
        f_to_str,
        rewardf,
        testf,
        unsafe_string,
        ds,
        us,
        Q,
        R,
        s_min,
        s_max,
        u_max=u_max,
        u_min=u_min,
        timestep=h,
    )
    return env
