import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from DDPG import DDPG
from metrics import timeit
from envs import ENV_CLASSES


def policy_distance(env, policy, coffset, len_episodes):
    s = env.reset()
    distance = 0
    for i in range(len_episodes):
        a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
        a_linear = coffset.dot(s)
        distance -= np.abs(a_policy - a_linear)
        s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))

    return distance/len_episodes

# TODO: add logger
@timeit
def evolution_policy(env, policy, n_vars, len_episodes, n_population=50, n_iterations=2500, sigma=0.1, alpha=0.05):
    mean_dist = []
    coffset = np.random.randn(n_vars)


    for iter in range(n_iterations):
        noise = np.random.randn(int(n_population/2), n_vars)
        noise = np.vstack((noise, -noise))

        distance = np.zeros(n_population)
        for p in range(n_population):
            new_coffset = coffset + sigma * noise[p]
            distance[p] = policy_distance(env, policy, new_coffset, len_episodes)

        # std_distance = (distance - np.mean(distance)) / np.std(distance)

        coffset = coffset + alpha / (n_population * sigma) * np.dot(noise.T, distance)
        mean_dist.append(np.mean(distance))

        print(coffset)
        print(np.mean(distance))
    plt.plot(mean_dist)
    plt.show()
    plt.savefig("1.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Options")
    parser.add_argument("--env", default="pendulum", type=str, help="The selected environment.")
    parser.add_argument("--do_eval", action="store_true", help="Test RL controller")
    parser.add_argument("--test_episodes", default=50, help="test_episodes", type=int)
    parser.add_argument("--do_retrain", action="store_true", help="retrain RL controller")
    args = parser.parse_args()

    env = ENV_CLASSES[args.env]()
    with open("configs.json") as f:
        configs = json.load(f)

    DDPG_args = configs[args.env]
    DDPG_args["enable_retrain"] = args.do_retrain
    DDPG_args["enable_eval"] = args.do_eval
    DDPG_args["enable_fuzzing"] = False
    DDPG_args["enable_falsification"] = False

    DDPG_args["test_episodes"] = args.test_episodes

    policy = DDPG(env, DDPG_args)
    evolution_policy(env, policy, 2, 10)
    policy.sess.close()



