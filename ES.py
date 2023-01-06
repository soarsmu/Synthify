import json
import argparse
import numpy as np

from tqdm import tqdm
from DDPG import DDPG
from metrics import timeit
from envs import ENV_CLASSES

# TODO: add logger
@timeit
def evolution_policy(env, policy, n_vars, len_episodes, n_population=100, n_iterations=200, sigma=0.1, alpha=0.05):

    coffset = np.random.randn(n_vars)

    for iter in tqdm(range(n_iterations)):
        noise = np.random.randn(int(n_population/2), n_vars)
        noise = np.vstack((noise, -noise))
        distance = np.zeros(n_population)

        s = env.reset()
        for i in range(len_episodes):
            a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
            for p in range(n_population):
                new_coffset = coffset + sigma * noise[p]
                a_linear = new_coffset[:n_vars-1].dot(s)+ new_coffset[n_vars-1]
                distance[p] = - np.abs(a_policy - a_linear)
            std_distance = (distance - np.mean(distance)) / np.std(distance)
            coffset = coffset + alpha / (n_population * sigma) * np.dot(noise.T, std_distance)
            s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))


# TODO: add logger
@timeit
def evolution_dynamics(env, para, policy, n_vars, len_episodes, n_population=100, n_iterations=200, sigma=0.1, alpha=0.05):

    coffset = np.random.randn(n_vars)

    for iter in tqdm(range(n_iterations)):
        noise = np.random.randn(int(n_population/2), n_vars)
        noise = np.vstack((noise, -noise))
        distance = np.zeros(n_population)

        s = env.reset()
        for i in range(len_episodes):
            s_old = s
            a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
            s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
            for p in range(n_population):
                new_coffset = coffset + sigma * noise[p]
                d_linear = new_coffset[:n_vars-1].dot(s_old)+ new_coffset[n_vars-1]
                dynamic = s[para]
                distance[p] = - np.abs(dynamic - d_linear)
            std_distance = (distance - np.mean(distance)) / np.std(distance)

            coffset = coffset + alpha / (n_population * sigma) * np.dot(noise.T, std_distance)
            print(coffset)
            print(np.mean(distance))
            s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))

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
    # evolution_policy(env, policy, 3, 1000)
    evolution_dynamics(env, 0, policy, 3, 1000)
    policy.sess.close()





