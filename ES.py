import json
import logging
import argparse
import numpy as np

from tqdm import tqdm
from DDPG import DDPG

from envs import ENV_CLASSES

logging.getLogger().setLevel(logging.INFO)

def evolution_policy(env, policy, n_vars, len_episodes, n_population=50, n_iterations=50, sigma=0.1, alpha=0.05):

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
                a_linear = new_coffset[:n_vars-1].dot(s**2)+ new_coffset[n_vars-1]
                distance[p] = - np.abs(a_policy - a_linear)
            std_distance = (distance - np.mean(distance)) / np.std(distance)
            coffset = coffset + alpha / (n_population * sigma) * np.dot(noise.T, std_distance)
            s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
    return coffset


def evolution_dynamics(env, para, policy, n_vars, len_episodes, n_population=50, n_iterations=20, sigma=0.1, alpha=0.05):

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
                d_linear = new_coffset[:n_vars-1].dot(np.vstack((s_old[para], a_policy)))+ new_coffset[n_vars-1]
                dynamic = s[para]
                distance[p] = - np.abs(dynamic - d_linear)
            std_distance = (distance - np.mean(distance)) / np.std(distance)

            coffset = coffset + alpha / (n_population * sigma) * np.dot(noise.T, std_distance)

    return coffset

def evolution_dynamic(env, para, policy, n_vars, len_episodes, n_population=50, n_iterations=20, sigma=0.1, alpha=0.05):


    coffset = np.random.randn(n_vars+1)

    for iter in tqdm(range(n_iterations)):
        noise = np.random.randn(int(n_population/2), n_vars+1)
        noise = np.vstack((noise, -noise))
        distance = np.zeros(n_population)

        s = env.reset()
        for i in range(len_episodes):
            s_old = s
            a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
            s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
            for p in range(n_population):
                new_coffset = coffset + sigma * noise[p]
                d_linear = new_coffset.dot(np.vstack((s_old, a_policy)))
                dynamic = s[para]
                distance[p] = - np.abs(dynamic - d_linear)
            std_distance = (distance - np.mean(distance)) / np.std(distance)
            print(dynamic - d_linear)
            coffset = coffset + alpha / (n_population * sigma) * np.dot(noise.T, std_distance)
    print(np.mean(distance))
    return coffset

def evolution(env, para, policy, n_vars, len_episodes, n_population=50, n_iterations=100, sigma=1, alpha=0.05):

    coffset = np.random.randn(n_vars+1, n_vars+1)

    for iter in tqdm(range(n_iterations)):
        noise = np.random.randn(int(n_population/2), n_vars+1, n_vars+1)
        noise = np.vstack((noise, -noise))
        distance = np.zeros(n_population)

        s = env.reset()
        a_old = 0
        for i in range(len_episodes):
            s_old = s
            a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
            s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
            for p in range(n_population):
                new_coffset = coffset + sigma * noise[p]
                a_linear = new_coffset[0].dot(np.vstack((s_old, a_old)))
                distance[p] = - np.abs(a_policy - a_linear)
                d_linear = new_coffset[1:].dot(np.vstack((s_old, a_policy)))
                distance[p] += - np.sum(np.abs(s - d_linear))
                print(distance[p])
            std_distance = (distance - np.mean(distance)) / np.std(distance)

            coffset = coffset + alpha / (n_population * sigma) * np.dot(noise.T, std_distance)
            a_old = a_policy
    print(np.mean(distance))
    return coffset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Options")
    parser.add_argument("--env", default="cartpole", type=str, help="The selected environment.")
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
    coffset = evolution_policy(env, policy, 5, 250)

    s = env.reset()
    for i in tqdm(range(250)):
        # a_linear = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
        a_linear = coffset[:5-1].dot(s**2)+ coffset[5-1]
        s, r, terminal = env.step(a_linear.reshape(policy.a_dim, 1))
        if terminal:
            break
    # print(evolution_dynamics(env, 0, policy, 3, 250))
    # print(evolution_dynamic(env, 0, policy, 4, 250))
    policy.sess.close()





