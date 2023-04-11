from z3 import *
set_option(max_args=10000000, max_lines=100000000, max_depth=10000000, max_visited=100000000, precision=2)

import math
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
from DDPG import DDPG, get_params
import time
from envs import ENV_CLASSES
from scipy.stats import mannwhitneyu
from scipy.optimize import rosen, differential_evolution

logging.getLogger().setLevel(logging.INFO)


def evolution_policy_with_checker(env, policy, n_vars, len_episodes, threshold, n_population=50, n_iterations=50, sigma=0.1, alpha=0.05):

    coffset = np.random.randn(n_vars)

    s_set = []
    a_set = []

    def checker():
        all_dis = 0
        for s, a in tqdm(zip(s_set, a_set)):
            all_dis += np.abs(coffset[:n_vars-1].dot(s)+ coffset[n_vars-1] - a)
        return all_dis/len(s_set)

    dis = 100
    while True:
        for iter in range(n_iterations):
            s = env.reset()
            for i in range(len_episodes):
                s_set.append(s)
                a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
                a_set.append(a_policy)
                s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))

        for s, a in tqdm(zip(s_set, a_set)):
            noise = np.random.randn(int(n_population/2), n_vars)
            noise = np.vstack((noise, -noise))
            distance = np.zeros(n_population)

            for p in range(n_population):
                new_coffset = coffset + sigma * noise[p]
                a_linear = new_coffset[:n_vars-1].dot(s)+ new_coffset[n_vars-1]
                distance[p] = - np.abs(a - a_linear)
            std_distance = (distance - np.mean(distance)) / np.std(distance)
            coffset = coffset + alpha / (n_population * sigma) * np.dot(noise.T, std_distance)
        dis = checker()
        print(dis)

        if dis < threshold:
            break

    return coffset

def evolution_policy_with_verify(env, policy, n_vars, len_episodes, FullyConnected_W_params, FullyConnected_b_params, BatchNormalization_beta_params, BatchNormalization_gamma_params, n_population=50, n_iterations=50, sigma=0.1, alpha=0.05):

    def Relu(x):
        return np.vectorize(lambda y: If(y >= 0 , y, RealVal(0)))(x)
        # return np.maximum(0, x)

    if len(FullyConnected_W_params) == 4:
        w1, w2, w3, w4 = FullyConnected_W_params
        b1, b2, b3, b4 = FullyConnected_b_params
        # exit()
        beta1, beta2, beta3 = BatchNormalization_beta_params
        gamma1, gamma2, gamma3 = BatchNormalization_gamma_params
        u_max = env.u_max[0][0]
        e = math.e
        yyy = np.zeros(200)

        def net(x):
            # x1 = w1.T @ x + b1.reshape(-1, 1)
            # scale = gamma1 / np.sqrt(1 + 1e-5)
            # x1 = x1 * scale.reshape(-1, 1) + (beta1.reshape(-1, 1))
            # y1 = Relu(x1)

            # x2 = w2.T @ y1 + b2.reshape(-1, 1)
            # scale = gamma2 / np.sqrt(1 + 1e-5)
            # x2 = x2 * scale.reshape(-1, 1) + (beta2.reshape(-1, 1))
            # y2 = Relu(x2)

            # x3 = w3.T @ y2 + b3.reshape(-1, 1)
            # scale = gamma3 / np.sqrt(1 + 1e-5)
            # x3 = x3 * scale.reshape(-1, 1) + (beta3.reshape(-1, 1))
            # y3 = Relu(x3)

            x4 = w4 @ x + b4.reshape(-1, 1)
            x4 = Relu(x4)
            # we omit tanh function here but the results should be close enough
            x4 = x4 * u_max
            return x4[0][0]
    else:
        w1, w2, w3 = FullyConnected_W_params
        w1 = w1.T
        w2 = w2.T
        w3 = w3.T
        b1, b2, b3 = FullyConnected_b_params
        b1 = b1.reshape(-1, 1)
        b2 = b2.reshape(-1, 1)
        b3 = b3.reshape(-1, 1)
        # exit()
        beta1, beta2 = BatchNormalization_beta_params
        beta1 = beta1.reshape(-1, 1)
        beta2 = beta2.reshape(-1, 1)
        gamma1, gamma2 = BatchNormalization_gamma_params
        gamma1 = gamma1.reshape(-1, 1)
        gamma2 = gamma2.reshape(-1, 1)
        u_max = env.u_max[0][0]
        e = math.e

        def net(x):
            x1 = w1 @ x + b1
            scale = gamma1 / np.sqrt(1 + 1e-5)
            x1 = x1 * scale + beta1
            y1 = Relu(x1)

            x2 = w2 @ y1 + b2
            scale = gamma2 / np.sqrt(1 + 1e-5)
            x2 = x2 * scale + beta2
            y2 = Relu(x2)

            # # x3 = w3.T @ y2 + b3.reshape(-1, 1)
            # # scale = gamma3 / np.sqrt(1 + 1e-5)
            # # x3 = x3 * scale.reshape(-1, 1) + (beta3.reshape(-1, 1))
            # # y3 = Relu(x3)

            x3 = w3 @ y2 + b3
            # x4 = (e ** x4 - e ** (-x4)) / (e ** x4 + e ** (-x4))
            # we omit tanh function here but the results should be close enough
            x3 = x3 * u_max
            return x3[0][0]

    # s = env.reset()
    # print(net(np.array(s)))
    # a_linear = actor.predict(np.reshape(np.array(s), (1, actor.s_dim)))
    # print(a_linear)
    # exit()
    def Abs(x):
        return If(x <= 0, -x, x)

    if n_vars - 1 == 2:
        x = np.array([[Real('x')], [Real('y')]])
    elif n_vars - 1 == 4:
        # x = np.array([[Real('x')], [Real('y')], [Real('z')], [Real('q')]])
        x = np.array([[Real('x')]])

    coffset = np.random.randn(n_vars)

    s_set = []
    a_set = []

    def checker():
        all_dis = 0
        for s, a in tqdm(zip(s_set, a_set)):
            all_dis += np.abs(coffset[:n_vars-1].dot(s)+ coffset[n_vars-1] - a)
        return all_dis/len(s_set)

    # dis = 100
    # while not dis < threshold:

    for iter in range(n_iterations):
        s = env.reset()
        for i in range(len_episodes):
            s_set.append(s)
            a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
            a_set.append(a_policy)
            s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))


    for s, a in tqdm(zip(s_set, a_set)):
        noise = np.random.randn(int(n_population/2), n_vars)
        noise = np.vstack((noise, -noise))
        distance = np.zeros(n_population)

        for p in range(n_population):
            new_coffset = coffset + sigma * noise[p]
            a_linear = new_coffset[:n_vars-1].dot(s)+ new_coffset[n_vars-1]
            distance[p] = - np.abs(a - a_linear)
        std_distance = (distance - np.mean(distance)) / np.std(distance)
        coffset = coffset + alpha / (n_population * sigma) * np.dot(noise.T, std_distance)

    def func(x):
        ans = coffset[:n_vars-1].dot(x)+ coffset[n_vars-1]
        return ans[0]

    def verify():

        y_true = func(x)
        y_pred = net(x)
        sol = Solver()
        # sol.add(-0.05 == x[0][0] or 0.05 == x[0][0])
        # sol.add(-0.05 == x[1][0] or 0.05 == x[1][0])
        # sol.add(-0.05 == x[2][0])
        # sol.add(-0.05 == x[3][0])
        sol.add(-0.05 <= x[0][0], x[0][0] <= 0.05)
        sol.add(-0.05 <= x[1][0], x[1][0] <= 0.05)
        # sol.add(-0.05 <= x[2][0], x[2][0] <= 0.05)
        # sol.add(-0.05 <= x[3][0], x[3][0] <= 0.05)
        sol.add(Not(Abs(y_pred) <= 0.01))

        res = sol.check()

        print(res)
        if res == unsat:
            print("proved")
        else:
            print("counterexample")
            print(sol.model())
    time_v = time.time()
    verify()
    print(time.time() - time_v)

    return coffset

def evolution_policy(env, policy, n_vars, len_episodes, n_population=50, n_iterations=50, sigma=0.1, alpha=0.05):

    coffset = np.random.randn(n_vars)

    for iter in tqdm(range(n_iterations)):
        # print(policy_distance(env, policy, n_vars, coffset, len_episodes))
        if policy_distance(env, policy, n_vars, coffset, len_episodes) < 0.1:
            break
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
    # print(policy_distance(env, policy, n_vars, coffset, len_episodes))
    return coffset

def policy_distance(env, policy, n_vars, coffset, len_episodes):
     s = env.reset()
     distance = 0
     for i in range(len_episodes):
         a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
         a_linear = coffset[:n_vars-1].dot(s)+ coffset[n_vars-1]
         distance -= np.abs(a_policy - a_linear)
         s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))

     return abs(distance/len_episodes)

def policy_distance_stat(env, policy, n_vars, coffset, len_episodes):
     s = env.reset()
     distance = 0
     a_set_1 = []
     a_set_2 = []
     for i in range(len_episodes):
         a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
         a_linear = coffset[:n_vars-1].dot(s)+ coffset[n_vars-1]
         distance -= np.abs(a_policy - a_linear)
         s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
         a_set_1.append(a_policy)
         a_set_2.append(a_linear)

     return mannwhitneyu(a_set_1, a_set_2, method="asymptotic")

def refine(env, policy, coffset, n_vars, state, len_episodes, n_population=50, n_iterations=1, sigma=0.1, alpha=0.05):

    s_set = []
    a_set = []

    s = env.reset(np.reshape(np.array(state), (-1, 1)))
    for i in range(len_episodes):
        s_set.append(s)
        a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
        a_set.append(a_policy)
        s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))

    for s, a in tqdm(zip(s_set, a_set)):
        noise = np.random.randn(int(n_population/2), n_vars)
        noise = np.vstack((noise, -noise))
        distance = np.zeros(n_population)

        for p in range(n_population):
            new_coffset = coffset + sigma * noise[p]
            a_linear = new_coffset[:n_vars-1].dot(s)+ new_coffset[n_vars-1]
            distance[p] = - np.abs(a - a_linear)
        std_distance = (distance - np.mean(distance)) / np.std(distance)
        coffset = coffset + alpha / (n_population * sigma) * np.dot(noise.T, std_distance)

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
    FullyConnected_W_params, FullyConnected_b_params, BatchNormalization_beta_params, BatchNormalization_gamma_params, actor = get_params(env, DDPG_args)
    assert len(FullyConnected_W_params) == len(FullyConnected_b_params)
    assert len(BatchNormalization_beta_params) == len(BatchNormalization_gamma_params)
    assert len(FullyConnected_W_params) + len(FullyConnected_b_params) + len(BatchNormalization_beta_params) + len(BatchNormalization_gamma_params) == len(actor.network_params)

    # actor = DDPG(env, DDPG_args)
    coffset = evolution_policy_with_verify(env, actor, 3, 250, FullyConnected_W_params, FullyConnected_b_params, BatchNormalization_beta_params, BatchNormalization_gamma_params)
    print(policy_distance_stat(env, actor, 3, coffset, 250))
    # s = env.reset()
    # for i in tqdm(range(250)):
    #     # a_linear = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
    #     a_linear = coffset[:5-1].dot(s**2)+ coffset[5-1]
    #     s, r, terminal = env.step(a_linear.reshape(policy.a_dim, 1))
    #     if terminal:
    #         break
    # # print(evolution_dynamics(env, 0, policy, 3, 250))
    # # print(evolution_dynamic(env, 0, policy, 4, 250))
    # policy.sess.close()





