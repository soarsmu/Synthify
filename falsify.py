import json
import time
import rtamt
import logging
import argparse

import numpy as np

from tqdm import tqdm
from DDPG import DDPG
from envs import ENV_CLASSES
from ES import evolution_policy, evolution_dynamics

from numpy.typing import NDArray

from staliro.models import ModelData, SignalTimes, SignalValues, StaticInput, blackbox
from staliro.optimizers import DualAnnealing, UniformRandom
from staliro.options import Options
from staliro.specifications import RTAMTDense
from staliro.staliro import staliro

def init_monitor(vars, specification):
    monitor = rtamt.STLDiscreteTimeSpecification()
    for var in vars:
        monitor.declare_var(var, 'float')
    monitor.spec = specification

    monitor.parse()
    monitor.pastify()

    return monitor

def false_checker(policy, env, state):
    s = env.reset(state)
    for i in range(1000):
        a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
        s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
        if terminal and i < 1000:
            return True
    return False

@blackbox(sampling_interval=1.0)
def model(static: StaticInput, times: SignalTimes, signals: SignalValues):
    states = []
    s = env.reset(np.reshape(np.array(static), (-1, 1)))
    for i in range(len(times)):
        states.append(np.array(s))
        a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
        s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))

    states = np.hstack(states)
    return ModelData(states, np.asarray(times))

def global_search():

    DataT = ModelData[NDArray[np.float_], None]
    @blackbox(sampling_interval=1.0)
    def model(static: StaticInput, times: SignalTimes, signals: SignalValues) -> DataT:
        states = []
        s = env.reset(np.reshape(np.array(static), (-1, 1)))
        for i in range(len(times)):
            states.append(np.array(s))
            a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
            s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))

        states = np.hstack(states)
        return ModelData(states, np.asarray(times))

    initial_conditions = policy_args["initial_conditions"]
    phi = policy_args["specification"]
    specification = RTAMTDense(phi, policy_args["var_map"])

    failures = []

    start = time.time()
    itertimes = 0
    while True:
        options = Options(runs=1, iterations=50, interval=(0, 5000), static_parameters=initial_conditions)
        if args.algo == "SA":
            optimizer = DualAnnealing()
        elif args.algo == "UR":
            optimizer = UniformRandom()

        result = staliro(model, specification, optimizer, options)
        for run in result.runs:
            itertimes += len(run.history)
            for evaluation in run.history:
                if evaluation.cost < 0:
                    failures.append(evaluation.sample)

        logging.info("find %d failures by %d iterations", len(failures), itertimes)

        if time.time() - start > 3600:
            break

    logging.info("\n find %d failures by %d iterations in total", len(failures), itertimes)


def local_search():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Options")
    parser.add_argument("--env", default="pendulum", type=str, help="The selected environment.")
    args = parser.parse_args()

    env = ENV_CLASSES[args.env]()
    with open("configs.json") as f:
        configs = json.load(f)

    policy_args = configs[args.env]

    policy = DDPG(env, policy_args)

    vars = list(policy_args["var_map"].keys())
    n_vars = len(vars)

    syn_policy = evolution_policy(env, policy, n_vars+1, 1000)

    syn_dynamics = []
    for i in range(n_vars):
        syn_dynamics.append(evolution_dynamics(env, 0, policy, n_vars+1, 1000))

    min_robs = [100] * policy_args["spec_lens"]
    prob = [0] * policy_args["spec_lens"]
    times = [0] * policy_args["spec_lens"]

    specifications = policy_args["slice_spec"]

    def sample_spec(specifications, prob, eps=0.5):
        p = np.random.uniform(0, 1)
        if(p > eps):
            arm_to_pull = np.argmax(prob)
        else:
            arm_to_pull = np.random.randint(0, policy_args["spec_lens"], 1)[0]

        return arm_to_pull

    # for p in range(len(specifications)):
    #     times[p] += 1
    #     prob[p] += 1
    #     monitor = init_monitor(vars, specifications[p])

    #     falied_tests = []

    #     s = env.reset()
    #     for iter in tqdm(range(20)):
    #         noise = np.array([0.01, 0.01, 0.01, 0.01])
    #         for sign in [-1, 1]:
    #             s += sign * noise.reshape(4, 1)
    #             s_inital = s
    #             robs = []
    #             for i in range(1000):
    #                 a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
    #                 s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
    #                 rob = monitor.update(i, [('a', s[0]), ('b', s[1]), ('c', s[2]), ('d', s[3])])
    #                 robs.append(rob)
    #                 if rob < 0:
    #                     falied_tests.append(s)
    #                     break

    #             min_robs[p] = -min(min_robs[p], min(robs))
    #     prob[p] = prob[p] / (1 / times[p]) * (min_robs[p] - prob[p])

    failures = []

    for i in range(10):
        spec_index = sample_spec(specifications, prob)
        times[spec_index] += 1
        prob[spec_index] += 1
        print(specifications[spec_index])
        monitor = init_monitor(vars, specifications[spec_index])

        s = env.reset()
        for iter in tqdm(range(20)):
            flag = False
            noise = np.array([0.01, 0.01, 0.01, 0.01])
            # print(s)
            for sign in [-1, 1]:
                s = env.reset()
                s += sign * noise.reshape(4, 1)
                s_inital = s
                robs = []
                for i in range(1000):
                    a_linear = syn_policy[:n_vars].dot(s)+ syn_policy[n_vars]
                    s = np.vstack([syn_dynamic[:n_vars].dot(s)+ syn_dynamic[n_vars] for syn_dynamic in syn_dynamics])
                    rob = monitor.update(i, [(var, float(s_v)) for var, s_v in zip(vars, s)])
                    robs.append(rob)
                    if rob < 0:
                        if false_checker(policy, env, s):
                            print("Falsified at step {}".format(i))
                            failures.append(s)
                            flag = True
                            break
                if flag:
                    break
            min_robs[spec_index] = -min(min_robs[spec_index], min(robs))
            if flag:
                break

        print(prob)
        prob[spec_index] = prob[spec_index] / (1 / times[spec_index]) * (min_robs[spec_index] - prob[spec_index])
        print(prob)
        print(min_robs)
    print(len(failures))
    policy.sess.close()
