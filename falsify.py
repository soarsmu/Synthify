import json
import time
import rtamt
import logging
import argparse

import numpy as np
from numpy.random import default_rng

from tqdm import tqdm
from DDPG import DDPG
from scipy.optimize import minimize
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
    start = time.time()
    s = env.reset(np.reshape(np.array(state), (-1, 1)))
    for i in range(250):
        a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
        s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
        if terminal and i < 250:
            return True, time.time() - start
    return False, time.time() - start

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

    syn_policy = evolution_policy(env, policy, n_vars+1, 250)

    syn_dynamics = []
    for i in range(n_vars):
        syn_dynamics.append(evolution_dynamics(env, 0, policy, 3, 250))

    min_robs = [0] * policy_args["spec_lens"]
    prob = [0] * policy_args["spec_lens"]
    times = [0] * policy_args["spec_lens"]
    specifications = policy_args["slice_spec"]

    def sample_spec(specifications, prob, eps=0.6):
        p = np.random.uniform(0, 1)
        if(p > eps):
            arm_to_pull = np.argmax(prob)
        else:
            arm_to_pull = np.random.randint(0, policy_args["spec_lens"], 1)[0]
        return arm_to_pull

    failures = []
    sim_time = 0
    falsification_time = 0

    itertimes = []
    for budget in tqdm(range(50), desc="Falsification of %s" % args.env):

        spec_index = sample_spec(specifications, prob)
        times[spec_index] += 1
        prob[spec_index] += 1

        initial_conditions = policy_args["initial_conditions"]
        phi = specifications[spec_index]
        RTAMT_offline = RTAMTDense(phi, policy_args["var_map"])

        def cost(s):
            states = []
            s = np.reshape(np.array(s), (-1, 1))
            for i in range(250):
                states.append(np.array(s))
                a_linear = syn_policy[:n_vars].dot(s) + syn_policy[n_vars]
                s = np.vstack([syn_dynamic[:2].dot(np.vstack((s[id], a_linear))) + syn_dynamic[2] for id, syn_dynamic in enumerate(syn_dynamics)])
            states = np.hstack(states)
            return RTAMT_offline.evaluate(states, np.arange(0, 250, 1))

        rng = default_rng()
        input_costs = []
        simulations = 0
        start = time.time()
        success = False
        for iter in range(300):
            inputs = [rng.uniform(bound[0], bound[1]) for bound in initial_conditions]
            rob = cost(inputs)
            input_costs.append((inputs, rob))
            if rob < 0:
                real, time_cost = false_checker(policy, env, inputs)
                simulations += 1
                sim_time += time_cost
                if real:
                    failures.append(inputs)
                    success = True
                    min_robs[spec_index] = -min(min_robs[spec_index], max(rob, -1))
                    break

        if success:
            falsification_time += time.time() - start
            itertimes.append(simulations)
            prob[spec_index] = prob[spec_index] + (1 / times[spec_index]) * (min_robs[spec_index] - prob[spec_index])
            continue

        falsification_time += time.time() - start
        best_sample = min(input_costs, key=lambda x: x[1])[0]

        bounds = []
        for index, initial_var in enumerate(best_sample):
            bounds.append([np.clip(initial_var-0.01, initial_conditions[i][0], initial_conditions[i][1]), np.clip(initial_var+0.01, initial_conditions[i][0], initial_conditions[i][1])])
            np.clip(initial_var, initial_conditions[i][0], initial_conditions[i][1])
        start = time.time()
        res = minimize(cost, np.array(best_sample), bounds=bounds, method='Nelder-Mead')

        if res.fun < 0:
            real, time_cost = false_checker(policy, env, res.x)
            simulations += 1
            sim_time += time_cost
            if real:
                failures.append(res.x)
                min_robs[spec_index] = -min(min_robs[spec_index], max(res.fun, -1))
                itertimes.append(simulations)

        falsification_time += time.time() - start
        prob[spec_index] = prob[spec_index] + (1 / times[spec_index]) * (min_robs[spec_index] - prob[spec_index])


    logging.info("%d successful trials over 50 trials", len(failures))
    logging.info("falsification rate wrt. 50 trials is %f", len(failures)/50)
    logging.info("mean number of simulations over successful trials is %f", np.mean(itertimes))
    logging.info("median number of simulations over successful trials is %f", np.median(itertimes))
    logging.info("simulation time is %f", sim_time)
    logging.info("falsification time is %f", falsification_time)
    logging.info("non-simulation time ratio %f", (falsification_time - sim_time)/falsification_time)

    coverage = [i for i in min_robs if i > 0]
    logging.info("coverage of slice specifications is %s", len(coverage)/len(min_robs))

