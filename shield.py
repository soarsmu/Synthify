import json
import time
import rtamt
import logging
import argparse

import numpy as np
from numpy.random import default_rng

from tqdm import tqdm
from DDPG import DDPG
from envs import ENV_CLASSES
from ES import evolution_policy, refine, evolution_policy_with_checker

from numpy.typing import NDArray
from staliro.models import ModelData, SignalTimes, SignalValues, StaticInput, blackbox
# from optimizer import UniformRandom, DualAnnealing
from optimizer import DualAnnealing
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

# def false_checker(policy, env, state):
#     start = time.time()
#     s = env.reset(np.reshape(np.array(state), (-1, 1)))
#     for i in range(5000):
#         a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
#         s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
#         if terminal and i < 5000:
#             return True, time.time() - start
#     return False, time.time() - start

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Options")
    parser.add_argument("--env", default="pendulum", type=str, help="The selected environment.")
    args = parser.parse_args()

    env = ENV_CLASSES[args.env]()
    with open("configs.json") as f:
        configs = json.load(f)

    policy_args = configs[args.env]
    policy = DDPG(env, policy_args)

    DataT = ModelData[NDArray[np.float_], None]

    sim_time = 0
    vars = list(policy_args["var_map"].keys())
    n_vars = len(vars)
    syn_policy = evolution_policy(env, policy, n_vars+1, 100)

    @blackbox(sampling_interval=1.0)
    def model(static: StaticInput, times: SignalTimes, signals: SignalValues) -> DataT:
        states = []
        global sim_time
        start_time = time.time()
        if args.env == "biology":
            static = (static[0], 0.0, static[1])
        elif args.env == "oscillator":
            static = (static[0], static[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0 ,0.0 ,0.0 ,0.0)
        s = env.reset(np.reshape(np.array(static), (-1, 1)))
        for i in range(len(times)):
            # a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
            a_linear = syn_policy[:n_vars].dot(s) + syn_policy[n_vars]
            s, r, terminal = env.step(a_linear.reshape(policy.a_dim, 1))
            states.append(np.array(s))
        sim_time += time.time() - start_time
        states = np.hstack(states)
        return ModelData(states, np.asarray(times))
    
    # syn_dynamics = []
    # for i in range(n_vars):
    #     syn_dynamics.append(evolution_dynamics(env, 0, policy, 3, 250))

    min_robs = [0] * policy_args["spec_lens"]
    prob = [0] * policy_args["spec_lens"]
    times = [0] * policy_args["spec_lens"]
    specifications = policy_args["slice_spec"]

    def sample_spec(specifications, prob, eps=0.9):
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
    linear_itertimes = []
    count = 0
    for budget in tqdm(range(50), desc="Falsification of %s" % args.env):

        spec_index = sample_spec(specifications, prob)
        times[spec_index] += 1
        prob[spec_index] += 1

        initial_conditions = policy_args["initial_conditions"]
        # phi = specifications[spec_index]
        phi = policy_args["specification"]
        RTAMT_offline = RTAMTDense(phi, policy_args["var_map"])

        def cost(s):
            states = []
            s = np.reshape(np.array(s), (-1, 1))
            for i in range(100):
                # a_linear = syn_policy[:n_vars].dot(s) + syn_policy[n_vars]
                a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
                # s = np.vstack([syn_dynamic[:2].dot(np.vstack((s[id], a_linear))) + syn_dynamic[2] for id, syn_dynamic in enumerate(syn_dynamics)])
                s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
                states.append(np.array(s))
            states = np.hstack(states)
            return RTAMT_offline.evaluate(states, np.arange(0, 100, 1))
        
        def false_checker(state):
            start = time.time()
            real_cost = cost(state)    
            if real_cost < 0 or np.isnan(real_cost) or np.isinf(real_cost):
                    return True, time.time() - start
            return False, time.time() - start

        real_simulations = 0
        linear_simulations = 0
        start = time.time()
        while real_simulations < 300:
            spec_index = sample_spec(specifications, prob)
            times[spec_index] += 1
            prob[spec_index] += 1

            initial_conditions = policy_args["initial_conditions"]
            phi = specifications[spec_index]
            # phi = policy_args["specification"]
            RTAMT_offline = RTAMTDense(phi, policy_args["var_map"])
            success = False
            options = Options(runs=1, iterations=300, interval=(0, 100), static_parameters=initial_conditions)
            optimizer = DualAnnealing()
            result = staliro(model, RTAMT_offline, optimizer, options)
            for run in result.runs:
                for id, evaluation in enumerate(run.history):
                    # print(evaluation.cost)
                    min_robs[spec_index] = -min(min_robs[spec_index], max(evaluation.cost, -1))
                    prob[spec_index] = prob[spec_index] + (1 / times[spec_index]) * (min_robs[spec_index] - prob[spec_index])
            
            evaluation = result.runs[0].history[-1]
            if evaluation.cost < 0 or np.isnan(evaluation.cost) or np.isinf(evaluation.cost):
                real, time_cost = false_checker(evaluation.sample)
                real_simulations += 1
                linear_simulations += len(result.runs[0].history)
                sim_time += time_cost
                if real:
                    failures.append(evaluation.sample)
                    success = True
                    itertimes.append(real_simulations)
                    linear_itertimes.append(linear_simulations)
                    # break
                        # else:
                        #     syn_policy = refine(env, policy, syn_policy, n_vars+1, evaluation.sample, 500)
                        # failures.append(evaluation.sample)
                        # itertimes.append(id+1)
            if success:
                logging.info("%d successful trials over 50 trials", len(failures))
                logging.info("mean number of simulations over successful trials is %f", np.mean(itertimes))
                logging.info("median number of simulations over successful trials is %f", np.median(itertimes))
                logging.info("mean number of linear simulations over successful trials is %f", np.mean(linear_itertimes))
                logging.info("median number of linear simulations over successful trials is %f", np.median(linear_itertimes))
                falsification_time += time.time() - start
                break
    print(prob)

    logging.info("%d successful trials over 50 trials", len(failures))
    logging.info("falsification rate wrt. 50 trials is %f", len(failures)/50)
    logging.info("mean number of simulations over successful trials is %f", np.mean(itertimes))
    logging.info("median number of simulations over successful trials is %f", np.median(itertimes))
    logging.info("simulation time is %f", sim_time)
    logging.info("falsification time is %f", falsification_time)
    logging.info("non-simulation time ratio %f", (falsification_time - sim_time)/falsification_time)
    coverage = [i for i in min_robs if i > 0]
    logging.info("coverage of slice specifications is %s", len(coverage)/len(min_robs))


    dp = []
    for f in failures:
        dp.append((list(f.values), 0))

    while len(dp) < 100:
        s_init = env.reset()
        s = s_init
        flag = 1
        for i in range(100):
            a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
            # a_linear = syn_policy[:n_vars].dot(s) + syn_policy[n_vars]
            s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
            if terminal and i < 100:
                flag = 0
                print(flag, i)
                break
        if flag == 1:
            dp.append((np.reshape(np.array(s_init), (policy.s_dim, 1)).squeeze().tolist(), 1))

    import random
    random.shuffle(dp)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for d in dp[:int(len(dp)*0.8)]:
        train_x.append(d[0])
        train_y.append(d[1])
    
    print(train_x, train_y)
    for d in dp[int(len(dp)*0.8):]:
        test_x.append(d[0])
        test_y.append(d[1])

    def svm_classifer(train_x, train_y):
        from sklearn.svm import SVC
        model = SVC()
        model.fit(train_x, train_y)
        return model

    classifer = svm_classifer(train_x, train_y)

    res = []
    for d in test_x:
        res.append(classifer.predict(d))
    from sklearn.metrics import accuracy_score
    print(accuracy_score(res, d))


