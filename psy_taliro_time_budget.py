import time
import json
import logging
import argparse

import numpy as np

from tqdm import tqdm
from DDPG import DDPG
from envs import ENV_CLASSES
from numpy.typing import NDArray

from staliro.models import ModelData, SignalTimes, SignalValues, StaticInput, blackbox
# from staliro.optimizers import DualAnnealing
from optimizer import DualAnnealing
from staliro.options import Options
from staliro.specifications import RTAMTDense
from staliro.staliro import staliro, simulate_model

logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Running Options")
    parser.add_argument("--env", default="pendulum", type=str, help="The selected environment.")
    parser.add_argument("--algo", default="SA", type=str, help="The selected algorithm.")
    args = parser.parse_args()

    env = ENV_CLASSES[args.env]()
    with open("configs.json") as f:
        configs = json.load(f)

    policy_args = configs[args.env]
    policy = DDPG(env, policy_args)

    DataT = ModelData[NDArray[np.float_], None]

    sim_time = 0

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
            a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
            s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
            states.append(np.array(s))
        sim_time += time.time() - start_time
        states = np.hstack(states)
        return ModelData(states, np.asarray(times))

    initial_conditions = policy_args["initial_conditions"]
    phi = policy_args["specification"]
    specification = RTAMTDense(phi, policy_args["var_map"])

    failures = []

    logging.info("Falsification of %s", args.env)
    itertimes = []
    falsification_time = 0
    # start = time.time()

    while (falsification_time < 600):
    # for budget in tqdm(range(50), desc="Falsification of %s" % args.env):
        start = time.time()
        options = Options(runs=1, iterations=100, interval=(0, 200), static_parameters=initial_conditions)
        optimizer = DualAnnealing()
        # optimizer = UniformRandom()

        result = staliro(model, specification, optimizer, options)
        
        evaluation = result.runs[0].history[-1]
        if evaluation.cost < 0 or np.isnan(evaluation.cost) or np.isinf(evaluation.cost):
            failures.append(evaluation.sample)
            itertimes.append(len(result.runs[0].history))
            logging.info("%d successful trials over 50 trials", len(failures))
            logging.info("mean number of simulations over successful trials is %f", np.mean(itertimes))
            logging.info("median number of simulations over successful trials is %f", np.median(itertimes))

        falsification_time += time.time() - start

    logging.info("%d successful trials over 50 trials", len(failures))
    logging.info("falsification rate wrt. 50 trials is %f", len(failures)/50)
    logging.info("mean number of simulations over successful trials is %f", np.mean(itertimes))
    logging.info("median number of simulations over successful trials is %f", np.median(itertimes))
    logging.info("simulation time is %f", sim_time)
    logging.info("falsification time is %f", falsification_time)
    logging.info("non-simulation time ratio %f", (falsification_time - sim_time)/falsification_time)

    coverage = [0] * len(policy_args["slice_spec"])
    for failure in tqdm(failures, desc="Coverage of %s" % args.env):
        sample = simulate_model(model, options, failure)
        for id, spec in enumerate(policy_args["slice_spec"]):
            specification = RTAMTDense(spec, policy_args["var_map"])
            if specification.evaluate(sample.states, sample.times) < 0:
                coverage[id] += 1

    logging.info("coverage of slice specifications is %s", np.count_nonzero(coverage)/len(coverage))
