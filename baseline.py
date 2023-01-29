import time
import json
import logging
import argparse

import numpy as np

from DDPG import DDPG
from envs import ENV_CLASSES
from numpy.typing import NDArray

from staliro.models import ModelData, SignalTimes, SignalValues, StaticInput, blackbox
from staliro.optimizers import DualAnnealing, UniformRandom
from staliro.options import Options
from staliro.specifications import RTAMTDense
from staliro.staliro import staliro

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
        options = Options(runs=1, iterations=100, interval=(0, 1000), static_parameters=initial_conditions)
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