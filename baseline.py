import sys
import json
import rtamt
import argparse

import numpy as np

from tqdm import tqdm
from DDPG import DDPG
from envs import ENV_CLASSES

# from staliro.core import best_eval, best_run
from staliro.optimizers import UniformRandom
from staliro.options import Options
from staliro.models import State, ode
from staliro.specifications import RTAMTDense
from staliro.staliro import simulate_model, staliro

env = ENV_CLASSES["cartpole"]()
with open("configs.json") as f:
    configs = json.load(f)

DDPG_args = configs["cartpole"]
DDPG_args["enable_retrain"] = False
DDPG_args["enable_eval"] = False
DDPG_args["enable_fuzzing"] = False
DDPG_args["enable_falsification"] = False

DDPG_args["test_episodes"] = 5000

policy = DDPG(env, DDPG_args)

@ode()
def model(time: float, state: State, _) -> State:
    a_policy = policy.predict(np.reshape(np.array(state), (1, policy.s_dim)))
    s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))

    return np.ndarray(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Options")
    parser.add_argument("--env", default="pendulum", type=str, help="The selected environment.")
    parser.add_argument("--do_eval", action="store_true", help="Test RL controller")
    parser.add_argument("--test_episodes", default=50, help="test_episodes", type=int)
    parser.add_argument("--do_retrain", action="store_true", help="retrain RL controller")
    args = parser.parse_args()

    # env = ENV_CLASSES[args.env]()
    with open("configs.json") as f:
        configs = json.load(f)

    DDPG_args = configs[args.env]
    DDPG_args["enable_retrain"] = args.do_retrain
    DDPG_args["enable_eval"] = args.do_eval
    DDPG_args["enable_fuzzing"] = False
    DDPG_args["enable_falsification"] = False

    DDPG_args["test_episodes"] = args.test_episodes

    # policy = DDPG(env, DDPG_args)

    s = env.reset()
    initial_conditions = s.tolist()

    print(initial_conditions)
    phi = r"always[0:50] (a >= -0.05 and a <= 0.05 and b >= -0.05 and b <= 0.05 and c >= -0.05 and c <= 0.05 and d >= -0.05 and d <= 0.05)"
    specification = RTAMTDense(phi, {"a": 0, "b": 0, "c": 0, "d": 0})
    options = Options(runs=1, iterations=100, interval=(0, 2), static_parameters=initial_conditions)
    optimizer = UniformRandom()

    # for iter in tqdm(range(20)):
    #     noise = np.array([0.01, 0.01, 0.01, 0.01])
    #     # print(s)
    #     for sign in [-1, 1]:
    #         s += sign * noise.reshape(4, 1)
    #         s_inital = s
    #         robs = []
    #         for i in range(100):
    #             a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
    #             s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))

    result = staliro(model, specification, optimizer, options)

    best_run = result.best_run
    print(best_run)
    best_sample = best_run.best_eval.sample
    best_result = simulate_model(model, options, best_sample)
    # sample_xs = [evaluation.sample[0] for evaluation in best_run_.history]
    # sample_ys = [evaluation.sample[1] for evaluation in best_run_.history]