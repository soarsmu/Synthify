import sys
import json
import rtamt
import argparse

import numpy as np

from DDPG import DDPG
from envs import ENV_CLASSES

def monitor(vars, specification):
    # # stl
    spec = rtamt.STLDiscreteTimeSpecification()
    for var in vars:
        spec.declare_var(var, 'float')
    spec.spec = specification

    try:
        spec.parse()
        spec.pastify()
    except rtamt.RTAMTException as err:
        print('RTAMT Exception: {}'.format(err))
        sys.exit()

    rob = spec.update(0, [('a', 100.0), ('b', 20.0)])


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

    vars = ['a']
    specification = 'always[0,5000] (0.5 >= a)'
    spec = rtamt.STLDiscreteTimeSpecification()
    for var in vars:
        spec.declare_var(var, 'float')
    spec.spec = specification

    try:
        spec.parse()
        spec.pastify()
    except rtamt.RTAMTException as err:
        print('RTAMT Exception: {}'.format(err))
        sys.exit()

    s = env.reset()
    for i in range(5000):
        # print(s)
        a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
        s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
        rob = spec.update(i, [('a', s[0])])
        print(rob)
        if rob < 0:
            print("Falsified at step {}".format(i))
            print(rob)
            print(s)
            break

    policy.sess.close()

