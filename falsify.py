import sys
import json
import rtamt
import argparse

import numpy as np

from tqdm import tqdm
from DDPG import DDPG
from envs import ENV_CLASSES

def init_monitor(vars, specification):
    monitor = rtamt.STLDiscreteTimeSpecification()
    for var in vars:
        monitor.declare_var(var, 'float')
    monitor.spec = specification

    try:
        monitor.parse()
        monitor.pastify()
    except rtamt.RTAMTException as err:
        print('RTAMT Exception: {}'.format(err))
        sys.exit()

    return monitor


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

    vars = ['a', 'b', "c", "d"]
    specifications = ['always[0,5000] (a <= 0.05)', 'always[0,5000] (a >= -0.05)', 'always[0,5000] (b <= 0.1)', 'always[0,5000] (b >= -0.1)', 'always[0,5000] (c <= 0.05)', 'always[0,5000] (c >= -0.05)', 'always[0,5000] (d <= 0.05)', 'always[0,5000] (d >= -0.05)']

    min_robs = [100, 100, 100, 100, 100, 100, 100, 100]
    prob = [0, 0, 0, 0, 0, 0, 0, 0]
    times = [0, 0, 0, 0, 0, 0, 0, 0]

    def sample_spec(specifications, prob, eps=0.5):
        p = np.random.uniform(0, 1)
        if(p > eps):
            arm_to_pull = np.argmax(prob)
        else:
            arm_to_pull = np.random.randint(0, len(specifications), 1)[0]

        return arm_to_pull

    for p in range(len(specifications)):
        times[p] += 1
        prob[p] += 1
        print(prob)
        monitor = init_monitor(vars, specifications[p])

        falied_tests = []

        s = env.reset()
        for iter in tqdm(range(20)):
            noise = np.array([0.01, 0.01, 0.01, 0.01])
            # print(s)
            for sign in [-1, 1]:
                s += sign * noise.reshape(4, 1)
                s_inital = s
                robs = []
                for i in range(100):
                    a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
                    s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
                    rob = monitor.update(i, [('a', s[0]), ('b', s[1]), ('c', s[2]), ('d', s[3])])
                    robs.append(rob)
                    if rob < 0:
                        # print("Falsified at step {}".format(i))
                        falied_tests.append(s)
                        break

                min_robs[p] = -min(min_robs[p], min(robs))
        prob[p] = prob[p] / (1 / times[p]) * (min_robs[p] - prob[p])
    
    falied_tests = []
    for i in range(10):
        spec_index = sample_spec(specifications, prob)
        times[spec_index] += 1
        prob[spec_index] += 1
        
        monitor = init_monitor(vars, specifications[spec_index])
        s = env.reset()
        for iter in tqdm(range(20)):
            flag = False
            noise = np.array([0.01, 0.01, 0.01, 0.01])
            # print(s)
            for sign in [-1, 1]:
                s += sign * noise.reshape(4, 1)
                s_inital = s
                robs = []
                for i in range(100):
                    a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
                    s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
                    rob = monitor.update(i, [('a', s[0]), ('b', s[1]), ('c', s[2]), ('d', s[3])])
                    robs.append(rob)
                    if rob < 0:
                        print("Falsified at step {}".format(i))
                        falied_tests.append(s)
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
    print(len(falied_tests))
    policy.sess.close()
