import json
import argparse

import numpy as np

from DDPG import DDPG
from envs import ENV_CLASSES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Options")
    parser.add_argument("--env", default="cartpole", type=str, help="The selected environment.")
    parser.add_argument("--do_train", action="store_true", help="Train RL controller.")
    parser.add_argument("--do_test", action="store_true", help="Test RL controller")
    parser.add_argument("--test_episodes", default=50, help="test_episodes", type=int)
    # parser.add_argument("--do_retrain", action="store_true", help="retrain RL controller")
    args = parser.parse_args()

    env = ENV_CLASSES[args.env]()
    with open("configs.json") as f:
        configs = json.load(f)
    DDPG_args = configs[args.env]
    DDPG_args["enable_test"] = args.do_test
    DDPG_args["test_episodes"] = args.test_episodes

    cur_seq = np.matrix(
            [[np.random.uniform(env.s_min[i, 0], env.s_max[i, 0])] for i in range(env.state_dim)],
        )
    
    actor = DDPG(env, cur_seq, DDPG_args)
    actor.sess.close()

