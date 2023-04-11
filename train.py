import json
import argparse

from envs import ENV_CLASSES
from DDPG import DDPG

if __name__ == "__main__":
    print(1)
    parser = argparse.ArgumentParser(description="Running Options")
    parser.add_argument("--env", default="pendulum", type=str, help="The selected environment.")
    parser.add_argument("--train", action="store_true", help="Whether to train RL.")
    args = parser.parse_args()

    env = ENV_CLASSES[args.env]()
    with open("configs.json") as f:
        configs = json.load(f)

    policy_args = configs[args.env]

    if args.train:
        policy_args["enable_train"] = True
    print("policy_args:\n", policy_args)
    policy = DDPG(env, policy_args)
    policy.close()