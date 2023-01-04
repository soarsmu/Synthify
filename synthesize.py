import os
import json
import argparse

import numpy as np

from DDPG import DDPG
from envs import ENV_CLASSES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Options")
    parser.add_argument("--env", default="cartpole", type=str, help="The selected environment.")
    args = parser.parse_args()

    env = ENV_CLASSES[args.env]()
    with open("configs.json") as f:
        configs = json.load(f)

    DDPG_args = configs[args.env]
    DDPG_args["enable_retrain"] = False
    DDPG_args["enable_eval"] = False
    DDPG_args["enable_fuzzing"] = False
    DDPG_args["enable_falsification"] = False

    DDPG_args["test_episodes"] = args.test_episodes

    model = DDPG(env, args)

    model_path = os.path.split(args['model_path'])[0]+'/'
    linear_func_model_name = 'K.model'
    model_path = model_path+linear_func_model_name+'.npy'

    shield = Shield(env, model, model_path, force_learning=retrain_shield, debug=retrain_shield)
    shield.train_shield(learning_method, number_of_rollouts, simulation_steps, eq_err=0, explore_mag = 1.0, step_size = 1.0)
    if shield_test:
        shield.test_shield(test_episodes, 1000, mode="single")

    actor.sess.close()


