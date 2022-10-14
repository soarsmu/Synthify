import json
import argparse

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
    DDPG_args["enable_test"] = args.do_test, 
    DDPG_args["test_episodes"] = args.test_episodes

    actor = DDPG(env, DDPG_args)
    actor.sess.close()



    # parser.add_argument("--do_train", action="store_true", help="Train RL controller")
    # parser_res = parser.parse_args()
    # nn_test = parser_res.nn_test
    # retrain_shield = parser_res.retrain_shield
    # shield_test = parser_res.shield_test
    # test_episodes = parser_res.test_episodes if parser_res.test_episodes is not None else 100
    # retrain_nn = parser_res.retrain_nn

    # cartpole("random_search", 100, 200, 0, [300, 200], [300, 250, 200], "ddpg_chkp/cartpole/continuous/300200300250200/", \
    #          nn_test=nn_test, retrain_shield=retrain_shield, shield_test=shield_test, test_episodes=test_episodes, retrain_nn=retrain_nn)





