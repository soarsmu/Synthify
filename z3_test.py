
from z3 import *

import math
import json
import logging
import argparse
import numpy as np
import tensorflow as tf

set_option(max_args=100000, max_lines=100000, max_depth=100000, max_visited=100000)


from tqdm import tqdm
from DDPG import DDPG, get_params

from envs import ENV_CLASSES

logging.getLogger().setLevel(logging.INFO)



def verify(policy, func):
    
    
    exit()
    y_pred = policy.predict(x_)
    s = Solver()
    s.add(Abs(y_pred[0] - y_true[0]) == 0.0)
    r = s.check()
    if r == unsat:
        print("proved")
    else:
        print("counterexample")
        print(s.model())
#     # x, y = Ints('x y')
# # F = And(x >= 1, x == 2*y)
# # G = And(2*y - x == 0, x >= 0)
# # s = Solver()
# # s.add(Not(F == G))
# # r = s.check()
# # if r == unsat:
# #     print("proved")
# # else:
# #     print("counterexample")
# #     print(s.model())
#     res = s.check()
#     print(res)
#     if res == sat:
#         m = s.model()
#         print("Bad x value:", m[x[0]])
#         x_bad = m[x[0]].numerator_as_long() / m[x[0]].denominator_as_long() 
#         print("Error of prediction: ", abs(model.predict(np.array([x_bad])) - cheb(x_bad)))

   
# w1, b1, w2, b2, w3, b3 = model.get_weights() # unpack weights from model

# def Relu(x):
#     return np.vectorize(lambda y: If(y >= 0 , y, RealVal(0)))(x)
# def Abs(x):
#     return If(x <= 0, -x, x)
# def net(x):
#     x1 = w1.T @ x + b1
#     y1 = Relu(x1)
#     x2 = w2.T @ y1 + b2
#     y2 = Relu(x2)
#     x3 = w3.T @ y2 + b3
#     return x3

# x = np.array([Real('x')])
# y_true = cheb(x)
# y_pred = net(x)
# s = Solver()
# s.add(-1 <= x[0], x[0] <= 1)
# s.add(Abs( y_pred[0] - y_true[0] ) >= 0.5)
# #prove(Implies( And(-1 <= x[0], x[0] <= 1),  Abs( y_pred[0] - y_true[0] ) >= 0.2))
# res = s.check()
# print(res)
# if res == sat:
#     m = s.model()
#     print("Bad x value:", m[x[0]])
#     x_bad = m[x[0]].numerator_as_long() / m[x[0]].denominator_as_long() 
#     print("Error of prediction: ", abs(model.predict(np.array([x_bad])) - cheb(x_bad)))


# x, y = Ints('x y')
# F = And(x >= 1, x == 2*y)
# G = And(2*y - x == 0, x >= 0)
# s = Solver()
# s.add(Not(F == G))
# r = s.check()
# if r == unsat:
#     print("proved")
# else:
#     print("counterexample")
#     print(s.model())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Options")
    parser.add_argument("--env", default="cartpole", type=str, help="The selected environment.")
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

    FullyConnected_W_params, FullyConnected_b_params, BatchNormalization_beta_params, BatchNormalization_gamma_params, actor = get_params(env, DDPG_args)
    assert len(FullyConnected_W_params) == len(FullyConnected_b_params) 
    assert len(BatchNormalization_beta_params) == len(BatchNormalization_gamma_params)
    assert len(FullyConnected_W_params) + len(FullyConnected_b_params) + len(BatchNormalization_beta_params) + len(BatchNormalization_gamma_params) == len(actor.network_params)

    def Relu(x):
        return np.vectorize(lambda y: If(y >= 0 , y, RealVal(0)))(x)
        # return np.maximum(0, x)

    w1, w2, w3, w4 = FullyConnected_W_params
    b1, b2, b3, b4 = FullyConnected_b_params
    # exit()
    beta1, beta2, beta3 = BatchNormalization_beta_params
    gamma1, gamma2, gamma3 = BatchNormalization_gamma_params

    def net(x):
        x1 = w1.T @ x + b1.reshape(-1, 1)
        scale = gamma1 / np.sqrt(1 + 1e-5)
        x1 = x1 * scale.reshape(-1, 1) + (beta1.reshape(-1, 1))
        y1 = Relu(x1)

        x2 = w2.T @ y1 + b2.reshape(-1, 1)
        scale = gamma2 / np.sqrt(1 + 1e-5)
        x2 = x2 * scale.reshape(-1, 1) + (beta2.reshape(-1, 1))
        y2 = Relu(x2)

        x3 = w3.T @ y2 + b3.reshape(-1, 1)
        scale = gamma3 / np.sqrt(1 + 1e-5)
        x3 = x3 * scale.reshape(-1, 1) + (beta3.reshape(-1, 1))
        y3 = Relu(x3)

        x4 = w4.T @ y3 + b4.reshape(-1, 1)
        x4 = (math.e ** x4 - math.e ** (-x4)) / (math.e ** x4 + math.e ** (-x4))
        x4 = x4 * env.u_max[0]
        return x4
    
    # s = env.reset()
    # print(net(np.array(s)))
    # a_linear = actor.predict(np.reshape(np.array(s), (1, actor.s_dim)))
    # print(a_linear)
    # exit()
    def Abs(x):
        return If(x <= 0, -x, x)
    # exit()
    def func(x):
        return x[0]
    
    def func1(x):
        return x[0]
    # x = sy.symbols('x')
    # cheb = sy.lambdify(x, sy.chebyshevt(4,x))
    # print(cheb(0))
    # exit()

    x = np.array([[Real('x')], [Real('y')], [Real('z')], [Real('q')]])
    y_true = func(x)
    y_pred = net(x)
    s = Solver()
    # s = Solver()
    s.add(-0.5 <= x[0][0], x[0][0] <= 0.5)
    s.add(Abs(y_pred[0][0] - y_true[0]) <= 0.1)
    res = s.check()
    if res == unsat:
        print("proved")
    else:
        print("counterexample")
        print(s.model())
    # if res == sat:
    #     m = s.model()
    #     print("Bad x value:", m[x[0]])
    #     x_bad = m[x[0]].numerator_as_long() / m[x[0]].denominator_as_long() 
    #     print("Error of prediction: ", abs(model.predict(np.array([x_bad])) - cheb(x_bad)))
    # # add constraints
    # s.add(-1 <= x[0], x[0] <= 1)
    # res = s.check()
    # if res == unsat:
    #     print("proved")
    # else:
    #     print("counterexample")
    #     print(s.model())
