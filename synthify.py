import json
import time
import rtamt
import logging
import argparse
import numpy as np

from tqdm import tqdm
from DDPG import DDPG
from envs import ENV_CLASSES
from ES import evolution_policy, refine

from numpy.typing import NDArray
from staliro.models import ModelData, SignalTimes, SignalValues, StaticInput, blackbox
from optimizer import DualAnnealing
from staliro.options import Options
from staliro.specifications import RTAMTDense
from staliro.staliro import staliro, simulate_model


def init_monitor(vars, specification):
    monitor = rtamt.STLDiscreteTimeSpecification()
    for var in vars:
        monitor.declare_var(var, "float")
    monitor.spec = specification

    monitor.parse()
    monitor.pastify()

    return monitor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Options")
    parser.add_argument(
        "--env", default="pendulum", type=str, help="The selected environment."
    )
    args = parser.parse_args()

    env = ENV_CLASSES[args.env]()
    with open("configs.json") as f:
        configs = json.load(f)

    policy_args = configs[args.env]
    policy = DDPG(env, policy_args)

    DataT = ModelData[NDArray[np.float_], None]

    sim_time = 0
    vars = list(policy_args["var_map"].keys())
    n_states = policy.s_dim
    n_actions = policy.a_dim
    syn_time = time.time()
    syn_policy = evolution_policy(env, policy, n_states + 1, n_actions, 100)
    logging.info("Synthesis time: %f" % (time.time() - syn_time))
    inter_time = 0
    linear_time = 0
    drl_time = 0

    @blackbox(sampling_interval=1.0)
    def model(static: StaticInput, times: SignalTimes, signals: SignalValues) -> DataT:
        states = []
        global sim_time
        global inter_time
        start_time = time.time()
        if args.env == "biology":
            static = (static[0], 0.0, static[1])
        elif args.env == "oscillator":
            static = (
                static[0],
                static[1],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        s = env.reset(np.reshape(np.array(static), (policy.s_dim, 1)))
        for i in range(len(times)):
            start_time = time.time()
            a_linear = syn_policy[:, :n_states].dot(s) + syn_policy[:, n_states:]
            sim_time += time.time() - start_time
            start_time = time.time()

            s, r, terminal = env.step(a_linear.reshape(policy.a_dim, 1))
            inter_time += time.time() - start_time
            states.append(np.array(s))
        states = np.hstack(states)
        return ModelData(states, np.asarray(times))

    min_robs = [0] * policy_args["spec_lens"]
    prob = [0] * policy_args["spec_lens"]
    times = [0] * policy_args["spec_lens"]
    specifications = policy_args["slice_spec"]

    def sample_spec(specifications, prob, eps=0.9):
        p = np.random.uniform(0, 1)
        if p > eps:
            arm_to_pull = np.argmin(prob)
        else:
            arm_to_pull = np.random.randint(0, policy_args["spec_lens"], 1)[0]
        return arm_to_pull

    failures = []
    # sim_time = 0
    real_sim_time = 0
    falsification_time = 0
    refine_time = 0

    itertimes = []
    linear_itertimes = []
    count = 0
    for budget in tqdm(range(50), desc="Falsification of %s" % args.env):
        spec_index = sample_spec(specifications, min_robs)
        times[spec_index] += 1
        prob[spec_index] += 1

        initial_conditions = policy_args["initial_conditions"]
        phi = policy_args["specification"]
        RTAMT_offline = RTAMTDense(phi, policy_args["var_map"])

        def cost(s):
            global inter_time
            time_cost = 0
            time_steps = 200
            states = []
            if args.env == "biology":
                s = (s[0], 0.0, s[1])
            elif args.env == "oscillator":
                s = (
                    s[0],
                    s[1],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                time_steps = 300
            elif args.env == "quadcopter":
                time_steps = 300
            elif args.env == "lane_keeping":
                time_steps = 300
            s = np.reshape(np.array(s), (policy.s_dim, 1))

            for i in range(time_steps):
                start = time.time()
                a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
                time_cost += time.time() - start

                start = time.time()
                s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
                inter_time += time.time() - start

                states.append(np.array(s))
            states = np.hstack(states)
            return (
                RTAMT_offline.evaluate(states, np.arange(0, time_steps, 1)),
                time_cost,
            )

        def false_checker(state):
            real_cost, time_cost = cost(state)
            if real_cost < 0 or np.isnan(real_cost) or np.isinf(real_cost):
                return True, time_cost
            return False, time_cost

        real_simulations = 0
        linear_simulations = 0
        start = time.time()
        while real_simulations < 100:
            spec_index = sample_spec(specifications, min_robs)
            times[spec_index] += 1
            prob[spec_index] += 1

            initial_conditions = policy_args["initial_conditions"]
            phi = specifications[spec_index]
            phi = policy_args["specification"]
            RTAMT_offline = RTAMTDense(phi, policy_args["var_map"])
            success = False
            time_steps = 200
            if (
                args.env == "oscillator"
                or args.env == "quadcopter"
                or args.env == "lane_keeping"
            ):
                time_steps = 300
            options = Options(
                runs=1,
                iterations=100,
                interval=(0, time_steps),
                static_parameters=initial_conditions,
            )
            optimizer = DualAnnealing()
            result = staliro(model, RTAMT_offline, optimizer, options)
            for run in result.runs:
                for id, evaluation in enumerate(run.history):
                    min_robs[spec_index] = min(
                        min_robs[spec_index], max(evaluation.cost, -1)
                    )

            evaluation = result.runs[0].history[-1]
            if (
                evaluation.cost < 0
                or np.isnan(evaluation.cost)
                or np.isinf(evaluation.cost)
            ):
                real, time_cost = false_checker(evaluation.sample)
                real = True
                real_simulations += 1
                linear_simulations += len(result.runs[0].history)
                real_sim_time += time_cost

                if real:
                    failures.append(evaluation.sample)
                    success = True
                    itertimes.append(real_simulations)
                    linear_itertimes.append(linear_simulations)
                else:
                    logging.info("false negative, refining...")
                    st = time.time()
                    refine_steps = 5
                    learning_rate = 0.001
                    if args.env == "self_driving":
                        refine_steps = 100
                        learning_rate = 0.05
                    syn_policy = refine(
                        env,
                        policy,
                        syn_policy,
                        n_states + 1,
                        n_actions,
                        evaluation.sample,
                        refine_steps,
                        alpha=learning_rate,
                    )
                    refine_time += time.time() - st

            if success:
                logging.info("%d successful trials over 50 trials", len(failures))
                logging.info(
                    "mean number of simulations over successful trials is %f",
                    np.mean(itertimes),
                )
                logging.info(
                    "median number of simulations over successful trials is %f",
                    np.median(itertimes),
                )
                logging.info(
                    "mean number of linear simulations over successful trials is %f",
                    np.mean(linear_itertimes),
                )
                logging.info(
                    "median number of linear simulations over successful trials is %f",
                    np.median(linear_itertimes),
                )
                falsification_time += time.time() - start
                break

    logging.info("%d successful trials over 50 trials", len(failures))
    logging.info("falsification rate wrt. 50 trials is %f", len(failures) / 50)
    logging.info(
        "mean number of simulations over successful trials is %f", np.mean(itertimes)
    )
    logging.info(
        "median number of simulations over successful trials is %f",
        np.median(itertimes),
    )
    logging.info("linear simulation time is %f", sim_time)
    logging.info("DRL simulation time is %f", real_sim_time)
    logging.info("falsification time is %f", falsification_time)
    logging.info("refinement time is %f", refine_time)
    logging.info("interpolation time is %f", inter_time)
    logging.info(
        "non-simulation time ratio %f",
        (falsification_time - sim_time - real_sim_time) / falsification_time,
    )
    coverage = [i for i in min_robs if i < 0]
    logging.info(
        "coverage of slice specifications is %s", len(coverage) / len(min_robs)
    )
    coverage = [0] * len(policy_args["slice_spec"])
    for failure in tqdm(failures, desc="Coverage of %s" % args.env):
        sample = simulate_model(model, options, failure)
        for id, spec in enumerate(policy_args["slice_spec"]):
            specification = RTAMTDense(spec, policy_args["var_map"])
            if specification.evaluate(sample.states, sample.times) < 0:
                coverage[id] += 1
    logging.info(
        "coverage of slice specifications is %s",
        np.count_nonzero(coverage) / len(coverage),
    )
