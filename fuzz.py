import sys
import copy
import json
import time
import rtamt
import pickle
import argparse

import numpy as np

from tqdm import tqdm
from DDPG import DDPG
from MDPFuzz.fuzz import fuzzing
from envs import ENV_CLASSES

def init_monitor(vars, specification):
    # # stl
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

    # np.random.seed(2021)
    # states = np.random.randint(low=1, high=4, size=15)
    states = env.reset()

    '''
    video_length = 5000
    env = VecVideoRecorder(
    env,
    "./recording/test",
    record_video_trigger=lambda x: x == 0,
    video_length=video_length,
    name_prefix=f"{algo}-{env_id}",
    )
    obs = env.reset(states)
    '''

    episode_rewards, episode_lengths = [], []
    ep_len = 0
    successes = []
    fuzzer = fuzzing()
    seeds_num = 1000
    i = 0
    pbar = tqdm(total=seeds_num)
    while i < seeds_num:
        states = env.reset(states)
        state = None
        episode_reward = 0.0
        s = env.reset(states)
        sequences = [s]
        # print('states ', states)
        for _ in range(5000):
            a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
            s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
            sequences.append(s)
            episode_reward += r
            if terminal:
                break
        if not terminal:
            state = None
            episode_reward_mutate = 0.0
            delta_states = np.random.uniform(low=env.s_min, high=env.s_max)
            print(delta_states)
            # if np.sum(delta_states) == 0:
            #     delta_states[0] = 1
            mutate_states = states + delta_states
            # mutate_states = np.remainder(mutate_states, 4)
            mutate_states = np.clip(mutate_states, env.s_min, env.s_max)

            s = env.reset(mutate_states)
            print('mutate states ', mutate_states)

            for _ in range(5000):
                a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
                s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
                episode_reward_mutate += r
                if terminal:
                    break
            entropy = np.abs(episode_reward_mutate - episode_reward) / np.sum(delta_states)
            cvg = fuzzer.state_coverage(sequences)
            fuzzer.further_mutation(states, episode_reward, entropy, cvg, states)
            print(entropy, episode_reward, episode_reward_mutate, terminal, cvg)
            i += 1
            pbar.update(1)


    with open('./results/corpus_EM.pkl', 'wb') as handle:
        pickle.dump(fuzzer.corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./results/rewards_EM.pkl', 'wb') as handle:
        pickle.dump(fuzzer.rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./results/entropy_EM.pkl', 'wb') as handle:
        pickle.dump(fuzzer.entropy, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./results/cvg_EM.pkl', 'wb') as handle:
        pickle.dump(fuzzer.coverage, handle, protocol=pickle.HIGHEST_PROTOCOL)
    fuzzer.count = [5] * len(fuzzer.corpus)
    fuzzer.original = copy.deepcopy(fuzzer.corpus)

    # HACK: start fuzzing
    start_fuzz_time = time.time()
    cvg_threshold = 0.02

    current_time = time.time()
    pbar1 = tqdm(total=seeds_num)
    time_of_env = 0
    time_of_fuzzer = 0
    time_of_DynEM = 0
    while current_time - start_fuzz_time < 3600 * 12 and len(fuzzer.corpus) > 0:
        temp1_time = time.time()
        states = fuzzer.get_pose()
        mutate_states = fuzzer.mutation(states)
        state = None
        episode_reward = 0.0
        s = env.reset(mutate_states)
        sequences = [s]
        for _ in range(5000):
            a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
            s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
            sequences.append(s)
            if not args.no_render:
                env.render("human")
            episode_reward += r
            if terminal:
                break
        temp2_time = time.time()
        time_of_env += temp2_time - temp1_time
        cvg = fuzzer.state_coverage(sequences)
        temp3_time = time.time()
        time_of_DynEM += temp3_time - temp2_time
        local_sensitivity = np.abs(episode_reward - fuzzer.current_reward)
        if terminal or episode_reward < 10:
            pbar1.update(1)
            fuzzer.add_crash(mutate_states)
            print('Found: ', len(fuzzer.result))
        elif args.em:
            if cvg < cvg_threshold or episode_reward < fuzzer.current_reward:
                current_pose = copy.deepcopy(mutate_states)
                orig_pose = fuzzer.current_original
                fuzzer.further_mutation(current_pose, episode_reward, local_sensitivity, cvg, orig_pose)
        else:
            if episode_reward < fuzzer.current_reward:
                current_pose = copy.deepcopy(mutate_states)
                orig_pose = fuzzer.current_original
                fuzzer.further_mutation(current_pose, episode_reward, local_sensitivity, cvg, orig_pose)
        current_time = time.time()
        time_of_fuzzer += current_time - temp2_time
        print('total reward: ', episode_reward, ', coverage: ', cvg, ', passed time: ', current_time - start_fuzz_time, ', corpus size: ', len(fuzzer.corpus), 'time_of_fuzzer: ', time_of_fuzzer, 'time_of_env: ', time_of_env)

    if args.em:
        file_name = './results/crash_EM.pkl'
    else:
        file_name = './results/crash_noEM.pkl'
    with open(file_name, 'wb') as handle:
        pickle.dump(fuzzer.result, handle, protocol=pickle.HIGHEST_PROTOCOL)


    if args.verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    # if not args.no_render:
    #     if args.n_envs == 1 and "Bullet" not in env_id and not is_atari and isinstance(env, VecEnv):
    #         while isinstance(env, VecEnvWrapper):
    #             env = env.venv
    #         if isinstance(env, DummyVecEnv):
    #             env.envs[0].env.close()
    #         else:
    #             env.close()
    #     else:
    #         env.close()


# if __name__ == "__main__":
#     f = open('./results/fuzz.txt', 'w', buffering=1)
#     sys.stdout = f
#     main()