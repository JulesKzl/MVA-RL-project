# coding: utf-8
""" Implementation of an xperiment """

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def run_experiment(agent, env, nb_simu, N_samples, seed=1):
    """
    A simple script to run an experiment

    Args:
        agent - agent
        env - MDP
        nb_simu - number of simulation
        N_samples - number of samples to compute regret
        seed - 1 - for reproductibility
    """
    np.random.seed(seed)
    max_duration = 500

    # Learning
    regrets_list=[]
    for _ in tqdm(range(N_samples)):
        agent.initialize()
        for _ in tqdm(range(nb_simu), desc="Simulating {}".format(agent.name)):
            # Compute policy π ̃k:
            agent.update_policy()
            # Execute policy π ̃k on environnement during a max of max_duration:
            agent.execute_policy(env, max_duration)

        T_max = len(agent.reward_list)
        regrets = agent.gain_opt*np.arange(1, T_max+1) - np.cumsum(np.array(agent.reward_list))
        regrets_list.append(regrets)


    T_max_min = len(min(regrets_list,key=len))
    regrets_list = np.array([l[:T_max_min] for l in regrets_list])
    regrets_mean = np.mean(regrets_list, 0)

    return regrets_mean
    # Evaluation of learned policy
    # T_max = 100
    # x = np.arange(1, T_max+1)
    # for state in range(env.n_states):
    #     rewards_list = []
    #     for i in range(100):
    #         rewards, _ = env.compute_cumul_reward(T_max, agent.policy, init_state=state)
    #         rewards_list.append(rewards)
    #     # Display cumulative reward
    #     rewards_mean = np.mean(np.array(rewards_list), 0)
    #     cumul_reward = np.cumsum(rewards_mean)
    #     plt.figure(1)
    #     plt.plot(x, cumul_reward, label="From state:"+str(state))
    #     plt.ylabel('Cumulative reward')
    #     plt.xlabel('Rounds')
    #     plt.legend()
