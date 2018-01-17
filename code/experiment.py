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
    max_duration = 5000

    # Learning
    for k in tqdm(range(nb_simu), desc="Simulating {}".format(agent.name)):
        # Compute policy π ̃k:
        agent.update_policy()
        # Execute policy π ̃k on environnement during a max of max_duration:
        agent.execute_policy(env, max_duration)

    # Evaluation of learned policy
    T_max = 100
    x = np.arange(1, T_max+1)
    for state in range(env.n_states):
        rewards_list = []
        regret_list = []
        for i in range(N_samples):
            rewards, regret = env.compute_regret(T_max, agent.policy, agent.policy_opt, init_state=state)
            rewards_list.append(rewards)
            regret_list.append(regret)

        # Display cumulative reward
        rewards_mean = np.mean(np.array(rewards_list), 0)
        cumul_reward = np.cumsum(rewards_mean)
        plt.figure(1)
        plt.plot(x, cumul_reward, label="From state:"+str(state))
        plt.ylabel('Cumulative reward')
        plt.xlabel('Rounds')
        plt.legend()
        # Display regret
        if (agent.policy_opt != []):
            regrets_mean = np.mean(np.array(regret_list), 0)
            plt.figure(2)
            plt.plot(x, regrets_mean, label="From state:"+str(state))
            plt.ylabel('Regret')
            plt.xlabel('Rounds')
            plt.legend()

    plt.show()
