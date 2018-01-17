# coding: utf-8
""" Implementation of an xperiment """

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def run_experiment(agent, env, T_max, nb_simu, N_samples, seed=1):
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
    regret_list = np.empty(nb_simu, T_max)
    
    for i in range(nb_simu):
        agent.run(env, T_max)
        regret_list[i,:] = agent.compute_regret(env)
        

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
