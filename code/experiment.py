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
    max_duration = 5000#50

    # T_max = 100000
    # E_eps_mean = []

    for k in tqdm(range(nb_simu), desc="Simulating {}".format(agent.name)):
        # Compute policy π ̃k:
        agent.update_policy()

        # Execute policy π ̃k on environnement during a max of max_duration:
        agent.execute_policy(env, max_duration)

        # Computing regret at eatch step for plotting
        if (agent.verbose > 0):
            print("> Computing Regret (", N_samples, "samples).")
        time0 = time.time()

        # E_list = []
        # for i in range(N_samples):
        #     # E = env.compute_LTAR(agent.policy, T_max/N_samples)
        #     E = env.compute_regret(agent.policy, T_max/N_samples)
        #     E_list.append(E)
        # E_mean = np.array(E_list).mean()
        #
        # if (agent.verbose > 1):
        #     print(" took", time.time() - time0, "s.")
        # if (agent.verbose > 0):
        #     print(" Regret = ", E_mean)
        # E_var = np.array(E_list).var()
        # E_eps_mean.append(E_mean)

    # x = np.arange(1, nb_simu+1)
    # plt.plot(x, E_eps_mean, label = agent.name)
    # plt.xlabel('Rounds')
    # # plt.ylabel('Average Long-term reward')
    # plt.ylabel('Regret')
    # plt.legend()
    # plt.show()

    T_max = 50
    x = np.arange(1, T_max+1)
    for state in range(env.n_states):
        rewards_list = []
        for i in range(N_samples):
            rewards = env.compute_cumul_reward(agent.policy, T_max, init_state=state)
            rewards_list.append(rewards)
        rewards_mean = np.mean(np.array(rewards_list), 0)
        cumul_reward = np.cumsum(rewards_mean)
        plt.plot(x, cumul_reward, label="state "+str(state))

    plt.xlabel('Rounds')
    plt.ylabel('Cumulative reward')
    plt.legend()
    plt.show()
