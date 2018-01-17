# coding: utf-8
""" Script to be run from terminal to launch experiment """

import env_examples as env
import algos as ag
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

# env_TP1 = env.make_MDP_TP1()
#env_TP1 = env.make_riverSwim()
# env_TP1 = env.make_MDP_Pierre()
env_TP1 = env.generate_random_MDP(10, 5, 10)
env_TP1.compute_gain(1e-5)


print("bias :",  env_TP1.bias)
# env_TP1.augment_MDP()

max_reward = 10

nb_simu = 25
T_max = 10000
bias = env_TP1.bias
for C in tqdm([bias+1, bias+10, bias+20, bias+50, float("inf"), None]):
    regret_list = np.empty((nb_simu, T_max))
    for i in range(nb_simu):
        agent = ag.PSRL(env_TP1, max_reward, verbose=0, C=C)
        agent.run(env_TP1, T_max+1)
        regret_list[i,:] = agent.compute_regret(env_TP1)

    regret_mean = np.mean(regret_list, 0)

    plt.figure(1)
    plt.plot(regret_mean, label = "C : {}".format(agent.C))
    plt.ylabel('Regret')
    plt.xlabel('Rounds')
    plt.legend()
plt.show()
# agent = ag.UCRL2(env_TP1, max_reward, verbose=3)

#policy_opt = agent.compute_optimal_policy(env_TP1, 1e-4)
#print(policy_opt)

#N = 100
#exp.run_experiment(agent, env_TP1, N, 100)
#print(agent.policy)
