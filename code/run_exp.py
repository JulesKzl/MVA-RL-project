# coding: utf-8
""" Script to be run from terminal to launch experiment """

import env_examples as env
import algos as ag
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
from math import *

# env_TP1 = env.make_MDP_TP1()
# env_TP1 = env.make_MDP_Pierre()
#env_TP1 = env.generate_random_MDP(20, 2, 1)
n= 15
R0 = 0.5
R1 = 0.75
R = 0.75
delta = 0.05
env_TP1 = env.make_trap(n, R0, R, R1, delta, x0 = 7)
#env_TP1 = env.make
#env_TP1 = env.make_StarRiverSwim(5, 6)
env_TP1.compute_gain(1e-5)
#print("Bias:", env_TP1.bias)


max_reward = 1

nb_simu = 50
T_max = 20000
span_bias = env_TP1.span_bias
print("span_bias :", span_bias)
H,S,A = env_TP1.bias, env_TP1.n_states, env_TP1.n_actions

opt_check = env_TP1.check_optimality()
print("gain", env_TP1.max_gain)

for C in tqdm([0.01,0.5,1, float("inf")]):
    regret_list = np.empty((nb_simu, T_max))
    for i in range(nb_simu):
        agent = ag.PSRL(env_TP1, max_reward, verbose=0, C=C)
        agent.run(env_TP1, T_max+1)
        # print(agent.policy)
        regret_list[i,:] = agent.compute_regret(env_TP1)

    regret_mean = np.mean(regret_list, 0)

    plt.figure(1)
    plt.plot(regret_mean, label = "C : {}".format(agent.C))
    plt.ylabel('Regret')
    plt.xlabel('Rounds')
    plt.legend()

upper_bound = lambda T : H*S*sqrt(A*T)
upper_bound = np.vectorize(upper_bound)
#plt.plot(np.arange(1,T_max), upper_bound(np.arange(1,T_max)), label = "upper_bound")
plt.show()
