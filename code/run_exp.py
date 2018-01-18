# coding: utf-8
""" Script to be run from terminal to launch experiment """

import env_examples as env
import algos as ag
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

# env_TP1 = env.make_MDP_TP1()
# env_TP1 = env.make_MDP_Pierre()
env_TP1 = env.generate_random_MDP(10, 2, 1)

# env_TP1 = env.make_StarRiverSwim(5, 6)
# env_TP1 = env.make_riverSwim(5)
env_TP1.compute_gain(1e-5)
print("Bias:", env_TP1.bias)


max_reward = 1

nb_simu = 200
T_max = 10000

H,S,A = env_TP1.bias, env_TP1.n_states, env_TP1.n_actions

for C in tqdm([H, 1.5*H, float("inf"), None]):
    regret_list = np.empty((nb_simu, T_max))
    for i in tqdm(range(nb_simu)):
        agent = ag.PSRL(env_TP1, max_reward, verbose=0, C=C)
        agent.run(env_TP1, T_max+1)
        # print(agent.policy)
        regret_list[i,:] = agent.compute_regret(env_TP1)

    regret_mean = np.mean(regret_list, 0)
    # regret_var = np.var(regret_list, 0)

    plt.figure(1)
    plt.plot(np.arange(1, T_max+1), regret_mean, label = "C : {}".format(agent.C))
    plt.ylabel('Regret')
    plt.xlabel('Rounds')

upper_bound = lambda T : H*S*np.sqrt(A*T)
upper_bound = np.vectorize(upper_bound)
# plt.plot(np.arange(1,T_max), upper_bound(np.arange(1,T_max)), label = "upper_bound")
plt.legend()
plt.show()
