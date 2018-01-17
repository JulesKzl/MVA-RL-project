# coding: utf-8
""" Script to be run from terminal to launch experiment """
from tqdm import tqdm
import matplotlib.pyplot as plt

import env_examples as env
import algos as ag
import matplotlib.pyplot as plt
import numpy as np

def run_exp(env1, agent, T_max, nb_simu):
    regret_list = np.empty((nb_simu, T_max))
    for i in tqdm(range(nb_simu)):
        agent.initialize()
        agent.run(env1, T_max+1)
        var1 = agent.compute_regret(env1)
        print(np.shape(var1))
        print(regret_list[i,:])

    regret_mean = np.mean(regret_list, 0)

    if (agent.name == "PSRL"):
        label = "C:"+str(agent.C)
    elif (agent.name == "UCRL2"):
        label = "UCRL2"
    else:
        label = "?"

    plt.figure(1)
    plt.plot(regret_mean, label=label)
    plt.ylabel('Regret')
    plt.xlabel('Rounds')
    plt.legend()
    return regret_mean

# env_TP1 = env.make_MDP_TP1()
env_TP1 = env.make_riverSwim()
# env_TP1 = env.make_MDP_Pierre()
# env_TP1 = env.generate_random_MDP(10, 5, 10)

# env_TP1.augment_MDP()

max_reward = 10

nb_simu = 25
T_max = 10000

C_list = [None, 0.01, 0.1, 1, 10, 100]
for C in tqdm(C_list, desc='Testing REGAL:'):
    agent = ag.PSRL(env_TP1, max_reward, verbose=0, C=C)
    run_exp(env_TP1, agent, T_max, nb_simu)
plt.figure(1)
plt.show()

# # Display cumulative reward
# rewards_mean = np.mean(np.array(rewards_list), 0)
# cumul_reward = np.cumsum(rewards_mean)
# plt.figure(1)
# plt.plot(x, cumul_reward, label="From state:"+str(state))
# plt.ylabel('Cumulative reward')
# plt.xlabel('Rounds')
# plt.legend()
