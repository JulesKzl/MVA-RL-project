# coding: utf-8
""" Script to be run from terminal to launch experiment """
from tqdm import tqdm
import matplotlib.pyplot as plt

import env_examples as env
import algos as ag
import experiment as exp

# env_TP1 = env.make_MDP_TP1()
env_TP1 = env.make_riverSwim()
# env_TP1 = env.make_MDP_Pierre()
# env_TP1 = env.generate_random_MDP(10, 5, 10)

# env_TP1.augment_MDP()

max_reward = 10
agent = ag.PSRL(env_TP1, max_reward, verbose=0)
# agent = ag.UCRL2(env_TP1, max_reward, verbose=3)
policy_opt = agent.compute_optimal_policy(env_TP1, 1e-4)
print(policy_opt)

N = 10
C_list = [None, 0, 0.001, 0.01, 0.1, 1, 10, 100]
for C in tqdm(C_list, desc='Testing REGAL:'):
    agent = ag.PSRL(env_TP1, max_reward, verbose=0, C=C)
    regrets_mean = exp.run_experiment(agent, env_TP1, N, 1)
    print(agent.policy)

    # Display regret
    plt.plot(np.arange(1, T_max+1), regrets_mean, label=str(agent.C))
    plt.ylabel('Regret')
    plt.xlabel('Rounds')
    plt.legend()
plt.show()
