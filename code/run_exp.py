# coding: utf-8
""" Script to be run from terminal to launch experiment """

import env_examples as env
import algos as ag
import experiment as exp

env_TP1 = env.make_MDP_TP1()
# env_TP1 = env.make_riverSwim()

max_reward = 5
agent = ag.PSRL(env_TP1, max_reward, verbose=3)
# agent = ag.UCRL2(env_TP1, max_reward, verbose=3)

# N = 99
# N = 100
# print(env_TP1.pi_star)
# exp.run_experiment(agent, env_TP1, N, 100)
# print(agent.policy)

R1 = env_TP1.R
R = {key: R1[key][0] for key in R1.keys()}
print(R)
P = env_TP1.P

_, _, v = agent.value_iteration(P,R, 1e-4)
policy_op = agent.policy
print(v)
print(policy_op)
