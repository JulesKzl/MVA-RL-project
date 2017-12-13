'''
Script to run simple RL experiments.

author: iosband@stanford.edu
'''

import numpy as np
import pandas as pd

from shutil import copyfile


def run_experiment(agent, env, num_eps, seed=1,
                    fileFreq=1000, targetPath='tmp.csv'):
    '''
    A simple script to run a finite tabular MDP experiment

    Args:
        agent - finite tabular agent
        env - finite TabularMDP
        num_eps - number of episodes to run
        seed - numpy random seed
        fileFreq - how many episodes between writing file
        targetPath - where to write the csv

    Returns:
        NULL - data is output to targetPath as csv file
    '''
    data = []
    np.random.seed(seed)
    max_duration = 50

    cum_reward = []
    regret = []

    for ep in range(1, num_eps + 2):
        print("ep:", ep)
        # Compute policy π ̃k:
        agent.update_policy()

        # Execute policy π ̃k:
        ep_cum_reward = 0
        ep_regret = 0

        agent.visited_sa.clear()
        state = env.reset()
        t = 0
        while (agent.nu_k[state][agent.policy_indices[state]] \
                < max(1, agent.nb_observations[state][agent.policy_indices[state]])\
                and t < max_duration):
            # Select action
            action = agent.pick_action(state)

            # Step through the episode
            new_state, reward, absorb = env.step(state, action)

            # Store total reward and regret
            t = env.timestep
            ep_cum_reward += reward
            curr_regret = t* agent.rho_star - ep_cum_reward
            #ep_regret.append(curr_regret)

            # Update estimations at each step
            agent.update_estimations(state, new_state, reward, absorb, t)
            state = new_state
        # Update estimated probabilities
        agent.update_obs()

        cum_reward.append(ep_cum_reward)
        print(curr_regret)
        #regret.append(ep_regret)

        # Logging to dataframe
        if ep < 1e4:
            recFreq = 100
        elif ep < 1e5:
            recFreq = 1000
        else:
            recFreq = 10000

        # if ep % recFreq == 0:
        #     data.append([ep, ep_reward, cumReward, cumRegret, empRegret])
        #     print 'episode:', ep, 'ep_reward:', ep_reward, 'cumRegret:', cumRegret
        #
        # if ep % max(fileFreq, recFreq) == 0:
        #     dt = pd.DataFrame(data,
        #                       columns=['episode', 'ep_reward', 'cumReward',
        #                                'cumRegret', 'empRegret'])
        #     print 'Writing to file ' + targetPath
        #     dt.to_csv('tmp.csv', index=False, float_format='%.2f')
        #     copyfile('tmp.csv', targetPath)
        #     print '****************************'

    print('Experiment complete')
