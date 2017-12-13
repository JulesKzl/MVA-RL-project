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

    cumRegret = 0
    cumReward = 0
    #empRegret = 0

    for ep in range(1, num_eps + 2):
        # Reset the environment
        state = env.reset()
        agent.update_policy(ep)

        ep_reward = 0
        ep_regret = 0
        absorb = False

        while (not absorb):
            # Step through the episode
            t = env.timestep
            action = agent.pick_action(state, t)
            new_state, reward, absorb = env.step(state, action)

            ep_reward += reward
            ep_regret -= reward

            agent.update_obs(state, action, reward, new_state, absorb, t)
            state = new_state

        cumReward += ep_reward
        cumRegret += ep_regret
        #empRegret += (epMaxVal - ep_reward)


        # Logging to dataframe
        # Variable granularity
        # recFreq - how many episodes between logging
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

    print '**************************************************'
    print 'Experiment complete'
    print '**************************************************'
