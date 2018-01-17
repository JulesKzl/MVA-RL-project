# coding: utf-8
"""
Implementation of a basic RL environments.

author: iosband@stanford.edu
Edit : Jules Kozolinsky
"""

import numpy as np
import numbers #Gridwold
import copy #Gridwold

#-------------------------------------------------------------------------------
class Environment(object):
    """General RL environment"""

    def __init__(self, n_states, n_actions):
        """
        Initialize Env

        Args:
            n_states  - int - number of states
            n_actions - int - number of actions

        Returns:
            Environment object
        """
        self.n_states = n_states
        self.n_actions = n_actions

    def reset(self):
        """
        Returns:
            An initial state randomly drawn from
            the initial distribution
        """
        pass

    def step(self, state, action):
        """
        Moves one step in the environment.

        Args:
            state (int): the amount of good
            action (int): the action to be executed

        Returns:
            next_state (int): the state reached by performing the action
            reward (float): a scalar value representing the immediate reward
            absorb (boolean): True if the next_state is absorsing, False otherwise
        """
        pass



#-------------------------------------------------------------------------------
class MDP(Environment):
    """
    MDP
    Rewards are all normal.
    Transitions are multinomial.

    R - dict by (s,a) - each R[s,a] = (meanReward, sdReward)
    P - dict by (s,a) - each P[s,a] = transition vector size S
    """

    def __init__(self, n_states, n_actions):
        """
        Initialize MDP

        Args:
            n_states  - int - number of states
            n_actions - int - number of actions

        Returns:
            Environment object
        """
        super(MDP, self).__init__(n_states, n_actions)

        self.timestep = 0

        # Now initialize R and P
        self.R = {}
        self.P = {}
        for state in range(n_states):
            for action in range(n_actions):
                self.R[state, action] = (1, 1)
                self.P[state, action] = np.ones(n_states) / n_states
        self.augmented = False

    def get_R(self):
        if (self.augmented):
            return self.R_augm
        else:
            return self.R

    def set_R(self, R):
        if (self.augmented):
            self.R_augm = R
        else:
            self.R = R

    def get_P(self):
        if (self.augmented):
            return self.P_augm
        else:
            return self.P

    def set_P(self, P):
        if (self.augmented):
            self.P_augm = P
        else:
            self.P = P

    def reset(self):
        """
        Reset the environment

        Returns:
            An initial state randomly drawn from
            the initial distribution
        """
        self.timestep = 0
        x_0 = np.random.randint(0, self.n_states)
        return x_0

    def step(self, state, action):
        """
        Moves one step in the environment.

        Args:
            state (int): the amount of good
            action (int): the action to be executed

        Returns:
            next_state (int): the state reached by performing the action
            reward (float): a scalar value representing the immediate reward
            absorb (boolean): True if the next_state is absorsing, False otherwise
        """
        R = self.get_R()
        if R[state, action][1] < 1e-9:
            reward = R[state, action][0]
        else:
            reward = np.random.normal(loc=R[state, action][0],
                                      scale=R[state, action][1])

        # Update the environment
        self.timestep += 1
        P = self.get_P()
        next_state = np.random.choice(self.n_states, p=P[state, action])
        absorb = False

        return next_state, reward, absorb

    def compute_cumul_reward(self, T_max, policy, init_state=None):
        """ Compute long-term average reward """
        t = 0
        cumul_reward = []
        if (init_state == None):
            state = self.reset()
        else:
            state = init_state
        action = policy[state]
        while (t < T_max): #or terminal
            t += 1
            nexts, reward, term = self.step(state, action)
            cumul_reward.append(reward)
            state = nexts
            action = policy[state]
        LTAR = np.sum(cumul_reward)/T_max
        return cumul_reward, LTAR

    def compute_regret(self, T_max, policy, policy_opt, init_state=None):
        cumul_reward, _ = self.compute_cumul_reward(T_max, policy, init_state)
        if (policy_opt == []):
            regret = None
        else:
            _, LTAR_opt = self.compute_cumul_reward(T_max, policy_opt, init_state)
            regret = T_max*LTAR_opt - np.array(cumul_reward)
        return cumul_reward, regret

    def augment_MDP(self):
        """ Transform MDP into augmented MDP """
        n_actions_augm = self.n_actions*2
        R_augm = {}
        P_augm = {}
        # Copy of MDP
        for s in range(self.n_states):
            for a in range(self.n_actions):
                R_augm[s, a] = self.R[s, a]
                P_augm[s, a] = self.P[s, a]
                R_augm[s, a+self.n_actions] = (0, 0)
                P_augm[s, a+self.n_actions] = self.P[s, a]

        self.MDP_augm = MDP(self.n_states, n_actions_augm)
        self.MDP_augm.R = R_augm
        self.MDP_augm.P = P_augm
        self.augmented = True

    def transform_policy(self, policy_augm):
        """ Transform a policy from the augmented MDP to a policy of MDP """
        policy = policy_augm.copy()
        for s in range(self.n_states):
            if (policy_augm[s] >= self.n_actions):
                policy[s] = policy_augm[s]-self.n_actions
        return policy
