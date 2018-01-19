# coding: utf-8
"""
MDP Implementation
"""

import numpy as np

class MDP:
    """
    MDP
    Rewards are all normal.
    Transitions are multinomial.

    R - dict by (s,a) - each R[s,a] = (meanReward, sdReward)
    P - dict by (s,a) - each P[s,a] = transition vector size S
    """

    def __init__(self, n_states, n_actions, x0 = None):
        """
        Initialize MDP

        Args:
            n_states  - int - number of states
            n_actions - int - number of actions

        Returns:
            Environment object
        """
        self.n_states = n_states
        self.n_actions = n_actions

        self.timestep = 0

        # Now initialize R and P
        self.R = {}
        self.P = {}
        self.x0 = None #initial state

        self.max_gain = None
        self.bias = None
        self.span_bias = None
        self.pi_star = np.empty(self.n_states)

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

    def reset(self, seed=None):
        """
        Reset the environment

        Returns:
            An initial state randomly drawn from
            the initial distribution
        """
        if (seed != None):
            np.random.seed(seed)
        self.timestep = 0
        if self.x0 is not None:
            return self.x0
        else:
            x_0 = np.random.randint(0, self.n_states)
            return x_0

    def step(self, state, action, seed=None):
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
        if (seed != None):
            np.random.seed(seed)
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

    def compute_cumul_reward(self, policy, T_max , init_state=None):
        """ Compute long-term average reward """
        t = 0
        cumul_reward = []
        if (init_state == None):
            state = self.reset()
        else:
            state = init_state
        action = np.random.choice(self.n_actions, p=policy[state,:])
        while (t < T_max): #or terminal
            t += 1
            nexts, reward, term = self.step(state, action)
            cumul_reward.append(reward)
            state = nexts
            action = np.random.choice(self.n_actions, p=policy[state,:])
        return cumul_reward

    def compute_gain(self, epsilon):
        """
        :param epsilon: desired accuracy
        """
        u1 = np.zeros(self.n_states)
        u2 = np.zeros(self.n_states)
        policy_opt = np.zeros(self.n_states)

        P_samp = self.P
        R_samp = self.R
        counter = 0
        while True:
            counter += 1
            for s in range(0, self.n_states):
                first_action = True
                for a in range(self.n_actions):
                    vec = P_samp[s, a]
                    r_optimal = R_samp[s, a][0]
                    v = r_optimal + np.dot(vec, u1)
                    if first_action or v > u2[s]:  # optimal policy = argmax
                        u2[s] = v
                        policy_opt[s] = a
                    first_action = False

            if (max(u2-u1)-min(u2-u1) < epsilon or counter > 100):  # stopping condition of EVI

                max_gain = 0.5 * (max(u2 - u1) + min(u2 - u1))
                self.max_gain = max_gain
                self.pi_star = policy_opt
                self.bias = u2
                self.span_bias = max(u2) - min(u2)
                break
            else:
                u1 = u2
                u2 = np.empty(self.n_states)

    def compute_gain_constraints(self, epsilon, C=float("inf")):
        """
        :param epsilon: desired accuracy
        """
        u1 = np.empty(self.n_states)
        u2 = np.empty(self.n_states)
        policy_opt = np.zeros((self.n_states, self.n_actions))

        P_samp = self.P
        R_samp = self.R
        min_u2 = float("inf")
        counter = 0
        while True:
            counter += 1
            for s in range(0, self.n_states):
                first_action = True
                for a in range(self.n_actions):
                    vec = P_samp[s, a]
                    r_optimal = R_samp[s, a][0]
                    v = r_optimal + np.dot(vec, u1)
                    if first_action or v > u2[s]:
                        u2[s] = v
                        first_action = False

                if u2[s] < min_u2:
                    min_u2 = u2[s]

            u2 = np.clip(u2, None, min_u2 + C)
            if (max(u2-u1)-min(u2-u1) < epsilon):# or counter > 10):  # stopping condition of EVI
                #update policy
                for s in range(self.n_states):
                    span_break = False
                    a_p = -1
                    a_m = -1
                    v_p = float("inf")
                    v_m = -float("inf")
                    for a in range(self.n_actions):
                        vec = P_samp[s, a]
                        r_optimal = R_samp[s, a][0]
                        v = r_optimal + np.dot(vec, u2)

                        if v_m < v < min_u2 +C:
                            v_m = v
                            a_m = a

                        elif v_p > v > min_u2 +C:
                            span_break = True
                            v_p = v
                            a_p = a

                    policy_opt[s, :] = np.zeros(self.n_actions)
                    if not span_break:
                        policy_opt[s, :] = np.zeros(self.n_actions)
                        policy_opt[s][a_m] = 1 #truncation/interpolation not needed.
                    else:
                        q = (v_p -(min_u2 +C))/(v_p - v_m)
                        policy_opt[s][a_m] = q
                        policy_opt[s][a_p] = 1-q
                # return
                return policy_opt, max(u2) - min(u2)

            else:
                u1 = u2
                u2 = np.empty(self.n_states)
                min_u2 = float("inf")



    def check_optimality(self):
        opt_check=[]
        for s in range(self.n_states):
            qt = np.array([self.R[s,a][0] + np.dot(self.P[s,a], self.bias) - self.bias[s] for a in range(self.n_actions)])

            opt_check.append(max(qt))
        return opt_check
