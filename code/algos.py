# coding: utf-8
"""
Implementation of PSRL and UCRL2

Used Ian Osband and Ronan Fruit's code
Edit : Jules Kozolinsky
"""

import numpy as np
import math as m
import time

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def dict_to_matrix(dict_to_transform, n_states, n_actions):
    M = np.zeros((n_states, n_actions))
    for i in range(n_states):
        for j in range(n_actions):
            M[i, j] = dict_to_transform[i, j][0]
    return M

def print_dict_array(dict1, dict2):
    for (i, j) in dict1.keys():
        dit1_ij = dict1[i, j]/np.sum(dict1[i, j])
        print("(", i, ",", j, "):", dit1_ij, dict2[i, j])

class Agent:
    """
    Learning Agent (Childs: UCRL2 and PSRL)
    """

    def __init__(self, env, r_max, verbose=0, seed=None):
        """
        Initialize our learning agent

        :param env: RL environment
        :r_max: upper bound of environment reward
        """
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        n_states = self.n_states
        n_actions = self.n_actions
        self.r_max = float(r_max)

        self.reward_list=[]
        self.t = 1
        self.iteration = 0
        self.delta = 1

        # mu0 - prior mean rewards
        self.mu0 = self.r_max
        # tau0 - precision of prior mean rewards
        self.tau0 = 1.
        # alpha0 - prior weight for uniform Dirichlet
        self.alpha0 = 0.1

        # Now make the prior beliefs
        self.R_prior = {}
        self.P_prior = {}
        for state in range(n_states):
            for action in range(n_actions):
                self.R_prior[state, action] = (self.mu0, self.tau0)
                self.P_prior[state, action] = (
                    self.alpha0 * np.ones(self.n_states, dtype=np.float32))

        # Keep trace of number of observations
        self.nb_observations = np.zeros((n_states, n_actions), dtype=np.int64)
        self.nu_k = np.zeros((n_states, n_actions), dtype=np.int64)

        # initialize policy
        self.policy = np.zeros((n_states, n_actions))
        self.policy_opt = []
        self.verbose = verbose
        self.seed = seed

    def pick_action(self, state):
        """
        Use policy for action selection

        :param state: current state in which we want to know the corresponding
        action
        :return: action from policy
        """
        if (self.seed != None):
            seed = self.seed + 100*self.t
            np.random.seed(seed)
        action = np.random.choice(self.n_actions, p=self.policy[state,:])
        return action

    def update_models(self, s, a, s2, r, absorb):
        """
        Update model during execution of policy

        :param s: current state
        :param s2: new state
        :param r: reward
        """
        scale_f = self.nb_observations[s][a] + self.nu_k[s][a]
        mu0, tau0 = self.R_prior[s, a]
        tau = tau0 / max(1, scale_f) #TODO Why?

        # mu1 = (mu0 * tau0 + r * tau) / (tau0 + tau)
        mu1 = (mu0*scale_f +r) / (scale_f + 1.)
        tau1 = tau0 + tau
        self.R_prior[s, a] = mu1, tau1

        if (not absorb):
            # print("Update P_prior:", s, a, "->", s2)
            self.P_prior[s, a][s2] += 1

        self.nu_k[s][a] += 1
        self.iteration += 1

    def execute_policy(self, env, T_max):
        """
        Execute policy in the environment during max time of max_duration

        :param env: given RL environment
        :param max_duration: maximum execution time
        """
        #print(self.policy[0], self.policy[self.n_states-1])
        if (self.verbose > 0):
            print("> Execute policy.")
        time0 = time.time()

        # Initialize env
        seed = self.seed
        if (seed != None):
            seed += 100*self.t
        state = env.reset(seed)
        action = self.pick_action(state)

        while (self.nu_k[state][action] < max(1, self.nb_observations[state][action]) and self.t<T_max):
            self.t += 1 # Avoid infinite loop
            if (seed != None):
                seed = self.seed + 100*self.t
            # Step through the episode
            new_state, reward, absorb = env.step(state, action, seed)
            self.reward_list.append(reward)
            # Update estimations at each step
            self.update_models(state, action, new_state, reward, absorb)
            state = new_state

            # Select next action
            action = self.pick_action(state)

        # Update nb of observations
        self.nb_observations += self.nu_k
        self.nu_k.fill(0)

        if (self.verbose > 1):
            print(" took", time.time() - time0, "s.")
        if (self.verbose > 0):
            print(" in", t, "iterations.")
        if (self.verbose > 2):
            print("|R_prior - R|:")
            R_prior_mat = dict_to_matrix(self.R_prior, self.n_states, self.n_actions)
            R_mat = dict_to_matrix(env.R, self.n_states, self.n_actions)
            print(np.array(R_prior_mat))
            print(np.array(R_mat))
            print("|P_prior - P|:")
            print_dict_array(self.P_prior, env.P)

    def value_iteration(self, P_samp, R_samp, epsilon):
        """
        :param P_samp: sampled probability
        :param R_samp: sampled rewads
        :param epsilon: desired accuracy
        """
        self.policy = np.zeros((self.n_states, self.n_actions))
        u1 = np.zeros(self.n_states)
        u2 = np.zeros(self.n_states)
        counter = 0
        while True:
            counter += 1
            for s in range(0, self.n_states):
                first_action = True
                for a in range(self.n_actions):
                    vec = P_samp[s, a]
                    r_optimal = R_samp[s, a]
                    v = r_optimal + np.dot(vec, u1)
                    if first_action or v > u2[s]:  # optimal policy = argmax
                        u2[s] = v
                        self.policy[s, :] = np.zeros(self.n_actions)
                        self.policy[s][a] = 1
                    first_action = False
            if (max(u2-u1)-min(u2-u1) < epsilon):  # stopping condition of EVI
                return max(u1) - min(u1), u1, u2
            else:
                u1 = u2
                u2 = np.empty(self.n_states)

    def run(self, env, T_max):
        if (self.seed != None):
            np.random.seed(self.seed)
        while(self.t < T_max):
            self.update_policy()
            self.execute_policy(env, T_max)

    def compute_regret(self, env):
        T = len(self.reward_list)
        cumul_reward = np.cumsum(self.reward_list)
        gain = env.max_gain*np.arange(1,T+1)
        return (gain - cumul_reward)


#-----------------------------------------------------------------------------
# PSRL
#-----------------------------------------------------------------------------

class PSRL(Agent):
    """
    Posterior Sampling for Reinforcement Learning
    """
    def __init__(self, env, r_max, verbose=0, C=None, seed=None):
        super(PSRL, self).__init__(env, r_max, verbose, seed=seed)
        self.name = "PSRL"
        self.C = C

    def value_iteration_modified(self, P_samp, R_samp, epsilon, C):
            """
            Implement value_iteration with modified Bellman Operator that converges to a solution respecting the constraint on the
            span of the bias vector

            :param P_samp: sampled probability
            :param R_samp: sampled rewads
            :param epsilon: desired accuracy
            """
            self.policy = np.zeros((self.n_states, self.n_actions))
            u1 = np.empty(self.n_states)
            u2 = np.empty(self.n_states)
            min_u2 = float("inf")
            counter = 0
            while True:
                counter += 1
                for s in range(0, self.n_states):
                    first_action = True
                    for a in range(self.n_actions):
                        vec = P_samp[s, a]
                        r_optimal = R_samp[s, a]
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
                            r_optimal = R_samp[s, a]
                            v = r_optimal + np.dot(vec, u2)

                            if v_m < v < min_u2 +C:
                                v_m = v
                                a_m = a

                            elif v_p > v > min_u2 +C:
                                span_break = True
                                v_p = v
                                a_p = a

                        self.policy[s, :] = np.zeros(self.n_actions)
                        if not span_break:
                            self.policy[s, :] = np.zeros(self.n_actions)
                            self.policy[s][a_m] = 1 #truncation/interpolation not needed.
                        else:
                            q = (v_p -(min_u2 +C))/(v_p - v_m)
                            self.policy[s][a_m] = q
                            self.policy[s][a_p] = 1-q
                    # return
                    return max(u2) - min(u2), u1, u2

                else:
                    u1 = u2
                    u2 = np.empty(self.n_states)
                    min_u2 = float("inf")

    def sample_mdp(self):
        """
        Returns a single sampled MDP from the posterior.

        Returns:
            R_samp - R_samp[s, a] is the sampled mean reward for (s,a)
            P_samp - P_samp[s, a] is the sampled transition vector for (s,a)
        """
        if (self.seed != None):
            seed = self.seed + 100*self.t
            np.random.seed(seed)
        R_samp = {}
        P_samp = {}
        for s in range(self.n_states):
            for a in range(self.n_actions):
                mu, tau = self.R_prior[s, a]
                R_samp[s, a] = mu + np.random.normal() * 1./np.sqrt(tau)
                P_samp[s, a] = np.random.dirichlet(self.P_prior[s, a])
        return P_samp, R_samp

    def update_policy(self):
        """
        Compute PSRL via value iteration.
        """
        if (self.verbose > 0):
            print("> Update policy.")
        time0 = time.time()
        self.delta = 1 / np.sqrt(self.iteration + 1)
        # Approximate MDP
        P_samp, R_samp = self.sample_mdp()

        #Augmentation
        P_samp_augm, R_samp_augm = self.augment_MDP(P_samp, R_samp)

        # Compute optimistic policy
        epsilon = self.delta # desired accuracy
        if (self.C == None):
            span_value = self.value_iteration(P_samp_augm, R_samp_augm, epsilon)
            #span_value = self.value_iteration(P_samp, R_samp, epsilon)
        else:
            span_value = self.value_iteration_modified(P_samp_augm, R_samp_augm, epsilon, self.C)
            #span_value = self.value_iteration_modified(P_samp, R_samp, epsilon, self.C)

        # Desaugment
        self.policy = self.transform_policy(self.policy)

        if (self.verbose > 1):
            print(" took", time.time() - time0, "s.")
        if (self.verbose > 0):
            print(" -> New policy:", self.policy)

    def augment_MDP(self, P, R):
        """ Transform MDP into augmented MDP """
        n_actions_augm = self.n_actions*2
        R_augm = {}
        P_augm = {}
        # Copy of MDP
        for s in range(self.n_states):
            for a in range(self.n_actions):
                R_augm[s, a] = R[s, a]
                P_augm[s, a] = P[s, a]
                R_augm[s, a+self.n_actions] = 0
                P_augm[s, a+self.n_actions] = P[s, a]
        self.n_actions = self.n_actions*2
        return P_augm, R_augm

    def transform_policy(self, policy_augm):
        """ Transform a policy from the augmented MDP to a policy of MDP """
        self.n_actions = int(self.n_actions/2)
        policy = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    policy[s][a] = policy_augm[s][a] + policy_augm[s][self.n_actions+a]
        return policy

#-----------------------------------------------------------------------------
# UCRL2
#-----------------------------------------------------------------------------

class UCRL2(Agent):
    def __init__(self, env, r_max, verbose=0):
        super(UCRL2, self).__init__(env, r_max, verbose)
        self.name = "UCRL2"

    def chernoff(self, it, N, delta, sqrt_C, log_C, range=1.):
        ci = range * np.sqrt(sqrt_C * m.log(log_C * (it + 1) / delta) / np.maximum(1,N))
        return ci

    def beta_r(self):
        """ Confidence bounds on the reward

        Returns:
            np.array: the vector of confidence bounds on the reward function (|S| x |A|)

        """
        S = self.n_states
        A = self.n_actions
        beta = self.chernoff(it=self.iteration, N=self.nb_observations,
                             range=self.r_max, delta=self.delta,
                             sqrt_C=3.5, log_C=2*S*A)
        return beta

    def beta_p(self):
        """ Confidence bounds on transition probabilities

        Returns:
            np.array: the vector of confidence bounds on the transition matrix (|S| x |A|)

        """
        S = self.n_states
        A = self.n_actions
        beta = self.chernoff(it=self.iteration, N=self.nb_observations,
                               range=1., delta=self.delta,
                               sqrt_C=14*S, log_C=2*A)
        return beta.reshape([S, A, 1])

    def max_proba(self, P_hat, P_slack, sorted_indices):
        """
        :param P_hat: probability distribution with toys support
        :param P_slack: confidence bound on the empirical probability
        :param sorted_indices: argsort of value function
        :return: optimal probability
        """
        n = np.size(sorted_indices)
        min1 = min(1, P_hat[sorted_indices[n-1]] + P_slack/2)
        if min1 == 1:
            p2 = np.zeros(self.n_states)
            p2[sorted_indices[n-1]] = 1
        else:
            sorted_p = P_hat[sorted_indices]
            support_sorted_p = np.nonzero(sorted_p)[0]
            restricted_sorted_p = sorted_p[support_sorted_p]
            support_p = sorted_indices[support_sorted_p]
            p2 = np.zeros(self.n_states)
            p2[support_p] = restricted_sorted_p
            p2[sorted_indices[n-1]] = min1
            s = 1 - P_hat[sorted_indices[n-1]] + min1
            s2 = s
            for i, proba in enumerate(restricted_sorted_p):
                max1 = max(0, 1 - s + proba)
                s2 += (max1 - proba)
                p2[support_p[i]] = max1
                s = s2
                if s <= 1: break
        return p2

    def extended_value_iteration(self, P_hat, R_hat, P_slack, R_slack, epsilon):
        """
        :param P_hat: estimated probability
        :param R_hat: estimated rewads
        :param P_slack: confidence bounds on rewards
        :param R_slack: confidence bounds on transition probabilities
        :param epsilon: desired accuracy
        """
        u1 = np.zeros(self.n_states)
        sorted_indices = np.arange(self.n_states)
        u2 = np.zeros(self.n_states)
        counter = 0
        while True:
            counter += 1
            for s in range(0, self.n_states):
                first_action = True
                for a in range(self.n_actions):
                    vec = self.max_proba(P_hat[s, a], P_slack[s][a], sorted_indices)
                    vec[s] -= 1
                    r_optimal = R_hat[s, a] + R_slack[s][a]
                    v = r_optimal + np.dot(vec, u1)
                    if first_action or v + u1[s] > u2[s]:  # optimal policy = argmax
                        u2[s] = v + u1[s]
                        self.policy[s, :] = np.zeros(self.n_actions)
                        self.policy[s, a] = 1
                    first_action = False
            if (max(u2-u1)-min(u2-u1) < epsilon or counter > 10):  # stopping condition of EVI
                return max(u1) - min(u1), u1, u2
            else:
                u1 = u2
                u2 = np.empty(self.n_states)
                sorted_indices = np.argsort(u1)

    def estimated_mdp(self):
        """
        Returns estimated MDP from the prior.
        """
        R_hat = {}
        P_hat = {}
        for s in range(self.n_states):
            for a in range(self.n_actions):
                R_hat[s, a] = self.R_prior[s, a][0]
                P_hat[s, a] = self.P_prior[s, a] / np.sum(self.P_prior[s, a])
        return P_hat, R_hat

    def update_policy(self):
        """
        Compute UCRL2 via extended value iteration.
        """
        if (self.verbose > 0):
            print("> Update policy.")
        time0 = time.time()
        self.delta = 1 / np.sqrt(self.iteration + 1)
        # Approximate MDP
        P_hat, R_hat = self.estimated_mdp()
        P_slack, R_slack = self.beta_p(), self.beta_r()

        # Compute optimistic policy
        epsilon = self.delta # desired accuracy
        span_value = self.extended_value_iteration(P_hat, R_hat, P_slack, R_slack, epsilon)
        if (self.verbose > 1):
            print(" took", time.time() - time0, "s.")
        if (self.verbose > 0):
            print(" -> New policy:", self.policy)
