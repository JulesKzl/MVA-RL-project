"""
This is a collection of some of the classic benchmark algorithms for efficient
reinforcement learning
We provide implementations of:

- PSRL
- UCRL2

author: iosband@stanford.edu
Edit : Jules Kozolinsky
"""

import numpy as np
import math as m
import time

def argmax(seq, fn):
    best = seq[0]; best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score > best_score:
            best, best_score = x, x_score
    return best

class Agent:
    """
    Simple tabular Bayesian learner from Tabula Rasa.

    Child agents will mainly implement:
        update_policy

    Important internal representation is given by qVals and qMax.
        qVals - qVals[state, timestep] is vector of Q values for each action
        qMax - qMax[timestep] is the vector of optimal values at timestep

    """

    def __init__(self, env, r_max):
        """
        Learner for MDP.

        Args:
            env: environment
            alpha0 - prior weight for uniform Dirichlet
            mu0 - prior mean rewards
            tau0 - precision of prior mean rewards

        Returns:
            learner, to be inherited from
        """
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        n_states = self.n_states
        n_actions = self.n_actions
        self.r_max = float(r_max)

        self.iteration = 0
        self.delta = 1

        self.mu0 = self.r_max + 99.
        self.tau0 = 1.
        self.alpha0 = 1.

        # Now make the prior beliefs
        self.R_prior = {}
        self.P_prior = {}
        for state in range(n_states):
            for action in range(n_actions):
                self.R_prior[state, action] = (self.mu0, self.tau0)
                self.P_prior[state, action] = (
                    self.alpha0 * np.ones(self.n_states, dtype=np.float32))

        self.nb_observations = np.zeros((n_states, n_actions), dtype=np.int64)
        self.nu_k = np.zeros((n_states, n_actions), dtype=np.int64)

        # initialize policy
        self.policy = np.zeros((n_states,), dtype=np.int_) # initial policy

        self.rho_star = env.compute_LTAR(env.pi_star, 100000)

    def pick_action(self, state):
        """
        Use policy for action selection
        """
        action = self.policy[state]
        return action

    def update_estimations(self, s, s2, r, absorb, t):
        """
        :param s: current state
        :param s2: new state
        :param r: reward
        :param t: timestep
        """
        a = self.policy[s]

        scale_f = self.nb_observations[s][a] + self.nu_k[s][a]
        mu0, tau0 = self.R_prior[s, a]
        tau = tau0 / max(1, scale_f) #TODO Why?

        # mu1 = (mu0 * tau0 + r * tau) / (tau0 + tau)
        mu1 = (mu0*scale_f +r) / (scale_f + 1.)
        tau1 = tau0 + tau
        self.R_prior[s, a] = mu1, tau1

        if (not absorb):
            self.P_prior[s, a][s2] += 1

        self.nu_k[s][a] += 1
        self.iteration += 1


#-----------------------------------------------------------------------------
# PSRL
#-----------------------------------------------------------------------------

class PSRL(Agent):
    """
    Posterior Sampling for Reinforcement Learning
    """
    def value_iteration(self, P_samp, R_samp, epsilon):
        """
        :param P_samp: sampled probability
        :param R_samp: sampled rewads
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
                    vec = P_samp[s, a]
                    # vec[s] -= 1
                    r_optimal = R_samp[s, a]
                    v = r_optimal + np.dot(vec, u1)
                    if first_action or v + u1[s] > u2[s] or m.isclose(v + u1[s], u2[s]):  # optimal policy = argmax
                        u2[s] = v + u1[s]
                        self.policy[s] = a
                    first_action = False
            if (max(u2-u1)-min(u2-u1) < epsilon or counter > 10):  # stopping condition of EVI
                return max(u1) - min(u1), u1, u2
            else:
                u1 = u2
                u2 = np.empty(self.n_states)
                sorted_indices = np.argsort(u1)


    def sample_mdp(self):
        """
        Returns a single sampled MDP from the posterior.

        Args:
            NULL

        Returns:
            R_samp - R_samp[s, a] is the sampled mean reward for (s,a)
            P_samp - P_samp[s, a] is the sampled transition vector for (s,a)
        """
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
        self.delta = 1 / np.sqrt(self.iteration + 1)
        # Approximate MDP
        P_samp, R_samp = self.sample_mdp()

        # Compute optimistic policy
        epsilon = self.delta # desired accuracy
        t0 = time.time()
        span_value = self.value_iteration(P_samp, R_samp, epsilon)
        t1 = time.time()

#-----------------------------------------------------------------------------
# UCRL2
#-----------------------------------------------------------------------------

class UCRL2(Agent):
    """Classic benchmark optimistic algorithm"""

    # def __init__(self, env, r_max):
    #     """
    #     Args:
    #         env
    #         r_max
    #     """
    #     super(UCRL2, self).__init__(env, r_max)
    #

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
                    if first_action or v + u1[s] > u2[s] or m.isclose(v + u1[s], u2[s]):  # optimal policy = argmax
                        u2[s] = v + u1[s]
                        self.policy[s] = a
                    first_action = False
            if (max(u2-u1)-min(u2-u1) < epsilon or counter > 10):  # stopping condition of EVI
                return max(u1) - min(u1), u1, u2
            else:
                u1 = u2
                u2 = np.empty(self.n_states)
                sorted_indices = np.argsort(u1)

    def estimated_mdp(self):
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
        self.delta = 1 / np.sqrt(self.iteration + 1)
        # Approximate MDP
        P_hat, R_hat = self.estimated_mdp()
        P_slack, R_slack = self.beta_p(), self.beta_r()

        # Compute optimistic policy
        epsilon = self.delta # desired accuracy
        t0 = time.time()
        span_value = self.extended_value_iteration(P_hat, R_hat, P_slack, R_slack, epsilon)
        t1 = time.time()
