"""
This is a collection of some of the classic benchmark algorithms for efficient
reinforcement learning in a tabular MDP with little/no prior knowledge.
We provide implementations of:

- PSRL
- Gaussian PSRL
- UCBVI
- UCRL2

author: iosband@stanford.edu
"""

import numpy as np
import math as m
import time


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

        Returns:
            learner, to be inherited from
        """
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        n_states = self.n_states
        n_actions = self.n_actions
        self.r_max = float(r_max)

        self.iteration = 0

        self.estimated_P_counter = np.zeros((n_states, n_actions, n_states), dtype=np.int64)
        self.estimated_P = np.ones((n_states, n_actions, n_states)) / n_states
        self.visited_sa = set()

        self.estimated_rewards = np.ones((n_states, n_actions)) * (self.r_max + 99)
        self.estimated_holding_times = np.ones((n_states, n_actions))

        self.nb_observations = np.zeros((n_states, n_actions), dtype=np.int64)
        self.nu_k = np.zeros((n_states, n_actions), dtype=np.int64)
        self.tau = 0.9
        self.tau_max = 1
        self.tau_min = 1

        # initialize policy
        self.policy = np.zeros((n_states,), dtype=np.int_) # initial policy
        self.policy_indices = np.zeros((n_states,), dtype=np.int_)


    def update_obs(self, state, action, reward, new_state, absorb, t):
        """
        Update the posterior belief based on one transition.

        Args:
            state - int
            action - int
            reward - double
            new_state - int
            absorb - 0/1
            t - int - time within episode (not used)

        Returns:
            NULL - updates in place
        """
        mu0, tau0 = self.R_prior[state, action]
        tau1 = tau0 + self.tau
        mu1 = (mu0 * tau0 + reward * self.tau) / tau1
        self.R_prior[state, action] = (mu1, tau1)

        if (not absorb):
            self.P_prior[state, action][new_state] += 1

    def pick_action(self, state):
        """
        Use policy for action selection
        """
        action = self.policy[state]
        return action


#-----------------------------------------------------------------------------
# PSRL
#-----------------------------------------------------------------------------

class PSRL(Agent):
    """
    Posterior Sampling for Reinforcement Learning
    """
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

        return R_samp, P_samp

    def update_policy(self, h=False):
        """
        Sample a single MDP from the posterior and solve for optimal Q values.

        Works in place with no arguments.
        """
        # Sample the MDP
        R_samp, P_samp = self.sample_mdp()

        # Solve the MDP via value iteration
        qVals, qMax = self.compute_qVals(R_samp, P_samp)

        # Update the Agent's Q-values
        self.qVals = qVals
        self.qMax = qMax

#-----------------------------------------------------------------------------
# UCRL2
#-----------------------------------------------------------------------------

class UCRL2(Agent):
    """Classic benchmark optimistic algorithm"""

    def __init__(self, env, r_max):
        """
        Args:
            env
            r_max
        """
        super(UCRL2, self).__init__(env, r_max)
        self.delta = 1.
        self.rho_star = env.compute_LTAR(env.pi_star, 100000)

    def update_estimations(self, s, s2, r, absorb, t):
        """
        :param s: current state
        :param s2: new state
        :param r: reward
        :param t: timestep
        """
        curr_act_idx = self.policy_indices[s]
        scale_f = self.nb_observations[s][curr_act_idx] + self.nu_k[s][curr_act_idx]

        self.estimated_rewards[s, curr_act_idx] *= scale_f / (scale_f + 1.)
        self.estimated_rewards[s, curr_act_idx] += r / (scale_f + 1.)
        self.estimated_holding_times[s, curr_act_idx] *= scale_f / (scale_f + 1.)
        self.estimated_holding_times[s, curr_act_idx] += t / (scale_f + 1)

        if (not absorb):
            self.estimated_P_counter[s, curr_act_idx, s2] += 1
        self.visited_sa.add((s,curr_act_idx))

        self.nu_k[s][curr_act_idx] += 1
        self.iteration += 1

    def update_obs(self):
        self.nb_observations += self.nu_k

        for (s,a) in self.visited_sa:
            self.estimated_P[s,a]  = self.estimated_P_counter[s,a] / self.nb_observations[s,a]

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


    def max_proba(self, p, sorted_indices, beta):
        """
        :param p: probability distribution with toys support
        :param sorted_indices: argsort of value function
        :param beta: confidence bound on the empirical probability
        :return: optimal probability
        """
        n = np.size(sorted_indices)
        min1 = min(1, p[sorted_indices[n-1]] + beta/2)
        if min1 == 1:
            p2 = np.zeros(self.n_states)
            p2[sorted_indices[n-1]] = 1
        else:
            sorted_p = p[sorted_indices]
            support_sorted_p = np.nonzero(sorted_p)[0]
            restricted_sorted_p = sorted_p[support_sorted_p]
            support_p = sorted_indices[support_sorted_p]
            p2 = np.zeros(self.n_states)
            p2[support_p] = restricted_sorted_p
            p2[sorted_indices[n-1]] = min1
            s = 1 - p[sorted_indices[n-1]] + min1
            s2 = s
            for i, proba in enumerate(restricted_sorted_p):
                max1 = max(0, 1 - s + proba)
                s2 += (max1 - proba)
                p2[support_p[i]] = max1
                s = s2
                if s <= 1: break
        return p2

    def extended_value_iteration(self, beta_r, beta_p, epsilon):
        """
        :param beta_r: confidence bounds on rewards
        :param beta_p: confidence bounds on transition probabilities
        :param epsilon: desired accuracy
        """
        u1 = np.zeros(self.n_states)
        sorted_indices = np.arange(self.n_states)
        u2 = np.zeros(self.n_states)
        P = self.estimated_P
        counter = 0
        while True:
            counter += 1
            for s in range(0, self.n_states):
                first_action = True
                for c, a in enumerate(range(self.n_actions)):
                    vec = self.max_proba(P[s][c], sorted_indices, beta_p[s][c])
                    vec[s] -= 1
                    r_optimal = min(self.tau_max*self.r_max,
                                    self.estimated_rewards[s][c] + beta_r[s][c])
                    v = r_optimal + np.dot(vec, u1) * self.tau
                    tau_optimal = min(self.tau_max, max(max(self.tau_min, r_optimal/self.r_max),
                                  self.estimated_holding_times[s][c]))
                    if first_action or v/tau_optimal + u1[s] > u2[s] or m.isclose(v/tau_optimal + u1[s], u2[s]):  # optimal policy = argmax
                        u2[s] = v/tau_optimal + u1[s]
                        self.policy_indices[s] = c
                        self.policy[s] = a
                    first_action = False
            if (max(u2-u1)-min(u2-u1) < epsilon or counter > 10):  # stopping condition of EVI
                # print("-{}".format(counter))
                return max(u1) - min(u1), u1, u2
            else:
                u1 = u2
                u2 = np.empty(self.n_states)
                sorted_indices = np.argsort(u1)

    def update_policy(self):
        """
        Compute UCRL2 via extended value iteration.
        """
        self.delta = 1 / np.sqrt(self.iteration + 1)

        beta_r = self.beta_r()  # confidence bounds on rewards
        beta_p = self.beta_p()  # confidence bounds on transition probabilities
        epsilon = self.delta # desired accuracy

        t0 = time.time()
        span_value = self.extended_value_iteration(beta_r, beta_p, epsilon)
        t1 = time.time()
