from .cython.max_proba import maxProba
from .cython.ExtendedValueIteration import extended_value_iteration
from .evi.evi import EVI
from .logging import default_logger
from . import bounds as bounds
from . import __version__ as ucrl_version

import numbers
import math as m
import numpy as np
import time


class AbstractUCRL(object):

    def __init__(self, environment,
                 r_max, random_state,
                 alpha_r=None, alpha_p=None,
                 solver=None,
                 verbose = 0,
                 logger=default_logger,
                 bound_type="chernoff"):
        self.environment = environment
        self.r_max = float(r_max)

        if alpha_r is None:
            self.alpha_r = 1
        else:
            self.alpha_r = alpha_r
        if alpha_p is None:
            self.alpha_p = 1
        else:
            self.alpha_p = alpha_p

        if solver is None:
            # create solver for optimistic model
            self.opt_solver = EVI(nb_states=self.environment.nb_states,
                                  actions_per_state=self.environment.get_state_actions(),
                                  bound_type= bound_type,
                                  random_state = random_state
                                  )
        else:
            self.opt_solver = solver

        # initialize matrices
        self.policy = np.zeros((self.environment.nb_states,), dtype=np.int_) #[0]*self.environment.nb_states  # initial policy
        self.policy_indices = np.zeros((self.environment.nb_states,), dtype=np.int_) #[0]*self.environment.nb_states

        # initialization
        self.total_reward = 0
        self.total_time = 0
        self.regret = [0]  # cumulative regret of the learning algorithm
        self.regret_unit_time = [0]
        self.unit_duration = [1]  # ratios (nb of time steps)/(nb of decision steps)
        self.span_values = []
        self.span_times = []
        self.iteration = 0
        self.episode = 0
        self.delta = 1.  # confidence
        self.bound_type = bound_type

        self.verbose = verbose
        self.logger = logger
        self.version = ucrl_version
        self.random_state = random_state

    def clear_before_pickle(self):
        del self.opt_solver
        del self.logger

    def description(self):
        desc = {
            "alpha_p": self.alpha_p,
            "alpha_r": self.alpha_r,
            "r_max": self.r_max,
            "version": self.version
        }
        return desc

class UcrlMdp(AbstractUCRL):
    """
    Implementation of Upper Confidence Reinforcement Learning (UCRL) algorithm for toys state/actions MDPs with
    positive bounded rewards.
    """

    def __init__(self, environment, r_max, alpha_r=None, alpha_p=None, solver=None,
                 bound_type="chernoff", verbose = 0,
                 logger=default_logger, random_state=None):
        """
        :param environment: an instance of any subclass of abstract class Environment which is an MDP
        :param r_max: upper bound
        :param alpha_r: multiplicative factor for the concentration bound on rewards (default is r_max)
        :param alpha_p: multiplicative factor for the concentration bound on transition probabilities (default is 1)
        """

        assert bound_type in ["chernoff",  "bernstein"]

        super(UcrlMdp, self).__init__(environment=environment,
                                      r_max=r_max, alpha_r=alpha_r,
                                      alpha_p=alpha_p, solver=solver,
                                      verbose=verbose,
                                      logger=logger, bound_type=bound_type,
                                      random_state=random_state)
        nb_states = self.environment.nb_states
        max_nb_actions = self.environment.max_nb_actions_per_state
        # self.estimated_probabilities = np.ones((nb_states, max_nb_actions, nb_states)) / nb_states
        self.P_counter = np.zeros((nb_states, max_nb_actions, nb_states), dtype=np.int64)
        self.P = np.ones((nb_states, max_nb_actions, nb_states)) / nb_states
        self.visited_sa = set()

        self.estimated_rewards = np.ones((nb_states, max_nb_actions)) * (r_max + 99)
        self.estimated_holding_times = np.ones((nb_states, max_nb_actions))

        self.nb_observations = np.zeros((nb_states, max_nb_actions), dtype=np.int64)
        self.nu_k = np.zeros((nb_states, max_nb_actions), dtype=np.int64)
        self.tau = 0.9
        self.tau_max = 1
        self.tau_min = 1

    def learn(self, duration, regret_time_step, render=False):
        """ Run UCRL on the provided environment

        Args:
            duration (int): the algorithm is run until the number of time steps
                            exceeds "duration"
            regret_time_step (int): the value of the cumulative regret is stored
                                    every "regret_time_step" time steps

        """
        if self.total_time >= duration:
            return
        threshold = self.total_time + regret_time_step
        threshold_span = threshold

        self.solver_times = []
        self.simulation_times = []

        t_star_all = time.perf_counter()
        while self.total_time < duration:
            self.episode += 1
            # print(self.total_time)

            # initialize the episode
            self.nu_k.fill(0)
            self.delta = 1 / m.sqrt(self.iteration + 1)

            if self.verbose > 0:
                self.logger.info("{}/{} = {:3.2f}%".format(self.total_time, duration, self.total_time / duration *100))

            # solve the optimistic (extended) model
            t0 = time.time()
            span_value = self.solve_optimistic_model()
            t1 = time.time()

            if render:
                self.environment.render_policy(self.policy)

            span_value *= self.tau / self.r_max
            if self.verbose > 0:
                self.logger.info("span({}): {:.9f}".format(self.episode, span_value))
                curr_regret = self.total_time * self.environment.max_gain - self.total_reward
                self.logger.info("regret: {}, {:.2f}".format(self.total_time, curr_regret))
                self.logger.info("evi time: {:.4f} s".format(t1-t0))

            if self.total_time > threshold_span:
                self.span_values.append(span_value)
                self.span_times.append(self.total_time)
                threshold_span = self.total_time + regret_time_step

            # execute the recovered policy
            t0 = time.perf_counter()
            self.visited_sa.clear()
            while self.nu_k[self.environment.state][self.policy_indices[self.environment.state]] \
                    < max(1, self.nb_observations[self.environment.state][self.policy_indices[self.environment.state]]) \
                    and self.total_time < duration:
                self.update()
                if self.total_time > threshold:
                    curr_regret = self.total_time * self.environment.max_gain - self.total_reward
                    self.regret.append(curr_regret)
                    self.regret_unit_time.append(self.total_time)
                    self.unit_duration.append(self.total_time/self.iteration)
                    threshold = self.total_time + regret_time_step
            self.nb_observations += self.nu_k

            for (s,a) in self.visited_sa:
                self.P[s,a]  = self.P_counter[s,a] / self.nb_observations[s,a]
                # assert np.sum(self.P_counter[s,a]) == self.nb_observations[s,a]
            # assert np.allclose(self.estimated_probabilities, self.P)

            t1 = time.perf_counter()
            self.simulation_times.append(t1-t0)
            if self.verbose > 0:
                self.logger.info("expl time: {:.4f} s".format(t1-t0))

        t_end_all = time.perf_counter()
        self.speed = t_end_all - t_star_all
        self.logger.info("TIME: %.5f s" % self.speed)

    def beta_r(self):
        """ Confidence bounds on the reward

        Returns:
            np.array: the vector of confidence bounds on the reward function (|S| x |A|)

        """
        S = self.environment.nb_states
        A = self.environment.max_nb_actions_per_state
        ci = bounds.chernoff(it=self.iteration, N=self.nb_observations,
                             range=self.r_max, delta=self.delta,
                             sqrt_C=3.5, log_C=2*S*A)
        # beta = self.r_max * np.sqrt(7/2 * m.log(2 * self.environment.nb_states * self.environment.max_nb_actions_per_state *
        #                                         (self.iteration+1)/self.delta) / np.maximum(1, self.nb_observations))
        # assert np.allclose(ci, beta)
        return self.alpha_r * ci

    def beta_tau(self):
        """ Confidence bounds on holding times

        Returns:
            np.array: the vecor of confidence bounds on the holding times (|S| x |A|)

        """
        return np.zeros((self.environment.nb_states, self.environment.max_nb_actions_per_state))

    def beta_p(self):
        """ Confidence bounds on transition probabilities

        Returns:
            np.array: the vector of confidence bounds on the transition matrix (|S| x |A|)

        """
        S = self.environment.nb_states
        A = self.environment.max_nb_actions_per_state
        if self.bound_type != "bernstein":
            beta = bounds.chernoff(it=self.iteration, N=self.nb_observations,
                                   range=1., delta=self.delta,
                                   sqrt_C=14*S, log_C=2*A)
            # ci = np.sqrt(14 * S * m.log(2 * A
            #         * (self.iteration + 1)/self.delta) / np.maximum(1, self.nb_observations))
            # assert np.allclose(ci, beta)
            return self.alpha_p * beta.reshape([S, A, 1])
        else:
            beta = bounds.bernstein(it=self.iteration, N=self.nb_observations,
                                    delta=self.delta, P=self.P, log_C=S,
                                    alpha_1=1., alpha_2=self.alpha_p)
            # Z = m.log(S * m.log(self.iteration + 2) / self.delta)
            # n = np.maximum(1, self.nb_observations)
            # Va = np.sqrt(2 * self.P * (1-self.P) * Z / n[:,:,np.newaxis])
            # Vb = Z * 7 / (3 * n)
            # ci = (Va + Vb[:, :, np.newaxis])
            # assert np.allclose(beta, ci)
            return beta

    def update(self):
        s = self.environment.state  # current state
        self.environment.execute(self.policy[s])
        s2 = self.environment.state  # new state
        r = self.environment.reward
        t = self.environment.holding_time

        curr_act_idx = self.policy_indices[s]
        scale_f = self.nb_observations[s][curr_act_idx] + self.nu_k[s][curr_act_idx]

        self.estimated_rewards[s, curr_act_idx] *= scale_f / (scale_f + 1.)
        self.estimated_rewards[s, curr_act_idx] += r / (scale_f + 1.)
        self.estimated_holding_times[s, curr_act_idx] *= scale_f / (scale_f + 1.)
        self.estimated_holding_times[s, curr_act_idx] += t / (scale_f + 1)

        self.estimated_probabilities[s][curr_act_idx] *= scale_f / (scale_f + 1.)
        self.estimated_probabilities[s][curr_act_idx][s2] += 1. / (scale_f + 1.)

        self.P_counter[s, curr_act_idx, s2] += 1
        self.visited_sa.add((s,curr_act_idx))

        self.nu_k[s][curr_act_idx] += 1
        self.total_reward += r
        self.total_time += t
        self.iteration += 1


    def solve_optimistic_model(self):

        beta_r = self.beta_r()  # confidence bounds on rewards
        beta_tau = self.beta_tau()  # confidence bounds on holding times
        beta_p = self.beta_p()  # confidence bounds on transition probabilities

        span_value = self.extended_value_iteration(beta_r, beta_p, beta_tau, 1 / m.sqrt(self.iteration + 1))  # python implementation: slow
        return span_value


    def extended_value_iteration(self, beta_r, beta_p, beta_tau, epsilon):
        """
        Use compiled .so file for improved speed.
        :param beta_r: confidence bounds on rewards
        :param beta_p: confidence bounds on transition probabilities
        :param beta_tau: confidence bounds on holding times
        :param epsilon: desired accuracy
        """
        u1 = np.zeros(self.environment.nb_states)
        sorted_indices = np.arange(self.environment.nb_states)
        u2 = np.zeros(self.environment.nb_states)
        p = self.estimated_probabilities
        counter = 0
        while True:
            counter += 1
            for s in range(0, self.environment.nb_states):
                first_action = True
                for c, a in enumerate(self.environment.get_available_actions_state(s)):
                    vec = self.max_proba(p[s][c], sorted_indices, beta_p[s][c])  # python implementation: slow
                    # vec = maxProba(p[s][c], sorted_indices, beta_p[s][c])  # cython implementation: fast
                    vec[s] -= 1
                    r_optimal = min(self.tau_max*self.r_max,
                                    self.estimated_rewards[s][c] + beta_r[s][c])
                    v = r_optimal + np.dot(vec, u1) * self.tau
                    tau_optimal = min(self.tau_max, max(max(self.tau_min, r_optimal/self.r_max),
                                  self.estimated_holding_times[s][c] - np.sign(v) * beta_tau[s][c]))
                    if first_action or v/tau_optimal + u1[s] > u2[s] or m.isclose(v/tau_optimal + u1[s], u2[s]):  # optimal policy = argmax
                        u2[s] = v/tau_optimal + u1[s]
                        self.policy_indices[s] = c
                        self.policy[s] = a
                    first_action = False
            if max(u2-u1)-min(u2-u1) < epsilon:  # stopping condition of EVI
                print("---{}".format(counter))
                return max(u1) - min(u1), u1, u2
            else:
                u1 = u2
                u2 = np.empty(self.environment.nb_states)
                sorted_indices = np.argsort(u1)

    def max_proba(self, p, sorted_indices, beta):
        """
        Use compiled .so file for improved speed.
        :param p: probability distribution with toys support
        :param sorted_indices: argsort of value function
        :param beta: confidence bound on the empirical probability
        :return: optimal probability
        """
        n = np.size(sorted_indices)
        min1 = min(1, p[sorted_indices[n-1]] + beta/2)
        if min1 == 1:
            p2 = np.zeros(self.environment.nb_states)
            p2[sorted_indices[n-1]] = 1
        else:
            sorted_p = p[sorted_indices]
            support_sorted_p = np.nonzero(sorted_p)[0]
            restricted_sorted_p = sorted_p[support_sorted_p]
            support_p = sorted_indices[support_sorted_p]
            p2 = np.zeros(self.environment.nb_states)
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
