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

class Agent:
    """
    Simple tabular Bayesian learner from Tabula Rasa.

    Child agents will mainly implement:
        update_policy

    Important internal representation is given by qVals and qMax.
        qVals - qVals[state, timestep] is vector of Q values for each action
        qMax - qMax[timestep] is the vector of optimal values at timestep

    """

    def __init__(self, n_states, n_actions, time_horizon,
                 alpha0=1., mu0=0., tau0=1., tau=1., **kwargs):
        """
        Tabular episodic learner for time-homoegenous MDP.
        Must be used together with true state feature extractor.

        Args:
            n_states - int - number of states
            n_actions - int - number of actions
            alpha0 - prior weight for uniform Dirichlet
            mu0 - prior mean rewards
            tau0 - precision of prior mean rewards
            tau - precision of reward noise

        Returns:
            tabular learner, to be inherited from
        """
        # Instantiate the Bayes learner
        self.n_states = n_states
        self.n_actions = n_actions
        self.time_horizon = time_horizon
        self.alpha0 = alpha0
        self.mu0 = mu0
        self.tau0 = tau0
        self.tau = tau

        self.qVals = {}
        self.qMax = {}

        # Now make the prior beliefs
        self.R_prior = {}
        self.P_prior = {}

        for state in xrange(n_states):
            for action in xrange(n_actions):
                self.R_prior[state, action] = (self.mu0, self.tau0)
                self.P_prior[state, action] = (
                    self.alpha0 * np.ones(self.n_states, dtype=np.float32))

    def update_obs(self, oldState, action, reward, newState, pContinue, h):
        """
        Update the posterior belief based on one transition.

        Args:
            oldState - int
            action - int
            reward - double
            newState - int
            pContinue - 0/1
            h - int - time within episode (not used)

        Returns:
            NULL - updates in place
        """
        mu0, tau0 = self.R_prior[oldState, action]
        tau1 = tau0 + self.tau
        mu1 = (mu0 * tau0 + reward * self.tau) / tau1
        self.R_prior[oldState, action] = (mu1, tau1)

        if pContinue == 1:
            self.P_prior[oldState, action][newState] += 1

    def egreedy(self, state, timestep, epsilon=0):
        """
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        """
        Q = self.qVals[state, timestep]
        n_actions = Q.size
        noise = np.random.rand()

        if noise < epsilon:
            action = np.random.choice(n_actions)
        else:
            action = np.random.choice(np.where(Q == Q.max())[0])

        return action

    def pick_action(self, state, timestep):
        """
        Default is to use egreedy for action selection
        """
        action = self.egreedy(state, timestep)
        return action

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
        for s in xrange(self.n_states):
            for a in xrange(self.n_actions):
                mu, tau = self.R_prior[s, a]
                R_samp[s, a] = mu + np.random.normal() * 1./np.sqrt(tau)
                P_samp[s, a] = np.random.dirichlet(self.P_prior[s, a])

        return R_samp, P_samp

    def map_mdp(self):
        """
        Returns the maximum a posteriori MDP from the posterior.

        Args:
            NULL

        Returns:
            R_hat - R_hat[s, a] is the MAP mean reward for (s,a)
            P_hat - P_hat[s, a] is the MAP transition vector for (s,a)
        """
        R_hat = {}
        P_hat = {}
        for s in xrange(self.n_states):
            for a in xrange(self.n_actions):
                R_hat[s, a] = self.R_prior[s, a][0]
                P_hat[s, a] = self.P_prior[s, a] / np.sum(self.P_prior[s, a])

        return R_hat, P_hat

    def compute_qVals(self, R, P):
        """
        Compute the Q values for a given R, P estimates

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        """
        qVals = {}
        qMax = {}

        qMax[self.time_horizon] = np.zeros(self.n_states, dtype=np.float32)

        for i in range(self.time_horizon):
            j = self.time_horizon - i - 1
            qMax[j] = np.zeros(self.n_states, dtype=np.float32)

            for s in range(self.n_states):
                qVals[s, j] = np.zeros(self.n_actions, dtype=np.float32)

                for a in range(self.n_actions):
                    qVals[s, j][a] = R[s, a] + np.dot(P[s, a], qMax[j + 1])

                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax

    def compute_qVals_opt(self, R, P, R_bonus, P_bonus):
        """
        Compute the Q values for a given R, P estimates + R/P bonus

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions
            R_bonus - R_bonus[s,a] = bonus for rewards
            P_bonus - P_bonus[s,a] = bonus for transitions

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        """
        qVals = {}
        qMax = {}

        qMax[self.time_horizon] = np.zeros(self.n_states, dtype=np.float32)

        for i in range(self.time_horizon):
            j = self.time_horizon - i - 1
            qMax[j] = np.zeros(self.n_states, dtype=np.float32)

            for s in range(self.n_states):
                qVals[s, j] = np.zeros(self.n_actions, dtype=np.float32)

                for a in range(self.n_actions):
                    qVals[s, j][a] = (R[s, a] + R_bonus[s, a]
                                      + np.dot(P[s, a], qMax[j + 1])
                                      + P_bonus[s, a] * i)
                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax

    def compute_qVals_EVI(self, R, P, R_slack, P_slack):
        """
        Compute the Q values for a given R, P by extended value iteration

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions
            R_slack - R_slack[s,a] = slack for rewards
            P_slack - P_slack[s,a] = slack for transitions

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        """
                # Extended value iteration
        qVals = {}
        qMax = {}
        qMax[self.time_horizon] = np.zeros(self.n_states)

        for i in range(self.time_horizon):
            j = self.time_horizon - i - 1
            qMax[j] = np.zeros(self.n_states)

            for s in range(self.n_states):
                qVals[s, j] = np.zeros(self.n_actions)

                for a in range(self.n_actions):
                    rOpt = R[s, a] + R_slack[s, a]

                    # form pOpt by extended value iteration, pInd sorts the values
                    pInd = np.argsort(qMax[j + 1])
                    pOpt = P[s, a]
                    if pOpt[pInd[self.n_states - 1]] + P_slack[s, a] * 0.5 > 1:
                        pOpt = np.zeros(self.n_states)
                        pOpt[pInd[self.n_states - 1]] = 1
                    else:
                        pOpt[pInd[self.n_states - 1]] += P_slack[s, a] * 0.5

                    # Go through all the states and get back to make pOpt a real prob
                    sLoop = 0
                    while np.sum(pOpt) > 1:
                        worst = pInd[sLoop]
                        pOpt[worst] = max(0, 1 - np.sum(pOpt) + pOpt[worst])
                        sLoop += 1

                    # Do Bellman backups with the optimistic R and P
                    qVals[s, j][a] = rOpt + np.dot(pOpt, qMax[j + 1])

                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax

#-----------------------------------------------------------------------------
# PSRL
#-----------------------------------------------------------------------------

class PSRL(Agent):
    """
    Posterior Sampling for Reinforcement Learning
    """

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
# PSRL
#-----------------------------------------------------------------------------

class PSRLunif(PSRL):
    """
    Posterior Sampling for Reinforcement Learning with spread prior
    """

    def __init__(self, n_states, n_actions, time_horizon,
                 alpha0=1., mu0=0., tau0=1., tau=1., **kwargs):
        """
        Just like PSRL but rescale alpha between successor states

        Args:
            nSamp - int - number of samples to use for optimism
        """
        newAlpha = alpha0 / n_states
        super(PSRLunif, self).__init__(n_states, n_actions, time_horizon, alpha0=newAlpha,
                                       mu0=mu0, tau0=tau0, tau=tau)

#-----------------------------------------------------------------------------
# Optimistic PSRL
#-----------------------------------------------------------------------------

class OptimisticPSRL(PSRL):
    """
    Optimistic Posterior Sampling for Reinforcement Learning
    """
    def __init__(self, n_states, n_actions, time_horizon,
                 alpha0=1., mu0=0., tau0=1., tau=1., nSamp=10, **kwargs):
        """
        Just like PSRL but we take optimistic over multiple samples

        Args:
            nSamp - int - number of samples to use for optimism
        """
        super(OptimisticPSRL, self).__init__(n_states, n_actions, time_horizon,
                                             alpha0, mu0, tau0, tau)
        self.nSamp = nSamp

    def update_policy(self):
        """
        Take multiple samples and then take the optimistic envelope.

        Works in place with no arguments.
        """
        # Sample the MDP
        R_samp, P_samp = self.sample_mdp()
        qVals, qMax = self.compute_qVals(R_samp, P_samp)
        self.qVals = qVals
        self.qMax = qMax

        for i in xrange(1, self.nSamp):
            # Do another sample and take optimistic Q-values
            R_samp, P_samp = self.sample_mdp()
            qVals, qMax = self.compute_qVals(R_samp, P_samp)

            for timestep in xrange(self.time_horizon):
                self.qMax[timestep] = np.maximum(qMax[timestep],
                                                 self.qMax[timestep])
                for state in xrange(self.n_states):
                    self.qVals[state, timestep] = np.maximum(qVals[state, timestep],
                                                             self.qVals[state, timestep])

#-----------------------------------------------------------------------------
# UCRL2
#-----------------------------------------------------------------------------

class UCRL2(Agent):
    """Classic benchmark optimistic algorithm"""

    def __init__(self, n_states, n_actions, time_horizon,
                 delta=0.05, scaling=1., **kwargs):
        """
        As per the tabular learner, but prior effect --> 0.

        Args:
            delta - double - probability scale parameter
            scaling - double - rescale default confidence sets
        """
        super(UCRL2, self).__init__(n_states, n_actions, time_horizon,
                                    alpha0=1e-5, tau0=0.0001)
        self.delta = delta
        self.scaling = scaling


    def get_slack(self, time):
        """
        Returns the slackness parameters for UCRL2

        Args:
            time - int - grows the confidence sets

        Returns:
            R_slack - R_slack[s, a] is the confidence width for UCRL2 reward
            P_slack - P_slack[s, a] is the confidence width for UCRL2 transition
        """
        R_slack = {}
        P_slack = {}
        delta = self.delta
        scaling = self.scaling
        for s in xrange(self.n_states):
            for a in xrange(self.n_actions):
                nObsR = max(self.R_prior[s, a][1] - self.tau0, 1.)
                R_slack[s, a] = scaling * np.sqrt((4 * np.log(2 * self.n_states * self.n_actions * (time + 1) / delta)) / float(nObsR))

                nObsP = max(self.P_prior[s, a].sum() - self.alpha0, 1.)
                P_slack[s, a] = scaling * np.sqrt((4 * self.n_states * np.log(2 * self.n_states * self.n_actions * (time + 1) / delta)) / float(nObsP))
        return R_slack, P_slack

    def update_policy(self, time=100):
        """
        Compute UCRL2 Q-values via extended value iteration.
        """
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Compute the slack parameters
        R_slack, P_slack = self.get_slack(time)

        # Perform extended value iteration
        qVals, qMax = self.compute_qVals_EVI(R_hat, P_hat, R_slack, P_slack)

        self.qVals = qVals
        self.qMax = qMax
