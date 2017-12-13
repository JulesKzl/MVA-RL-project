import numpy as np
import matplotlib.pyplot as plt
import random

###############################################################################

class MDP:
    def __init__(self, n_states, n_actions, transition_matrix, reward_matrix,
                 gamma, pi_star):
        self.n_states = n_states
        self.n_actions = n_actions
        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix
        self.gamma = gamma
        self.pi_star = pi_star

    def R(self, state, action):
        """ Return the reward in the given state """
        return self.reward_matrix[state, action]

    def T(self, state, action):
        """Transition model.  From a state and an action, return a list
        of (result-state, probability) pairs."""
        return self.transition_matrix[state, action]

n_states = 3
n_actions = 2
transition_matrix = np.zeros((n_states, n_actions, n_states))
reward_matrix = np.zeros((n_states, n_actions))

transition_matrix[0,0,0] = 0.45
transition_matrix[0,0,2] = 0.55
# reward_matrix[0,0] = -0.4
reward_matrix[0,0] = 1.5
transition_matrix[0,1,2] = 1
# reward_matrix[0,1] = 0
reward_matrix[0,1] = 2
transition_matrix[1,0,2] = 1
# reward_matrix[1,0] = 2
reward_matrix[1,0] = 4
transition_matrix[1,1,0] = 0.5
transition_matrix[1,1,1] = 0.4
transition_matrix[1,1,2] = 0.1
# reward_matrix[1,1] = 0
reward_matrix[1,1] = 2
transition_matrix[2,0,0] = 0.6
transition_matrix[2,0,2] = 0.4
# reward_matrix[2,0] = -1
reward_matrix[2,0] = 1
transition_matrix[2,1,1] = 0.9
transition_matrix[2,1,2] = 0.1
# reward_matrix[2,1] = -0.5
reward_matrix[2,1] = 1.5

gamma = 0.95
pi_star = {0: 1, 1: 0, 2: 1}

mdp = MDP(n_states, n_actions, transition_matrix, reward_matrix, gamma, pi_star)


###############################################################################

def argmax(seq, fn):
    best = seq[0]; best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score > best_score:
            best, best_score = x, x_score
    return best

###############################################################################
def Q_function(s, a, V, mdp):
    return mdp.R(s,a) + mdp.gamma * sum([p * V[s1] for (s1, p) in enumerate(mdp.T(s, a))])

###############################################################################
### Value iteration

def policy_evaluation(pi, mdp, K):
    V = [0 for s in range(mdp.n_states)]
    for i in range(K):
        for s in range(mdp.n_states):
            V[s] = mdp.R(s, pi[s]) + mdp.gamma * sum([p * V[s1] for (s1, p) in enumerate(mdp.T(s, pi[s]))])
    return V

def value_iteration(mdp, epsilon=0.01):
    V_list = []
    V_list.append([0 for s in range(mdp.n_states)])
    K = 0
    while True:
        delta = 0
        V_list.append([0 for s in range(mdp.n_states)])
        for s in range(mdp.n_states):
            V_list[K+1][s] = max([mdp.R(s,a) + mdp.gamma * sum([p * V_list[K][s1] for (s1, p) in enumerate(mdp.T(s, a))])
                            for a in range(mdp.n_actions)])
            delta = max(delta, abs(V_list[K+1][s] - V_list[K][s]))
        K += 1
        if delta < epsilon * (1 - gamma) / (2*gamma):
            return V_list

def best_policy(mdp, V):
    """Given an MDP and a value function V, determine the best policy """
    pi = {}
    for s in range(mdp.n_states):
        pi[s] = argmax(range(mdp.n_actions), lambda a:Q_function(s, a, V, mdp))
    return pi

V_list = value_iteration(mdp)
V = V_list[-1]
print("V from value iteration:", V, "finished after", len(V_list),"iterations.")
pi = best_policy(mdp, V)
print("Best policy from value iteration:", pi)
V_star = policy_evaluation(mdp.pi_star, mdp, 100)
print("V_star from policy evaluation:", V_star)
diff_V = list(abs(np.array(V_list[:])-V_star).max(1))
plt.title('Value iteration')
plt.ylabel('||v_k - v*||')
plt.xlabel('number of iterations')
plt.plot(range(len(V_list)),diff_V)
# plt.show()

###############################################################################


def policy_iteration(mdp):
    V = [0 for s in range(mdp.n_states)]
    pi = dict([(s, random.choice(range(mdp.n_actions))) for s in range(mdp.n_states)])
    finished = False
    K = 0
    while (not finished):
        finished = True
        V = policy_evaluation(pi, mdp, 100)
        for s in range(mdp.n_states):
            a = argmax(range(mdp.n_actions), lambda a:Q_function(s, a, V, mdp))
            if a != pi[s]:
                pi[s] = a
                finished = False
        K += 1
    print("Policy iteration converge in",K, "iterations.")
    return pi

pi = policy_iteration(mdp)
print("Policy iteration", pi)
