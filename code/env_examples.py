# coding: utf-8
""" Benchmark environments """

from environment import *

def make_MDP_TP1():
    """ MDP from TP1 RL """
    n_states = 3
    n_actions = 2

    R_true = {}
    P_true = {}

    for s in range(n_states):
        for a in range(n_actions):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(n_states)

    P_true[0,0][0] = 0.45
    P_true[0,0][2] = 0.55
    # R_true[0,0] = (-0.4, 0)
    R_true[0,0] = (16, 0)
    P_true[0,1][2] = 1
    # R_true[0,1] = (0, 0)
    R_true[0,1] = (20, 0)
    P_true[1,0][2] = 1
    # R_true[1,0] = (2, 0)
    R_true[1,0] = (40, 0)
    P_true[1,1][0] = 0.5
    P_true[1,1][1] = 0.4
    P_true[1,1][2] = 0.1
    # R_true[1,1] = (0, 0)
    R_true[1,1] = (20, 0)
    P_true[2,0][0] = 0.6
    P_true[2,0][2] = 0.4
    # R_true[2,0] = (-1, 0)
    R_true[2,0] = (10, 0)
    P_true[2,1][1] = 0.9
    P_true[2,1][2] = 0.1
    # R_true[2,1] = (-0.5, 0)
    R_true[2,1] = (15, 0)

    MDP_TP1 = MDP(n_states, n_actions)
    MDP_TP1.R = R_true
    MDP_TP1.P = P_true

    MDP_TP1.pi_star = [1, 0, 1]

    return MDP_TP1

def make_MDP_Pierre():
    """ MDP from Pierre """
    n_states = 4
    n_actions = 2

    R_true = {}
    P_true = {}

    for s in range(n_states):
        for a in range(n_actions):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(n_states)

    P_true[0,0][1] = 1
    R_true[0,0] = (1, 0)
    P_true[0,1][0] = 0.8
    P_true[0,1][1] = 0.2
    R_true[0,1] = (5, 0)

    P_true[1,0][0] = 0.4
    P_true[1,0][2] = 0.6
    R_true[1,0] = (1, 0)
    P_true[1,1][0] = 0.7
    P_true[1,1][2] = 0.3
    R_true[1,1] = (1, 0)

    P_true[2,0][3] = 0.4
    P_true[2,0][1] = 0.6
    R_true[2,0] = (1, 0)
    P_true[2,1][3] = 0.7
    P_true[2,1][1] = 0.3
    R_true[2,1] = (1, 0)

    P_true[3,0][2] = 1
    R_true[3,0] = (1, 0)
    P_true[3,1][2] = 0.2
    P_true[3,1][3] = 0.8
    R_true[3,1] = (10, 0)

    MDP_Pierre = MDP(n_states, n_actions)
    MDP_Pierre.R = R_true
    MDP_Pierre.P = P_true

    # MDP_Pierre.pi_star = [1, 0, 1]

    return MDP_Pierre


def make_riverSwim(n_states=6):
    """
    Makes the benchmark RiverSwim MDP.

    Args:
        NULL - works for default implementation

    Returns:
        riverSwim - Tabular MDP environment
    """
    n_actions = 2
    R_true = {}
    P_true = {}

    for s in range(n_states):
        for a in range(n_actions):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(n_states)

    # Rewards
    R_true[0, 0] = (0.2, 0)
    R_true[n_states - 1, 1] = (1, 0)

    # Transitions
    for s in range(n_states):
        P_true[s, 0][max(0, s-1)] = 1.

    for s in range(1, n_states - 1):
        P_true[s, 1][min(n_states - 1, s + 1)] = 0.35
        P_true[s, 1][s] = 0.6
        P_true[s, 1][max(0, s-1)] = 0.05

    P_true[0, 1][0] = 0.4
    P_true[0, 1][1] = 0.6
    P_true[n_states - 1, 1][n_states - 1] = 0.6
    P_true[n_states - 1, 1][n_states - 2] = 0.4


    riverSwim = MDP(n_states, n_actions)
    riverSwim.R = R_true
    riverSwim.P = P_true
    riverSwim.x_0 = 0
    return riverSwim

def make_StarRiverSwim(n_branch=3, n_states=6):
    n_total_states = n_branch*n_states + 1
    n_actions = 2
    R_true = {}
    P_true = {}

    # INITIALIZATION
    # For branches
    for n in range(n_branch):
        for s in range(n_states):
            for a in range(n_actions):
                R_true[1+(n*n_states)+s, a] = (0, 0)
                P_true[1+(n*n_states)+s, a] = np.zeros(n_total_states)
    # For center
    for a in range(n_actions):
        P_true[0, a] = np.zeros(n_total_states)

    # REWARDS
    # For nodes
    for n in range(n_branch):
        R_true[1+(n*n_states)+n_states - 1, 1] = (1, 0)
    # For center
    R_true[0, 0] = (5. / 1000, 0)
    R_true[0, 1] = (0, 0)

    # Transitions
    for n in range(n_branch):
        for s in range(0, n_states - 1):
            P_true[1+(n*n_states)+s, 0][1+(n*n_states)+s-1] = 1.
            P_true[1+(n*n_states)+s, 1][min(1+(n*n_states)+n_states - 1, 1+(n*n_states)+s + 1)] = 0.35
            P_true[1+(n*n_states)+s, 1][1+(n*n_states)+s] = 0.6
            P_true[1+(n*n_states)+s, 1][1+(n*n_states)+s-1] = 0.05

        P_true[1+(n*n_states)+n_states - 1, 0][1+(n*n_states)+n_states - 2] = 1.
        P_true[1+(n*n_states)+n_states - 1, 1][1+(n*n_states)+n_states - 1] = 0.6
        P_true[1+(n*n_states)+n_states - 1, 1][1+(n*n_states)+n_states - 2] = 0.4

    P_true[0, 0][0] = 1.
    P_true[0, 1][0] = 0.4
    for n in range(n_branch):
        P_true[0, 1][1+(n*n_states)] = 0.6/n_branch

    StarRiverSwim = MDP(n_total_states, n_actions)
    StarRiverSwim.R = R_true
    StarRiverSwim.P = P_true
    return StarRiverSwim

def generate_random_MDP(n_states, n_actions, max_reward):
    R_random = max_reward*np.random.rand(n_states, n_actions)

    R_true = {}
    P_true = {}

    for s in range(n_states):
        for a in range(n_actions):
            R_true[s, a] = (R_random[s, a], 0)
            P_sa_random = np.random.rand(n_states)
            P_true[s, a] = P_sa_random/np.sum(P_sa_random)

    random_MDP = MDP(n_states, n_actions)
    random_MDP.R = R_true
    random_MDP.P = P_true
    return random_MDP


def make_riverSwim(n_states=6):
    """
    Makes the benchmark RiverSwim MDP.

    Args:
        NULL - works for default implementation

    Returns:
        riverSwim - Tabular MDP environment
    """
    n_actions = 2
    R_true = {}
    P_true = {}

    for s in range(n_states):
        for a in range(n_actions):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(n_states)

    # Rewards
    R_true[0, 0] = (0.2, 0)
    R_true[n_states - 1, 1] = (1, 0)

    # Transitions
    for s in range(n_states):
        P_true[s, 0][max(0, s-1)] = 1.

    for s in range(1, n_states - 1):
        P_true[s, 1][min(n_states - 1, s + 1)] = 0.35
        P_true[s, 1][s] = 0.6
        P_true[s, 1][max(0, s-1)] = 0.05

    P_true[0, 1][0] = 0.4
    P_true[0, 1][1] = 0.6
    P_true[n_states - 1, 1][n_states - 1] = 0.6
    P_true[n_states - 1, 1][n_states - 2] = 0.4


    riverSwim = MDP(n_states, n_actions)
    riverSwim.R = R_true
    riverSwim.P = P_true
    riverSwim.x_0 = 0
    return riverSwim

def make_trap(n, R0, R, R1, delta, x0=None):
    n_states = n
    n_actions = 2
    R_true = {}
    P_true = {}

    # INITIALIZATION
    for s in range(n_states):
        for a in range(n_actions):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(n_states)

    # REWARDS
    # For nodes
    for s_inside in range(1, n_states-1): #inside nodes
        R_true[s_inside, 0] = (R, 0)
        R_true[s_inside, 1] = (R, 0)
        
        P_true[s_inside, 0][s_inside-1] = 1-delta
        P_true[s_inside, 0][s_inside] = delta  
        
        P_true[s_inside, 1][s_inside+1] = 1-delta
        P_true[s_inside, 1][s_inside] = delta
    
    # For first and last nodes
    R_true[0, 0] = (R0, 0)
    R_true[0, 1] = (R, 0)
    
    P_true[0,0][0] = 1
    P_true[0,1][0] = delta
    P_true[0,1][1] = 1-delta

    R_true[n_states-1, 0] = (R, 0)
    R_true[n_states-1, 1] = (R1, 0)

    P_true[n_states -1, 1][n_states-1] = 1
    P_true[n_states-1,0][n_states-1] = delta
    P_true[n_states-1,0][n_states-2] = 1-delta


    trap = MDP(n_states, n_actions)
    trap.R = R_true
    trap.P = P_true
    trap.x0 = x0
    return trap