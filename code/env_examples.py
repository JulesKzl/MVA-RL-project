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
    R_true[0,0] = (1.6, 0)
    P_true[0,1][2] = 1
    # R_true[0,1] = (0, 0)
    R_true[0,1] = (2, 0)
    P_true[1,0][2] = 1
    # R_true[1,0] = (2, 0)
    R_true[1,0] = (4, 0)
    P_true[1,1][0] = 0.5
    P_true[1,1][1] = 0.4
    P_true[1,1][2] = 0.1
    # R_true[1,1] = (0, 0)
    R_true[1,1] = (2, 0)
    P_true[2,0][0] = 0.6
    P_true[2,0][2] = 0.4
    # R_true[2,0] = (-1, 0)
    R_true[2,0] = (1, 0)
    P_true[2,1][1] = 0.9
    P_true[2,1][2] = 0.1
    # R_true[2,1] = (-0.5, 0)
    R_true[2,1] = (1.5, 0)

    MDP_TP1 = MDP(n_states, n_actions)
    MDP_TP1.R = R_true
    MDP_TP1.P = P_true

    MDP_TP1.pi_star = [1, 0, 1]

    return MDP_TP1


def make_riverSwim(n_states=6, time_horizon=20):
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
    R_true[0, 0] = (5. / 1000, 0)
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

    riverSwim = TabularMDP(n_states, n_actions, time_horizon)
    riverSwim.R = R_true
    riverSwim.P = P_true

    return riverSwim


def make_deterministicChain(n_states, time_horizon):
    """
    Creates a deterministic chain MDP with two actions.

    Args:
        n_states - int - number of states
        time_horizon - int - episode length

    Returns:
        chainMDP - Tabular MDP environment
    """
    n_actions = 2

    R_true = {}
    P_true = {}

    for s in range(n_states):
        for a in range(n_actions):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(n_states)

    # Rewards
    R_true[0, 0] = (0, 1)
    R_true[n_states - 1, 1] = (1, 1)

    # Transitions
    for s in range(n_states):
        P_true[s, 0][max(0, s-1)] = 1.
        P_true[s, 1][min(n_states - 1, s + 1)] = 1.

    chainMDP = TabularMDP(n_states, n_actions, time_horizon)
    chainMDP.R = R_true
    chainMDP.P = P_true

    return chainMDP


def make_stochasticChain(chain_length):
    """
    Creates a difficult stochastic chain MDP with two actions.

    Args:
        chainLen - int - total number of states

    Returns:
        chainMDP - Tabular MDP environment
    """
    n_states = chain_length
    time_horizon = chain_length
    n_actions = 2
    pNoise = 1. / chain_length

    R_true = {}
    P_true = {}

    for s in range(n_states):
        for a in range(n_actions):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(n_states)

    # Rewards
    R_true[0, 0] = (0, 1)
    R_true[n_states - 1, 1] = (1, 1)

    # Transitions
    for s in range(n_states):
        P_true[s, 0][max(0, s-1)] = 1.

        P_true[s, 1][min(n_states - 1, s + 1)] = 1. - pNoise
        P_true[s, 1][max(0, s-1)] += pNoise

    stochasticChain = TabularMDP(n_states, n_actions, time_horizon)
    stochasticChain.R = R_true
    stochasticChain.P = P_true

    return stochasticChain


def make_bootDQNChain(n_states=6, n_actions=2, time_horizon=15):
    """
    Creates the chain from Bootstrapped DQN

    Returns:
        bootDQNChain - Tabular MDP environment
    """
    R_true = {}
    P_true = {}

    for s in range(n_states):
        for a in range(n_actions):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(n_states)

    # Rewards
    R_true[0, 0] = (0.01, 1)
    R_true[n_states - 1, 1] = (1, 1)

    # Transitions
    for s in range(n_states):
        P_true[s, 0][max(0, s-1)] = 1.

        P_true[s, 1][min(n_states - 1, s + 1)] = 0.5
        P_true[s, 1][max(0, s-1)] = 0.5

    bootDQNChain = TabularMDP(n_states, n_actions, time_horizon)
    bootDQNChain.R = R_true
    bootDQNChain.P = P_true

    return bootDQNChain


def make_hardBanditMDP(time_horizon, n_actions=2, gap=0.01, pSuccess=0.5):
    """
    Creates a difficult bandit-style MDP which is hard to distinguish.

    Args:
        time_horizon - int
        gap - double - how much better is best arm
        n_actions - int - how many actions

    Returns:
        hardBanditMDP - Tabular MDP environment
    """
    n_states = 3

    R_true = {}
    P_true = {}

    for a in range(n_actions):
        # Rewards are independent of action
        R_true[0, a] = (0.5, 1)
        R_true[1, a] = (1, 0)
        R_true[2, a] = (0, 0)

        # Transitions are like a bandit
        P_true[0, a] = np.array([0, pSuccess, 1 - pSuccess])
        P_true[1, a] = np.array([0, 1, 0])
        P_true[2, a] = np.array([0, 0, 1])

    # The first action is a better action though
    P_true[0, 0] = np.array([0, pSuccess + gap, 1 - (pSuccess + gap)])

    hardBanditMDP = TabularMDP(n_states, n_actions, time_horizon)
    hardBanditMDP.R = R_true
    hardBanditMDP.P = P_true

    return hardBanditMDP


def make_stateBanditMDP(stateMul, gap=0.1):
    """
    Creates a bandit-style MDP which examines dependence on states.

    Args:
        stateMul - int
        gap - double - how much better is best arm

    Returns:
        stateBanditMDP - Tabular MDP environment
    """
    time_horizon = 2
    n_actions = 2
    n_states = 1 + 2 * stateMul

    R_true = {}
    P_true = {}

    for a in range(n_actions):
        R_true[0, a] = (0, 0)
        P_true[0, a] = np.zeros(n_states)

        for k in range(stateMul):
            for i in range(2):
                s = 1 + (2 * k) + i
                P_true[s, a] = np.zeros(n_states)
                P_true[s, a][s] = 1
                R_true[s, a] = (1-i, 0)

    # Important piece is where the transitions go
    P_true[0, 0] = np.ones(n_states) / (n_states - 1)
    P_true[0, 0][0] = 0

    # Rewarding states
    inds = (np.arange(n_states) % 2) > 0
    P_true[0, 1][inds] = (0.5 + gap) / stateMul
    P_true[0, 1][~inds] = (0.5 - gap) / stateMul
    P_true[0, 1][0] = 0

    stateBanditMDP = TabularMDP(n_states, n_actions, time_horizon)
    stateBanditMDP.R = R_true
    stateBanditMDP.P = P_true

    return stateBanditMDP

def make_confidenceMDP(stateMul, gap=0.1):
    """
    Creates a bandit-style MDP which examines dependence on states.

    Args:
        stateMul - int
        gap - double - how much better is best arm

    Returns:
        confidenceMDP - Tabular MDP environment
    """
    time_horizon = 2
    n_actions = 1
    n_states = 1 + 2 * stateMul

    R_true = {}
    P_true = {}

    for a in range(n_actions):
        R_true[0, a] = (0, 0)
        P_true[0, a] = np.zeros(n_states)

        for k in range(stateMul):
            for i in range(2):
                s = 1 + (2 * k) + i
                P_true[s, a] = np.zeros(n_states)
                P_true[s, a][s] = 1
                R_true[s, a] = (1-i, 0)

    # Important piece is where the transitions go
    P_true[0, 0] = np.ones(n_states) / (n_states - 1)
    P_true[0, 0][0] = 0

    # Rewarding states
    inds = (np.arange(n_states) % 2) > 0

    confidenceMDP = TabularMDP(n_states, n_actions, time_horizon)
    confidenceMDP.R = R_true
    confidenceMDP.P = P_true

    return confidenceMDP


def make_HconfidenceMDP(time_horizon):
    """
    Creates a H-dependence bandit confidence.

    Args:
        time_horizon - int
        gap - double - how much better is best arm
        n_actions - int - how many actions

    Returns:
        hardBanditMDP - Tabular MDP environment
    """
    n_states = 3

    R_true = {}
    P_true = {}

    # Rewards are independent of action
    R_true[0, 0] = (0.5, 0)
    R_true[1, 0] = (1, 0)
    R_true[2, 0] = (0, 0)

    # Transitions are like a bandit
    P_true[0, 0] = np.array([0, 0.5, 0.5])
    P_true[1, 0] = np.array([0, 1, 0])
    P_true[2, 0] = np.array([0, 0, 1])

    hardBanditMDP = TabularMDP(n_states, 1, time_horizon)
    hardBanditMDP.R = R_true
    hardBanditMDP.P = P_true

    return hardBanditMDP

def make_GridWold():
    grid1 = [
        ['', '', '', 1],
        ['', 'x', '', -1],
        ['', '', '', '']
    ]
    GridWorld1 = GridWorld(gamma=0.95, grid=grid1)

    return GridWorld1
