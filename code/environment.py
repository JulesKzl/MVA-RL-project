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

    def __init__(self, n_states, n_actions, pi_star=None):
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
        self.pi_star = None

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
        if self.R[state, action][1] < 1e-9:
            reward = self.R[state, action][0]
        else:
            reward = np.random.normal(loc=self.R[state, action][0],
                                      scale=self.R[state, action][1])

        # Update the environment
        self.timestep += 1
        next_state = np.random.choice(self.n_states, p=self.P[state, action])
        absorb = False

        return next_state, reward, absorb


    def compute_LTAR(self, policy, T_max):
        """ Compute long-term average reward """
        # N = np.zeros((self.n_states, self.n_actions))
        t = 0
        cumul_reward = 0
        state = self.reset()
        action = policy[state]
        while (t < T_max): #or terminal
            t += 1
            nexts, reward, term = self.step(state, action)
            cumul_reward += reward
            state = nexts
            action = policy[state]
        return cumul_reward/T_max

    def compute_regret(self, policy, T_max):
        if (self.pi_star != None):
            E_star = self.compute_LTAR(self.pi_star, T_max)
            E = self.compute_LTAR(policy, T_max)
            return T_max*(E_star - E)
        else:
            return 0

#-------------------------------------------------------------------------------

class TabularMDP(MDP):
    """
    Tabular MDP
    Rewards are all normal.
    Transitions are multinomial.

    R - dict by (s,a) - each R[s,a] = (meanReward, sdReward)
    P - dict by (s,a) - each P[s,a] = transition vector size S
    """
    def __init__(self, n_states, n_actions, time_horizon):
        super(TabularMDP, self).__init__(n_states, n_actions)
        self.time_horizon = time_horizon

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
        next_state, reward, absorb = super(TabularMDP, self).step(state, action)
        if (self.timestep == self.time_horizon):
            next_state = self.reset()
            absorb = True
        return next_state, reward, absorb

#-------------------------------------------------------------------------------

class GridWorld(Environment):
    def __init__(self, gamma=0.95, grid=None, render=False):
        self.grid = grid

        self.action_names = np.array(['right', 'down', 'left', 'up'])

        self.n_rows, self.n_cols = len(self.grid), max(map(len, self.grid))

        # Create a map to translate coordinates [r,c] to scalar index
        # (i.e., state) and vice-versa
        self.coord2state = np.empty_like(self.grid, dtype=np.int)
        self.n_states = 0
        self.state2coord = []
        for i in range(self.n_rows):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] != 'x':
                    self.coord2state[i, j] = self.n_states
                    self.n_states += 1
                    self.state2coord.append([i, j])
                else:
                    self.coord2state[i, j] = -1

        # compute the actions available in each state
        self.compute_available_actions()
        self.gamma = gamma
        self.proba_succ = 0.9
        self.render = render
        super(GridWorld, self).__init__(n_states, 4)

    def reset(self):
        """
        Returns:
            An initial state randomly drawn from
            the initial distribution
        """
        x_0 = np.random.randint(0, self.n_states)
        return x_0

    def step(self, state, action):
        """
        Args:
            state (int): the amount of good
            action (int): the action to be executed

        Returns:
            next_state (int): the state reached by performing the action
            reward (float): a scalar value representing the immediate reward
            absorb (boolean): True if the next_state is absorsing, False otherwise
        """
        r, c = self.state2coord[state]
        assert action in self.state_actions[state]
        if isinstance(self.grid[r][c], numbers.Number):
            return state, 0, True
        else:
            failed = np.random.rand(1) > self.proba_succ
            if action == 0:
                c = min(self.n_cols - 1, c + 1) if not failed else max(0, c - 1)
            elif action == 1:
                r = min(self.n_rows - 1, r + 1) if not failed else max(0, r - 1)
            elif action == 2:
                c = max(0, c - 1) if not failed else min(self.n_cols - 1, c + 1)
            elif action == 3:
                r = max(0, r - 1) if not failed else min(self.n_rows - 1, r + 1)

            if self.grid[r][c] == 'x':
                next_state = state
                r, c = self.state2coord[next_state]
            else:
                next_state = self.coord2state[r, c]
            if isinstance(self.grid[r][c], numbers.Number):
                reward = self.grid[r][c]
                absorb = True
            else:
                reward = 0.
                absorb = False

        if self.render:
            self.show(state, action, next_state, reward)

        return next_state, reward, absorb

    def show(self, state, action, next_state, reward):
        dim = 200
        rows, cols = len(self.grid) + 0.5, max(map(len, self.grid))
        if not hasattr(self, 'window'):
            root = Tk()
            self.window = gui.GUI(root)

            self.window.config(width=cols * (dim + 12), height=rows * (dim + 12))
            my_font = tkfont.Font(family="Arial", size=32, weight="bold")
            for s in range(self.n_states):
                r, c = self.state2coord[s]
                x, y = 10 + c * (dim + 4), 10 + r * (dim + 4)
                if isinstance(self.grid[r][c], numbers.Number):
                    self.window.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                                               fill='blue', width=2)
                    self.window.create_text(x + dim / 2., y + dim / 2., text="{:.1f}".format(self.grid[r][c]),
                                            font=my_font, fill='white')
                else:
                    self.window.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                                               fill='white', width=2)
            self.window.pack()

        my_font = tkfont.Font(family="Arial", size=32, weight="bold")

        r0, c0 = self.state2coord[state]
        r0, c0 = 10 + c0 * (dim + 4), 10 + r0 * (dim + 4)
        x0, y0 = r0 + dim / 2., c0 + dim / 2.
        r1, c1 = self.state2coord[next_state]
        r1, c1 = 10 + c1 * (dim + 4), 10 + r1 * (dim + 4)
        x1, y1 = r1 + dim / 2., c1 + dim / 2.

        if hasattr(self, 'oval2'):
            # self.window.delete(self.line1)
            # self.window.delete(self.oval1)
            self.window.delete(self.oval2)
            self.window.delete(self.text1)
            self.window.delete(self.text2)

        # self.line1 = self.window.create_arc(x0, y0, x1, y1, dash=(3,5))
        # self.oval1 = self.window.create_oval(x0 - dim / 20., y0 - dim / 20., x0 + dim / 20., y0 + dim / 20., dash=(3,5))
        self.oval2 = self.window.create_oval(x1 - dim / 5., y1 - dim / 5., x1 + dim / 5., y1 + dim / 5., fill='red')
        self.text1 = self.window.create_text(dim, (rows - 0.25) * (dim + 12), font=my_font,
                                             text="r= {:.1f}".format(reward), anchor='center')
        self.text2 = self.window.create_text(2 * dim, (rows - 0.25) * (dim + 12), font=my_font,
                                             text="action: {}".format(self.action_names[action]), anchor='center')
        self.window.update()

    def compute_available_actions(self):
        # define available actions in each state
        # actions are indexed by: 0=right, 1=down, 2=left, 3=up
        self.state_actions = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if isinstance(self.grid[i][j], numbers.Number):
                    self.state_actions.append([0])
                elif self.grid[i][j] != 'x':
                    actions = [0, 1, 2, 3]
                    if i == 0:
                        actions.remove(3)
                    if j == self.n_cols - 1:
                        actions.remove(0)
                    if i == self.n_rows - 1:
                        actions.remove(1)
                    if j == 0:
                        actions.remove(2)

                    for a in copy.copy(actions):
                        r, c = i, j
                        if a == 0:
                            c = min(self.n_cols - 1, c + 1)
                        elif a == 1:
                            r = min(self.n_rows - 1, r + 1)
                        elif a == 2:
                            c = max(0, c - 1)
                        else:
                            r = max(0, r - 1)
                        if self.grid[r][c] == 'x':
                            actions.remove(a)

                    self.state_actions.append(actions)
