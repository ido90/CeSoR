import numpy as np
import pandas as pd
from gym import core, spaces
from gym.utils import seeding
import utils

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="precision lowered by casting")


class RoomsEnv(core.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, rows=8, cols=None, action_noise=0, max_steps=None, seed=None,
                 goal_in_state=True, detailed_r=False, force_motion=False, collect=False,
                 fixed_reset=False, init_state=None, goal_state=None, continuous=True,
                 kill_prob=0):
        if cols is None: cols = rows
        if cols != rows:
            raise ValueError('Only square boards are currently supported.')
        if goal_state is None: goal_state = (-2,rows//3)
        self.rng = np.random.RandomState(seed)
        self.continuous = continuous
        self.kill_prob = kill_prob
        self.detailed_r = detailed_r
        self.rows, self.cols = rows, cols
        self.L = 2 * self.rows
        self.max_steps = 10*self.L if max_steps is None else max_steps
        self.force_motion = force_motion

        n_channels = 2 + goal_in_state
        self.action_space = spaces.Discrete(4)
        if self.continuous:
            # self.observation_space = spaces.Tuple((
            #     spaces.Box(low=0, high=1, shape=(self.rows, self.cols), dtype=np.float32),
            #     spaces.Box(low=0, high=self.rows, shape=(2,), dtype=np.float32)
            # ))
            self.observation_space = spaces.Box(
                low=np.float32(0), high=np.float32(self.rows),
                shape=(2,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(
                low=np.float32(0), high=np.float32(1),
                shape=(n_channels, self.rows, self.cols), dtype=np.float32)
        self.directions = [np.array((-1, 0)), np.array((1, 0)),
                           np.array((0, -1)), np.array((0, 1))]  # l, r, d, u

        self.goal_in_state = goal_in_state
        self.action_noise = action_noise

        self.map = self._randomize_walls()
        self.goal_cell, self.goal = self._random_from_map(goal_state)
        self.goal_cell = self.goal_cell % self.rows
        self.state_cell, self.state = self._random_from_map(init_state)
        self.state_xy = self.state_cell.copy()

        self.fixed_reset = fixed_reset
        if fixed_reset:
            self.reset_state_cell = self.state_cell.copy()
            self.reset_state = self.state.copy()
        else:
            self.reset_state_cell, self.reset_state = None, None

        self.killed = False
        self.tot_reward = 0
        self.episode_count = 0
        self._nep, self._states, self._actions, self._rewards, self._terminals = \
            [], [], [], [], []
        self.collect = collect

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.rng = np.random.RandomState(seed)
        return [seed]

    def get_obs(self):
        if self.continuous:
            return self.state_xy
        return self._im_from_state()

    def reset(self, kill_prob=None, init_state=None):
        if kill_prob is not None:
            self.kill_prob = kill_prob
        if self.fixed_reset:
            self.state_cell, self.state = self.reset_state_cell, self.reset_state
        else:
            self.state_cell, self.state = self._random_from_map(init_state)
        self.state_xy = self.state_cell.copy()
        self.state_traj = [np.reshape(self.state_xy,(1,self.state_xy.size))]
        self.nsteps = 0
        self.tot_reward = 0
        self.killed = False
        if self.collect:
            self.episode_count += 1
            self._nep.append(self.episode_count)
            self._rewards.append(self.get_reward(False))
            self._terminals.append(False)
            self._states.append(np.reshape(self.state_cell,(1,self.state_cell.size)))

        obs = self.get_obs()
        return self.kill_prob, obs

    def collect_on(self):
        self._states = []
        self._actions = []
        self._rewards = []
        self._terminals = []
        self._nep = []
        self.episode_count = 0
        self.collect = True

    def collect_off(self):
        self._states = []
        self._actions = []
        self._rewards = []
        self._terminals = []
        self._nep = []
        self.episode_count = 0
        self.collect = False

    def get_collected(self):
        return pd.DataFrame(np.concatenate((
            np.array(self._nep)[:,np.newaxis],
            np.concatenate(self._states,axis=0),
            np.array(self._actions)[:,np.newaxis],
            np.array(self._rewards)[:,np.newaxis],
            np.array(self._terminals)[:,np.newaxis],
        ), axis=1), columns=('episode','x','y','a','r','terminal'))

    def get_reward(self, done=False, moved=True, mid_cell=None):
        if self.detailed_r:
            if done:
                r = 0
            else:
                r = -np.linalg.norm(np.abs(self.state_cell - self.goal_cell), ord=1) \
                    / (self.rows + self.cols)
        else:
            r = 1*self.L if done else (
                -1 if self.nsteps<2*self.L else 0)

        if self.force_motion and not moved:
            r -= 1

        if self.kill_prob and not self.killed:
            if self.is_kill_zone() or self.is_kill_zone(mid_cell):
                if self.rng.random() < self.kill_prob:
                    r -= 4*self.L
                    self.killed = True

        return r

    def step(self, action: int):
        next_xy = self.state_xy + self.directions[action]
        if self.action_noise:
            next_xy = next_xy + self.rng.normal(0, self.action_noise, size=2)
        next_xy = np.clip(next_xy, 0, self.rows-1)
        next_cell = np.round(next_xy).astype(int)
        mid_xy = 0.5 * (self.state_xy + next_xy)
        mid_cell = np.round(mid_xy).astype(int)

        moved = False
        if self.map[next_cell[0], next_cell[1]] != 1 and \
                self.map[mid_cell[0], mid_cell[1]] != 1:
            moved = True
            self.state_xy = next_xy
            self.state_cell = next_cell
            self.state = np.zeros_like(self.map)
            self.state[self.state_cell[0], self.state_cell[1]] = 1

        self.state_traj.append(np.reshape(self.state_xy,(1,self.state_xy.size)))
        done = np.all(self.state_cell == self.goal_cell)

        obs = self.get_obs()
        r = self.get_reward(done, moved, mid_cell)

        if self.nsteps >= self.max_steps:
            done = True

        if self.collect:
            self._nep.append(self.episode_count)
            self._states.append(np.reshape(self.state_xy,(1,self.state_xy.size)))
            self._actions.append(action)
            self._rewards.append(r)
            self._terminals.append(done)
            if done:
                self._actions.append(-1)

        self.tot_reward += r
        self.nsteps += 1
        info = dict()

        if done:
            info = {'r': self.tot_reward, 'l': self.nsteps}

        return obs, r, done, info

    def get_local_map(self, rad=1):
        pad = max(0, rad - 1)
        n = self.rows + 2*pad
        padded_map = np.zeros((n, n))
        padded_map[pad:n-pad, pad:n-pad] = self.map
        x = pad + self.state_cell[0]
        y = pad + self.state_cell[1]
        return padded_map[x-rad:x+rad+1, y-rad:y+rad+1]

    def _random_from_map(self, cell=None, radius=0):
        if cell is None:
            cell = self.rng.choice(self.rows), self.rng.choice(self.cols)
            while self.map[cell[0], cell[1]] != 0:
                cell = self.rng.choice(self.rows), self.rng.choice(self.cols)
        else:
            cell = np.array(cell).astype(int)

        map = np.zeros_like(self.map)
        map[cell[0]-radius : cell[0]+radius+1,
            cell[1]-radius : cell[1]+radius+1] = 1

        return np.array(cell), map

    def _im_from_state(self, for_plot=False):
        # map
        m = self.map
        if for_plot:
            m = np.array([[0.,0.,1.][int(x)] for x in m.reshape(-1)]).reshape(m.shape)
        im_list = [m]

        # goal
        goal = self.goal if self.goal_in_state else np.zeros(m.shape)
        if self.goal_in_state or for_plot:
            im_list.append(goal)

        # state (agent)
        im_list.append(self.state)

        im = np.stack(im_list, axis=0)

        if for_plot:
            walls = np.array([[0,0.7,0][int(x)]
                              for x in self.map.reshape(-1)]).reshape(self.map.shape)
            walls = np.stack(3*[walls], axis=0)
            im += walls

        return im  #.astype(np.int8)

    def _show_state(self, ax=None, color_scale=0.6, show_traj=True, traj_col='b'):
        if ax is None: ax = utils.Axes(1, 1, grid=False)[0]
        im = self._im_from_state(for_plot=True).swapaxes(0, 2)
        ax.imshow(color_scale*im)
        ax.invert_yaxis()
        if show_traj:
            track = np.concatenate(self.state_traj)
            ax.plot(track[:, 0], track[:, 1], f'{traj_col}.-')
            ax.plot(track[:1, 0], track[:1, 1], f'{traj_col}>', markersize=12)
        if self.rows == 16:
            ticks = [0,5,10,15]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        return ax

    def is_kill_zone(self, cell=None):
        if cell is None:
            cell = self.state_cell
        return self.map[cell[0], cell[1]] == -1

    def _randomize_walls(self):
        W = self.cols
        H = self.rows
        map = np.zeros((H, W))
        # outer walls
        map[0, :] = 1
        map[:, 0] = 1
        map[-1:, :] = 1
        map[:, -1:] = 1
        # inner walls
        # map[:W//2, H//2-1:H//2+1] = 1
        map[W//4+1:3*W//4, 3*H//4-1:3*H//4] = 1
        map[3*W//4-1:3*W//4, H//4:3*H//4] = 1
        # kill zone
        if self.kill_prob:
            map[3*W//4-1:3*W//4, 1:H//4] = -1

        return map

    def render(self, mode='human', close=False):
        return 0

def _evaluate_strategies(n=8, max_cost=2, goal_val=1, kill_cost=4,
                         short_dist=1):
    pk = np.arange(0,1.01,0.01) # kill probabilities
    L = 2 * n # going from one end to the other
    max_cost *= L
    goal_val *= L

    # costs of different strategies
    stay = max_cost + 0*pk # staying out of trouble
    late_reach = max_cost -goal_val + 0*pk # random walk to the goal
    long = (4-short_dist)*(n-2)*1 - goal_val + 0*pk # long way to the goal
    greed = short_dist*(n-2)*1 - goal_val + kill_cost*L*pk # short (and risky) way

    axs = utils.Axes(1, 1)
    ax = axs[0]
    ax.plot(pk, stay, label='stay')
    ax.plot(pk, late_reach, label='late long')
    ax.plot(pk, long, label='long')
    ax.plot(pk, greed, label='greedy')
    ax.legend()
    axs.labs(0, 'p_kill', 'E[loss]')
    return ax
