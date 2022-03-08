'''
Driving simulator based on:
https://rss2017.lids.mit.edu/static/papers/62.pdf
https://arxiv.org/pdf/1711.10055.pdf
https://github.com/StanfordASL/RSIRL

The agent controls a car that follows another car, and should keep
as close as possible without collision. The leader acts randomly and
independently of the follower.

Written by Ido Greenberg, 2022
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from gym import core, spaces
from gym.utils import seeding
import imageio
from PIL import Image
import utils


class DrivingSim(core.Env):
    metadata = {'render.modes': ['none', 'human']}

    def __init__(self, seed=None, T=None, leader_probs=(.3,.3,.3,.0,.1), p_oil=0,
                 rewards_conf=None, nine_actions=False, reaction_delay=7):
        # General
        self.rng = np.random.RandomState(seed)

        # Configuration
        self.dt = 0.1
        self.T = 30 if T is None else T
        self.leader_action_freq = 15
        self.agent_action_freq = 5
        self.reaction_delay = reaction_delay
        self.max_leader_acc = 1
        self.max_leader_dec = 2
        self.leader_hard_brake_dec = 6
        self.min_leader_speed = 20
        self.leader_init = np.array((0., 40., 2., 0., 0.))  # x,vx,y,vy,ay
        self.agent_init = np.array((-10., 2., 40., 0.))  # x,y,v,theta
        self.leader_probs = leader_probs
        self.p_oil = p_oil
        self.oil_factor = 2
        self.l = 3.476
        self.lag = -2.5
        if rewards_conf is None:
            # collision, forward, backward, speed, acc, lane, road
            # original values: (-,   1., 0.05, 0.1, 1.0, 0.1, 0.5)
            rewards_conf =     (5., 0.5, 0.05, 0.1, 0.5, 1.0, 0.5)
        self.rconf = rewards_conf
        self.max_deviation = 10

        # Constants
        self.observation_space = spaces.Box(
            low=np.float32(-1e6), high=np.float32(1e6),
            shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(9 if nine_actions else 5)
        self.leader_dynamics = np.array([
            [1., self.dt, 0., 0., 0.],  # x = x + dt*vx
            [0., 1., 0., 0., 0.],       # vx = vx
            [0., 0., 1., self.dt, 0.5*self.dt**2],  # y = y + dt*vy + 0.5*ay*dt^2
            [0., 0., 0., 1., self.dt],  # vy = vy + dt*ay
            [0., 0., 0., 0., 1.]        # ay = ay
        ])
        self.leader_control = np.array([  # ax, wy(=d(ay)/dt)
            [0.5*self.dt**2, 0.],
            [self.dt, 0.],
            [0., (self.dt**3)/6.],
            [0., 0.5*self.dt**2],
            [0, self.dt]
        ])
        self.permitted_leader_actions = []
        self.set_leader_actions()
        turn_power = 0.03486
        if nine_actions:
            self.agent_actions_map = np.array([
                [-6, -turn_power], [-6, 0.0], [-6, turn_power],
                [ 0, -turn_power], [ 0, 0.0], [ 0, turn_power],
                [ 4, -turn_power], [ 4, 0.0], [ 4, turn_power],
            ])
        else:
            self.agent_actions_map = np.array(
                [[-6, 0], [0, -turn_power], [0, 0], [0, turn_power], [4, 0]])

        # State variables
        self.i = 0
        self.t = 0
        self.state_of_no_return = False
        self.oil = None
        self.leader_states = None
        self.leader_actions = []
        self.agent_state = None
        self.agent_states = []
        self.agent_actions = []
        self.prev_acc = 0
        self.curr_acc = 0
        self.rewards = []
        self.full_rewards = []

        # render
        self.ax = None
        self.plots = []

    def reset(self, probs=None, leader_actions=None, L=None, init_state=None,
              p_oil=None, oil=None, seed=None):
        if seed is not None:
            self.rng.seed(seed)

        # conf
        if L is not None:
            self.T = L
        if probs is not None:
            self.leader_probs = probs
        if p_oil is not None:
            self.p_oil = p_oil
        if self.p_oil and oil is None:
            oil = [self.rng.random()<self.p_oil for _ in range(
                int(np.ceil(self.T/self.dt/self.leader_action_freq)))]
        self.oil = oil

        # init
        self.i = 0
        self.t = 0
        self.state_of_no_return = False
        self.generate_leader_traj(leader_actions)
        self.agent_state = self.agent_init.copy()
        if init_state is not None:
            self.agent_state += init_state
        self.agent_states = [self.agent_state]
        self.agent_actions = []
        self.prev_acc = 0
        self.curr_acc = 0
        self.rewards = []
        self.full_rewards = []

        # render
        self.ax = None
        self.plots = []

        return self.leader_probs, self.get_obs()

    def seed(self, seed=None):
        seed = seeding.np_random(seed)[1]
        self.rng = np.random.RandomState(seed)
        return [seed]

    def get_obs(self):
        step = max(self.i-self.reaction_delay, 0)
        leader = self.leader_states[step]
        return np.concatenate((self.agent_state, leader))

    def get_reward(self, i=None):
        if i is None: i = self.i
        log2 = np.log(2)
        y = self.agent_state[1]
        dx = self.agent_state[0] - self.leader_states[i,0]
        dy = self.agent_state[1] - self.leader_states[i,2]
        dvx = self.agent_state[2]*np.cos(self.agent_state[3]) - \
              self.leader_states[i,1]

        rc = self.rconf
        # accident
        collision = np.abs(dx)<self.l and np.abs(dy)<(self.l/2)
        r0 = collision * rc[0]
        # stay behind leader
        r1 = (dx>self.lag) * (np.log(1+np.exp(rc[1]*(dx-self.lag))) - log2 + 1.)
        # don't stay too far behind
        r2 = (dx<self.lag) * (np.log(1+np.exp(rc[2]*(self.lag-dx))) - log2)
        # keep similar speed
        r3 = np.log(1+np.exp(rc[3]*np.abs(dvx))) - log2
        # keep smooth acceleration
        r4 = rc[4] * np.sum(np.abs(self.curr_acc-self.prev_acc))
        # stay in the same lane
        r5 = np.log(1+np.exp(rc[5]*np.abs(dy))) - log2
        # stay on road
        r6 = (y> 2) * (np.log(1+np.exp( rc[6]*(y-2))) - log2) + \
             (y<-2) * (np.log(1+np.exp(-rc[6]*(y+2))) - log2)

        r = -np.array((r0,r1,r2,r3,r4,r5,r6))
        self.full_rewards.append(r)
        return np.sum(r)

    def check_status_and_get_reward(self):
        y = self.agent_state[1]
        dx = self.agent_state[0] - self.leader_states[self.i,0]
        if y<-self.max_deviation or y>self.max_deviation or dx>self.max_deviation:
            self.state_of_no_return = True
            return np.sum([self.get_reward(i) for i in
                           range(self.i, int(np.ceil(self.T/self.dt)))])
        return self.get_reward()

    def step(self, action):
        self.agent_actions.append(action)
        a = self.agent_actions_map[action].copy()
        if self.oil is not None and self.oil[int(
                self.i//self.leader_action_freq)]:
            a[0] /= self.oil_factor
        tot_r = 0
        done = False
        for j in range(self.agent_action_freq):
            x,y,v,th = self.agent_state
            v_mean = v + 0.5 * self.dt * a[0]
            a1_scaled = a[1]*(40/v)**2
            omega = v_mean/self.l * np.tan(a1_scaled)
            th_mean = th - 0.5 * self.dt * omega
            self.agent_state = np.array([
                x + self.dt * v_mean * np.cos(th_mean),
                y + self.dt * v_mean * np.sin(th_mean),
                v + self.dt * a[0],
                th - self.dt * omega,
            ])
            self.agent_states.append(self.agent_state)

            self.i += 1
            self.t = np.round(self.t + self.dt, 3)
            time_ended = self.t >= self.T

            self.prev_acc = self.curr_acc
            self.curr_acc = a[0]
            reward = self.check_status_and_get_reward()
            self.rewards.append(reward)
            tot_r += reward

            done = self.state_of_no_return or time_ended
            if done:
                break

        return self.get_obs(), tot_r, done, {}

    def set_leader_actions(self):
        turn = 4/3 * np.array((44.117, 25.21, 9.211, -3.878, -14.05,
                               -21.33, -25.69, -27.14, -25.69, -21.33,
                               -14.05, -3.878, 9.211, 25.21, 44.117))
        K = self.leader_action_freq
        a = self.max_leader_acc
        d = self.max_leader_dec
        dh = self.leader_hard_brake_dec
        T = self.leader_action_freq
        T2 = T/2
        if len(turn) != T:
            # if the turn should last other than 15 steps, interpolate its values
            h = 1/(T-1)
            turn = utils.quantile(turn, np.arange(0, 1+h/2, h))

        self.permitted_leader_actions = [
            [np.array([0, 0]) for _ in range(K)],  # do nothing
            [np.array([a*t/T2 if t<T2 else a*(1-(t-T2)/(T-T2)), 0])
             for t in range(K)],  # acc
            [np.array([-d*t/T2 if t<T2 else -d*(1-(t-T2)/(T-T2)), 0])
             for t in range(K)],  # dec
            [np.array([-dh*t/T2 if t<T2 else -dh*(1-(t-T2)/(T-T2)), 0])
             for t in range(K)],  # hard brake
            [np.array([0, -turn[t]]) for t in range(K)],  # turn left
            [np.array([0, turn[t]]) for t in range(K)],  # turn right
        ]

    def init_leader(self, dvx_scale=3):
        s = self.leader_init.copy()
        # random speed
        s[1] = self.rng.normal(s[1], dvx_scale)
        # random lane
        if self.rng.random() < 0.5:
            s[3] = -s[3]
        return s

    def generate_leader_traj(self, leader_actions=None):
        s = self.init_leader()
        on_left = False
        traj = [s]
        actions = []

        i, t = 0, 0
        while t < self.T:
            # choose leader action
            if s[1] < self.min_leader_speed:
                # accelerate
                j = 1
            else:
                if leader_actions is None:
                    j = self.rng.choice(np.arange(5), p=self.leader_probs)
                else:
                    j = leader_actions[i]
                if j==4:
                    # turn
                    if on_left:
                        j = 5
                    on_left = not on_left

            action_fun = self.permitted_leader_actions[j]
            actions.append(j)

            # update states interval
            for k in range(self.leader_action_freq):
                action = action_fun[k]
                s = np.dot(self.leader_dynamics, s) + np.dot(
                    self.leader_control, action)
                traj.append(s)
                t = np.round(t + self.dt, 3)

            s[-1] = 0  # enforce to end turn with ay=0
            i += 1

        self.leader_states, self.leader_actions = np.stack(traj), actions

    def render(self, mode='human'):
        return 0

    def get_both_cars(self, i=None):
        if i is None: i = self.i
        xl, vxl, yl, vyl = self.leader_states[i, :4]
        thl = np.arctan2(vyl, vxl)
        pl = self.get_car_points(xl, yl, thl)

        x, y, _, th = self.agent_states[i][:4]
        p = self.get_car_points(x, y, th)

        return pl, p

    def get_car_points(self, x, y, theta):
        dx, dy = self.l/2, self.l/4
        p = np.array([[-dx,dy],[dx,dy],[dx,-dy],[-dx,-dy]]).T
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        p = np.dot(R, p)
        p[0, :] += x
        p[1, :] += y
        return p

    def get_trajectory(self):
        agent = np.array(self.agent_states)
        x, y = agent[:, 0], agent[:, 1]
        xl, yl = self.leader_states[:, 0], self.leader_states[:, 2]
        return x, y, xl, yl

    def show_trajectory(self, ax=None, show_future=False):
        if ax is None:
            ax = utils.Axes(1, 1, (6, 1))[0]

        pl, p = self.get_both_cars()
        x, y, xl, yl = self.get_trajectory()
        if not show_future:
            xl, yl = xl[:self.i+1], yl[:self.i+1]

        ax.plot(xl, yl, 'r-')
        ax.plot(pl[0,:], pl[1,:], 'r.-', linewidth=2, label='leader')
        ax.plot(x, y, 'b-')
        ax.plot(p[0, :], p[1, :], 'b.-', linewidth=2, label='agent')

        ax.legend(fontsize=13, loc='center')

        return ax

    def show_rewards(self, ax=None, accumulated=False):
        if ax is None: ax = utils.Axes(1,1)[0]
        r = np.array(self.full_rewards)
        if accumulated:
            r = np.cumsum(r, axis=0)
        labs = ('no collision', 'stay behind', 'stay close', 'similar speed',
                'smooth accel', 'same lane', 'on road')
        for i in range(len(labs)):
            ax.plot(r[:,i], label=labs[i])
        utils.labels(ax, 't', 'return' if accumulated else 'reward', fontsize=15)
        ax.legend(fontsize=13)
        return ax

    def show_rewards_and_returns(self, axs, a0=0):
        if axs is None: axs = utils.Axes(2, 2)
        self.show_rewards(axs[a0+0])
        self.show_rewards(axs[a0+1], True)
        plt.tight_layout()
        return axs

    def show_frame(self, i=None, ax=None, show_info=True, lab='agent',
                   extra_cars=None):
        if i is None: i = self.i
        if ax is None: ax = utils.Axes(1, 1, (4,3.5))[0]

        # get cars
        xl, vxl, yl, vyl = self.leader_states[i, :4]
        thl = np.arctan2(vyl, vxl)
        x, y, v, th = self.agent_states[i]

        # plot cars
        ax.plot([yl], [xl], 'r.-', label='leader')
        ax.plot([y], [x], 'b.-', label=lab)
        if extra_cars is not None:
            for nm, (xx,yy,cc) in extra_cars.items():
                ax.plot([yy], [xx], cc, markersize=10, label=nm)

        l = 0.45*self.l
        img = Image.open('images/car1.png').rotate(-180/np.pi*thl)
        ax.imshow(img, extent=(yl-l/2,yl+l/2,xl-l,xl+l), zorder=3)
        img = Image.open('images/car2.png').rotate(-180/np.pi*th)
        ax.imshow(img, extent=(y-l,y+l,x-l,x+l), zorder=3)

        # focus and scale camera
        x0 = 0.5 * (x + xl)
        d = 1.2 * max(np.abs(x-xl)/2, np.abs(y-yl))
        d = max(d, 3.5*self.l)
        if extra_cars is not None:
            for nm, (xx,yy,cc) in extra_cars.items():
                d_cand = 1.2 * max(np.abs(xx-xl)/2, np.abs(yy-yl))
                if d_cand > d:
                    x0 = 0.5 * (xx + xl)
                    d = d_cand

        # plot road
        H = 2*d
        y0 = x0 - d
        road = mpl.patches.Rectangle((-4, y0), 8, H, color='gray')
        marg1 = mpl.patches.Rectangle((-5, y0), 1, H, color='tomato')
        marg2 = mpl.patches.Rectangle((4, y0), 1, H, color='tomato')
        ax.set_facecolor('palegreen')
        ax.add_patch(road)
        ax.add_patch(marg1)
        ax.add_patch(marg2)

        l = 1.5*self.l
        l2 = 2*l
        for y in (0,):
            lane0 = l2*((x0-d)//l2)
            for j in range(int(np.ceil((2*d)/l2))):
                ax.plot([y,y], [lane0+j*l2, lane0+j*l2+l], 'w-', linewidth=3)

        ax.set_ylim((x0-d, x0+d))
        ax.set_xlim((-d, d))

        # figure design
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if show_info:
            ax.set_yticks([x, xl])
            ax.set_yticklabels([f'$\Delta$={int(np.round(xl-x-self.l)):02d}m', ''],
                               fontsize=13, rotation=90,
                               verticalalignment='bottom')
            ax.set_title(f't={self.dt*i:.1f}s', fontsize=14)
        ax.legend(fontsize=12, loc='upper left')

        return ax

    def create_gif(self, frames=None, ax=None, ff=3, frame_freq=1,
                   outname='outputs/demo.gif', tmpname='outputs/tmp.png'):
        if ax is None: ax = utils.Axes(1, 1, (3.8, 3.5))[0]
        if frames is None:
            frames = []
            for i in range(self.i):
                if (i % frame_freq) == 0:
                    ax.clear()
                    self.show_frame(i, ax)
                    plt.savefig(tmpname)
                    frames.append(imageio.imread(tmpname))
        imageio.mimsave(outname, frames, duration=self.dt/ff)
        return frames

    def get_features(self):
        # raw state
        # agent:  x,y,v,theta
        # leader: x,vx,y,vy,ay
        x,y,v,th, xl,vxl,yl,vyl,ayl = self.get_obs()

        # features
        dx = x - xl
        dvx = v*np.cos(th) - vxl
        ax = self.curr_acc
        dy = y - yl

        # normalized features (to scale ~2)
        return np.array([dx/5, dvx/2, ax/2, dy/2, 10*th])
