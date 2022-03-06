'''
A simulator (built as a gym environment) of servers handling users requests.
The decision making is with respect to the number of active servers.
The goal is to minimize the waiting time of the users, while not paying too much
for many servers.
The number of new requests in a certain time-step is exponentially-distributed,
and its expectation is an exp. moving average (EMA) over the latent users-interest.
The users-interest is at a default rate most of the time, but peaks once in a while.

Written by Ido Greenberg, 2022.
'''

import numpy as np
import matplotlib.pyplot as plt
from gym import core, spaces
from gym.utils import seeding
import utils

MINUTE = 60
HOUR = 60*MINUTE
DAY = 24*HOUR

class ServersSim(core.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=None, max_servers=10, min_servers=3, init_servers=5,
                 T=HOUR, dt=1, act_freq=MINUTE, upload_len=2*MINUTE, req_len=1,
                 rate1=3, rate2=6, event_len=5*MINUTE, event_freq=1/(3*DAY),
                 tts_cost=1, server_cost=2, tts_on_assignment=True):
        # General
        self.rng = np.random.RandomState(seed)

        # Configuration
        self.M = max_servers
        self.M_min = min_servers
        self.m0 = init_servers
        self.dt = dt
        self.T = HOUR if T is None else T
        self.act_freq = act_freq  # decision-making turns frequency
        self.upload_len = upload_len  # time for a new server to be uploaded
        self.req_len = req_len  # average time to handle a request
        self.rate1 = rate1  # default rate of requests arrival
        self.rate2 = rate2  # peak rate of requests arrival
        self.event_len = event_len  # duration of peaks (inverse exp. decay rate)
        self.event_freq = event_freq  # prob. that a new peak starts this second
        self.tts_cost = tts_cost
        self.server_cost = server_cost
        # Whether to include the serving itself (in addition to the waiting) in
        # the tts metric. We typically ignore the serving itself (i.e., determine
        # the tts already upon task assignment), since in the simulation the
        # duration of serving itself is independent of decision-making.
        self.tts_on_assignment = tts_on_assignment

        # Constants
        # observation = (n_curr_servers, waiting requests, servers times-to-upload)
        self.observation_space = spaces.Box(
            low=0, high=np.float32(1e6), shape=(2+self.M,), dtype=np.float32)
        # action = nothing / add server / remove server
        self.action_space = spaces.Discrete(3)

        # State variables
        self.i = 0
        self.t = 0
        self.m = 3
        self.queue_len = 0
        self.time_to_active = self.upload_len * np.ones(self.M)
        self.arrival_rate = 3
        self.arrival_time_per_server = -np.ones(self.M)
        self.remaining_work = np.zeros(self.M)
        self.next_request = 0
        self.events = None

        # Recording
        self.n_servers_paid = []  # T
        self.n_servers_active = []  # T
        self.n_arrivals = []  # T
        self.n_requests_in_system = []  # T
        self.arrival_times = []  # N
        self.tts = []  # N (note: may be ordered differently than arrival_times)
        self.events_count = 0

    def reset(self, event_freq=None, n_events=None, T=None, m0=None, seed=None):
        if seed is not None:
            self.rng.seed(seed)

        # Conf
        if m0 is not None:
            self.m0 = m0
        if T is not None:
            self.T = T
        if event_freq is not None:
            self.event_freq = event_freq
        self.events = None
        if n_events is not None:
            N = int(np.ceil(self.T/self.dt))
            self.events = set(list(
                self.rng.choice(np.arange(N), n_events, replace=False)))

        # State variables
        self.i = 0
        self.t = 0
        self.m = self.m0
        self.queue_len = 0
        self.time_to_active = self.upload_len * np.ones(self.M)
        self.time_to_active[:self.m] = 0
        self.arrival_rate = 3
        self.arrival_time_per_server = -np.ones(self.M)
        self.remaining_work = np.zeros(self.M)
        self.next_request = 0

        # Recording
        self.n_servers_paid = []  # T
        self.n_servers_active = []  # T
        self.n_arrivals = []  # T
        self.n_requests_in_system = []  # T
        self.arrival_times = []  # N
        self.tts = []  # N
        self.events_count = 0

        return self.get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.rng = np.random.RandomState(seed)
        return [seed]

    def get_obs(self):
        return np.concatenate((
            [self.m, self.queue_len/self.rate1/10],
            self.time_to_active/self.upload_len))

    def get_total_cost(self, detailed=False, cvar=0):
        if cvar:
            n = self.tts
            n = int(np.ceil(cvar*n))
            tts = sorted(self.tts, reverse=True)[:n]
        else:
            tts = self.tts

        # note: tts_cost scales as n_requests but is normalized by time
        tts_cost = self.tts_cost * np.sum(tts) / self.T
        servers_cost = self.server_cost * np.sum(self.n_servers_paid) / self.T
        if detailed:
            return tts_cost, servers_cost
        return tts_cost + servers_cost

    def step(self, action):
        time_ended = self.t >= self.T

        # update servers
        if not time_ended:
            if action == 1 and self.m < self.M:
                self.add_server()
            elif action == 2 and self.m > self.M_min:
                if self.is_busy(self.m-1):
                    self.mark_for_removal()
                else:
                    self.remove_server()

        # run simulation
        reward, done = None, None
        for _ in range(self.act_freq):
            reward, done = self.inner_step()
            if done:
                break

        return self.get_obs(), reward, done, {}

    def inner_step(self):
        time_ended = self.t >= self.T

        # upload servers
        for i in range(self.m):
            if self.is_uploading(i):
                self.time_to_active[i] = max(0, self.time_to_active[i]-self.dt)

        # arrivals
        n_arrivals = 0
        if not time_ended:
            self.update_arrival_rate()
            n_arrivals = self.rng.poisson(self.arrival_rate)
            self.queue_len += n_arrivals
            self.arrival_times.extend(n_arrivals*[self.t])

        # serving
        for i in range(self.m):
            if not self.is_uploading(i):
                if not self.is_busy(i):
                    if self.queue_len > 0:
                        self.assign_task(i)
                if self.is_busy(i):
                    self.serve(i)

        # update time
        self.i += 1
        self.t = np.round(self.t + self.dt, 3)

        # update info
        n_working = np.sum([self.is_busy(i) for i in range(self.m)])
        done = time_ended and self.queue_len==0 and n_working==0
        reward = -self.get_total_cost() if done else 0
        self.n_servers_paid.append(self.m)
        self.n_servers_active.append(np.sum(self.time_to_active<=0))
        self.n_arrivals.append(n_arrivals)
        self.n_requests_in_system.append(n_working + self.queue_len)

        return reward, done

    def is_uploading(self, i):
        return i < self.m and self.time_to_active[i] > 0

    def is_busy(self, i):
        return self.arrival_time_per_server[i] >= 0

    def is_for_removal(self, i):
        return self.time_to_active[i] == -1

    def mark_for_removal(self, i=None):
        if i is None:
            i = self.m - 1
        self.time_to_active[i] = -1

    def remove_server(self, i=None):
        if i is not None and i != self.m-1:
            raise ValueError(f'{i}!={self.m-1}')
        self.time_to_active[self.m-1] = self.upload_len
        self.m -= 1

    def add_server(self):
        for i in range(self.m):
            if self.is_for_removal(i):
                self.time_to_active[i] = 0
                break
        else:
            self.m += 1

    def is_event_start(self):
        if self.events is None:
            return self.rng.random() < self.event_freq
        # TODO expensive - maybe sort self.events in advance
        #  (then only need to check the current one)
        return self.i in self.events

    def update_arrival_rate(self):
        rate = self.rate1
        if self.is_event_start():
            # Set new rate such that it is translated to a peak of height rate2
            #  in the arrival rate.
            rate = self.event_len * (self.rate2-self.rate1)
            self.events_count += 1
        self.arrival_rate = (1-1/self.event_len)*self.arrival_rate + \
                            1/self.event_len*rate

    def assign_task(self, i, time_spent=0):
        self.arrival_time_per_server[i] = self.arrival_times[self.next_request]
        self.next_request += 1
        self.queue_len -= 1
        # TODO expensive - consider doing randomization in batches
        self.remaining_work[i] = self.rng.exponential(self.req_len)
        if self.tts_on_assignment:
            # TODO expensive - consider allocating a long list in advance
            self.tts.append(self.t+time_spent - self.arrival_time_per_server[i])

    def serve(self, i, time_spent=0):
        time_left = self.dt - time_spent
        if self.remaining_work[i] < time_left:
            time_spent += self.remaining_work[i]
            # end task
            if not self.tts_on_assignment:
                self.tts.append(
                    self.t+time_spent - self.arrival_time_per_server[i])
            self.arrival_time_per_server[i] = -1

            if self.is_for_removal(i):
                self.remove_server(i)
            elif self.queue_len > 0:
                self.assign_task(i, time_spent)
                self.serve(i, time_spent)
        else:
            self.remaining_work[i] -= time_left

    def get_last_events_freq(self):
        return self.events_count / (self.T/self.dt)

    def analyze_episode(self, axs=None, a=0):
        if axs is None: axs = utils.Axes(6, 6, (3.5,2.5))

        n = len(self.tts)
        mean = np.mean(self.tts)
        cvar1 = np.mean(sorted(self.tts, reverse=True)[:int(np.ceil(0.01 * n))])
        utils.plot_quantiles(self.tts, ax=axs[a])
        axs.labs(a, 'task quantile [%]', 'tts [s]',
                 f'Mean={mean:.0f}, $CVaR_{{1\%}}$={cvar1:.0f}')
        a += 1

        axs[a].plot(np.arange(n), self.tts, '-')
        axs.labs(a, 'task id', 'tts [s]')
        a += 1

        n_smooth = 10
        T = len(self.n_arrivals)
        axs[a].plot(self.dt * np.arange(T),
                    utils.smooth(self.n_arrivals, n_smooth), '-')
        axs.labs(a, 't [s]', f'arrivals ({n_smooth}-smoothed)')
        a += 1

        axs[a].plot(self.dt * np.arange(T), self.n_servers_paid,
                    '-', label='paid')
        axs[a].plot(self.dt * np.arange(T), self.n_servers_active,
                    '-', label='active')
        axs.labs(a, 't [s]', f'servers')
        axs[a].legend(fontsize=13)
        a += 1

        mean = np.mean(self.n_requests_in_system)
        cvar1 = np.mean(sorted(self.n_requests_in_system, reverse=True)[
                        :int(np.ceil(0.01 * T))])
        axs[a].plot(self.dt * np.arange(T), self.n_requests_in_system, '-')
        axs.labs(a, 't [s]', f'tasks in system',
                 f'Mean={mean:.0f}, $CVaR_{{1\%}}$={cvar1:.0f}')
        a += 1

        tts_cost = self.tts_cost * np.sum(self.tts) / self.T
        servers_cost = self.server_cost * np.sum(self.n_servers_paid) / self.T
        axs[a].bar(('tts', 'servers'), (tts_cost, servers_cost))
        axs.labs(a, 'source', 'loss', f'{tts_cost:.0f} + {servers_cost:.0f} '
                                      f'= {tts_cost+servers_cost:.0f}')

        plt.tight_layout()
        return axs
