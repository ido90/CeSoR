'''

'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import multiprocessing as mp
import torch
import torch.optim as optim
import gym

import Agents, Optim, CEM
import utils

STATE_DIM = dict(xy=2, local_map=9, both=11, mid_map=25, mid_map2=27)

class Experiment:

    ###############   INITIALIZATION & SETUP   ###############

    def __init__(self, agents=None, train_episodes=500, valid_episodes=20,
                 test_episodes=100, global_seed=0, maze_size=16, max_episode_steps=None,
                 detailed_rewards=False, valid_freq=10, save_best_model=True,
                 kill_prob=0.05, beta_kill_dist=True, clip_factor=None,
                 action_noise=0.2,
                 optimizer=optim.Adam, optim_freq=100, episodic_loss=True,
                 cvar=1, normalize_returns=True, max_distributed=0,
                 gamma=1.0, lr=1e-2, weight_decay=0.0, state_mode='xy',
                 use_ce=False, ce_perc=0.2, ce_source_perc=0, ce_dyn=True, ce_s0=False,
                 ce_IS=True, ce_ref=False, log_freq=1000, T0=1, Tgamma=1, title=''):
        self.title = title
        self.global_seed = global_seed
        np.random.seed(self.global_seed)
        torch.manual_seed(self.global_seed)

        self.n_train = train_episodes
        self.n_valid = valid_episodes
        self.n_test = test_episodes
        self.maze_size = maze_size
        self.max_episode_steps = max_episode_steps
        self.state_mode = state_mode
        self.detailed_rewards = detailed_rewards

        self.T0 = T0  # initial train temperature
        self.Tgamma = Tgamma  # temperature decay
        self.temp = self.T0
        self.valid_freq = valid_freq
        self.save_best_model = save_best_model

        self.dynamics = np.array([kill_prob])
        self.clip_factor = None if clip_factor is None else np.array(clip_factor)
        # If true - model killing probability using Beta(a,2-a), whose mean
        # is kill_prob=a/2 and variance is a*(2-a)/12.
        # https://en.wikipedia.org/wiki/Beta_distribution
        self.use_beta_dist = beta_kill_dist
        self.action_noise = action_noise

        self.optimizer_constructor = optimizer
        self.optim_freq = optim_freq
        self.episodic_loss = episodic_loss
        self.cvar = cvar
        self.normalize_returns = normalize_returns
        self.gamma = gamma # note: 0.98^100=13%, 0.99^200=13%
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_freq = log_freq
        self.max_distributed = int(max_distributed)

        self.env = None
        self.dd = pd.DataFrame()
        self.episode_map = None
        self.test_episodes_params = None
        self.agents = None
        self.agents_names = None
        self.last_train_success = {}
        self.valid_scores = {}
        self.samples_usage = {}
        # https://en.wikipedia.org/wiki/Effective_sample_size#Weighted_samples
        self.eff_samples_usage = {}

        self.ce = CEM.CEM(dyn_dist=self.dynamics, modify_dyn=ce_dyn, modify_s0=ce_s0,
                          update_freq=use_ce*self.optim_freq, update_perc=ce_perc,
                          source_sample_perc=ce_source_perc, IS=ce_IS,
                          valid_ref=ce_ref, mode='Rooms')
        self.ce_history = {}
        self.ce_ws = {}

        self.make_env()
        self.generate_test_episodes()
        if agents is not None:
            self.set_agents(agents, generate_tables=True)

    def register_env(self):
        # first unregister if needed
        if 'RoomsEnv-v0' in gym.envs.registry.env_specs:
            del gym.envs.registry.env_specs['RoomsEnv-v0']
        gym.envs.registration.register(
            id='RoomsEnv-v0',
            entry_point='rooms:RoomsEnv',
            kwargs=dict(
                kill_prob=self.dynamics[0], action_noise=self.action_noise,
                rows=self.maze_size, max_steps=self.max_episode_steps,
                detailed_r=self.detailed_rewards),
        )

    def make_env(self):
        self.register_env()
        self.env = gym.make('RoomsEnv-v0')

    def init_dd(self, n=1, agent='', group='', episode0=0, ag_updates=-1, ag_hash='',
                temperature=0, inplace=True, reset=False, update_map=True):
        dd = pd.DataFrame(dict(
            agent          = (n * [agent]) if isinstance(agent, str) else agent,
            group          = (n * [group]) if isinstance(group, str) else group,
            episode        = np.arange(episode0, episode0+n),
            ag_updates     = n * [ag_updates],
            ag_hash        = n * [ag_hash],
            ag_temperature = n * [temperature],
            dyn_kp         = n * [np.nan],
            s0_x           = n * [np.nan],
            s0_y           = n * [np.nan],
            log_prob       = n * [np.nan],
            ret_loss       = n * [np.nan],
            score          = n * [np.nan],
        ))
        if inplace:
            if reset:
                self.dd = dd
            else:
                self.dd = pd.concat((self.dd, dd))
                self.dd.reset_index(drop=True, inplace=True)
            if update_map:
                self.build_episode_map()
        return dd

    def build_episode_map(self):
        get_key = lambda i: (self.dd.agent[i], self.dd.group[i], self.dd.episode[i])
        self.episode_map = {get_key(i):i for i in range(len(self.dd))}

    def draw_episode_params(self, dyn_dist=None, s0_dist=None,
                            dyn_clip=None, seed=None):
        if seed is not None: np.random.seed(seed)
        if s0_dist is None: s0_dist = (0.6,0.6,6.6,6.6) if self.maze_size>14 \
            else (0.6,0.6,3.6,3.6)
        if dyn_dist is None: dyn_dist = self.dynamics
        if dyn_clip is None: dyn_clip = self.clip_factor
        dyn_dist = np.array(dyn_dist)

        if self.use_beta_dist:
            dyn = np.random.beta(2*dyn_dist, 2-2*dyn_dist)
        else:
            dyn = np.random.exponential(np.array(dyn_dist), size=(len(dyn_dist),))
        if dyn_clip:
            dyn = np.clip(dyn, self.dynamics / dyn_clip, dyn_clip * self.dynamics)

        a = np.array(s0_dist[:2])
        b = np.array(s0_dist[2:])
        state = a + (b - a) * np.random.random(size=(2,))

        return dyn, state

    def generate_test_episodes(self):
        '''Draw dynamics & initial states for the test (and validation) episodes.'''
        n = self.n_valid + self.n_test
        dynamics, states = [], []
        for i in range(n):
            d, s = self.draw_episode_params()
            dynamics.append(d)
            states.append(s)
        self.test_episodes_params = np.array(dynamics), np.array(states)

    def generate_train_dd(self, agents_names=None):
        if agents_names is None: agents_names = self.agents_names
        if isinstance(agents_names, str): agents_names = [agents_names]
        dd_base = self.init_dd(self.n_train, temperature=1, group='train',
                               inplace=False)

        for agent_nm in agents_names:
            dd = dd_base.copy()
            dd['agent'] = agent_nm
            self.dd = pd.concat((self.dd, dd))
        self.dd.reset_index(drop=True, inplace=True)
        self.build_episode_map()

    def generate_test_dd(self, agents_names=None):
        if agents_names is None: agents_names = self.agents_names
        if isinstance(agents_names, str): agents_names = [agents_names]
        n = self.n_valid + self.n_test
        dd_base = self.init_dd(
            n,
            temperature=0,  # valid/test -> deterministic randomization
            group=self.n_valid * ['valid'] + self.n_test * ['test'],
            inplace=False
        )
        dd_base.iloc[:,6:7] = self.test_episodes_params[0]  # dynamics
        dd_base.iloc[:,7:9] = self.test_episodes_params[1]  # init states

        for agent_nm in agents_names:
            dd = dd_base.copy()
            dd['agent'] = agent_nm
            self.dd = pd.concat((self.dd, dd))
        self.dd.reset_index(drop=True, inplace=True)
        self.build_episode_map()

    def set_agents(self, agents, generate_tables=True):
        if not isinstance(agents, (tuple, list, dict)):
            agents = [agents]

        if isinstance(agents, dict):
            # dict of agents confs
            self.agents_names = []
            self.agents = {}
            for nm, (const, args) in agents.items():
                args['state_dim'] = self.maze_size if self.state_mode=='full' \
                    else STATE_DIM[self.state_mode]
                args['act_dim'] = 4
                self.agents_names.append(nm)
                a = const(**args)
                a.title = nm
                self.agents[nm] = a
        else:
            # assuming list (or another iterable) of actual agents
            self.agents_names = [a.title for a in agents]
            self.agents = {a.title:a for a in agents}

        self.last_train_success = {a:True for a in self.agents_names}

        if generate_tables:
            self.generate_train_dd()
            self.generate_test_dd()

    def add_agent(self, agent, title=None, generate_tables=True):
        '''agent = agent OR (constructor, kwargs)'''
        if isinstance(agent, (tuple,list)): agent = agent[0](**agent[1])
        if title is None:
            title = agent.title
        else:
            agent.title = title
        self.agents_names.append(title)
        self.agents[title] = agent
        if generate_tables:
            self.generate_train_dd(title)
            self.generate_test_dd(title)

    ###############   RUN ENV   ###############

    def init_episode(self, episode, update=False):
        # get params if already known
        dyn = self.dd.iloc[episode, 6:7].values.astype(np.float)
        s0 = self.dd.iloc[episode, 7:9].values.astype(np.float)
        if np.any(np.isnan(dyn)): dyn = None
        if np.any(np.isnan(s0)): s0 = None

        # reset env with params
        out_dyn, out_s0 = self.env.reset(kill_prob=dyn[0], init_state=s0)

        # save params if were not set before
        if update:
            if dyn is None:
                self.dd.iloc[episode, 6:7] = out_dyn
            if s0 is None:
                self.dd.iloc[episode, 7:9] = out_s0

        return out_s0

    def run_episode(self, episode, agent, render=False, update_res=False,
                    temperature=None, verbose=0, ax=None, **kwargs):
        if isinstance(episode, (tuple, list)):
            episode = self.episode_map[tuple(episode)]
        if temperature is None:
            temperature = self.dd.ag_temperature[episode]

        # seed (may only affect random agent behavior... episode is already set)
        seed = int(self.dd.episode[episode])
        self.env.seed(seed)
        np.random.seed(seed)

        if verbose >= 1:
            print('Episode info:')
            print(self.dd.iloc[episode:episode+1,:])

        agent.init_state()

        log_probs = []
        rewards = []

        try:
            # initialize
            observation = self.init_episode(episode, update=update_res)

            # run episode
            for i in range(self.env.max_steps):
                if verbose >= 2:
                    print(observation)

                agent_input = observation / self.maze_size
                if self.state_mode == 'local_map':
                    agent_input = self.env.get_local_map().reshape(-1)
                elif self.state_mode == 'mid_map':
                    agent_input = self.env.get_local_map(rad=2).reshape(-1)
                elif self.state_mode == 'both':
                    agent_input = np.concatenate((
                        agent_input, self.env.get_local_map().reshape(-1)))
                elif self.state_mode == 'mid_map2':
                    agent_input = np.concatenate((
                        agent_input, self.env.get_local_map(rad=2).reshape(-1)))
                elif self.state_mode == 'full':
                    agent_input = self.env._im_from_state()

                action, log_prob, _ = agent.act(
                    agent_input, T=temperature, verbose=verbose-2, **kwargs)
                observation, reward, done, info = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)

                if done:
                    break

            # summarize results
            T = len(rewards)
            score = np.sum(rewards)
            returns = np.zeros(T)
            returns[T-1] = rewards[-1]
            for t in range(T):
                returns[T-1-t] = rewards[T-1-t-1] + self.gamma*returns[T-1-t]
            losses = [-lp * R for lp, R in zip(log_probs, returns)]
            ret_loss = torch.cat(losses).sum()
            log_prob = torch.cat(log_probs).sum()
            if update_res:
                self.dd.loc[episode,'agent'] = agent.title
                self.dd.loc[episode,'ag_updates'] = agent.n_updates
                self.dd.loc[episode,'ag_hash'] = agent.get_params_hash()
                self.dd.loc[episode,'ag_temperature'] = temperature
                self.dd.loc[episode,'log_prob'] = log_prob.item()
                self.dd.loc[episode,'ret_loss'] = ret_loss.item()
                self.dd.loc[episode,'score'] = score
            if verbose >= 1:
                print(f'Score = {np.sum(rewards)}')

        except:
            self.env.close()
            raise

        if render:
            self.env.close()
            self.env._show_state(ax=ax)

        return score, log_prob, ret_loss, T

    def show_episode(self, idx=None, episode=None, agent_nm=None, group=None,
                     verbose=2, **kwargs):
        if idx is None:
            if episode is None:
                ids = len(self.dd) * [True]
                if agent_nm is not None and group is not None:
                    ids = (self.dd.agent == agent_nm) & (self.dd.group == group)
                elif agent_nm is not None:
                    ids = (self.dd.agent == agent_nm)
                elif group is not None:
                    ids = (self.dd.group == group)
                idx = np.random.choice(np.arange(len(self.dd))[ids])
                agent_nm = self.dd.agent[episode]
            if agent_nm is None:
                agent_nm = np.random.choice(self.agents_names)
        else:
            if isinstance(idx, (tuple, list)):
                idx = self.episode_map[tuple(idx)]
            agent_nm = self.dd.agent[idx]

        assert agent_nm == self.dd.agent[idx]
        agent = self.agents[agent_nm]

        if verbose >= 1:
            print(f'Agent:\t{agent_nm:s}')
            if verbose >= 2:
                self.print_hash_info(agent_nm=agent_nm, verbose=verbose-1)

        agent.eval()
        score, _, _, T = self.run_episode(idx, agent, render=True, update_res=False,
                                          verbose=verbose-1, **kwargs)
        agent.train()
        return score, T

    def print_hash_info(self, hash=None, agent_nm=None, verbose=1):
        if hash is None:
            hash = self.agents[agent_nm].get_params_hash()
        ids = self.dd.ag_hash == hash
        n = np.sum(ids)
        dd = self.dd[ids]
        print(f'Agent hash:\t{hash:s}')
        print(f'Matches:\t{n:d}/{len(self.dd):d}')
        if verbose >= 2:
            if n >= 1:
                print(f'\tFirst match:')
                print(dd.iloc[0:1,:4])
                if n >= 2:
                    print(f'\tLast match:')
                    print(dd.iloc[-1:,:4])
        print()

    ###############   TRAIN   ###############

    def draw_train_episode_params(self, idx, ce=None):
        if ce is None: ce = self.ce
        dyn, s0, w = ce.sample(self.clip_factor)
        self.dd.iloc[idx, 6:7], self.dd.iloc[idx, 7:9] = dyn, s0
        self.dd.loc[idx, 'ag_temperature'] = self.T
        return w

    def ce_update(self, idx, score, ce=None, cvar=1, ref_scores=None):
        if ce is None: ce = self.ce
        dyn = self.dd.iloc[idx, 6:7].values.astype(np.float)
        s0 = self.dd.iloc[idx, 7:9].values.astype(np.float)
        q_ref = np.percentile(ref_scores, 100*cvar) if 0<cvar<1 else None
        ce.update_recording(score, dyn, s0, update_dists=True,
                            update_q_ref=q_ref, cvar=cvar)

    def get_train_hparams(self, agent=None):
        # optimization hparams
        get_value = lambda x: agent.train_hparams[x] \
            if agent.train_hparams is not None and x in agent.train_hparams \
            else getattr(self, x)
        lr = get_value('lr')
        weight_decay = get_value('weight_decay')
        optim_freq = get_value('optim_freq')
        episodic_loss = get_value('episodic_loss')
        normalize_returns = get_value('normalize_returns')
        T0 = get_value('T0')
        Tgamma = get_value('Tgamma')
        cvar = get_value('cvar')

        # CE hparams
        if isinstance(agent.train_hparams, dict) and \
                np.any([s.startswith('ce_') for s in agent.train_hparams]):
            get_value = lambda x: agent.train_hparams[x] if \
                x in agent.train_hparams else getattr(self.ce, x[3:])
            modify_dyn = get_value('ce_'+'modify_dyn')
            modify_s0 = get_value('ce_'+'modify_s0')
            update_freq = get_value('ce_'+'update_freq')
            update_perc = get_value('ce_'+'update_perc')
            valid_ref = get_value('ce_'+'valid_ref')
            source_perc = get_value('ce_'+'source_perc')
            IS = get_value('ce_'+'IS')
            ce = CEM.CEM(mode='Rooms', dyn_dist=self.dynamics, modify_dyn=modify_dyn,
                         modify_s0=modify_s0, update_freq=update_freq*self.optim_freq,
                         source_sample_perc=source_perc,
                         update_perc=update_perc, IS=IS, valid_ref=valid_ref)
        else:
            self.ce.reset()
            ce = self.ce

        return lr, weight_decay, optim_freq, episodic_loss, normalize_returns, \
               T0, Tgamma, cvar, ce

    def train_with_dependencies(self, agents_names=None, **kwargs):
        if agents_names is None: agents_names = self.agents_names.copy()
        if isinstance(agents_names, str): agents_names = [agents_names]

        trained_agents = set()
        while True:
            curr_agents = [a for a in agents_names if (
                    self.agents[a].pretrained_filename is None or
                    self.agents[a].pretrained_filename in trained_agents)]
            if not curr_agents:
                break
            if self.max_distributed:
                curr_agents = curr_agents[:self.max_distributed]
            print(f'Training {len(curr_agents)}/{len(agents_names)} '
                  'remaining agents...')
            self.train(curr_agents, **kwargs)
            for a in curr_agents:
                trained_agents.add(a)
                agents_names.remove(a)

        if agents_names:
            print('Not all agents were reached in the training-dependencies tree!',
                  agents_names)

    def train(self, agents_names=None, distributed=True, log_freq=None,
              verbose=1, **kwargs):
        if log_freq is None: log_freq = self.log_freq
        if agents_names is None: agents_names = self.agents_names
        if isinstance(agents_names, str): agents_names = [agents_names]

        if verbose >= 1:
            print(f'Training {len(agents_names):d} agents...')

        if not distributed or len(agents_names)==1:
            for agent_nm in agents_names:
                self.train_agent(agent_nm, log_freq, verbose, **kwargs)

        else:
            t0 = time.time()
            ctx = mp.get_context('spawn')
            q = ctx.Queue()
            procs = []

            # run trainings
            for agent_nm in agents_names:
                p = ctx.Process(target=Experiment.train_agent_wrapper,
                                args=(q, self, agent_nm, kwargs))
                procs.append(p)
                if verbose >= 2:
                    print(f'Training {agent_nm:s}...')
                p.start()

            # wait for results
            if verbose >= 1:
                print('Waiting for trainings...')
            res = [q.get() for _ in range(len(procs))]
            names = [d[0] for d in res]
            dd = [d[1] for d in res]
            if verbose >= 1:
                print('Waiting for processes...')
            for i, p in enumerate(procs):
                if verbose >= 2:
                    print(f'Waiting for {agents_names[i]:s}...')
                p.join()
            if verbose >= 1:
                print(f'Done.\t({time.time() - t0:.0f}s)')

            # merge results
            for agent_nm, d in zip(names, dd):
                self.last_train_success[agent_nm] = d[0]
                self.dd[(self.dd.agent == agent_nm) & (self.dd.group == 'train')] = d[1]
                self.valid_scores[agent_nm] = d[2]
                self.samples_usage[agent_nm] = d[3]
                self.eff_samples_usage[agent_nm] = d[4]
                self.ce_history[agent_nm] = d[5]
                self.ce_ws[agent_nm] = d[6]
                # update agent
                self.agents[agent_nm].load()
                self.agents[agent_nm].n_updates = d[1].ag_updates.values[-1] + 1

    @staticmethod
    def train_agent_wrapper(q, E, agent_nm, kwargs):
        try:
            print(f'Training {agent_nm}...')
            E.train_agent(agent_nm, **kwargs)
        except:
            q.put((agent_nm, (
                False,
                E.dd[(E.dd.agent==agent_nm)&(E.dd.group=='train')],
                E.valid_scores[agent_nm],
                E.samples_usage[agent_nm],
                E.eff_samples_usage[agent_nm],
                E.ce_history[agent_nm],
                E.ce_ws[agent_nm],
            )))
            print(f'Error in {agent_nm}.')
            raise
        q.put((agent_nm, (
            True,
            E.dd[(E.dd.agent == agent_nm) & (E.dd.group == 'train')],
            E.valid_scores[agent_nm],
            E.samples_usage[agent_nm],
            E.eff_samples_usage[agent_nm],
            E.ce_history[agent_nm],
            E.ce_ws[agent_nm],
        )))

    def train_agent(self, agent_nm, log_freq=None, verbose=1, **kwargs):
        if log_freq is None: log_freq = self.log_freq
        t0 = time.time()
        agent = self.agents[agent_nm]
        if agent.pretrained_filename is not None:
            agent.load(agent.pretrained_filename)
        agent.train()

        lr, weight_decay, optim_freq, episodic_loss, normalize_returns, T0, Tgamma, cvar, ce = \
            self.get_train_hparams(agent)
        valid_fun = (lambda x: np.mean(sorted(x)[:int(np.ceil(cvar*len(x)))])) \
            if (0<cvar<1) else np.mean

        optimizer = self.optimizer_constructor(
            agent.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_wrap = Optim.Optimizer(
            optimizer, optim_freq, episodic_loss, normalize_returns, cvar)

        # get episodes
        ids = np.arange(len(self.dd))[
            (self.dd.agent == agent_nm) & (self.dd.group == 'train')]

        self.T = T0
        self.valid_scores[agent_nm] = [
            self.test(agent_nm, 'valid', update_inplace=False, temperature=0,
                      verbose=0)]
        valid_score = valid_fun(self.valid_scores[agent_nm][-1])
        best_valid_score = valid_score
        if self.save_best_model:
            agent.save()

        for i, idx in enumerate(ids):
            # draw episode params
            w = self.draw_train_episode_params(idx, ce)

            # run episode
            score, log_prob, ret_loss, n_steps = self.run_episode(
                idx, agent, update_res=True, temperature=self.T, **kwargs)

            self.ce_update(idx, score, ce, cvar, self.valid_scores[agent_nm][-1])

            # get reference scores for distribution estimation
            ref_scores = None
            if ce.valid_ref:
                if ce.n_source > 1:
                    ref_scores = optimizer_wrap.curr_scores[:ce.n_source]
                else:
                    ref_scores = self.valid_scores[agent_nm][-1]

            # optimize
            if optimizer_wrap.step(w, log_prob, score, ret_loss, ref_scores):
                agent.n_updates += 1
                self.T *= Tgamma
                # validation
                if (agent.n_updates % self.valid_freq) == 0:
                    self.valid_scores[agent_nm].append(
                        self.test(agent_nm, 'valid', update_inplace=False,
                                  temperature=0, verbose=0))
                    valid_score = valid_fun(self.valid_scores[agent_nm][-1])
                    # save model
                    if best_valid_score < valid_score:
                        if self.save_best_model:
                            agent.save()
                        best_valid_score = valid_score

            self.samples_usage[agent_nm] = optimizer_wrap.samples_per_step
            self.eff_samples_usage[agent_nm] = optimizer_wrap.eff_samples_per_step
            self.ce_history[agent_nm] = ce.history
            self.ce_ws[agent_nm] = ce.w_history

            # log
            if log_freq > 0 and ((i + 1) % log_freq) == 0:
                print(f'\t[{i + 1:03d}/{len(ids):03d}, {agent_nm}] valid_loss='
                      f'{valid_score:.1f}\t({time.time() - t0:.0f}s)')

        # load best model
        if self.save_best_model:
            agent.load()
        else:
            agent.save()

        if verbose >= 1:
            print(f'{agent_nm:s} trained ({time.time() - t0:.0f}s).')

    ###############   TEST   ###############

    def test_multiple_agents(self, agents=None, distributed=True,
                             group='test', verbose=1, **kwargs):
        if agents is None: agents = self.agents_names
        if isinstance(agents, str): agents = [agents]

        if not distributed or len(agents)==1:
            return {a: self.test(a, **kwargs) for a in agents}

        # run distributed
        t0 = time.time()
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        procs = []

        # run tests
        for agent_nm in agents:
            p = ctx.Process(target=Experiment.test_agent_wrapper,
                            args=(q, self, agent_nm, kwargs, group))
            procs.append(p)
            if verbose >= 2:
                print(f'Testing {agent_nm:s}...')
            p.start()

        # wait for results
        if verbose >= 1:
            print('Waiting for tests...')
        res = [q.get() for _ in range(len(procs))]
        names = [d[0] for d in res]
        dd = [d[1] for d in res]
        scores = [d[2] for d in res]
        if verbose >= 1:
            print('Waiting for processes...')
        for i, p in enumerate(procs):
            if verbose >= 2:
                print(f'Waiting for {agents[i]:s}...')
            p.join()
        if verbose >= 1:
            print(f'Done.\t({time.time() - t0:.0f}s)')

        # merge results
        for agent_nm, d in zip(names, dd):
            self.dd[(self.dd.agent == agent_nm) & (self.dd.group == group)] = d[0]

        return {a: s for a, s in zip(names,scores)}

    @staticmethod
    def test_agent_wrapper(q, E, agent_nm, kwargs, group='test'):
        try:
            scores = E.test(agent_nm, group=group, **kwargs)
        except:
            q.put((agent_nm, (E.dd[(E.dd.agent==agent_nm)&(E.dd.group==group)],), []))
            print(f'Error in {agent_nm}.')
            raise
        q.put((agent_nm, (E.dd[(E.dd.agent == agent_nm) & (E.dd.group == group)],), scores))

    def test(self, agent_nm=None, group='test', update_inplace=True, temperature=0,
             verbose=1, **kwargs):
        if agent_nm is None:
            return self.test_multiple_agents(
                self.agents_names, group=group, update_inplace=update_inplace,
                verbose=verbose, **kwargs)

        t0 = time.time()
        scores = []
        dd = self.dd[(self.dd.agent==agent_nm) & (self.dd.group==group)]
        if len(dd) == 0:
            return []
        agent = self.agents[agent_nm]
        agent.eval()
        for episode in dd.episode:
            idx = self.episode_map[(agent_nm,group,episode)]
            score = self.run_episode(
                idx, agent, update_res=update_inplace, temperature=temperature,
                verbose=verbose-2, **kwargs)[0]
            scores.append(score)
        agent.train()

        if verbose >= 1:
            print(f'{agent_nm:s} tested ({time.time()-t0:.0f}s).')

        return scores

    ###############   ANALYSIS   ###############

    def analysis_preprocessing(self, agents=None):
        train_df = self.dd[self.dd.group == 'train']
        train_df = pd.DataFrame(dict(
            agent=train_df.agent,
            group='train',
            train_iteration=train_df.ag_updates,
            episode=train_df.episode,
            score=train_df.score
        ))

        valid_df = pd.DataFrame()
        for agent in self.valid_scores:
            # valid_scores[agent][iteration][episode]=score
            valid_scores = self.valid_scores[agent]
            valid_df = pd.concat((valid_df, pd.DataFrame(dict(
                agent=agent,
                group='valid',
                train_iteration=self.valid_freq * np.repeat(np.arange(len(valid_scores)),
                                                            len(valid_scores[0])),
                episode=len(valid_scores) * list(np.arange(len(valid_scores[0]))),
                score=np.concatenate(valid_scores)
            ))))

        train_valid_df = pd.concat((train_df, valid_df))
        train_valid_df['agent/group'] = [f'{a}_{g}' for a,g in zip(
            train_valid_df.agent,train_valid_df.group)]

        test_df = self.dd[self.dd.group == 'test']

        # training samples used per iteration
        samples_usage = pd.DataFrame()
        for agent in self.agents_names:
            if agents is None or agent in agents:
                samples_usage = pd.concat((samples_usage, pd.DataFrame(dict(
                    agent = agent,
                    train_iteration = np.arange(len(self.samples_usage[agent])),
                    samples_used = 100*np.array(self.samples_usage[agent]),
                    effective_samples_used=100*np.array(self.eff_samples_usage[agent]),
                ))))

        # sampler weights
        weights = pd.DataFrame()
        for agent in self.agents_names:
            if agents is None or agent in agents:
                ws = self.ce_ws[agent]
                n_iters = len(ws)
                n_samps = len(ws[0])
                ce = self.get_train_hparams(self.agents[agent])[-1]
                if ce.update_freq > 0:
                    if ce.n_source > 0:
                        source_dist = ce.n_source*[True] + (n_samps-ce.n_source)*[False]
                    else:
                        source_dist = n_samps*[False]
                else:
                    source_dist = n_samps*[True]

                weights = pd.concat((weights, pd.DataFrame(dict(
                    agent = agent,
                    train_iteration = np.repeat(np.arange(n_iters), n_samps),
                    source_distribution = n_iters*list(source_dist),
                    weight = np.concatenate(ws),
                ))))

        if agents is not None:
            train_valid_df = train_valid_df[train_valid_df.agent.isin(agents)]
            test_df = test_df[test_df.agent.isin(agents)]

        train_valid_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        samples_usage.reset_index(drop=True, inplace=True)
        return train_valid_df, test_df, samples_usage, weights

    def analyze(self, agents=None, W=3, axsize=(6,4), Q=100,
                verify_train_success=False):
        train_valid_df, test_df, samples_usage, weights = \
            self.analysis_preprocessing(agents)

        axs = utils.Axes(6, W, axsize, fontsize=15)
        a = 0

        sns.lineplot(data=train_valid_df, x='train_iteration', hue='agent',
                     style='group', y='score', ax=axs[a])
        axs[a].set_xlim((0,None))
        # axs[a].set_ylim((max(-200, axs[a].get_ylim()[0]), None))
        plt.setp(axs[a].get_legend().get_texts(), fontsize='13')
        axs.labs(a, 'train iteration', 'score')
        a += 1

        agents, ids = np.unique(test_df.agent.values, return_index=True)
        agents = agents[np.argsort(ids)]
        for agent in agents:
            if verify_train_success and not self.last_train_success[agent]:
                continue
            scores = test_df.score[test_df.agent==agent].values
            utils.plot_quantiles(scores, Q=np.arange(Q+1)/100, showmeans=True, ax=axs[a],
                                 label=f'{agent} ({np.mean(scores):.1f})')
        axs[a].set_xlim((0,Q))
        axs.labs(a, 'episode quantile [%]', 'score')
        axs[a].legend(fontsize=13)
        a += 1

        if verify_train_success:
            valid_agents = [a for a in self.agents_names if self.last_train_success[a]]
            weights = weights[weights.agent.isin(valid_agents)]
        iter_resol = 1 + weights.train_iteration.values[-1] // 4
        weights['iter_rounded'] = [iter_resol*(it//iter_resol) \
                                   for it in weights.train_iteration]
        sns.boxplot(data=weights, x='iter_rounded', hue='agent', y='weight',
                    showmeans=True, ax=axs[a])
        axs.labs(a, 'train iteration', 'weight')
        axs[a].legend(fontsize=13)
        a += 1

        sns.lineplot(data=samples_usage, x='train_iteration', hue='agent',
                     y='samples_used', ci=None, ax=axs[a])
        axs[a].set_xlim((0,None))
        plt.setp(axs[a].get_legend().get_texts(), fontsize='13')
        axs.labs(a, 'train iteration', 'samples (episodes) used [%]')
        a += 1

        sns.lineplot(data=samples_usage, x='train_iteration', hue='agent',
                     y='effective_samples_used', ci=None, ax=axs[a])
        axs[a].set_xlim((0,None))
        plt.setp(axs[a].get_legend().get_texts(), fontsize='13')
        axs.labs(a, 'train iteration', 'effective sample size [%]')
        a += 1

        for agent in agents:
            ceh = self.ce_history[agent]
            y = [100*yy[-1] for yy in ceh['dyn_dists']]
            axs[a].plot(np.arange(len(y)), y, '.-', label=agent)
        axs[a].set_ylim((0, 100))
        axs.labs(a, 'CE iteration', 'kill prob [%]')
        axs[a].legend(fontsize=13)
        a += 1

        plt.tight_layout()
        return axs

    def show_tests(self, agents=None, n=5, **kwargs):
        if agents is None: agents = self.agents_names
        if isinstance(agents, str): agents = [agents]

        axs = utils.Axes(n*len(agents), n, (2.7, 2.7), grid=False)
        a = 0

        for agent in agents:
            d = self.dd[(self.dd.agent == agent) & (self.dd.group == 'test')]
            d = d.sort_values('score')
            N = len(d)
            qs = np.linspace(0, 1, n)
            ids = ((N-1)*qs).astype(int)
            episodes = d.episode.values[ids]
            scores = d.score.values[ids]

            for i in range(n):
                s, T = self.show_episode((agent, 'test', episodes[i]), ax=axs[a],
                                         verbose=0, **kwargs)
                axs.labs(a, None, None,
                         f'{agent} ({ids[i]+1}/{N})\nscore={scores[i]:.0f} ({T} steps)'
                         f'\nreproduced={s:.0f}')
                a += 1

        plt.tight_layout()
        return axs

    def analyze_exposure(self, agents=None, good_threshold=-32):
        if agents is None: agents = self.agents_names
        axs = utils.Axes(2, 2, (7,4))
        good_episodes = {}
        for ag in agents:
            dd = self.dd[(self.dd.group=='train')&(self.dd.agent==ag)]
            n_samp = (np.array(self.samples_usage[ag]) * self.optim_freq).astype(int)
            n_iter = dd.ag_updates.values[-1] + 1

            # total "good" episodes
            used_scores = [dd[dd.ag_updates==i].score.values for i in range(n_iter)]
            good_episodes[ag] = [np.sum(np.array(s)>good_threshold) \
                                 for s in used_scores]
            axs[0].plot(good_episodes[ag], label=ag)

            # "good" episodes exposed to optimizer
            used_scores = [sorted(dd[dd.ag_updates==i].score.values)[:n_samp[i]] \
                           for i in range(n_iter)]
            good_episodes[ag] = [np.sum(np.array(s)>good_threshold) \
                                 for s in used_scores]
            axs[1].plot(good_episodes[ag], label=ag)
        axs.labs(0, 'train iteration', 'total "good" episodes')
        axs.labs(1, 'train iteration', '"good" episodes fed to optimizer')
        axs[0].legend()
        axs[1].legend()
        return axs

    def main(self, **analysis_args):
        self.train_with_dependencies()
        self.test()
        self.analyze(**analysis_args)


SAMPLE_AGENT_CONFS = dict(
    Vanilla = (Agents.NN, dict()),
    CVaR10 = (Agents.NN, dict(train_hparams=dict(cvar=0.1))),
    CVaR_CE_IS_REF = (Agents.NN, dict(train_hparams=dict(
        cvar=0.1, ce_update_freq=1, ce_perc=0.2, ce_IS=True,
        ce_valid_ref=True, ce_source_perc=0.4))),
)

if __name__ == '__main__':

    E = Experiment(SAMPLE_AGENT_CONFS, train_episodes=6000,
                   valid_episodes=40, test_episodes=500,
                   dynamics=(0.05,), clip_factor=(20,))
    E.main()
    print(E.dd.tail())
    E.show_tests()
    plt.show()
