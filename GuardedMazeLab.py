'''

'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time, warnings
import pickle as pkl
import multiprocessing as mp
import torch
import torch.optim as optim
import gym

import GuardedMaze
import Agents, GCVaR, CEM
import utils

STATE_DIM = dict(xy=2, local_map=9, both=11, mid_map=25, mid_map_xy=27)

class Experiment:

    ###############   INITIALIZATION & SETUP   ###############

    def __init__(self, agents=None, train_episodes=500, valid_episodes=20,
                 test_episodes=100, global_seed=0, maze_mode=1, maze_size=None,
                 max_episode_steps=None,
                 detailed_rewards=False, valid_freq=10, save_best_model=True,
                 kill_prob=0.05, beta_kill_dist=True,
                 action_noise=0.2, max_distributed=0,
                 optimizer=optim.Adam, optim_freq=100, optim_q_ref=True,
                 cvar=1, soft_cvar=0, zero_loss_tolerance=20,
                 gamma=1.0, lr=1e-2, weight_decay=0.0, state_mode='mid_map_xy',
                 use_ce=False, ce_alpha=0.2, ce_ref_mode='train', ce_ref_alpha=None,
                 ce_n_orig=None, ce_w_clip=5,
                 log_freq=2000, T0=1, Tgamma=1, title=''):
        self.title = title
        self.global_seed = global_seed
        np.random.seed(self.global_seed)
        torch.manual_seed(self.global_seed)

        self.n_train = train_episodes
        self.n_valid = valid_episodes
        self.n_test = test_episodes
        self.maze_mode = maze_mode
        if maze_size is None: maze_size = GuardedMaze.MAZE_SIZE[self.maze_mode]
        self.maze_size = maze_size
        self.max_episode_steps = max_episode_steps
        self.state_mode = state_mode
        self.detailed_rewards = detailed_rewards

        self.T0 = T0  # initial train temperature
        self.Tgamma = Tgamma  # temperature decay
        self.T = self.T0
        self.valid_freq = valid_freq
        self.save_best_model = save_best_model

        self.kill_prob = kill_prob
        # If true - model killing probability using Beta(a,2-a), whose mean
        # is kill_prob=a/2 and variance is a*(2-a)/12.
        # https://en.wikipedia.org/wiki/Beta_distribution
        self.use_beta_dist = beta_kill_dist
        self.action_noise = action_noise

        self.optimizers = {}
        self.optimizer_constructor = optimizer
        self.optim_freq = optim_freq
        self.optim_q_ref = optim_q_ref
        self.zero_loss_tolerance = zero_loss_tolerance
        self.cvar = cvar
        self.soft_cvar = soft_cvar
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

        self.CEs = {}
        if ce_n_orig is None:
            ce_n_orig = int(0.2*self.optim_freq) if ce_ref_mode=='train' else 0
        self.ce_update_freq = use_ce
        self.ce_w_clip = ce_w_clip
        self.ce_ref_mode = ce_ref_mode
        self.ce_ref_alpha = ce_ref_alpha
        self.ce_n_orig = ce_n_orig
        self.ce_internal_alpha = ce_alpha

        self.make_env()
        self.generate_test_episodes()
        if agents is not None:
            self.set_agents(agents, generate_tables=True)

    def register_env(self):
        # first unregister if needed
        if 'GuardedMazeEnv-v0' in gym.envs.registry.env_specs:
            del gym.envs.registry.env_specs['GuardedMazeEnv-v0']
        gym.envs.registration.register(
            id='GuardedMazeEnv-v0',
            entry_point='GuardedMaze:GuardedMaze',
            kwargs=dict(
                mode=self.maze_mode, kill_prob=self.kill_prob,
                action_noise=self.action_noise,
                rows=self.maze_size, max_steps=self.max_episode_steps,
                detailed_r=self.detailed_rewards),
        )

    def make_env(self):
        self.register_env()
        self.env = gym.make('GuardedMazeEnv-v0')

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

    def draw_episode_params(self, dyn_dist=None, s0_dist=None, seed=None):
        if seed is not None: np.random.seed(seed)
        if s0_dist is None:
            s0_dist = (0.6,0.6,6.4,6.4) if self.maze_mode==2 else (0.6,0.6,3.6,3.6)
        if dyn_dist is None: dyn_dist = self.kill_prob
        dyn_dist = np.array(dyn_dist)

        if self.use_beta_dist:
            dyn = np.random.beta(2*dyn_dist, 2-2*dyn_dist)
        else:
            dyn = np.random.exponential(np.array(dyn_dist), size=(len(dyn_dist),))

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

        init_states = np.array(
            [self.draw_episode_params()[1] for _ in range(self.n_train)])
        dd_base.iloc[:, 7:9] = init_states

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
        if isinstance(agent, (tuple,list)):
            args = agent[1]
            args['state_dim'] = self.maze_size if self.state_mode=='full' \
                else STATE_DIM[self.state_mode]
            args['act_dim'] = 4
            agent = agent[0](**args)
        if title is None:
            title = agent.title
        else:
            agent.title = title
        self.agents_names.append(title)
        self.agents[title] = agent
        if generate_tables:
            self.generate_train_dd(title)
            self.generate_test_dd(title)

    def save_agent(self, agent):
        nm = agent.title
        if self.title:
            nm = f'{self.title}_{nm}'
        agent.save(nm)

    def load_agent(self, agent):
        if isinstance(agent, str):
            nm = agent
            agent = self.agents[nm]
        else:
            nm = agent.title
        if self.title:
            nm = f'{self.title}_{nm}'
        agent.load(nm)

    def load_agents(self, agents=None):
        if agents is None: agents = self.agents_names
        for agent in agents:
            self.load_agent(agent)

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
                elif self.state_mode == 'mid_map_xy':
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
        if ce is None: ce = self.CEs[self.dd.agent[idx]]
        kp, w = ce.sample()
        self.dd.loc[idx, 'dyn_kp'] = kp
        self.dd.loc[idx, 'ag_temperature'] = self.T
        return w

    def prepare_training(self, agent=None):
        hparam_keys = {}
        if agent.train_hparams is not None:
            hparam_keys = set(agent.train_hparams.keys())
        def get_value(x):
            if x in hparam_keys:
                hparam_keys.remove(x)
                return agent.train_hparams[x]
            return getattr(self, x)

        # get optimization hparams
        lr = get_value('lr')
        weight_decay = get_value('weight_decay')
        optim_freq = get_value('optim_freq')
        optim_q_ref = get_value('optim_q_ref')
        T0 = get_value('T0')
        Tgamma = get_value('Tgamma')
        cvar = get_value('cvar')
        soft_cvar = get_value('soft_cvar')

        # prepare optimizer
        valid_fun = (lambda x: np.mean(sorted(x)[:int(np.ceil(cvar*len(x)))])) \
            if (0<cvar<1) else np.mean
        cvar_scheduler = (soft_cvar, self.n_train//self.optim_freq)
        optimizer = self.optimizer_constructor(
            agent.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_wrap = GCVaR.GCVaR(
            optimizer, optim_freq, cvar, cvar_scheduler, title=agent.title)
        self.optimizers[agent.title] = optimizer_wrap

        # get CE hparams
        update_freq = get_value('ce_update_freq')
        w_clip = get_value('ce_w_clip')
        ref_mode = get_value('ce_ref_mode')
        ref_alpha = get_value('ce_ref_alpha')
        n_orig = get_value('ce_n_orig')
        internal_alpha = get_value('ce_internal_alpha')

        # prepare CE
        if ref_alpha is None: ref_alpha = cvar
        ce = CEM.CEM_Beta_1D(
            self.kill_prob, batch_size=update_freq * self.optim_freq,
            ref_mode=ref_mode, ref_alpha=ref_alpha, w_clip=w_clip,
            n_orig_per_batch=n_orig, internal_alpha=internal_alpha,
            title=agent.title)
        self.CEs[agent.title] = ce

        if len(hparam_keys) > 0:
            warnings.warn(f'Unused hyper-params for {agent.title}: {hparam_keys}')
        return optimizer_wrap, valid_fun, optim_q_ref, T0, Tgamma, \
               ref_mode=='valid', ce

    def train_agent(self, agent_nm, log_freq=None, verbose=1, **kwargs):
        if log_freq is None: log_freq = self.log_freq
        t0 = time.time()
        agent = self.agents[agent_nm]
        if agent.pretrained_filename is not None:
            agent.load(agent.pretrained_filename)
        agent.train()

        optimizer_wrap, valid_fun, optim_q_ref, T0, Tgamma, ce_valid, ce = \
            self.prepare_training(agent)

        # get episodes
        ids = np.arange(len(self.dd))[
            (self.dd.agent == agent_nm) & (self.dd.group == 'train')]

        self.T = T0
        self.valid_scores[agent_nm] = [
            self.test(agent_nm, 'valid', update_inplace=False, temperature=0,
                      verbose=0)]
        valid_score = valid_fun(self.valid_scores[agent_nm][-1])
        valid_mean = np.mean(self.valid_scores[agent_nm][-1])
        best_valid_score = valid_score
        best_valid_mean = valid_mean
        if self.save_best_model:
            self.save_agent(agent)

        for i, idx in enumerate(ids):
            # draw episode params
            w = self.draw_train_episode_params(idx, ce)

            # run episode
            score, log_prob, ret_loss, n_steps = self.run_episode(
                idx, agent, update_res=True, temperature=self.T, **kwargs)

            # update CEM
            if ce_valid:
                ce.update_ref_scores(self.valid_scores[agent_nm][-1])
            ce.update(score, f'models/{self.title}_{agent_nm}')

            # get reference scores for distribution estimation
            ref_scores = None
            if ce.n_orig_per_batch > 1:
                ref_scores = optimizer_wrap.scores[-1][:ce.n_orig_per_batch]
            elif optim_q_ref:
                ref_scores = self.valid_scores[agent_nm][-1]

            # optimize
            if optimizer_wrap.step(w, log_prob, score, ref_scores,
                                   f'models/{self.title}_{agent_nm}'):
                agent.n_updates += 1
                self.T *= Tgamma
                # validation
                if (agent.n_updates % self.valid_freq) == 0:
                    self.valid_scores[agent_nm].append(
                        self.test(agent_nm, 'valid', update_inplace=False,
                                  temperature=0, verbose=0))
                    valid_score = valid_fun(self.valid_scores[agent_nm][-1])
                    valid_mean = np.mean(self.valid_scores[agent_nm][-1])
                    # save model
                    if best_valid_score < valid_score or \
                        (best_valid_score == valid_score and
                         best_valid_mean <= valid_mean):
                        if self.save_best_model:
                            self.save_agent(agent)
                        best_valid_score = valid_score
                        best_valid_mean = valid_mean

                # stop if loss==0 for a while (e.g. all returns are the same...)
                losses = optimizer_wrap.losses
                tol = self.zero_loss_tolerance
                if tol and tol<len(losses) and np.all([l==0 for l in losses[-tol:]]):
                    if verbose >= 1:
                        print(f'{agent_nm} training stopped after {i+1} episodes and '
                              f'{tol} steps with zero loss.')
                    break

            # log
            if log_freq > 0 and ((i + 1) % log_freq) == 0:
                print(f'\t[{i + 1:03d}/{len(ids):03d}, {agent_nm}] valid_mean='
                      f'{valid_mean:.1f}\t({time.time() - t0:.0f}s)')

        # load best model
        if self.save_best_model:
            self.load_agent(agent)
        else:
            self.save_agent(agent)

        if verbose >= 1:
            print(f'{agent_nm:s} trained ({time.time() - t0:.0f}s).')

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

                # update agent
                self.load_agent(self.agents[agent_nm])
                self.agents[agent_nm].n_updates = d[1].ag_updates.values[-1] + 1

                # update optimizer
                filename = f'models/{self.title}_{agent_nm}'
                self.optimizers[agent_nm] = GCVaR.GCVaR(None, 0)
                try:
                    self.optimizers[agent_nm].load(filename)
                except:
                    print(f'Missing optimizer file {filename}.opt')

                # update CE sampler
                self.CEs[agent_nm] = CEM.CEM_Beta_1D(self.kill_prob)
                hp = self.agents[agent_nm].train_hparams
                if self.ce_update_freq or (
                        hp is not None and 'ce_update_freq' in hp and
                        hp['ce_update_freq'] > 0):
                    try:
                        self.CEs[agent_nm].load(filename)
                    except:
                        print(f'Missing CE file {filename}.cem')

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
            )))
            print(f'Error in {agent_nm}.')
            raise
        q.put((agent_nm, (
            True,
            E.dd[(E.dd.agent == agent_nm) & (E.dd.group == 'train')],
            E.valid_scores[agent_nm],
        )))

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

    def save_results(self, fname=None, agents=False, optimizers=False, CEs=False):
        if fname is None: fname=f'outputs/{self.title}'
        fname += '.pkl'
        with open(fname, 'wb') as h:
            pkl.dump((self.dd.copy(), self.valid_scores.copy()), h)

        for anm in self.agents_names:
            if agents: self.save_agent(anm)
            if optimizers: self.optimizers[anm].save(f'models/{self.title}_{anm}')
            if CEs: self.CEs[anm].save(f'models/{self.title}_{anm}')

    def load_results(self, fname=None, agents=False, optimizers=False, CEs=False):
        if fname is None: fname=f'outputs/{self.title}'
        fname += '.pkl'
        with open(fname, 'rb') as h:
            self.dd, self.valid_scores = pkl.load(h)

        for anm in self.agents_names:
            if agents: self.load_agent(anm)
            if optimizers: self.optimizers[anm].load(f'models/{self.title}_{anm}')
            if CEs: self.CEs[anm].load(f'models/{self.title}_{anm}')

    def analysis_preprocessing(self, agents=None):
        if agents is None:
            agents = self.agents_names

        train_df = self.dd[(self.dd.group=='train') & (self.dd.agent.isin(agents))]
        train_df = pd.DataFrame(dict(
            agent=train_df.agent,
            group='train',
            train_iteration=train_df.ag_updates,
            episode=train_df.episode,
            score=train_df.score
        ))

        valid_df = pd.DataFrame()
        for agent in agents:
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

        test_df = self.dd[(self.dd.group=='test') & (self.dd.agent.isin(agents))]

        # optimizer data
        opt_batch_data, opt_sample_data = pd.DataFrame(), pd.DataFrame()
        for ag in agents:
            d1, d2 = self.optimizers[ag].get_data()
            opt_batch_data = pd.concat((opt_batch_data, d1))
            opt_sample_data = pd.concat((opt_sample_data, d2))
        opt_batch_data['agent'] = opt_batch_data.title
        opt_sample_data['agent'] = opt_sample_data.title

        # CE data
        ce_batch_data, ce_sample_data = pd.DataFrame(), pd.DataFrame()
        for ag in agents:
            ce = self.CEs[ag]
            if not ce.batch_size:
                continue
            d1, d2 = ce.get_data('hit_prob', 'hit_prob')
            ce_batch_data = pd.concat((ce_batch_data, d1))
            ce_sample_data = pd.concat((ce_sample_data, d2))
        if len(ce_batch_data) > 0:
            ce_batch_data['agent'] = ce_batch_data.title
            ce_sample_data['agent'] = ce_sample_data.title
            ce_batch_data.reset_index(drop=True, inplace=True)
            ce_sample_data.reset_index(drop=True, inplace=True)

        train_valid_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        opt_batch_data.reset_index(drop=True, inplace=True)
        opt_sample_data.reset_index(drop=True, inplace=True)

        return train_valid_df, test_df, opt_batch_data, opt_sample_data, \
               ce_batch_data, ce_sample_data

    def build_estimators(self, est_names):
        if isinstance(est_names, dict):
            return est_names

        if isinstance(est_names, str):
            est_names = [est_names]

        ests = {}
        for est_name in est_names:
            if est_name == 'mean':
                est = est_name
                est_name = f'${est_name}$'
            elif est_name.startswith('cvar'):
                alpha = float(est_name[len('cvar'):]) / 100
                est = lambda x: np.mean(np.sort(x)[:int(np.ceil(alpha * len(x)))])
                est_name = f'$CVaR_{{{alpha:.2f}}}$'
            else:
                raise ValueError(est_name)
            ests[est_name] = est

        return ests

    def analyze(self, agents=None, W=3, axsize=(6,4), Q=100,
                train_estimators=('mean','cvar05'), verify_train_success=False):
        if agents is None: agents = self.agents_names
        train_estimators = self.build_estimators(train_estimators)
        train_valid_df, test_df, opt_batch_data, opt_sample_data, \
        ce_batch_data, ce_sample_data = self.analysis_preprocessing(agents)
        if verify_train_success:
            agents = [ag for ag in agents if self.last_train_success[ag]]

        axs = utils.Axes(7+len(train_estimators), W, axsize, fontsize=15)
        a = 0

        # Train scores
        for est_name, est in train_estimators.items():
            sns.lineplot(data=train_valid_df, x='train_iteration', hue='agent',
                         style='group', y='score', estimator=est, ax=axs[a])
            axs[a].set_xlim((0,None))
            plt.setp(axs[a].get_legend().get_texts(), fontsize='13')
            axs.labs(a, 'train iteration', f'{est_name} score')
            a += 1

        # Test scores
        cvar = lambda x, alpha: np.mean(np.sort(x)[:int(np.ceil(alpha*len(x)))])
        for agent in agents:
            scores = test_df.score[test_df.agent==agent].values
            utils.plot_quantiles(
                scores, q=np.arange(Q+1)/100, showmeans=True, ax=axs[a],
                label=f'{agent} ({np.mean(scores):.1f})')
            print(f'{agent}:\tmean={np.mean(scores):.1f}\t'
                  f'CVaR10={cvar(scores,0.1):.1f}\tCVaR05={cvar(scores,0.05):.1f}')
        axs[a].set_xlim((0,Q))
        axs.labs(a, 'episode quantile [%]', 'score')
        axs[a].legend(fontsize=13)
        a += 1

        # Optimizer: sample sizes and weights
        filter_trivial_agents = False
        if filter_trivial_agents:
            keep_agents = [ag for ag in agents if np.any(
                opt_batch_data.sample_size_perc[opt_batch_data.agent==ag]!=100)]
            opt_batch_data = opt_batch_data[opt_batch_data.agent.isin(keep_agents)]

        if len(opt_batch_data) > 0:
            sns.lineplot(data=opt_batch_data, x='batch', hue='agent',
                         y='sample_size_perc', ci=None, ax=axs[a])
            axs[a].set_xlim((0,None))
            plt.setp(axs[a].get_legend().get_texts(), fontsize='13')
            axs.labs(a, 'train iteration', 'sample size [%]')
            a += 1

            sns.lineplot(data=opt_batch_data, x='batch', hue='agent',
                         y='eff_sample_size_perc', ci=None, ax=axs[a])
            axs[a].set_xlim((0,None))
            plt.setp(axs[a].get_legend().get_texts(), fontsize='13')
            axs.labs(a, 'train iteration', 'effective sample size [%]')
            a += 1

        # CE: weights
        if len(ce_sample_data) > 0:
            iter_resol = 1 + ce_sample_data.batch.values[-1] // 4
            ce_sample_data['iter_rounded'] = [
                iter_resol*(it//iter_resol) for it in ce_sample_data.batch]
            sns.boxplot(data=ce_sample_data, x='iter_rounded', hue='agent',
                        y='weight', showmeans=True, ax=axs[a])
            axs.labs(a, 'train iteration', 'weight')
            axs[a].legend(fontsize=13)
            a += 1

        self.analyze_exposure(opt_sample_data, agents, axs=axs, a0=a)
        a += 2

        if len(ce_batch_data) > 0:
            axs[a].axhline(100*self.kill_prob, color='k', label='original')
            for agent in agents:
                y = 100 * ce_batch_data.hit_prob[ce_batch_data.agent==agent].values
                if np.any(y!=100*self.kill_prob):
                    axs[a].plot(np.arange(len(y)), y, '.-', label=agent)
            axs[a].set_ylim((0, 100))
            axs.labs(a, 'train iteration', 'average hit probability [%]')
            axs[a].legend(fontsize=13)
            a += 1

        plt.tight_layout()
        return axs

    def analyze_exposure(self, dd, agents=None, good_threshold=None,
                         axs=None, a0=0):
        if agents is None: agents = self.agents_names
        dd = dd[dd.agent.isin(agents)]

        if good_threshold is None: good_threshold = -32 * self.maze_mode
        dd['success'] = 100 * (dd.score > good_threshold)
        dd['used_success'] = dd.success * dd.selected

        if axs is None: axs = utils.Axes(2, 2, (5.5,3.5), fontsize=15)

        sns.lineplot(data=dd, x='batch', hue='agent', y='success',
                     ci=None, ax=axs[a0+0])
        sns.lineplot(data=dd, x='batch', hue='agent', y='used_success',
                     ci=None, ax=axs[a0+1])

        axs.labs(a0+0, 'train iteration', 'successful episodes')
        axs.labs(a0+1, 'train iteration', 'successful episodes\nfed to optimizer')
        plt.setp(axs[a0+0].get_legend().get_texts(), fontsize='13')
        plt.setp(axs[a0+1].get_legend().get_texts(), fontsize='13')
        plt.tight_layout()

        return axs

    def show_tests(self, agents=None, n=5, clean_plot=False, **kwargs):
        if agents is None: agents = self.agents_names
        if isinstance(agents, str): agents = [agents]

        axs = utils.Axes(n*len(agents), n, (2.7, 2.7), fontsize=15, grid=False)
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
                if clean_plot or scores[i]==s:
                    axs.labs(a, None, None, f'{agent}: score={scores[i]:.0f}')
                    axs[a].axis('off')
                else:
                    axs.labs(a, None, None,
                             f'{agent} ({ids[i]+1}/{N})'
                             f'\nscore={scores[i]:.0f} ({T} steps)'
                             f'\nreproduced={s:.0f}')
                a += 1

        plt.tight_layout()
        return axs

    def main(self, **analysis_args):
        self.train_with_dependencies()
        self.test()
        self.analyze(**analysis_args)


SAMPLE_AGENT_CONFS = dict(
    PG = (Agents.FC, dict()),
    GCVaR = (Agents.FC, dict(train_hparams=dict(cvar=0.05))),
    CE_SGCVaR = (Agents.FC, dict(train_hparams=dict(
        cvar=0.05, ce_update_freq=1, soft_cvar=0.6))),
)

if __name__ == '__main__':

    E = Experiment(SAMPLE_AGENT_CONFS, train_episodes=6000,
                   valid_episodes=40, test_episodes=500,
                   optim_freq=400, kill_prob=0.05, title='demo')
    E.main()
    print(E.dd.tail())
    E.show_tests(clean_plot=True)
    plt.show()
