'''

'''

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time, warnings
import pickle as pkl
import multiprocessing as mp
import torch
import torch.optim as optim
import gym

import DrivingSim
import Agents, GCVaR, CEM
import utils

STATE_DIM = dict(default=6)

class Experiment:

    ###############   INITIALIZATION & SETUP   ###############

    def __init__(self, agents=None, train_episodes=100000, valid_episodes=200,
                 test_episodes=1000, global_seed=0, agent_mid_layers=(32,),
                 max_episode_len=30, episode_len_init=3, nine_actions=False,
                 p_oil=0, rewards_conf=None,
                 valid_freq=10, save_best_model=True, save_all_policies=False,
                 leader_probs=(0.35,0.3,0.248,0.002,0.1), max_distributed=0,
                 optimizer=optim.Adam, optim_freq=400, optim_q_ref=None,
                 cvar=1, soft_cvar=0, optimistic_q=False, no_change_tolerance=10,
                 gamma=1.0, lr=2e-2, lr_gamma=0.5, lr_step=100, weight_decay=0,
                 state_mode='default', ce_warmup_turns=0,
                 use_ce=False, ce_alpha=0.2, ce_ref_mode='train', ce_ref_alpha=None,
                 ce_n_orig=None, ce_w_clip=5, ce_constructor=None,
                 log_freq=4000, Ti=1, Tf=1, title=''):
        self.title = title
        self.global_seed = global_seed
        np.random.seed(self.global_seed)
        torch.manual_seed(self.global_seed)
        for dirname in ('models', 'outputs'):
            if not os.path.exists(f'./{dirname}/'):
                os.makedirs(f'./{dirname}/')

        self.n_train = train_episodes
        self.n_valid = valid_episodes
        self.n_test = test_episodes
        self.max_episode_len = max_episode_len
        self.state_mode = state_mode

        self.leader_probs = leader_probs
        self.dist = np.concatenate((
            [p_oil, 10., 0., 0.3], leader_probs,
            [np.ceil(self.max_episode_len/1.5)]))

        if episode_len_init is None:
            episode_len_init = self.max_episode_len
        self.L_init = episode_len_init
        self.L = episode_len_init
        self.nine_actions = nine_actions
        self.rewards_conf = rewards_conf
        self.Ti = Ti  # initial train temperature
        self.Tf = Tf  # final train temperature
        self.T = self.Ti
        self.valid_freq = valid_freq
        self.save_best_model = save_best_model
        self.save_all_policies = save_all_policies

        self.optimizers = {}
        self.optimizer_constructor = optimizer
        self.optim_freq = optim_freq
        self.optim_q_ref = optim_q_ref
        self.no_change_tolerance = no_change_tolerance
        self.cvar = cvar
        self.soft_cvar = soft_cvar
        self.optimistic_q = optimistic_q
        self.gamma = gamma
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.log_freq = log_freq
        self.max_distributed = int(max_distributed)

        self.env = None
        self.dd = pd.DataFrame()
        self.episode_map = None
        self.test_episodes_params = None
        self.agents = None
        self.agents_names = None
        self.best_train_iteration = {}
        self.valid_scores = {}
        self.test_actions = {}
        self.test_dx = {}
        self.test_dvx = {}
        self.test_dy = {}

        self.CEs = {}
        if ce_constructor is None:
            ce_constructor = CEM_Driving
        self.ce_constructor = ce_constructor
        if ce_n_orig is None:
            ce_n_orig = int(0.2*self.optim_freq) if ce_ref_mode=='train' else 0
        self.ce_update_freq = use_ce
        self.ce_warmup_turns = ce_warmup_turns
        self.ce_w_clip = ce_w_clip
        self.ce_ref_mode = ce_ref_mode
        self.ce_ref_alpha = ce_ref_alpha
        self.ce_n_orig = ce_n_orig
        self.ce_internal_alpha = ce_alpha

        self.make_env()
        self.generate_test_episodes()
        if agents is not None:
            self.set_agents(agents, mid_layers=agent_mid_layers,
                            generate_tables=True)

    def register_env(self):
        # first unregister if needed
        if 'DrivingSim-v0' in gym.envs.registry.env_specs:
            del gym.envs.registry.env_specs['DrivingSim-v0']
        gym.envs.registration.register(
            id='DrivingSim-v0',
            entry_point='DrivingSim:DrivingSim',
            kwargs=dict(T=self.max_episode_len, leader_probs=self.leader_probs,
                        p_oil=self.dist[0], rewards_conf=self.rewards_conf,
                        nine_actions=self.nine_actions)
        )

    def make_env(self):
        self.register_env()
        self.env = gym.make('DrivingSim-v0')

    def init_dd(self, n=1, agent='', group='', episode0=0, ag_updates=-1,
                ag_hash='', temperature=0, L=None, inplace=True, reset=False,
                update_map=True):
        if L is None: L = self.max_episode_len
        dd = pd.DataFrame(dict(
            agent          = (n * [agent]) if isinstance(agent, str) else agent,
            group          = (n * [group]) if isinstance(group, str) else group,
            episode        = np.arange(episode0, episode0+n),
            ag_updates     = n * [ag_updates],
            ag_hash        = n * [ag_hash],
            ag_temperature = n * [temperature],
            p_0            = n * [np.nan],
            p_acc          = n * [np.nan],
            p_dec          = n * [np.nan],
            p_hard         = n * [np.nan],
            p_turn         = n * [np.nan],
            x0             = n * [np.nan],
            y0             = n * [np.nan],
            v0             = n * [np.nan],
            th0            = n * [np.nan],
            brakes         = n * [np.nan],
            oil            = n * [np.nan],
            L              = n * [L],
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
        get_key = lambda i: (
            self.dd.agent[i], self.dd.group[i], self.dd.episode[i])
        self.episode_map = {get_key(i):i for i in range(len(self.dd))}

    def draw_episode_params(self, dyn_dist=None, seed=None):
        if seed is not None: np.random.seed(seed)
        if dyn_dist is None: dyn_dist = self.leader_probs
        dyn_dist = np.array(dyn_dist)
        return dyn_dist

    def generate_test_episodes(self):
        '''Draw dynamics for the test (and validation) episodes.'''
        n = self.n_valid + self.n_test
        dynamics, states = [], []
        for i in range(n):
            d = self.draw_episode_params()
            dynamics.append(d)
            states.append(Experiment.get_init_state(seed=i))
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
        dd_base.iloc[:,6:11] = self.test_episodes_params[0]
        dd_base.iloc[:,11:15] = self.test_episodes_params[1]

        for agent_nm in agents_names:
            dd = dd_base.copy()
            dd['agent'] = agent_nm
            self.dd = pd.concat((self.dd, dd))
        self.dd.reset_index(drop=True, inplace=True)
        self.build_episode_map()

    def set_agents(self, agents, mid_layers=(32,32), generate_tables=True):
        if not isinstance(agents, (tuple, list, dict)):
            agents = [agents]

        if isinstance(agents, dict):
            # dict of agents confs
            self.agents_names = []
            self.agents = {}
            for nm, (const, args) in agents.items():
                args['state_dim'] = STATE_DIM[self.state_mode]
                args['act_dim'] = 9 if self.nine_actions else 5
                if 'mid_sizes' not in args:
                    args['mid_sizes'] = mid_layers
                self.agents_names.append(nm)
                a = const(**args)
                a.title = nm
                self.agents[nm] = a
        else:
            # assuming list (or another iterable) of actual agents
            self.agents_names = [a.title for a in agents]
            self.agents = {a.title:a for a in agents}

        # apply the same initialization for all agents
        # (will fail if the architectures are different)
        agents_list = list(self.agents.values())
        for agent in agents_list[1:]:
            try:
                agent.load_state_dict(agents_list[0].state_dict())
            except:
                warnings.warn(
                    f'Cannot load model initialization for {agent.title}.')

        self.best_train_iteration = {a:-1 for a in self.agents_names}

        if generate_tables:
            self.generate_train_dd()
            self.generate_test_dd()

    def add_agent(self, agent, title=None, generate_tables=True):
        '''agent = agent OR (constructor, kwargs)'''
        if isinstance(agent, (tuple,list)):
            args = agent[1]
            args['state_dim'] = STATE_DIM[self.state_mode]
            args['act_dim'] = 9 if self.nine_actions else 5
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

    def save_agent(self, agent, nm=None, iter=False):
        if nm is None: nm = agent.title
        if self.title:
            nm = f'{self.title}_{nm}'
        if iter:
            nm = f'{nm}_iter{agent.n_updates:03d}'
        agent.save(nm)

    def load_agent(self, agent, nm=None, iter=None):
        if isinstance(agent, str):
            if nm is None: nm = agent
            agent = self.agents[agent]
        else:
            if nm is None: nm = agent.title
        if self.title:
            nm = f'{self.title}_{nm}'
        if iter is not None:
            nm = f'{nm}_iter{iter:03d}'
        agent.load(nm)

    def load_agents(self, agents=None):
        if agents is None: agents = self.agents_names
        for agent in agents:
            self.load_agent(agent)

    def load_optimizers(self, agents=None):
        if agents is None: agents = self.agents_names
        if isinstance(agents, str): agents = [agents]
        for agent_nm in agents:
            filename = f'models/{self.title}_{agent_nm}'
            if agent_nm not in self.optimizers:
                constructor = GCVaR.GCVaR
                hp = self.agents[agent_nm].train_hparams
                if 'nonepisodic_loss' in hp and hp['nonepisodic_loss']:
                    raise ValueError(
                        'Non-episodic loss is currently not supported.')
                self.optimizers[agent_nm] = constructor(None, 0)
                try:
                    self.optimizers[agent_nm].load(filename)
                except:
                    print(f'Cannot load optimizer: {filename}.opt')

    def load_CEs(self, agents=None):
        if agents is None: agents = self.agents_names
        if isinstance(agents, str): agents = [agents]
        for agent_nm in agents:
            filename = f'models/{self.title}_{agent_nm}'
            if agent_nm not in self.CEs:
                self.CEs[agent_nm] = self.ce_constructor(dist=self.dist)
            hp = self.agents[agent_nm].train_hparams
            if self.ce_update_freq or (
                    hp is not None and 'ce_update_freq' in hp and
                    hp['ce_update_freq'] > 0):
                try:
                    self.CEs[agent_nm].load(filename)
                except:
                    print(f'Cannot load CE: {filename}.cem')

    ###############   RUN ENV   ###############

    @staticmethod
    def get_init_state(seed=None, dx_scale=10, dv_shift=0, dv_scale=3,
                       mid_ratio=0.3, l=3.476):
        rng = np.random.RandomState(seed)
        # location
        x = 10 - 4*l - rng.exponential(dx_scale)
        # speed
        v = rng.normal(dv_shift, dv_scale)
        # lane
        y, th = -2, 0
        if rng.random() >= mid_ratio:
            # choose lane
            if rng.random() < 0.5:
                y = -4
            else:
                y = 0
        else:
            # choose tilt
            if rng.random() < 0.5:
                th = -0.2007 * 40/(40+v)
            else:
                th = 0.2007 * 40/(40+v)

        return np.array((x,y,v,th))

    def init_episode(self, episode, leader_actions=None, oil=None, L=None):
        # get params if already known
        seed = self.dd.episode.values[episode]
        dyn = self.dd.iloc[episode, 6:11].values.astype(np.float)
        s0 = self.dd.iloc[episode, 11:15].values.astype(np.float)
        if np.any(np.isnan(dyn)):
            dyn = None
        if L is None:
            L = self.max_episode_len

        # reset env with params
        out_dyn, out_s0 = self.env.reset(
            dyn, leader_actions=leader_actions, oil=oil, L=L,
            init_state=s0, seed=seed)

        return out_s0

    def get_agent_input(self, observation=None):
        if observation is None:
            observation = self.env.get_obs()
        if self.state_mode == 'default':
            agent_input = self.env.get_features()
        else:
            raise ValueError(self.state_mode)
        return agent_input

    def run_episode(self, episode, agent, render=False, leader_actions=None,
                    oil=None, update_res=False, temperature=None, L=None,
                    verbose=0, ax=None, gif=False, **kwargs):
        if isinstance(episode, (tuple, list)):
            episode = self.episode_map[tuple(episode)]
        if isinstance(agent, str):
            agent = self.agents[agent]
        if temperature is None:
            temperature = self.dd.ag_temperature[episode]
        if L is None:
            L = self.dd.L[episode]

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
            observation = self.init_episode(
                episode, leader_actions=leader_actions, oil=oil, L=L)

            # run episode
            for i in range(int(self.env.T//self.env.dt)):
                if verbose >= 2:
                    print(observation)

                agent_input = self.get_agent_input(observation)

                action, log_prob, _, _ = agent.act(
                    agent_input, T=temperature, verbose=verbose-2, **kwargs)
                observation, reward, done, info = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)

                if done:
                    break

            # summarize results
            T = len(rewards)
            n_brakes = np.sum(np.array(self.env.leader_actions)==3)
            n_oil = np.sum(self.env.oil)
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
                self.dd.loc[episode,'brakes'] = n_brakes
                self.dd.loc[episode,'oil'] = n_oil
                self.dd.loc[episode,'log_prob'] = log_prob.item()
                self.dd.loc[episode,'ret_loss'] = ret_loss.item()
                self.dd.loc[episode,'score'] = score
            if verbose >= 1:
                print(f'Score = {np.sum(rewards)}')

        except:
            self.env.close()
            raise

        if gif:
            i_ep = self.dd.episode.values[episode]
            self.env.create_gif(
                outname=f'outputs/{self.title}_{agent.title}_{i_ep}.gif')
        if render:
            self.env.show_trajectory(ax=ax)
            self.env.close()

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
        score, _, _, T = self.run_episode(
            idx, agent, render=True, verbose=verbose-1, **kwargs)
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
        (oil, s0, traj), w = ce.sample()
        self.dd.iloc[idx, 6:11] = ce.sample_dist[-1][4:-1]
        self.dd.iloc[idx, 11:15] = s0
        self.dd.loc[idx, 'oil'] = np.sum(oil)
        self.dd.loc[idx, 'ag_temperature'] = self.T
        self.dd.loc[idx, 'L'] = self.L
        return traj, oil, w

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
        optimizer_constructor = get_value('optimizer_constructor')
        lr = get_value('lr')
        weight_decay = get_value('weight_decay')
        optim_freq = get_value('optim_freq')
        optim_q_ref = get_value('optim_q_ref')
        Ti = get_value('Ti')
        Tf = get_value('Tf')
        cvar = get_value('cvar')
        soft_cvar = get_value('soft_cvar')
        optimistic_q = get_value('optimistic_q')
        L_init = get_value('L_init')

        # get CE hparams
        update_freq = get_value('ce_update_freq')
        ce_warmup_turns = get_value('ce_warmup_turns')
        w_clip = get_value('ce_w_clip')
        ref_mode = get_value('ce_ref_mode')
        ref_alpha = get_value('ce_ref_alpha')
        n_orig = get_value('ce_n_orig')
        internal_alpha = get_value('ce_internal_alpha')
        if update_freq == 0:
            ce_warmup_turns = 0
            n_orig = 0

        # prepare optimizer
        n_iters = self.n_train // optim_freq
        valid_fun = (lambda x: np.mean(sorted(x)[:int(np.ceil(cvar*len(x)))])) \
            if (0<cvar<1) else np.mean
        cvar_scheduler = (soft_cvar, n_iters)
        optimizer = optimizer_constructor(
            agent.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = None
        if self.lr_step:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.lr_step, gamma=self.lr_gamma)
        optimizer_wrap = GCVaR.GCVaR(
            optimizer, optim_freq, cvar, cvar_scheduler, skip_steps=ce_warmup_turns,
            optimistic_q=optimistic_q, lr_sched=lr_scheduler, title=agent.title)
        self.optimizers[agent.title] = optimizer_wrap

        # prepare CE
        if ref_alpha is None: ref_alpha = cvar
        ce = self.ce_constructor(
            dist=self.dist, l=self.env.l,
            batch_size=int(update_freq * optim_freq), w_clip=w_clip,
            ref_mode=ref_mode, ref_alpha=ref_alpha,
            n_orig_per_batch=n_orig, internal_alpha=internal_alpha,
            title=agent.title)
        self.CEs[agent.title] = ce

        if len(hparam_keys) > 0:
            warnings.warn(f'Unused hyper-params for {agent.title}: {hparam_keys}')
        return optimizer_wrap, valid_fun, optim_q_ref, \
               Ti, Tf, ref_mode=='valid', ce, L_init

    def train_log(self, agent, i_episode, tot_episodes, valid_mean, t0):
        # brake,left,nothing,right,gas -> nothing/brake/gas/left/right
        acts = 100*self.test_actions[agent]
        print(f'\t[{i_episode:03d}/{tot_episodes:03d}, {agent}] valid_mean='
              f'{valid_mean:.0f}\tactions: {acts[2]:.0f}/{acts[0]:.0f}/'
              f'{acts[4]:.0f}/{acts[1]:.0f}/{acts[3]:.0f}%\t'
              f'({time.time() - t0:.0f}s)')

    def train_agent(self, agent_nm, log_freq=None, verbose=1, **kwargs):
        if log_freq is None: log_freq = self.log_freq
        t0 = time.time()
        agent = self.agents[agent_nm]
        if agent.pretrained_filename is not None:
            self.load_agent(agent, agent.pretrained_filename)
        agent.train()

        optimizer_wrap, valid_fun, optim_q_ref, Ti, Tf, \
        ce_valid, ce, L_init = self.prepare_training(agent)

        # get episodes
        ids = np.arange(len(self.dd))[
            (self.dd.agent == agent_nm) & (self.dd.group == 'train')]

        self.T = Ti
        self.L = int(L_init)
        ce.original_dist[-1] = np.ceil(self.L/1.5)
        ce.sample_dist[-1][-1] = np.ceil(self.L/1.5)
        self.valid_scores[agent_nm] = [
            self.test(agent_nm, 'valid', update_inplace=False, temperature=0,
                      verbose=0)]
        valid_score = valid_fun(self.valid_scores[agent_nm][-1])
        valid_mean = np.mean(self.valid_scores[agent_nm][-1])
        best_valid_score = valid_score
        best_valid_mean = valid_mean
        self.best_train_iteration[agent_nm] = 0
        if self.save_best_model:
            self.save_agent(agent)
        if self.save_all_policies:
            self.save_agent(agent, iter=True)

        for i, idx in enumerate(ids):
            # draw episode params
            traj, oil, w = self.draw_train_episode_params(idx, ce)

            # run episode
            score, log_prob, ret_loss, n_steps = self.run_episode(
                idx, agent, leader_actions=traj, oil=oil, update_res=True,
                temperature=self.T, L=self.L, **kwargs)

            # update CEM
            if ce_valid:
                ce.update_ref_scores(self.valid_scores[agent_nm][-1])
            ce.update(score, f'models/{self.title}_{agent_nm}')

            # get reference scores for distribution estimation
            ref_scores = None
            if ce.n_orig_per_batch > 1:
                ref_scores = optimizer_wrap.scores[-1][:ce.n_orig_per_batch]
            elif optim_q_ref or (optim_q_ref is None and ce.batch_size):
                ref_scores = self.valid_scores[agent_nm][-1]

            # optimize
            if optimizer_wrap.step(w, log_prob, score, ref_scores,
                                   f'models/{self.title}_{agent_nm}'):
                agent.n_updates += 1
                if self.save_all_policies:
                    self.save_agent(agent, iter=True)
                self.T = Ti + (Tf-Ti) * i/len(ids)
                self.L = int(L_init + (self.max_episode_len-L_init) * min(
                    1.*i, 0.8*len(ids)) / (0.8*len(ids)) )
                ce.original_dist[-1] = np.ceil(self.L/1.5)
                ce.sample_dist[-1][-1] = np.ceil(self.L/1.5)

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
                        self.best_train_iteration[agent_nm] = agent.n_updates

                # stop if loss==0 for a while (e.g. all returns are the same...)
                losses = optimizer_wrap.losses
                tol = self.no_change_tolerance
                if tol and tol<len(losses) and len(np.unique(losses[-tol:]))==1:
                    if verbose >= 1:
                        print(f'{agent_nm} training stopped after {i+1} episodes'
                              f'and {tol} steps with constant loss.')
                    break

            # log
            if log_freq > 0 and (i % log_freq) == 0:
                self.train_log(agent_nm, i, len(ids), valid_mean, t0)

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
                self.best_train_iteration[agent_nm] = d[0]
                self.dd[(self.dd.agent == agent_nm) & (
                        self.dd.group == 'train')] = d[1]
                self.valid_scores[agent_nm] = d[2]

                # update agent
                self.load_agent(self.agents[agent_nm])
                self.agents[agent_nm].n_updates = d[1].ag_updates.values[-1] + 1

                # update optimizer
                self.load_optimizers(agent_nm)

                # update CE sampler
                self.load_CEs(agent_nm)

        try:
            self.enrich_results()
        except Exception as e:
            warnings.warn('Enrichment of optimizers and CEs data failed.')
            print(e)

    @staticmethod
    def train_agent_wrapper(q, E, agent_nm, kwargs):
        try:
            print(f'Training {agent_nm}...')
            E.train_agent(agent_nm, **kwargs)
        except:
            q.put((agent_nm, (
                -1,
                E.dd[(E.dd.agent==agent_nm)&(E.dd.group=='train')],
                E.valid_scores[agent_nm],
            )))
            print(f'Error in {agent_nm}.')
            raise
        q.put((agent_nm, (
            E.best_train_iteration[agent_nm],
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
            self.test_actions[agent_nm] = d[1]
            self.test_dx[agent_nm] = d[2]
            self.test_dvx[agent_nm] = d[3]
            self.test_dy[agent_nm] = d[4]

        return {a: s for a, s in zip(names,scores)}

    @staticmethod
    def test_agent_wrapper(q, E, agent_nm, kwargs, group='test'):
        try:
            scores = E.test(agent_nm, group=group, **kwargs)
        except:
            q.put((agent_nm, (
                E.dd[(E.dd.agent==agent_nm)&(E.dd.group==group)],
                E.test_actions[agent_nm],
                E.test_dx[agent_nm],
                E.test_dvx[agent_nm],
                E.test_dy[agent_nm],
            ), []))
            print(f'Error in {agent_nm}.')
            raise
        q.put((agent_nm, (
            E.dd[(E.dd.agent == agent_nm) & (E.dd.group == group)],
            E.test_actions[agent_nm],
            E.test_dx[agent_nm],
            E.test_dvx[agent_nm],
            E.test_dy[agent_nm],
        ), scores))

    def test(self, agent_nm=None, group='test', update_inplace=True, temperature=0,
             verbose=1, **kwargs):
        if agent_nm is None:
            return self.test_multiple_agents(
                self.agents_names, group=group, update_inplace=update_inplace,
                verbose=verbose, **kwargs)

        t0 = time.time()
        n_actions = self.env.action_space.n
        self.test_actions[agent_nm] = np.zeros(n_actions)
        self.test_dx[agent_nm] = []
        self.test_dvx[agent_nm] = []
        self.test_dy[agent_nm] = []
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
            # save episode results
            scores.append(score)
            x_ag, y_ag, x_l, y_l = self.env.get_trajectory()
            T = min(len(x_ag), len(x_l))
            dx = x_ag[:T] - x_l[:T]
            dvx = np.diff(x_ag[:T]) - np.diff(x_l[:T])
            dy = y_ag[:T] - y_l[:T]
            self.test_dx[agent_nm].append(dx-self.env.l)
            self.test_dvx[agent_nm].append(dvx)
            self.test_dy[agent_nm].append(dy)
            acts = np.array(self.env.agent_actions)
            self.test_actions[agent_nm] += np.array([
                np.mean(acts==i) for i in range(n_actions)])
        agent.train()
        self.test_actions[agent_nm] /= len(dd)

        if verbose >= 1:
            print(f'{agent_nm:s} tested ({time.time()-t0:.0f}s).')

        return scores

    ###############   ANALYSIS   ###############

    def enrich_results(self, optimizers=True, CEs=True):
        if optimizers:
            cols = dict(opt_selected='selected')
            for col in cols:
                self.dd[col] = np.nan
            for ag in self.optimizers:
                o1, o2 = self.optimizers[ag].get_data()
                if len(o2):
                    ids = (self.dd.agent==ag) & (self.dd.group=='train') & \
                          (~self.dd.score.isna())
                    for k,v in cols.items():
                        self.dd.loc[ids, k] = o2[v].values

        if CEs:
            cols = dict(ce_selected='selected', weight='weight')
            for col in cols:
                self.dd[col] = np.nan
            for ag in self.CEs:
                ce = self.CEs[ag]
                if ce.batch_count > 0:
                    c1, c2 = ce.get_data()
                    if len(c2):
                        ids = (self.dd.agent==ag) & (self.dd.group=='train') & \
                              (~self.dd.score.isna())
                        for k,v in cols.items():
                            self.dd.loc[ids, k] = c2[v].values

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
            if optimizers: self.load_optimizers(anm)
            if CEs: self.load_CEs(anm)

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
            try:
                # valid_scores[agent][iteration][episode]=score
                valid_scores = self.valid_scores[agent]
            except:
                warnings.warn(f'No validation data for {agent}.')
                continue
            valid_df = pd.concat((valid_df, pd.DataFrame(dict(
                agent=agent,
                group='valid',
                train_iteration=self.valid_freq * np.repeat(
                    np.arange(len(valid_scores)), len(valid_scores[0])),
                episode=len(valid_scores) * list(np.arange(len(valid_scores[0]))),
                score=np.concatenate(valid_scores)
            ))))

        train_valid_df = pd.concat((train_df, valid_df))
        train_valid_df['agent/group'] = [f'{a}_{g}' for a,g in zip(
            train_valid_df.agent,train_valid_df.group)]

        test_df = self.dd[(self.dd.group=='test') & (self.dd.agent.isin(agents))]

        # actions record
        actions = pd.DataFrame()
        for ag in agents:
            actions = pd.concat((actions, pd.DataFrame(dict(
                agent=ag,
                action=['brake','left','nothing','right','gas'],
                p=100*self.test_actions[ag],
            ))))

        # optimizer data
        opt_batch_data, opt_sample_data = pd.DataFrame(), pd.DataFrame()
        for ag in agents:
            try:
                d1, d2 = self.optimizers[ag].get_data()
            except:
                continue
            d1['agent'] = d1.title
            d2['agent'] = d2.title
            opt_batch_data = pd.concat((opt_batch_data, d1))
            opt_sample_data = pd.concat((opt_sample_data, d2))

        # CE data
        ce_batch_data, ce_sample_data = pd.DataFrame(), pd.DataFrame()
        for ag in agents:
            try:
                ce = self.CEs[ag]
                if not ce.batch_size:
                    continue
                d1, d2 = ce.get_data()
            except:
                continue
            ce_batch_data = pd.concat((ce_batch_data, d1))
            ce_sample_data = pd.concat((ce_sample_data, d2))
        if len(ce_batch_data) > 0:
            ce_batch_data['agent'] = ce_batch_data.title
            ce_sample_data['agent'] = ce_sample_data.title
            ce_batch_data.reset_index(drop=True, inplace=True)
            ce_sample_data.reset_index(drop=True, inplace=True)

        train_valid_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        actions.reset_index(drop=True, inplace=True)
        opt_batch_data.reset_index(drop=True, inplace=True)
        opt_sample_data.reset_index(drop=True, inplace=True)

        return train_valid_df, test_df, actions,\
               opt_batch_data, opt_sample_data, ce_batch_data, ce_sample_data

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
                est = lambda x: np.mean(-np.sort(-x)[:int(np.ceil(alpha * len(x)))])
                est_name = f'$CVaR_{{{alpha:.2f}}}$'
            else:
                raise ValueError(est_name)
            ests[est_name] = est

        return ests

    def analyze(self, agents=None, W=3, axsize=(6,4), Qs=(100,5),
                train_estimators=('mean','cvar05','cvar01'),
                verify_train_success=False):
        if agents is None: agents = self.agents_names
        train_estimators = self.build_estimators(train_estimators)
        train_valid_df, test_df, actions, opt_batch_data, opt_sample_data, \
        ce_batch_data, ce_sample_data = self.analysis_preprocessing(agents)
        if verify_train_success:
            agents = [ag for ag in agents if self.best_train_iteration[ag]>=0]

        axs = utils.Axes(6+2*len(train_estimators)+len(Qs)+(len(ce_sample_data)>0)+\
                         9*(len(ce_batch_data)>0)+(len(actions)>0),
                         W, axsize, fontsize=15)
        a = 0

        # Train scores
        train_valid_df['cost'] = -train_valid_df.score
        for group in ('train','valid'):
            for est_name, est in train_estimators.items():
                sns.lineplot(data=train_valid_df[train_valid_df.group==group],
                             x='train_iteration', hue='agent', y='cost',
                             estimator=est, ax=axs[a],
                             ci=None if group=='train' else 95)
                axs[a].set_xlim((0,None))
                axs[a].set_yscale('log')
                lg = axs[a].get_legend()
                if lg is not None:
                    plt.setp(lg.get_texts(), fontsize='13')
                axs.labs(a, 'train iteration', f'{est_name} cost', f'{group} data')
                if group == 'valid':
                    axs[a].set_ylim((None,2000))
                a += 1

        # Test scores
        n_iters = self.n_train // self.optim_freq
        cvar = lambda x, alpha: np.mean(np.sort(x)[:int(np.ceil(alpha*len(x)))])
        for agent in agents:
            scores = test_df.score[test_df.agent==agent].values
            print(f'{agent} ({self.best_train_iteration[agent]}/{n_iters}):\t'
                  f'mean={np.mean(scores):.1f}\t'
                  f'CVaR05={cvar(scores,0.05):.1f}\tCVaR01={cvar(scores,0.01):.1f}')
        for Q in Qs:
            for agent in agents:
                scores = test_df.score[test_df.agent==agent].values
                utils.plot_quantiles(
                    scores, q=np.linspace(0,Q/100,101), showmeans=True, ax=axs[a],
                    label=f'{agent} ({np.mean(scores):.1f})')
            axs[a].set_xlim((0,Q))
            axs.labs(a, 'episode quantile [%]', 'score')
            axs[a].legend(fontsize=13)
            a += 1

        # Scores per number of hard brakes
        sns.boxplot(data=test_df, x='brakes', hue='agent', y='score',
                    showmeans=True, ax=axs[a])
        axs.labs(a, 'emergency brakes in episode', 'score')
        axs[a].legend(fontsize=13)
        a += 1

        # Agent behavior analysis
        self.analyze_behavior(agents, axs, a)
        a += 3

        # Agent actions distribution
        if len(actions) > 0:
            sns.barplot(data=actions, x='action', hue='agent', y='p', ax=axs[a])
            axs.labs(a, 'action', 'action frequency [%]')
            axs[a].legend(fontsize=13)
            time.sleep(0.1)
            axs[a].set_xticklabels(axs[a].get_xticklabels(), fontsize=13)
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

        # CE: distributions and weights
        if len(ce_sample_data) > 0:
            iter_resol = 1 + ce_sample_data.batch.values[-1] // 4
            ce_sample_data['iter_rounded'] = [
                iter_resol*(it//iter_resol) for it in ce_sample_data.batch]
            sns.boxplot(data=ce_sample_data, x='iter_rounded', hue='agent',
                        y='weight', showmeans=True, ax=axs[a])
            axs.labs(a, 'train iteration', 'weight')
            axs[a].legend(fontsize=13)
            a += 1

        if len(ce_batch_data) > 0:
            ynms = self.dd.columns[6:11]
            ylabs = ('do-nothing', 'accelerate', 'deccelerate', 'emergency brake',
                     'turn')
            for i in range(len(ylabs)):
                axs[a].axhline(100*self.leader_probs[i], color='k', label='original')
                for agent in agents:
                    y = 100 * ce_batch_data[ynms[i]][
                        ce_batch_data.agent==agent].values
                    if np.any(y!=100*self.leader_probs[i]):
                        axs[a].plot(np.arange(len(y)), y, '-', label=agent)
                if ylabs[i] == 'emergency brake':
                    axs[a].set_ylim((0, None))
                else:
                    axs[a].set_ylim((0, 100))
                axs.labs(a, 'train iteration', f'{ylabs[i]} probability [%]')
                axs[a].legend(fontsize=13)
                a += 1

            for i, lab in enumerate(('x0','v0')):
                axs[a].axhline(self.dist[1+i], color='k', label='original')
                for agent in agents:
                    y = ce_batch_data[lab][ce_batch_data.agent==agent].values
                    if np.any(y!=self.dist[1+i]):
                        axs[a].plot(np.arange(len(y)), y, '-', label=agent)
                axs.labs(a, 'train iteration', f'average {lab}')
                axs[a].legend(fontsize=13)
                a += 1

            axs[a].axhline(100*self.dist[1+2], color='k', label='original')
            for agent in agents:
                y = 100 * ce_batch_data['mid'][ce_batch_data.agent==agent].values
                if np.any(y!=100*self.dist[1+2]):
                    axs[a].plot(np.arange(len(y)), y, '-', label=agent)
            axs.labs(a, 'train iteration', f'mid-lane probability [%]')
            axs[a].legend(fontsize=13)
            a += 1

            axs[a].axhline(100*self.dist[0], color='k', label='original')
            for agent in agents:
                y = 100 * ce_batch_data['oil'][ce_batch_data.agent==agent].values
                if np.any(y!=100*self.dist[0]):
                    axs[a].plot(np.arange(len(y)), y, '-', label=agent)
            axs.labs(a, 'train iteration', f'oil probability [%]')
            axs[a].legend(fontsize=13)
            a += 1

        plt.tight_layout()
        return axs

    def analyze_behavior(self, agents=None, axs=None, a0=0):
        if agents is None: agents = self.agents_names
        if axs is None: axs = utils.Axes(3, 3)
        a = a0

        for metric, lab in zip(
                (self.test_dx, self.test_dvx, self.test_dy), ('dx','dvx','dy')):
            Q = np.arange(1,101)/100 if lab=='dx' else None
            for agent in agents:
                if agent not in metric:
                    continue
                x = np.concatenate(metric[agent])
                utils.plot_quantiles(
                    x, q=Q, ax=axs[a],
                    label=f'{agent} (n={len(x)}, mean={np.mean(x):.1f})')
            axs[a].set_xlim((0,100))
            axs.labs(a, 'time-step quantile [%]', lab)
            axs[a].legend(fontsize=13)
            a += 1

        return axs

    def show_tests(self, agents=None, n=3, clean_plot=False, **kwargs):
        if agents is None: agents = self.agents_names
        if isinstance(agents, str): agents = [agents]
        dt = self.env.dt

        axs = utils.Axes(5*n*len(agents), 5, (5,3.5), fontsize=15)
        plt.tight_layout()
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
                s, _, _, T = self.run_episode(
                    (agent, 'test', episodes[i]), self.agents[agent],
                    update_res=False, verbose=0, **kwargs)
                x, y, xl, yl = self.env.get_trajectory()
                vx, vxl = np.diff(x)/dt, np.diff(xl)/dt
                dx = x+self.env.l-xl[:len(x)]
                axs[a].axhline(0, color='k')
                axs[a].plot(dt*np.arange(len(dx)), dx, '-')
                axs.labs(a, 't', '$\Delta$x', f'{agent}: score={scores[i]:.0f}')
                for j, (z,zl) in enumerate(((vx,vxl), (y,yl))):
                    axs[a+1+j].plot(dt*np.arange(len(z)), z, '-', label='agent')
                    axs[a+1+j].plot(dt*np.arange(len(zl)), zl, '-', label='leader')
                    axs[a+1+j].legend(fontsize=12)
                    if clean_plot or scores[i]==s:
                        axs.labs(a+1+j, 't', ['vx','y'][j],
                                 f'{agent}: score={scores[i]:.0f}')
                    else:
                        axs.labs(a+1+j, 't', ['vx','y'][j],
                                 f'{agent} ({ids[i]+1}/{N})\n'
                                 f'score={scores[i]:.0f} (L={T})\n'
                                 f'reproduced={s:.0f}')

                self.env.show_rewards_and_returns(axs, a+3)
                for j in range(2):
                    axs[a+3+j].set_title(
                        f'{agent} ({episodes[i]}: {ids[i]+1}/{N}), '
                        f'score={scores[i]:.0f}',
                        fontsize=15)

                a += 5

        plt.tight_layout()
        return axs

    def analyze_episode(self, episode, agents=None, group='test', axs=None, a0=0,
                        **kwargs):
        if agents is None: agents = self.agents_names
        if isinstance(agents, str): agents = [agents]
        if axs is None:
            axs = utils.Axes(4+2*len(agents), 4, (5,3.5), fontsize=15)
        a = a0 + 4

        dt = self.env.dt
        for i, agent in enumerate(agents):
            s, _, _, T = self.run_episode(
                (agent, group, episode), self.agents[agent],
                update_res=False, verbose=0, **kwargs)
            x, y, xl, yl = self.env.get_trajectory()
            t = dt*np.arange(max(len(x),len(xl)))
            vx, vxl = np.diff(x)/dt, np.diff(xl)/dt
            x -= 40*t[:len(x)]
            xl -= 40*t[:len(xl)]
            axs[a0].axhline(0, color='k')
            axs[a0].plot(dt*np.arange(len(x)), x+self.env.l-xl[:len(x)], '-',
                         label=f'{agent}: {s:.0f}')
            axs[a0].legend(fontsize=12)
            axs.labs(a0, 't', '$\Delta$x')
            for j, (z,zl) in enumerate(((x,xl), (vx,vxl), (y,yl))):
                if i==0:
                    axs[a0+1+j].plot(t[:len(zl)], zl, '-', label='leader')
                axs[a0+1+j].plot(t[:len(z)], z, '-', label=f'{agent}: {s:.0f}')
                axs[a0+1+j].legend(fontsize=12)
                axs.labs(a0+1+j, 't', ['x - 40$\cdot$t','vx','y'][j])

            self.env.show_rewards_and_returns(axs, a)
            for j in range(2):
                axs[a+j].set_title(
                    f'{agent} ({episode}), score={s:.0f}', fontsize=15)
            a += 2

        plt.tight_layout()
        return axs

    def main(self, **analysis_args):
        self.train_with_dependencies()
        self.test()
        return self.analyze(**analysis_args)


class CEM_Driving(CEM.CEM):
    '''CEM for discrete probability distribution.'''

    def __init__(self, update_s0=False, l=3.476, *args, **kwargs):
        super(CEM_Driving, self).__init__(*args, **kwargs)
        self.update_s0 = update_s0
        self.l = l
        self.default_dist_titles = ('oil', 'x0','v0','mid',
                                    'p_0','p_acc','p_dec','p_hard','p_turn','L')

    def do_sample(self, dist):
        p_oil = dist[0]
        x, v, mid = dist[1:4]
        p = dist[4:-1]
        n = int(dist[-1])

        oil = [np.random.random()<p_oil for _ in range(n)]
        s0 = Experiment.get_init_state(None, x, v, mid_ratio=mid)
        traj = np.random.choice(np.arange(len(p)), n, p=p)  # try non-iid actions?
        return oil, s0, traj

    def pdf_oil(self, oil, dist):
        n = len(oil)
        oil = int(np.sum(oil))
        return stats.binom.pmf(oil, n, dist[0])

    def pdf_init(self, s0, dist):
        p_x = stats.expon.pdf(10-4*self.l-s0[0], 0, dist[1])
        p_v = stats.norm.pdf(s0[2], dist[2], 3)
        p_th = 1-dist[3] if s0[3]==0 else dist[3]
        p = p_x * p_v * p_th
        return p

    def pdf(self, x, dist):
        oil, s0, traj = x
        p_oil = self.pdf_oil(oil, dist)
        p_init = self.pdf_init(s0, dist)
        p_traj = np.prod([dist[4+i]**np.sum(traj==i)
                          for i in range(len(dist)-4-1)])
        return p_oil * p_init * p_traj

    def log_p_traj(self, x, dist):
        return np.sum([np.sum(x==i) * np.log(dist[4+i])
                       for i in range(len(dist)-4-1)])

    def likelihood_ratio(self, x, use_original_dist=False):
        if use_original_dist:
            return 1

        dist0 = self.sample_dist[0]
        dist1 = self.sample_dist[-1]

        p_oil = self.pdf_oil(x[0], dist0) / self.pdf_oil(x[0], dist1)
        p_init = self.pdf_init(x[1],dist0) / self.pdf_init(x[1],dist1)

        logp0 = self.log_p_traj(x[2], dist0)
        logp1 = self.log_p_traj(x[2], dist1)
        p_traj = np.exp(logp0-logp1)

        return p_oil * p_init * p_traj

    def update_sample_distribution(self, samples, weights):
        w = np.array(weights)
        w_mean = np.mean(w)
        out = []

        # oil frequency
        oil0 = self.original_dist[0]
        if oil0:
            oil = np.mean(np.array([s[0] for s in samples]), axis=1)
            out.append(np.clip(np.mean(w*oil)/w_mean, oil0/10, 1-(1-oil0)/10))
        else:
            out.append(0)

        # init state
        if self.update_s0:
            s0 = np.array([s[1] for s in samples])
            x = 10-4*self.l - np.mean(w*s0[:,0]) / w_mean
            v = np.mean(w*s0[:,2]) / w_mean
            mid = np.mean(w*(s0[:,3]!=0)) / w_mean
            out.extend([np.clip(x,3,30), np.clip(v,-4,4),
                        np.clip(mid,0.05,0.95)])
        else:
            out.extend(list(self.original_dist[1:4]))

        # trajectory (actions)
        s = np.array([s[2] for s in samples])
        dim = s.shape[1]

        n = len(self.sample_dist[-1]) - 4 - 1
        dist = [np.mean(w*np.mean(s==i, axis=1))/w_mean for i in range(n-1)]
        # regularization: don't let the probabilities vanish
        reg = 0.1
        dist = [(1-reg)*p + reg/(len(dist)+1) for p in dist]
        # last prob completes to 1
        dist.append(1-np.sum(dist))
        # last entry represents the length
        dist.append(dim)
        out.extend(dist)

        return np.array(out)


SAMPLE_AGENT_CONFS = dict(
    PG = (Agents.FC, dict()),
    GCVaR = (Agents.FC, dict(train_hparams=dict(cvar=0.05))),
    CE_SGCVaR = (Agents.FC, dict(train_hparams=dict(
        cvar=0.05, ce_update_freq=1, soft_cvar=0.6))),
)

if __name__ == '__main__':

    E = Experiment(SAMPLE_AGENT_CONFS, train_episodes=6000,
                   valid_episodes=40, test_episodes=500,
                   optim_freq=400, title='demo')
    E.main()
    print(E.dd.tail())
    E.show_tests(clean_plot=True)
    plt.show()
