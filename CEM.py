
import numpy as np
from scipy import stats

class CEM:
    def __init__(self, mode='CartPole', dyn_dist=None, s0_dist=None,
                 modify_dyn=True, modify_s0=False, source_sample_perc=0,
                 update_freq=100, update_perc=0.2, IS=False, valid_ref=False,
                 use_beta=True, verbose=1):
        self.mode = mode
        self.verbose = verbose
        if dyn_dist is None:
            dyn_dist = (9.8, 1, 0.1, 0.5, 1e-3) if mode == 'CartPole' else (0.1,)
        if s0_dist is None:
            s0_dist = (0,0,0,0,0.05,0.05,0.05,0.05) if mode == 'CartPole' \
                else ((1,1,6,6))
        self.source_dyn_dist = np.array(dyn_dist, dtype=np.float32)
        self.source_s0_dist = np.array(s0_dist, dtype=np.float32)
        self.modify_dyn = modify_dyn
        self.modify_s0 = modify_s0
        self.update_freq = update_freq
        self.update_perc = update_perc
        self.IS = IS
        self.valid_ref = valid_ref
        self.n_source = int(np.ceil(source_sample_perc * self.update_freq)) \
            if source_sample_perc<=1 else source_sample_perc
        if self.update_freq and self.n_source:
            self.update_perc = update_perc * (1-self.n_source/self.update_freq)

        self.use_beta = use_beta
        self.sample_dyn_dist = self.source_dyn_dist
        self.sample_s0_dist = self.source_s0_dist
        self.curr_episodes = dict(ret=[], s0=[], dyn=[])
        self.history = dict(dyn_dists=[self.sample_dyn_dist],
                            s0_dists=[self.sample_s0_dist])
        self.w_history = []

    def reset(self):
        self.sample_dyn_dist = self.source_dyn_dist
        self.sample_s0_dist = self.source_s0_dist
        self.curr_episodes = dict(ret=[], s0=[], dyn=[])
        self.history = dict(dyn_dists=[self.sample_dyn_dist],
                            s0_dists=[self.sample_s0_dist])
        self.w_history = []

    def get_dists(self, use_source=False, dct=True):
        dyn_dist = self.source_dyn_dist if use_source else self.sample_dyn_dist
        s0_dist = self.source_s0_dist if use_source else self.sample_s0_dist
        if dct:
            return dict(dynamics_dist=dyn_dist, s0_dist=s0_dist)
        return dyn_dist, s0_dist

    def sample(self, dyn_clip=5, s0_clip=(1.2, np.inf, 0.1, np.inf), use_source=None):
        if use_source is None:
            # for the first n_source samples in the batch - use original distribution
            use_source = len(self.curr_episodes['ret']) < self.n_source
        dyn_dist = self.source_dyn_dist if use_source else self.sample_dyn_dist
        s0_dist = self.source_s0_dist if use_source else self.sample_s0_dist

        if self.use_beta:
            dyn = np.random.beta(2*dyn_dist, 2-2*dyn_dist)
        else:
            dyn = np.random.exponential(np.array(dyn_dist), size=(len(dyn_dist),))
        if dyn_clip:
            dyn = np.clip(dyn, self.source_dyn_dist / dyn_clip,
                          dyn_clip * self.source_dyn_dist)

        if self.mode == 'CartPole':
            means = s0_dist[:4]
            sigmas = s0_dist[4:]
            s0 = np.random.normal(means, sigmas, size=(4,))
            s0_clip = np.array(s0_clip)
            s0 = np.clip(s0, -s0_clip, s0_clip)
        else:
            s0 = np.random.randint(
                s0_dist[:2], np.array(s0_dist[2:])+1, size=(2,))

        w = self.sample_weight(dyn, s0, use_source)
        n = len(self.history['dyn_dists']) - 1
        if len(self.w_history) <= n:
            self.w_history.append([])
        self.w_history[n].append(w)

        return dyn, s0, w

    def update_dists(self, reset_recording=True, q_ref=None, cvar=None):
        if q_ref is not None and self.n_source>1 and cvar is not None and (0<cvar<1):
            # overwrite q_ref using recent training episodes
            q_ref_valid = q_ref
            q_ref = np.percentile(self.curr_episodes['ret'][:self.n_source], 100*cvar)
            if self.verbose >= 2:
                print(f'\t\t[{len(self.history["s0_dists"])}; alpha={cvar:.2f}] valid={q_ref_valid:.2f}, '
                      f'train({self.source_dyn_dist[0]:.2f})={q_ref:.1f}, '
                      f'train({self.sample_dyn_dist[0]:.2f})={np.percentile(self.curr_episodes["ret"][self.n_source:], 100*cvar):.1f}')

        # Sort recordings
        ids = np.argsort(self.curr_episodes['ret'])
        for k in self.curr_episodes:
            self.curr_episodes[k] = [self.curr_episodes[k][i] for i in ids]

        # Select worst episodes
        i0 = int(self.update_perc*len(self.curr_episodes['ret'])) + 1
        if q_ref is not None and self.curr_episodes['ret'][i0-1]<q_ref:
            # all i0 episodes are below the ref. quantile - can take more episodes
            ii = int(np.sum(np.array(self.curr_episodes['ret'])<q_ref))
            if self.verbose >= 2:
                print(f'\t\t[{len(self.history["s0_dists"])}] '
                      f'r[{i0-1}]={self.curr_episodes["ret"][i0-1]:.1f}<'
                      f'{q_ref:.1f}=q_alpha; taking {ii} samples instead.')
            i0 = ii

        # Update dists from recordings
        if self.modify_dyn:
            if self.use_beta:
                X = np.stack(self.curr_episodes['dyn'][:i0])
                self.sample_dyn_dist = np.mean(X, axis=0)
            else:
                self.sample_dyn_dist = np.mean(np.stack(self.curr_episodes['dyn'][:i0]), axis=0)
        if self.modify_s0:
            if self.mode == 'CartPole':
                ddof = 1  # for unbiased estimator
                self.sample_s0_dist = np.concatenate((
                    np.mean(np.stack(self.curr_episodes['s0'][:i0])[:,:4], axis=0),
                    np.std(np.stack(self.curr_episodes['s0'][:i0])[:,:4], axis=0, ddof=ddof)
                ))
            else:
                raise ValueError('smart sampling of initial state is not supported '
                                 f'for {self.mode}.')
        self.history['dyn_dists'].append(self.sample_dyn_dist)
        self.history['s0_dists'].append(self.sample_s0_dist)

        # Reset recordings
        if reset_recording:
            self.curr_episodes = dict(ret=[], s0=[], dyn=[])

    def update_recording(self, ret, dyn, s0, update_dists=True,
                         update_q_ref=None, cvar=None):
        self.curr_episodes['ret'].append(ret)
        self.curr_episodes['dyn'].append(dyn)
        self.curr_episodes['s0'].append(s0)
        if update_dists:
            if self.update_freq>0 and len(self.curr_episodes['ret']) >= self.update_freq:
                self.update_dists(reset_recording=True, q_ref=update_q_ref, cvar=cvar)

    def likelihood_ratio(self, dyn, s0, use_source=False):
        if use_source:
            return 1

        if self.use_beta:
            sorc = self.source_dyn_dist[0]
            samp = self.sample_dyn_dist[0]
            lr_dyn = stats.beta.pdf(dyn[0], 2*sorc, 2-2*sorc) / \
                     stats.beta.pdf(dyn[0], 2*samp, 2-2*samp)
        else:
            lr_dyn = np.prod( lr_exp(dyn, self.sample_dyn_dist, self.source_dyn_dist) )
        if self.mode == 'CartPole':
            lr_s0 = np.prod(
                pdf_norm(s0, self.source_s0_dist[:4], self.source_s0_dist[4:]) /
                pdf_norm(s0, self.sample_s0_dist[:4], self.sample_s0_dist[4:])
            )
        else:
            lr_s0 = 1

        return lr_dyn * lr_s0

    def sample_weight(self, dyn, s0, use_source=False, w_clip=5):
        if not self.IS:
            return 1
        lr = self.likelihood_ratio(dyn, s0, use_source)
        if w_clip:
            lr = np.clip(lr, 1/w_clip, w_clip)
        return lr

def pdf_exp(x, a):
    a = np.array(a)
    return np.exp(-x/a)/a

def lr_exp(x, a1, a2):
    a1 = np.array(a1)
    a2 = np.array(a2)
    return a1/a2*np.exp(-(1/a2-1/a1)*x)

def pdf_norm(x, mu, sigma):
    mu = np.array(mu)
    sigma = np.array(sigma)
    return stats.norm.pdf((x-mu)/sigma) / sigma

def beta_moments_method(mean=None, var=None, x=None):
    if mean is None:
        mean = np.mean(x)
    if var is None:
        var = np.var(x)
    var = min(var, 0.5*mean*(1-mean))
    alpha = mean * (mean*(1-mean)/var - 1)
    beta = (1-mean) * (mean*(1-mean)/var - 1)
    return np.array((alpha, beta)).reshape(-1)
