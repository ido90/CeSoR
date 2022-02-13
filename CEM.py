'''
This module implements the Cross Entropy Method (CEM) for sampling of low quantiles.

The user should have a stochastic process P(score; theta), whose distribution
depends on the parameter theta (the module was originally developed to sample
"difficult" conditions in reinforcement learning, so theta was the parameters
of the environment, and the score was the return of the episode).

The basic CEM class is abstract: any use of it requires inheritance with
implementation of the following methods:
- do_sample(dist): returning a sample from the distribution represented by dist.
- pdf(x, dist): the probability of x under the distribution dist.
- update_sample_distribution(samples, weights): updating dist given new samples.
- likelihood_ratio(x) (optional): implemented by default as
                                  pdf(x, orig_dist) / pdf(x, curr_dist).
                                  the user may provide a more efficient or stable
                                  implementation according to the underlying
                                  family of distributions.
Note that dist is an object that represents the distribution, and its type is
up to the user. A standard type may be a list of distribution parameters.

An example for an inheritance from CEM is provided below - see CEM_Beta_1D.
A simple usage example is provided in the bottom of this file - see __main__.

Module structure:
CEM:
    sample(): return a sample from the current distribution, along with a weight
              corresponding to the likelihood-ratio wrt the original distribution.
        do_sample(curr_dist): do the sampling.              [IMPLEMENTED BY USER]
        get_weight(x): calculate the LR weight.
            likelihood_ratio(x).
                pdf(x, dist).                               [IMPLEMENTED BY USER]
    update(score):
        update list of scores.
        if there're enough samples and it's time to update the distribution:
            select_samples().
            update_sample_distribution(samples, weights).   [IMPLEMENTED BY USER]
CEM_Beta_1D: an inherited class implementing CEM for a 1D Beta distribution.

Written by Ido Greenberg, 2022.
'''

import numpy as np
from scipy import stats
import pandas as pd
import pickle as pkl
import warnings


class CEM:
    def __init__(self, dist, batch_size=0, ref_mode='train', ref_alpha=0.05,
                 n_orig_per_batch=0.2, internal_alpha=0.2, w_clip=5, title='CEM'):
        self.title = title

        # An object defining the original distribution to sample from.
        # This can be any object (e.g., list of distribution parameters),
        # depending on the implementation of the inherited class.
        self.original_dist = dist

        # Number of samples to draw before updating distribution.
        # 0 is interpreted as infinity.
        self.batch_size = batch_size

        # Clip the likelihood-ratio weights to the range [1/w_clip, w_clip].
        # If None or 0 - no clipping is done.
        self.w_clip = w_clip

        # How to use reference scores to determine the threshold for the
        # samples selected for distribution update?
        # - 'none': ignore reference scores.
        # - 'train': every batch, draw the first n=n_orig_per_batch samples
        #            from the original distribution instead of the updated one.
        #            then use quantile(batch_scores[:n]; ref_alpha).
        # - 'valid': use quantile(external_validation_scores; ref_alpha).
        #            in this case, update_ref_scores() must be called to
        #            feed reference scores before updating the distribution.
        # In CVaR optimization, ref_alpha would typically correspond to
        # the CVaR risk level.
        self.ref_mode = ref_mode
        self.ref_alpha = ref_alpha

        # Number of samples to draw every batch from the original distribution
        # instead of the updated one. Either integer or ratio in (0,1).
        self.n_orig_per_batch = n_orig_per_batch
        if 0<self.n_orig_per_batch<1:
            self.n_orig_per_batch = int(self.n_orig_per_batch*self.batch_size)

        active_train_mode = self.ref_mode == 'train' and self.batch_size
        if active_train_mode and self.n_orig_per_batch < 1:
            raise ValueError('"train" reference mode must come with a positive'
                             'number of original-distribution samples per batch.')

        # In a distribution update, use from the current batch at least
        # internal_alpha * (batch_size - n_orig_per_batch)
        # samples. internal_alpha should be in the range (0,1).
        self.internal_alpha = internal_alpha
        if active_train_mode:
            self.internal_alpha *= 1 - self.n_orig_per_batch / self.batch_size

        # State
        self.batch_count = 0
        self.sample_count = 0
        self.update_count = 0
        self.ref_scores = None

        # Data
        self.sample_dist = []  # n_batches
        self.sampled_data = [[]]  # n_batches x batch_size
        self.weights = [[]]  # n_batches x batch_size
        self.scores = [[]]  # n_batches x batch_size
        self.ref_quantile = []  # n_batches
        self.internal_quantile = []  # n_batches
        self.selected_samples = [[]]  # n_batches x batch_size
        self.n_update_samples = []  # n_batches

        self.reset()

    def reset(self):
        self.batch_count = 0
        self.sample_count = 0
        self.update_count = 0
        self.ref_scores = None

        self.sample_dist = [self.original_dist]
        self.sampled_data = [[]]
        self.weights = [[]]
        self.scores = [[]]
        self.ref_quantile = []
        self.internal_quantile = []
        self.selected_samples = [[]]
        self.n_update_samples = []

    def save(self, filename=None):
        if filename is None: filename = f'models/{self.title}'
        filename += '.cem'
        obj = (
            self.title, self.original_dist, self.batch_size, self.w_clip,
            self.ref_mode, self.ref_alpha, self.n_orig_per_batch, self.internal_alpha,
            self.batch_count, self.sample_count, self.update_count, self.ref_scores,
            self.sample_dist, self.sampled_data, self.weights, self.scores,
            self.ref_quantile, self.internal_quantile, self.selected_samples,
            self.n_update_samples
        )
        with open(filename, 'wb') as h:
            pkl.dump(obj, h)

    def load(self, filename=None):
        if filename is None: filename = f'models/{self.title}'
        filename += '.cem'
        with open(filename, 'rb') as h:
            obj = pkl.load(h)
        self.title, self.original_dist, self.batch_size, self.w_clip, \
        self.ref_mode, self.ref_alpha, self.n_orig_per_batch, self.internal_alpha, \
        self.batch_count, self.sample_count, self.update_count, self.ref_scores, \
        self.sample_dist, self.sampled_data, self.weights, self.scores, \
        self.ref_quantile, self.internal_quantile, self.selected_samples, \
        self.n_update_samples = obj

    def is_original_dist(self):
        return self.sample_count < self.n_orig_per_batch

    ########   Sampling-related methods   ########

    def sample(self):
        orig_dist = self.is_original_dist()
        dist = self.sample_dist[0] if orig_dist else self.sample_dist[-1]
        x = self.do_sample(dist)
        w = self.get_weight(x, orig_dist)
        self.sampled_data[-1].append(x)
        self.weights[-1].append(w)
        self.sample_count += 1
        if 0 < self.batch_size < self.sample_count:
            warnings.warn(f'Drawn {self.sample_count}>{self.batch_size} samples without '
                          f'updating (only {self.update_count}<{self.batch_size} '
                          'scores for update)')
        return x, w

    def get_weight(self, x, use_original_dist=False):
        if use_original_dist:
            return 1

        lr = self.likelihood_ratio(x)
        if self.w_clip:
            lr = np.clip(lr, 1/self.w_clip, self.w_clip)
        return lr

    def likelihood_ratio(self, x, use_original_dist=False):
        if use_original_dist:
            return 1
        return self.pdf(x, self.sample_dist[0]) / \
               self.pdf(x, self.sample_dist[-1])

    def do_sample(self, dist):
        '''Given distribution parameters, return a sample drawn from the distribution.'''
        raise NotImplementedError()

    def pdf(self, x, dist):
        '''Given a sample x and distribution parameters dist, return P(x|dist).'''
        raise NotImplementedError()

    ########   Update-related methods   ########

    def update(self, score, save=True):
        self.scores[-1].append(score)
        self.update_count += 1

        if 0 < self.batch_size <= self.update_count:

            self.select_samples()
            samples = [self.sampled_data[-1][i] for i in range(self.sample_count)
                       if self.selected_samples[-1][i]]
            weights = [self.weights[-1][i] for i in range(self.sample_count)
                       if self.selected_samples[-1][i]]

            if len(samples) > 0:
                dist = self.update_sample_distribution(samples, weights)
            else:
                dist = self.sample_dist[-1]
            self.sample_dist.append(dist)

            self.reset_batch()
            if save:
                filename = save if isinstance(save, str) else None
                self.save(filename)

    def reset_batch(self):
        self.sampled_data.append([])
        self.scores.append([])
        self.weights.append([])
        self.batch_count += 1
        self.sample_count = 0
        self.update_count = 0

    def select_samples(self):
        # Get internal quantile
        q_int = quantile(self.scores[-1], self.internal_alpha)

        # Get reference quantile from "external" data
        q_ref = -np.inf
        if self.ref_mode == 'train':
            q_ref = quantile(self.scores[-1][:self.n_orig_per_batch],
                             self.ref_alpha)
        elif self.ref_mode == 'valid':
            if self.ref_scores is None:
                warnings.warn('ref_mode=valid, but no '
                              'validation scores were provided.')
            else:
                q_ref = quantile(self.ref_scores, 100*self.ref_alpha)
        elif self.ref_mode == 'none':
            q_ref = -np.inf
        else:
            warnings.warn(f'Invalid ref_mode: {self.ref_mode}')

        # Take the max over the two
        self.internal_quantile.append(q_int)
        self.ref_quantile.append(q_ref)
        q = max(q_int, q_ref)

        # Select samples
        selection = np.array(self.scores[-1]) < q
        self.selected_samples.append(selection)
        self.n_update_samples.append(int(np.sum(selection)))

    def update_ref_scores(self, scores):
        self.ref_scores = scores

    def update_sample_distribution(self, samples, weights):
        '''Return the parameters of a distribution given samples.'''
        raise NotImplementedError()

    ########   Analysis-related methods   ########

    def get_data(self, dist_obj_titles=None, sample_dimension_titles=None,
                 exclude_last_batch=True):
        n_batches = self.batch_count + 1 - bool(exclude_last_batch)
        bs = self.batch_size
        if bs == 0: bs = self.sample_count

        # Batch-level data
        # Create a map from table-titles to distribution parameters.
        sd = self.sample_dist[:-1] if exclude_last_batch else self.sample_dist
        dist_objs = {}
        if isinstance(dist_obj_titles, str):
            dist_objs = {dist_obj_titles:sd}
        elif isinstance(dist_obj_titles, (tuple, list)):
            dist_objs = {t:[ds[i] for ds in sd]
                         for i,t in enumerate(dist_obj_titles)}

        d1_dict = dict(
            title=self.title,
            batch=np.arange(n_batches),
            ref_quantile=self.ref_quantile,
            internal_quantile=self.internal_quantile,
            n_update_samples=self.n_update_samples,
            update_samples_perc=100*np.array(self.n_update_samples)/bs,
        )
        for k,v in dist_objs.items():
            d1_dict[k] = v
        d1 = pd.DataFrame(d1_dict)

        # Sample-level data
        # Create a map from table-titles to sampled dimensions.
        samples = {}
        if isinstance(sample_dimension_titles, str):
            samples = {sample_dimension_titles:np.concatenate(self.sampled_data)}
        elif isinstance(sample_dimension_titles, (tuple, list)):
            samples = {t:[sample[i] for batch in self.sampled_data
                          for sample in batch]
                         for i,t in enumerate(sample_dimension_titles)}

        d2_dict = dict(
            title=self.title,
            batch=np.repeat(np.arange(n_batches), bs),
            sample_id=n_batches*list(range(bs)),
            selected=np.concatenate(self.selected_samples),
            weight=np.concatenate(self.weights),
            score=np.concatenate(self.scores),
        )
        for k,v in samples.items():
            d2_dict[k] = v
        d2 = pd.DataFrame(d2_dict)

        return d1, d2


def quantile(x, q, w=None, is_sorted=False):
    if w is None:
        return np.percentile(x, 100*q)
    x = np.array(x)
    w = np.array(w)
    if not is_sorted:
        ids = np.argsort(x)
        x = x[ids]
        w = w[ids]
    w = np.cumsum(w) - 0.5*w
    w -= w[0]
    w /= w[-1]
    return np.interp(q, w, x)


class CEM_Beta_1D(CEM):
    '''Implementation example of the CEM for a 1D Beta distribution.'''

    # Note: in this 1D case, dist is a scalar in [0,1]. In general, dist may be
    #  any object that represents a distribution (e.g., any kind of array).

    def __init__(self, *args, **kwargs):
        super(CEM_Beta_1D, self).__init__(*args, **kwargs)

    def do_sample(self, dist):
        return np.random.beta(2*dist, 2-2*dist)

    def pdf(self, x, dist):
        return stats.beta.pdf(x, 2*dist, 2-2*dist)

    def update_sample_distribution(self, samples, weights):
        return np.clip(np.mean(np.array(weights)*np.array(samples)) / \
                       np.mean(weights), 0.001, 0.999)


if __name__ == '__main__':
    print('We draw numbers from U(0,1) (which is equivalent to Beta(1,1)). '
          'Every batch we update the distribution using the bottom half of '
          'the samples, or all the samples below the 10%-quantile of the '
          'original distribution (which is 0.1). Thus, we expect to converge '
          'to a distribution with E[X]=E[U(0,0.1)]=0.05. '
          'Below is the actual sample mean per Cross Entropy iteration:')

    n_steps = 10
    N = 1000

    ce = CEM_Beta_1D(dist=0.5, batch_size=N, w_clip=0,
                     internal_alpha=0.5, ref_alpha=0.1)
    for batch in range(n_steps):
        for iter in range(N):
            x, _ = ce.sample()
            # For this demonstration, we consider x as
            #  both sampled configuration and resulted score.
            score = x
            ce.update(score)
        # Take last batch scores (note: scores[-1] is the new empty batch).
        #  Exclude original-distribution scores.
        scores = ce.scores[-2][ce.n_orig_per_batch:]
        print(f'[{batch:d}] {np.mean(scores):.3f}')
