
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import utils


class GCVaR:
    def __init__(self, optimizer, batch_size, alpha=1, scheduler=None,
                 alpha1_normalizer=np.mean, skip_steps=0, check_nans=True,
                 optimistic_q=False, detailed_records=True, lr_sched=None,
                 title='GCVaR'):
        # Configuration
        self.title = title
        self.o = optimizer  # torch optimizer
        self.batch_size = batch_size
        self.alpha = alpha  # target risk level
        # scheduler should be one of the following:
        #  None; callable(iter, alpha); or tuple (soft_cvar, n_iterations).
        self.alpha_scheduler = scheduler
        # scores normalizer for alpha=1 (since alpha=1 is not CVaR,
        #  we don't have to use q_alpha anymore).
        self.alpha1_normalizer = alpha1_normalizer
        self.skip_steps = skip_steps
        self.optimistic_q = optimistic_q
        self.detailed_records = detailed_records
        self.lr_scheduler = lr_sched
        self.check_nans = check_nans

        # State
        self.batch_count = 0
        self.sample_count = 0

        # Data
        self.logprobs = [[]]  # n_batches x batch_size
        self.scores = [[]]  # n_batches x batch_size
        self.weights = [[]]  # n_batches x batch_size
        self.selected = [[]]  # n_batches x batch_size
        self.lr = []  # n_batches
        self.alphas = []  # n_batches
        self.q_alpha = []  # n_batches
        self.sample_size = []  # n_batches
        # https://en.wikipedia.org/wiki/Effective_sample_size#Weighted_samples
        self.eff_sample_size = []  # n_batches
        self.losses = []  # n_batches

        # Detailed data
        self.prev_params = None
        self.hashes = []
        self.mean_grads = []
        self.max_grads = []
        self.params_diff = []
        self.Rmin = []
        self.Rmax = []
        self.Rsmax = []

    def save(self, filename=None):
        if filename is None: filename = f'models/{self.title}'
        filename += '.opt'
        obj = (
            self.title, self.batch_size, self.alpha, self.alpha_scheduler,
            self.batch_count, self.sample_count, self.logprobs,
            self.scores, self.weights, self.selected,
            self.alphas, self.q_alpha, self.sample_size,
            self.eff_sample_size, self.losses, self.lr,
            self.hashes, self.mean_grads, self.max_grads, self.params_diff,
            self.Rmin, self.Rmax, self.Rsmax
        )
        with open(filename, 'wb') as h:
            pkl.dump(obj, h)

    def load(self, filename=None):
        if filename is None: filename = f'models/{self.title}'
        filename += '.opt'
        with open(filename, 'rb') as h:
            obj = pkl.load(h)
        self.title, self.batch_size, self.alpha, self.alpha_scheduler, \
        self.batch_count, self.sample_count, self.logprobs, \
        self.scores, self.weights, self.selected, \
        self.alphas, self.q_alpha, self.sample_size, \
        self.eff_sample_size, self.losses, self.lr, \
        self.hashes, self.mean_grads, self.max_grads, self.params_diff, \
        self.Rmin, self.Rmax, self.Rsmax = obj

    def reset_training(self):
        self.batch_count = 0
        self.sample_count = 0
        self.logprobs = [[]]
        self.scores = [[]]
        self.weights = [[]]
        self.selected = [[]]
        self.alphas = []
        self.q_alpha = []
        self.sample_size = []
        self.eff_sample_size = []
        self.losses = []

        self.prev_params = None
        self.hashes = []
        self.mean_grads = []
        self.max_grads = []
        self.params_diff = []
        self.Rmin = []
        self.Rmax = []
        self.Rsmax = []

    def reset_batch(self):
        self.batch_count += 1
        self.sample_count = 0

        self.logprobs[-1] = np.array([x.item() for x in self.logprobs[-1]])

        self.logprobs.append([])
        self.scores.append([])
        self.weights.append([])

    def get_alpha(self):
        if self.alpha_scheduler is None:
            return self.alpha
        if isinstance(self.alpha_scheduler, (list, tuple)):
            return alpha_scheduler(
                self.batch_count, self.alpha, *self.alpha_scheduler,
                skip_iters=self.skip_steps)
        if callable(self.alpha_scheduler):
            return self.alpha_scheduler(self.batch_count, self.alpha)
        raise IOError(self.alpha_scheduler)

    def step(self, weight, log_prob, score, ref_scores=None, save=True):
        # update metrics
        self.weights[-1].append(weight)      # np
        self.logprobs[-1].append(log_prob)   # torch
        self.scores[-1].append(score)        # np
        self.sample_count += 1

        # optimize
        if self.sample_count >= self.batch_size:
            self.select_samples(ref_scores)
            self.optimize()
            self.reset_batch()
            if save:
                filename = save if isinstance(save, str) else None
                self.save(filename)
            return True

        return False

    def select_samples(self, ref_scores=None):
        # get alpha
        alpha = self.get_alpha()

        # get q_alpha
        self.weights[-1] = np.array(self.weights[-1], dtype=np.float32)
        self.scores[-1] = np.array(self.scores[-1], dtype=np.float32)
        if alpha == 1:
            q = self.alpha1_normalizer(self.scores[-1])
        else:
            w = None
            if ref_scores is None:
                ref_scores = self.scores[-1]
                w = self.weights[-1]
            ref_scores = np.array(ref_scores)
            q = utils.quantile(ref_scores, alpha, w,
                               estimate_underlying_quantile=True)
            if self.optimistic_q and q < np.max(ref_scores):
                q = np.min(ref_scores[ref_scores>q])

        # get selected episodes for optimization
        selected = (self.scores[-1] < q) if alpha<1 else \
            np.ones_like(self.scores[-1], dtype=np.bool)

        # record everything
        self.alphas.append(alpha)
        self.q_alpha.append(q)
        self.selected.append(selected)
        self.sample_size.append(np.sum(selected))
        if self.sample_size[-1] > 0:
            w = self.selected[-1] * self.weights[-1]
            self.eff_sample_size.append(np.sum(w)**2 / np.sum(w**2))
        else:
            self.eff_sample_size.append(0)

    def optimize(self):
        logP = torch.stack(self.logprobs[-1])
        R = torch.tensor(self.scores[-1], dtype=torch.float32)
        w = torch.tensor(self.weights[-1], dtype=torch.float32)
        selected = torch.tensor(self.selected[-1], dtype=torch.float32)

        loss = - selected * w * (R-self.q_alpha[-1]) * logP
        loss = loss.mean()
        normalizer = (selected*w).mean()
        if normalizer.item() > 0:
            loss = loss / normalizer
        self.losses.append(loss.item())

        if self.check_nans and torch.isnan(loss):
            import pdb
            pdb.set_trace()

        self.o.zero_grad()
        loss.backward()
        if self.detailed_records:
            self.detailed_record(R, selected)

        if self.batch_count >= self.skip_steps:
            self.lr.append(self.o.param_groups[0]['lr'])
            self.o.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        else:
            self.lr.append(0)

        if self.check_nans and torch.any(torch.isnan(
                self.o.param_groups[0]['params'][0])):
            print(self.o.param_groups[0]['params'])
            import pdb
            pdb.set_trace()

    def detailed_record(self, R, selected):
        selected = selected.bool()
        params = torch.cat(
            [p.flatten() for g in self.o.param_groups
             for p in g['params']]).cpu().detach()
        params_hash = hex(hash(params.numpy().tostring()))
        if self.prev_params is not None:
            params_diff = (params-self.prev_params).abs().sum().item()
        else:
            params_diff = None
        self.prev_params = params
        grads = torch.cat(
            [p.grad.flatten() for g in self.o.param_groups
             for p in g['params']]).cpu().detach().abs().numpy()
        mean_grad = np.mean(grads)
        max_grad = np.max(grads)

        self.hashes.append(params_hash)
        self.mean_grads.append(mean_grad)
        self.max_grads.append(max_grad)
        self.params_diff.append(params_diff)
        self.Rmin.append(R.min().item())
        self.Rmax.append(R.max().item())
        self.Rsmax.append(R[selected].max().item() if selected.sum()>0 else None)

    def get_data(self):
        n_batches = self.batch_count
        bs = self.batch_size

        # Batch-level data
        d1 = pd.DataFrame(dict(
            title=self.title,
            batch=np.arange(n_batches),
            lr=self.lr,
            alpha=self.alphas,
            q_alpha=self.q_alpha,
            sample_size=self.sample_size,
            sample_size_perc=100*np.array(self.sample_size)/bs,
            eff_sample_size_perc=100*np.array(self.eff_sample_size)/bs,
            loss=self.losses,
        ))
        if self.detailed_records:
            d1['hash'] = self.hashes
            d1['mean_grad'] = self.mean_grads
            d1['max_grad'] = self.max_grads
            d1['params_diff'] = self.params_diff
            d1['Rmin'] = self.Rmin
            d1['Rmax'] = self.Rmax
            d1['Rsmax'] = self.Rsmax

        # Sample-level data
        d2 = pd.DataFrame(dict(
            title=self.title,
            batch=np.repeat(np.arange(n_batches), bs),
            sample_id=n_batches*list(range(bs)),
            selected=np.concatenate(self.selected),
            weight=np.concatenate(self.weights[:-1]),
            score=np.concatenate(self.scores[:-1]),
            logprob=np.concatenate(self.logprobs[:-1]),
        ))

        return d1, d2


def alpha_scheduler(iter, alpha, soft_cvar=0, n_iters=1, skip_iters=0,
                    resolution=0):
    if soft_cvar:
        ni = skip_iters
        nf = soft_cvar * n_iters
        if iter < ni:
            return 1
        if iter < nf:
            a = 1 - (iter-ni)/(nf-ni) * (1-alpha)
            if resolution:
                a /= resolution
                a = np.ceil(a)
                a *= resolution
            return a
        return alpha
    return alpha
