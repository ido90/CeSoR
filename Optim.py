
import numpy as np
from warnings import warn
import torch
import torch.optim as optim

DEBUG_MODE = False


class Optimizer:
    def __init__(self, optimizer, optim_freq=20, episodic_loss=False,
                 normalize_rets=False, cvar=1, cvar_w_bug=False, verbose=1):
        self.cvar_w_bug = cvar_w_bug
        self.o = optimizer
        self.optim_freq = optim_freq
        # loss per time-step (better) or per episode (applicable to cvar)
        self.episodic_loss = episodic_loss
        self.normalize_rets = normalize_rets
        self.is_cvar = (0<cvar<1)
        self.alpha = cvar
        self.verbose = verbose

        if self.normalize_rets and self.optim_freq == 1:
            if self.verbose >= 1:
                warn('A single-sample batch cannot be normalized: '
                     'disabling normalization.')
            self.normalize_rets = False
        # if self.normalize_rets and self.is_cvar:
        #     warn('CVaR does not support rewards normalization.')
        #     self.normalize_rets = False
        if self.is_cvar and not self.episodic_loss:
            if self.verbose >= 1:
                warn('CVaR requires episodic loss.')
            self.episodic_loss = True

        self.updates_count = 0
        self.episodes_count = 0
        self.curr_weights = []
        self.curr_log_probs = []
        self.curr_scores = []
        self.curr_losses = []

        self.n_updates = 0
        self.samples_per_step = []
        # https://en.wikipedia.org/wiki/Effective_sample_size#Weighted_samples
        self.eff_samples_per_step = []

    def reset_training(self):
        self.n_updates = 0
        self.samples_per_step = []
        self.eff_samples_per_step = []
        self.reset_metrics()

    def reset_metrics(self):
        self.episodes_count = 0
        self.curr_weights = []
        self.curr_log_probs = []
        self.curr_scores = []
        self.curr_losses = []

    def step(self, weight, log_prob, score, loss, ref_scores=None):
        # update metrics
        self.curr_weights.append(weight)      # np
        self.curr_log_probs.append(log_prob)  # torch
        self.curr_scores.append(score)        # np
        self.curr_losses.append(loss)         # torch
        self.episodes_count += 1

        # optimize
        if (self.episodes_count % self.optim_freq) == 0:
            self.optimize(ref_scores)
            self.updates_count += 1
            self.reset_metrics()
            return True

        return False

    def optimize(self, ref_scores=None):
        self.n_updates += 1
        self.o.zero_grad()

        w = np.array(self.curr_weights, dtype=np.float32)
        w = w / np.sum(w)

        if not self.is_cvar:
            self.samples_per_step.append(1)
            self.eff_samples_per_step.append(1/(w**2).sum()/len(w))
            # optimize E[score]
            if self.episodic_loss:
                if self.normalize_rets:
                    mean_score = np.mean(self.curr_scores)
                    self.curr_scores = [(s-mean_score) for s in self.curr_scores]
                losses = [-lp*s for lp,s in zip(self.curr_log_probs, self.curr_scores)]
            else:
                losses = self.curr_losses
            weighted_mean_score = (
                    torch.tensor(w, dtype=torch.float32) * torch.stack(losses, 0)).sum()
            if torch.isnan(weighted_mean_score): # TODO tmp
                import pdb
                pdb.set_trace()
            weighted_mean_score.backward()
        else:
            # optimize CVaR
            # sort episodes by score
            ids = np.argsort(self.curr_scores)
            scores = [self.curr_scores[i] for i in ids]
            log_probs = [self.curr_log_probs[i] for i in ids]
            w = [w[i] for i in ids]
            # find score's alpha-quantile
            if ref_scores is None:
                q, i0 = get_quantile(scores, self.alpha, w)
            else:
                q, i0 = get_quantile_from_reference(scores, self.alpha, ref_scores)
                if i0 < 0:
                    if self.verbose >= 1:
                        print(f'\t\t[iteration {self.n_updates:d}] {self.alpha}-quantile '
                              f'of reference scores is {q}, smaller than lowest train '
                              f'score ({np.min(scores)}).')
                    q, i0 = get_quantile(scores, self.alpha, w)
            self.samples_per_step.append((i0+1)/len(scores))
            self.eff_samples_per_step.append(
                np.sum(w[:i0+1])**2/np.sum(np.array(w[:i0+1])**2)/len(scores))
            # derive loss
            if not self.cvar_w_bug:
                w_sum = np.sum(w[:i0+1])
                loss = torch.stack([- w[i] * log_probs[i] * (scores[i]-q) \
                                    for i in range(i0+1)], 0).sum() / w_sum
            else:
                loss = torch.stack([-log_probs[i] * (scores[i]-q) \
                                    for i in range(i0+1)], 0).mean()
            loss.backward()

        if DEBUG_MODE:
            print('\nOptimization info:')
            print('\nweights:')
            print(w)
            print('\nlog_probs:')
            print(self.curr_log_probs)
            print('\nscores:')
            print(self.curr_scores)
            print('\nlosses:')
            print(losses)
            print('\nweighted mean loss:')
            print(weighted_mean_score)
            print('\ngrads:')
            print([p.grad for g in self.o.param_groups for p in g['params']])
            raise EOFError('Hello.')

        # TODO tmp
        if self.verbose >= 2:
            if torch.any(torch.isnan(self.o.param_groups[0]['params'][0])):
                print(1)
                print(self.o.param_groups[0]['params'])
                import pdb
                pdb.set_trace()
        self.o.step()
        if self.verbose >= 2:
            if torch.any(torch.isnan(self.o.param_groups[0]['params'][0])):
                print(2)
                print(self.o.param_groups[0]['params'])
                import pdb
                pdb.set_trace()


def get_quantile(scores, alpha, weights=None):
    n = len(scores)
    if weights is None:
        weights = np.ones(n) / n
    w_cum = 0
    i0 = 0
    for i0 in range(n):
        w_cum += weights[i0]
        if w_cum >= alpha:
            break
    q = scores[i0]
    return q, i0

def get_quantile_from_reference(scores, alpha, ref_scores):
    n = len(ref_scores)
    alpha_idx = (n + 1) * alpha - 1
    if alpha_idx < 0:
        warn(f'{n} reference samples can only estimate '
             f'quantile>={1 / (n + 1)}, not {alpha}.')
        alpha_idx = 0
    i1 = int(alpha_idx)
    i2 = i1 + 1 if alpha_idx < n - 1 else i1
    i2_weight = alpha_idx - i1
    q = ref_scores[i1] + i2_weight * (ref_scores[i2] - ref_scores[i1])
    i0 = int(np.sum(np.array(scores) <= q)) - 1
    return q, i0
