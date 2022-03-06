
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_PATH = 'models'


class NN(nn.Module):
    def __init__(self, state_dim=4, act_dim=2, eps_factor=0, train_hparams=None,
                 device='cpu', title='', pretrained_filename=None):
        super(NN, self).__init__()
        self.title = title
        self.device = device
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.eps_factor = eps_factor
        self.train_hparams = train_hparams if isinstance(train_hparams, dict) else None
        self.pretrained_filename = pretrained_filename

        self.n_updates = 0

    def init_state(self):
        pass

    def get_params_hash(self):
        return hex(hash(torch.cat(
            [par.flatten() for par in self.parameters()]
        ).cpu().detach().numpy().tostring()))

    def save(self, filename=None):
        if filename is None: filename = self.title
        torch.save(self.state_dict(), f'{MODELS_PATH:s}/{filename:s}.mdl')

    def load(self, filename=None):
        if filename is None: filename = self.title
        self.load_state_dict(torch.load(f'{MODELS_PATH:s}/{filename:s}.mdl'))

    def act(self, input, T=1, eps=None, verbose=0):
        if eps is None:
            eps = self.eps_factor*T
        # calculate actions scores
        input = torch.from_numpy(input).float().unsqueeze(0)
        if verbose >= 2:
            print(input)
        probs = self(input, T=T)
        if verbose >= 2:
            print('policy output:', probs)

        # choose action
        if eps and eps > np.random.random():
            choice = 'random'
            action = np.random.randint(0,self.act_dim)
        else:
            if T > 0:
                choice = 'soft'
                p = [probs[0][i].item() for i in range(self.act_dim)]
                ps = sum(p)
                assert np.abs(ps-1) < 1e-3, f"Probabilities don't sum to 1! ({ps})"
                action = np.random.choice(np.arange(self.act_dim), 1, p=np.array(p)/ps)[0]
            else:
                choice = 'max'
                action = probs.max(1)[1].item()
        if verbose >= 1:
            print(f'action = {action:d}\t({choice:s})')

        return action, probs[0,[action]].log(), probs.cpu().detach(), choice


class FC(NN):
    def __init__(self, mid_sizes=None, state_dim=4, act_dim=2, lstm=False,
                 activation='tanh', head_bias=True, dropout=0.0, **kwargs):
        super(FC, self).__init__(state_dim, act_dim, **kwargs)

        self.lstm = lstm
        if mid_sizes is None:
            mid_sizes = (16,) if self.lstm else (32,32)
        self.mid_sizes = mid_sizes
        sizes = [self.state_dim] + list(mid_sizes)
        layer = nn.LSTMCell if self.lstm else nn.Linear

        self.layers = nn.ModuleList([layer(a, b) for a,b in \
                                     zip(sizes[:-1],sizes[1:])]).to(self.device)
        self.activation = dict(relu=F.relu, tanh=torch.tanh)[activation]
        self.drop_p = dropout
        self.dropouts = None
        if self.drop_p:
            self.dropouts = [nn.Dropout(p=dropout).to(self.device) \
                             for _ in range(len(mid_sizes))]
        self.head = nn.Linear(sizes[-1], self.act_dim, bias=head_bias).to(self.device)

        self.h, self.c = None, None
        self.init_state()

    def init_state(self):
        if self.lstm:
            # hidden state
            self.h = [torch.zeros(1,m).to(self.device)
                      for m in self.mid_sizes]
            # memory cell
            self.c = [torch.zeros(1,m).to(self.device)
                      for m in self.mid_sizes]

    def forward(self, x, T=1):
        if T==0: T=1

        x = x.to(self.device)

        if self.lstm:
            for j, layer in enumerate(self.layers):
                self.h[j], self.c[j] = layer(
                    self.h[j-1] if j>0 else x, (self.h[j], self.c[j]))
            x = self.h[-1]
        else:
            for j, layer in enumerate(self.layers):
                x = layer(x)
                if self.drop_p:
                    x = self.dropouts[j](x)
                x = self.activation(x)

        action_scores = self.head(x)
        return F.softmax(action_scores/T, dim=1).cpu()


class CNN(NN):
    def __init__(self, mid_sizes=(16,16), state_dim=8, act_dim=4, in_channels=3,
                 **kwargs):
        super(CNN, self).__init__(state_dim, act_dim, **kwargs)

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, mid_sizes[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_sizes[0], mid_sizes[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(self.state_dim//4, self.state_dim//4),
            nn.Flatten(),
            nn.Linear(mid_sizes[1]*4*4, self.act_dim)
        ).to(self.device)

    def forward(self, x, T=1):
        if T==0: T=1
        x = x.to(self.device)
        action_scores = self.model(x)
        return F.softmax(action_scores/T, dim=1).cpu()
