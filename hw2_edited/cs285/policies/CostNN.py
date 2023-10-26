from cs285.infrastructure import pytorch_util as ptu
import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

class CostNN(nn.Module):
    def __init__(
        self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        learning_rate=1e-4,
        action_shape=1,
    ):


        super(CostNN, self).__init__()

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.action_shape = action_shape

        self.net = ptu.build_mlp(input_size=self.ob_dim+self.action_shape,
                                       output_size=1,
                                       n_layers=self.n_layers,
                                       size=self.size, activation = 'relu')
        self.net.to(ptu.device)
        self.optimizer = optim.Adam(self.net.parameters(),
                                    self.learning_rate)
    def forward(self, x):
        return self.net(x)

    def update(self,costs_samp, costs_demo, probs):
        #TODO --

        # LOSS CALCULATION FOR IOC (COST FUNCTION)
        loss_IOC = torch.mean(costs_demo) + \
                torch.log(torch.mean(torch.exp(-costs_samp)/(probs+1e-7)))
        # UPDATING THE COST FUNCTION
        self.optimizer.zero_grad()
        loss_IOC.backward()
        self.optimizer.step()

        train_log = {
            'TrainingLoss': ptu.to_numpy(loss_IOC),
            'CostDemo': np.sum(ptu.to_numpy(costs_demo)),
            'CostSamples': np.sum(ptu.to_numpy(costs_samp)),
        }


        return train_log
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
