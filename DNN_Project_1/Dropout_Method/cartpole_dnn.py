import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Bernoulli

class Net(nn.Module):
  def __init__(self, D_in1, n_hidden, D_out1, z_prob1, lam_mult1):
    super(Net, self).__init__()
    # Layers
    self.model_L_1 = torch.nn.Linear(D_in1, n_hidden, bias=True)
    self.model_D_1 = torch.nn.Dropout(p=z_prob1)
    self.model_L_2 = torch.nn.Linear(n_hidden, n_hidden, bias=True)
    self.model_D_2 = torch.nn.Dropout(p=z_prob1)
    self.model_L_3 = torch.nn.Linear(n_hidden, n_hidden, bias=True)
    self.model_D_3 = torch.nn.Dropout(p=z_prob1)
    self.model_L_4 = torch.nn.Linear(n_hidden, n_hidden, bias=True)
    self.model_D_4 = torch.nn.Dropout(p=z_prob1)
    self.model_L_5 = torch.nn.Linear(n_hidden, D_out1, bias=True)

  def forward(self, x):
    pred_1 = self.model_D_1(F.relu(self.model_L_1(x)))
    pred_2 = self.model_D_2(F.relu(self.model_L_2(pred_1)))
    pred_3 = self.model_D_3(F.relu(self.model_L_3(pred_2)))
    pred_4 = self.model_D_4(F.relu(self.model_L_4(pred_3)))
    y_pred = self.model_L_5(pred_4)
    return y_pred
