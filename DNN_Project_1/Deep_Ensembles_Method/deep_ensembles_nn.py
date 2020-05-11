import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
learning_rate = 0.001
class Net(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_units):
    super(Net,self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_units = hidden_units
    self.model = torch.nn.Sequential(
        torch.nn.Linear(self.input_dim, self.hidden_units, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(self.hidden_units, self.hidden_units, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(self.hidden_units, self.hidden_units, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(self.hidden_units, self.hidden_units, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(self.hidden_units, self.output_dim, bias=True),
    )
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
  
  def forward(self, inputs):
    out = self.model(inputs)
    mean, var = torch.split(out, self.output_dim//2, dim=1)
    var = F.softplus(var) + 1e-7 # add a minimum variance for numerical stability
    return  mean, var