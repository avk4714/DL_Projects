'''Dropout Method Implementation for training in PyTorch
By - Aman V. Kalia for CSE 571 Project'''

'''Import Packages for the Run'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Bernoulli
from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt
import csv
from cartpole_dnn import Net    # Loads defined neural network architecture
from numpy import genfromtxt

'''Define the training process as a function'''
def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    return train_step

'''Checking if Cuda capapble platform is available for training'''
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

'''Loading training dataset from CSV files and creating a TensorDataset for
batch runs'''
train_x = genfromtxt('train_x_data_4.csv', delimiter=',')
train_y = genfromtxt('train_y_data_4.csv', delimiter=',')
train_x_tnsr = torch.from_numpy(train_x).float()
train_y_tnsr = torch.from_numpy(train_y).float()
M = train_x_tnsr.shape[0]
train_data = TensorDataset(train_x_tnsr, train_y_tnsr)

'''Training Parameter Settings'''
n_samples = 500
n_hidden = 1000
z_prob = 0.2
lam_mult = 1e-2
d_in = 6            # Inputs are: [p, dp, dtheta, sin(theta), cos(theta), action]
d_out = 4           # Outputs are: [ddtheta, ddp, dtheta, dp]
n_epochs = 10000
batch_size = 100

'''Load data as mini-batches'''
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

'''PyTorch Model Invocation'''
model = Net(d_in, n_hidden, d_out, z_prob, lam_mult).to(device)

'''Loss function and Optimizer'''
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(),lr=0.00001,weight_decay=lam_mult)

'''Train Data'''
losses = []
train_step = make_train_step(model, criterion, optimizer)

for epoch in range(n_epochs):
  for x_batch, y_batch in train_loader:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    loss = train_step(x_batch, y_batch)
    losses.append(loss)

  if (epoch % 100) == 0:
    print('Epoch: %d -> Loss: %.5f' % (epoch, loss))

'''[UNCOMMENT IF WANT TO STORE LOSS DATA]'''
# with open ('train_loss_10.csv', mode='w') as loss_dat:
#   loss_write = csv.writer(loss_dat, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#   for loss_val in losses:
#     loss_write.writerow([loss_val])

'''[UNCOMMENT IF WANT TO SAVE THE TRAINED MODEL]'''
# PATH = './cartpole_ReLU_7.pth'
# torch.save(model.state_dict(), PATH)
