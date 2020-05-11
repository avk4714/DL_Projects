# Plot loss values from DNN training
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import csv

# Read loss data from DNN-Dropout File
fname1 = 'train_loss_10.csv'
dropoutLoss = genfromtxt(fname1, delimiter=',')

# Read loss data from Deep Ensembles File


# Plot Loss
for i in range(dropoutLoss.shape[0]):
	if i % 5000 == 0:
		plt.plot(i/5000,dropoutLoss[i],'b.')

plt.ylabel('Loss')
plt.xlabel('Epochs*100')
plt.title('DNN Dropout Method')
plt.grid()
plt.show()