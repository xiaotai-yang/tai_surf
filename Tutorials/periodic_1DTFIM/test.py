import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from TrainingRNN_1DTFIM import run_1DTFIM

#numsteps = number of training iterations
#systemsize = number of physical spins
#Bx = transverse magnetic field
#numsamples = number of samples used for training
numsamples = 200
#num_units = number of memory units of the hidden state of the RNN
#num_layers = number of vertically stacked RNN cells

#This function trains a pRNN wavefunction for 1DTFIM with the corresponding hyperparams
RNNEnergy, varRNNEnergy = run_1DTFIM(numsteps = 1000, systemsize = 10, Bx = +1, num_units = 10,  num_layers = 1, numsamples = numsamples, learningrate = 5e-3, seed = 111)

#RNNEnergy is a numpy array of the variational energy of the pRNN wavefunction
#varRNNEnergy is a numpy array of the variance of the variational energy of the pRNN wavefunction