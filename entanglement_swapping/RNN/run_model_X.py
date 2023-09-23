from TrainingRNN_X import run_

#numsteps = number of training iterations
#systemsize = the number of physical spins
#J1_, J2_ = the coupling parameters of the J1-J2 Hamiltonian
#Marshall_sign: True = A marshall sign is applied on top of the cRNN wavefunction, False = no prior sign is applied
#numsamples = number of samples used for training
#num_units = number of memory units of the hidden state of the RNN
#num_layers = number of vertically stacked RNN cells

RNNEnergy, varRNNEnergy = run_(numsteps = 10000, systemsize = 64, num_units = 64, num_layers = 1, numsamples = 128, learningrate = 0.0005, seed = 1)

#RNNEnergy is a numpy array of the variational energy of the cRNN wavefunction
#varRNNEnergy is a numpy array of the variance of the variational energy of the cRNN wavefunction
