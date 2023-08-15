from TrainingRNN_J1J2 import run_J1J2

#numsteps = number of training iterations
#systemsize = the number of physical spins
#J1_, J2_ = the coupling parameters of the J1-J2 Hamiltonian
#Marshall_sign: True = A marshall sign is applied on top of the cRNN wavefunction, False = no prior is applied
#numsamples = number of samples used for training
numsamples = 200
#num_units = number of memory units of the hidden state of the RNN
#num_layers = number of vertically stacked RNN cells

#This function trains a cRNN wavefunction for 1D J1J2 with the corresponding hyperparams
RNNEnergy, varRNNEnergy = run_J1J2(numsteps = 1000, systemsize = 10, J1_  = 1.0, J2_ = 0.2, Marshall_sign = False, num_units = 10, num_layers = 1, numsamples = numsamples, learningrate = 5e-4, seed = 111)

#RNNEnergy is a numpy array of the variational energy of the cRNN wavefunction
#varRNNEnergy is a numpy array of the variance of the variational energy of the cRNN wavefunction