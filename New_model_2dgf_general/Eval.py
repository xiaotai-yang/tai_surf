import jax
import pickle
import argparse
import itertools
import numpy as np
import optax
import jax
from jax import numpy as jnp
from Helperfunction import *
from RNNfunction import *
import pickle
from jax import make_jaxpr
import jax.config
jax.config.update("jax_enable_x64", False)

parser = argparse.ArgumentParser()
parser.add_argument('--L', type = int, default=6)
parser.add_argument('--numunits', type = int, default=128)
parser.add_argument('--lr', type = float, default=1e-4)
parser.add_argument('--J1', type = float, default=1.0)
parser.add_argument('--J2', type = float, default=0.2)
parser.add_argument('--J3', type = float, default=0.0)
parser.add_argument('--lrthreshold', type = float, default=5e-4)
parser.add_argument('--lrdecaytime', type = float, default=5000)
parser.add_argument('--mag_fixed', type = bool, default=True)
parser.add_argument('--Sz', type = int, default=0)
parser.add_argument('--spinparity_fixed', type = bool, default=False)
parser.add_argument('--spinparity_value', type = int, default=1)
parser.add_argument('--gradient_clip', type = bool, default=True)
parser.add_argument('--gradient_clipvalue', type = float, default=10.0)
parser.add_argument('--dotraining', type = bool, default=True)
parser.add_argument('--T0', type = float, default= 0.5)
parser.add_argument('--Nwarmup', type = int, default=2000)
parser.add_argument('--Nannealing', type = int, default=10) #10000
parser.add_argument('--Ntrain', type = int, default=2000)
parser.add_argument('--Nconvergence', type = int, default=50000)
parser.add_argument('--numsamples', type = int, default=256)
parser.add_argument('--testing_sample', type = int, default=5e+4)
parser.add_argument('--lrthreshold_convergence', type = float, default=5e-4)
parser.add_argument('--lrdecaytime_convergence', type = float, default=2500)
parser.add_argument('--seed', type = int, default=1)
parser.add_argument('--rnn_type', type = str, default="gru")

args = parser.parse_args()

units = args.numunits
numsamples = args.numsamples
lr=args.lr
lrdecaytime = args.lrdecaytime
lrdecaytime_conv = args.lrdecaytime_convergence
lrthreshold = args.lrthreshold
lrthreshold_conv = args.lrthreshold_convergence
T0 = args.T0
mag_fixed = args.mag_fixed
magnetization = 2*args.Sz
spinparity_fixed = args.spinparity_fixed
spinparity_value = args.spinparity_value
J1 = args.J1
J2 = args.J2
J3 = args.J3
gradient_clip = args.gradient_clip
gradient_clipvalue = args.gradient_clipvalue
dotraining = args.dotraining
Nwarmup = args.Nwarmup
Nannealing = args.Nannealing
Ntrain = args.Ntrain
Nconvergence = args.Nconvergence
numsteps = Nwarmup + (Nannealing+1)*Ntrain + Nconvergence
testing_sample = args.testing_sample
rnn_type = args.rnn_type
input_size = 2
L = args.L
Nx = L
Ny = L
key = PRNGKey(112)

meanEnergy=[]
varEnergy=[]


N = Nx*Ny
with open(f'params/params_L{L}_J1{J1}_J2{J2}_numsamples{numsamples}_numunits{units}_rnntype_{rnn_type}.pkl', 'rb') as file:
    params = pickle.load(file)
fixed_params = Ny, Nx, mag_fixed, magnetization, units

samples, sample_amp = sample_prob(numsamples, params, fixed_params, key)
    key, subkey1, subkey2 = split(key, 3)
    #t = time.time()
    matrixelements, log_diag_amp, sigmas, basis_where = J1J2J3_MatrixElements_numpy(np.array(samples), np.array(sample_amp), J1, J2, J3, Nx, Ny)
    #print("matrixelement_t:", time.time()-t)
    left_basis = (2*Nx*Ny)*numsamples-matrixelements.shape[0]
    if left_basis>0:
        sigmas = jnp.concatenate((sigmas, jnp.zeros((left_basis, Ny, Nx))), axis=0).astype(int)
        matrixelements = jnp.concatenate((matrixelements, jnp.zeros(left_basis)), axis=0)
        log_diag_amp = jnp.concatenate((log_diag_amp, jnp.zeros(left_basis)), axis=0)
    else:
        print("error")
    #t = time.time()
    log_all_amp = log_amp(sigmas, params, fixed_params)
    amp =jnp.exp(log_all_amp-log_diag_amp)

    #print("calculation_t:", time.time()-t)
    diag_local_E = matrixelements[:numsamples]*amp[:numsamples]
    matrixelements_off_diag, amp_off_diag = matrixelements[numsamples:-left_basis], amp[numsamples:-left_basis]
    basis_where = basis_where.reshape(-1, numsamples+1)
    ind1, ind2 = basis_where[:, :-1].astype(jnp.int32), basis_where[:, 1:].astype(jnp.int32)
    block_sum = compute_block_sum(amp_off_diag, matrixelements_off_diag, ind1, ind2)

    Eloc = jnp.sum(block_sum, axis=0)+diag_local_E
    meanE = jnp.mean(Eloc)
    varE = jnp.var(Eloc)
    print(meanE, varE)