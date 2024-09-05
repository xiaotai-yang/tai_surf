import argparse
import itertools

import numpy as np
import optax
import jax
from jax import numpy as jnp
from Helperfunction import *
from Helper_miscelluous import *
from patched_rnnfunction import *
import pickle
from jax import make_jaxpr
from jax.random import PRNGKey, categorical
import jax.config
jax.config.update("jax_enable_x64", False)

parser = argparse.ArgumentParser()
parser.add_argument('--L', type = int, default=4)
parser.add_argument('--px', type = int, default=1)
parser.add_argument('--py', type = int, default=1)
parser.add_argument('--numunits', type = int, default=64)
parser.add_argument('--lr', type = float, default=2e-4)
parser.add_argument('--lrthreshold', type = float, default=5e-4)
parser.add_argument('--lrdecaytime', type = float, default=5000)
parser.add_argument('--mag_fixed', type = bool, default=False)
parser.add_argument('--Sz', type = int, default=0)
parser.add_argument('--spinparity_fixed', type = bool, default=False)
parser.add_argument('--spinparity_value', type = int, default=1)
parser.add_argument('--gradient_clip', type = bool, default=True)
parser.add_argument('--gradient_clipvalue', type = float, default=10.0)
parser.add_argument('--dotraining', type = bool, default=True)
parser.add_argument('--T0', type = float, default= 0.0)
parser.add_argument('--Nwarmup', type = int, default=0)
parser.add_argument('--Nannealing', type = int, default=0) #10000
parser.add_argument('--Ntrain', type = int, default=0)
parser.add_argument('--Nconvergence', type = int, default=10000)
parser.add_argument('--numsamples', type = int, default=1)
parser.add_argument('--testing_sample', type = int, default=5e+4)
parser.add_argument('--lrthreshold_convergence', type = float, default=5e-4)
parser.add_argument('--lrdecaytime_convergence', type = float, default=2500)
parser.add_argument('--seed', type = int, default=3)
parser.add_argument('--rnn_type', type = str, default="tensor_gru")
parser.add_argument('--cmi_pattern', type = str, default="decay")
parser.add_argument('--sparsity', type = int, default=0)
parser.add_argument('--basis_rotation', type = bool, default=True)
parser.add_argument('--train_state', type = bool, default=True)
parser.add_argument('--angle', type = float, default=0.000001)
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
cmi_pattern = args.cmi_pattern
sparsity = args.sparsity
basis_rotation = args.basis_rotation
angle = args.angle
L = args.L
px = args.px
py = args.py
Nx = L
Ny = L
input_size = 2**(px*py)
key = PRNGKey(args.seed)
diag_bulk, diag_edge, diag_corner =False, False, False
meanEnergy=[]
varEnergy=[]
N = Ny*Nx
ny_nx_indices = jnp.array([[(i, j) for j in range(Nx)] for i in range(Ny)])
train_state = args.train_state
ang_ = jnp.array([0.0, 0.157, 0.314, 0.471, 0.628, 0.785, 0.942, 1.1, 1.257, 1.414, 1.571])

a = 0
for angle in (0.0*jnp.pi, 0.05*jnp.pi, 0.1*jnp.pi, 0.15*jnp.pi, 0.20*jnp.pi, 0.25*jnp.pi, 0.3*jnp.pi, 0.35*jnp.pi, 0.4*jnp.pi, 0.45*jnp.pi, 0.5*jnp.pi):

    gs = np.load("eig_vec_thata_"+str(ang_[a])+".npy")

    x, y = jnp.cos(angle), jnp.sin(angle)
    batch_total_samples_2d = vmap(total_samples_2d, (0, None), 0)
    batch_new_coe_2d = vmap(new_coe_2d, (0, None, None, None, None))
    batch_diag_coe = vmap(diag_coe, (0, None, None, None, None, None, None))
    batch_catgorical = vmap(categorical, (0, None))

    (xy_loc_bulk, xy_loc_edge, xy_loc_corner, yloc_bulk, yloc_edge, yloc_corner, zloc_bulk, zloc_edge,
     zloc_corner, off_diag_bulk_coe, off_diag_edge_coe, off_diag_corner_coe, zloc_bulk_diag, zloc_edge_diag,
     zloc_corner_diag, coe_bulk_diag, coe_edge_diag, coe_corner_diag) = vmc_off_diag_es(Ny, Nx, px ,py, angle, basis_rotation)

    total_samples = lax.cond( basis_rotation, lambda: numsamples * ((Nx * px - 2) * (Ny * py - 2) * (2 ** (5) - 1) + (Nx * px - 2) * 2 * (2 ** (4) - 1) + (Ny * py - 2) * 2 * (2 ** (4) - 1) + 4 * (2 ** (3) - 1))
    ,lambda: numsamples*Nx*px*Ny*py)
    prob = jnp.abs(gs)**2
    key_ = split(key, numsamples)

    samples_digit = batch_catgorical(key_, jnp.log(prob+1e-12))
    sample_log_amp = jnp.log(gs[samples_digit]+0.0*1j)

    samples = int_to_binary_array(samples_digit, L**2)
    samples = samples.reshape(-1, Ny*py, Nx*px)
    #print(samples)
    sigmas = jnp.concatenate((batch_total_samples_2d(samples, xy_loc_bulk),
                             batch_total_samples_2d(samples, xy_loc_edge),
                             batch_total_samples_2d(samples, xy_loc_corner)), axis=1).reshape(-1, Ny*py, Nx*px)
    #print(sigmas)
    #minus sign account for the minus sign of each term in the Hamiltonian
    matrixelements = jnp.concatenate((batch_new_coe_2d(samples, off_diag_bulk_coe, yloc_bulk, zloc_bulk, basis_rotation),
                                     batch_new_coe_2d(samples, off_diag_edge_coe, yloc_edge, zloc_edge, basis_rotation),
                                     batch_new_coe_2d(samples, off_diag_corner_coe, yloc_corner, zloc_corner, basis_rotation)), axis=1).reshape(numsamples, -1)
    #print(matrixelements)
    sigmas = sigmas.reshape(-1, Ny*py*Nx*px)

    sigmas_ = binary_array_to_int(sigmas, L**2)

    log_all_amp = jnp.log(gs[sigmas_]+0.0*1j)
    log_diag_amp = jnp.repeat(sample_log_amp, (jnp.ones(numsamples)*(matrixelements.shape[1])).astype(int), axis=0)
    amp = jnp.exp(log_all_amp.ravel()-log_diag_amp).reshape(numsamples, -1)
    Eloc = jnp.sum((amp*matrixelements), axis=1) + batch_diag_coe(samples, zloc_bulk_diag, zloc_edge_diag, zloc_corner_diag, coe_bulk_diag, coe_edge_diag, coe_corner_diag)
    meanE, varE = jnp.mean(Eloc), jnp.var(Eloc)
    a += 1
    print(meanE)
