import argparse
import itertools
import random

import numpy as np
import optax
import jax
from jax import numpy as jnp
from Helperfunction import *
from Helper_miscelluous import *
from patched_rnnfunction import *
import pickle
from jax import make_jaxpr
import jax.config

# jax.config.update("jax_enable_x64", False)
def create_alternating_matrix(n):
    # Create an n*n matrix where each element is 1
    matrix = np.ones((n, n), dtype=int)

    # Multiply by -1 at every other index
    matrix[1::2, ::2] = -1  # Change every other row starting from the second row
    matrix[::2, 1::2] = -1  # Change every other column starting from the second column

    return matrix
parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, default=4)
parser.add_argument('--px', type=int, default=2)
parser.add_argument('--py', type=int, default=2)
parser.add_argument('--numunits', type=int, default=128)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lrthreshold', type=float, default=5e-4)
parser.add_argument('--lrdecaytime', type=float, default=5000)
parser.add_argument('--mag_fixed', type=bool, default=False)
parser.add_argument('--Sz', type=int, default=0)
parser.add_argument('--spinparity_fixed', type=bool, default=False)
parser.add_argument('--spinparity_value', type=int, default=1)
parser.add_argument('--gradient_clip', type=bool, default=False)
parser.add_argument('--gradient_clipvalue', type=float, default=20.0)
parser.add_argument('--dotraining', type=bool, default=True)
parser.add_argument('--T0', type=float, default=0.0)
parser.add_argument('--Nwarmup', type=int, default=0)
parser.add_argument('--Nannealing', type=int, default=0)  # 10000
parser.add_argument('--Ntrain', type=int, default=0)
parser.add_argument('--Nconvergence', type=int, default=5000)
parser.add_argument('--numsamples', type=int, default=4096)
parser.add_argument('--testing_sample', type=int, default=5e+4)
parser.add_argument('--lrthreshold_convergence', type=float, default=5e-4)
parser.add_argument('--lrdecaytime_convergence', type=float, default=2500)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--rnn_type', type=str, default="RWKV")
parser.add_argument('--cmi_pattern', type=str, default="no_decay")
parser.add_argument('--sparsity', type=int, default=0)
parser.add_argument('--basis_rotation', type=bool, default=True)
parser.add_argument('--train_state', type=bool, default=True)
parser.add_argument('--angle', type=float, default=0.000001)
parser.add_argument('--emb_size', type=int, default=8)
parser.add_argument('--ff_size', type=int, default=2048)
parser.add_argument('--num_layer', type=int, default=2)
parser.add_argument('--num_head', type=int, default=8)
parser.add_argument('--Omega', type=float, default=1)
parser.add_argument('--delta', type=float, default=0)
parser.add_argument('--Rb', type=float, default=7 ** (1 / 6))
args = parser.parse_args()

units = args.numunits
numsamples = args.numsamples
lr = args.lr
lrdecaytime = args.lrdecaytime
lrdecaytime_conv = args.lrdecaytime_convergence
lrthreshold = args.lrthreshold
lrthreshold_conv = args.lrthreshold_convergence
T0 = args.T0
mag_fixed = args.mag_fixed
magnetization = 2 * args.Sz
spinparity_fixed = args.spinparity_fixed
spinparity_value = args.spinparity_value
gradient_clip = args.gradient_clip
gradient_clipvalue = args.gradient_clipvalue
dotraining = args.dotraining
Nwarmup = args.Nwarmup
Nannealing = args.Nannealing
Ntrain = args.Ntrain
Nconvergence = args.Nconvergence
numsteps = Nwarmup + (Nannealing + 1) * Ntrain + Nconvergence
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
input_size = 2 ** (px * py)
key = PRNGKey(args.seed)
diag_bulk, diag_edge, diag_corner = False, False, False
meanEnergy = []
varEnergy = []
N = Ny * Nx
ny_nx_indices = jnp.arange(Nx * Ny)
train_state = args.train_state
emb_size = args.emb_size
ff_size = args.ff_size
head = args.num_head
num_layer = args.num_layer
Omega = args.Omega
delta = args.delta
Rb = args.Rb
file_path = 'params/params_L4_numsamples512_numunits128_rnntype_RWKV_rotation_True_angle1e-06.pkl'
with open(file_path, 'rb') as file:
    # Load the data from the file
    params = pickle.load(file)
fixed_params = Ny, Nx, py, px, num_layer
batch_sample_prob = jax.jit(vmap(sample_prob, (None, None, None, 0)), static_argnames=['fixed_params'])
batch_int_E = jax.jit(vmap(int_E, (0, None, None)), static_argnames=['n'])
batch_dot = vmap(jnp.dot, (0, None))
sample_key = split(key, numsamples)
samples, sample_log_amp = batch_sample_prob(params, fixed_params, ny_nx_indices, sample_key)
samples = jnp.transpose(samples.reshape(numsamples, Ny, Nx, py, px), (0, 1, 3, 2, 4)).reshape(numsamples,
                                                                                              Ny * py, Nx * px)
samples = 2*samples-1
a = jnp.abs(batch_dot(samples.reshape(numsamples, -1), create_alternating_matrix(L*py).ravel().T))/64
print(jnp.mean(a), jnp.var(a))




