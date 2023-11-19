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


def clip_grad(g, clip_norm=10.0):
    norm = jnp.linalg.norm(g)
    scale = jnp.minimum(1.0, clip_norm / (norm + 1e-6))
    return g * scale


@partial(jax.jit, static_argnames=['fixed_parameters'])
def compute_cost(parameters, fixed_parameters, samples, Eloc, Temperature):
    samples = jax.lax.stop_gradient(samples)
    Eloc = jax.lax.stop_gradient(Eloc)

    # First term

    log_amps_tensor = log_amp(samples, parameters, fixed_parameters)
    term1 = 2 * jnp.real(jnp.mean(log_amps_tensor.conjugate() * (Eloc - jnp.mean(Eloc))))
    # Second term

    term2 = 4 * Temperature * (jnp.mean(jnp.real(log_amps_tensor) * jax.lax.stop_gradient(jnp.real(log_amps_tensor)))
                               - jnp.mean(jnp.real(log_amps_tensor)) * jnp.mean(
                jax.lax.stop_gradient(jnp.real(log_amps_tensor))))

    cost = term1 + term2

    return cost


def schedule(step: float, min_lr: float, max_lr: float, period: float) -> float:
    """Compute a learning rate that oscillates sinusoidally between min_lr and max_lr."""
    oscillation = (jnp.sin(jnp.pi * step / period) + 1) / 2  # Will be between 0 and 1
    return min_lr + (max_lr - min_lr) * oscillation


parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, default=10)
parser.add_argument('--numunits', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lrthreshold', type=float, default=5e-4)
parser.add_argument('--lrdecaytime', type=float, default=5000)
parser.add_argument('--mag_fixed', type=bool, default=False)
parser.add_argument('--Sz', type=int, default=0)
parser.add_argument('--spinparity_fixed', type=bool, default=False)
parser.add_argument('--spinparity_value', type=int, default=1)
parser.add_argument('--gradient_clip', type=bool, default=True)
parser.add_argument('--gradient_clipvalue', type=float, default=10.0)
parser.add_argument('--dotraining', type=bool, default=True)
parser.add_argument('--T0', type=float, default=0.0)
parser.add_argument('--Nwarmup', type=int, default=0)
parser.add_argument('--Nannealing', type=int, default=0)  # 10000
parser.add_argument('--Ntrain', type=int, default=0)
parser.add_argument('--Nconvergence', type=int, default=10000)
parser.add_argument('--numsamples', type=int, default=256)
parser.add_argument('--testing_sample', type=int, default=5e+4)
parser.add_argument('--lrthreshold_convergence', type=float, default=5e-4)
parser.add_argument('--lrdecaytime_convergence', type=float, default=2500)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--rnn_type', type=str, default="tensor_gru")
parser.add_argument('--cmi_pattern', type = str, default="random")
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
testing_sample = args.testing_sample
rnn_type = args.rnn_type
cmi_pattern = args.cmi_pattern
input_size = 2
L = args.L
Nx = L
Ny = L
N = Nx * Ny
key = PRNGKey(args.seed)
# create pauli matrices, 1 stands for pauli x and 3 stands for pauli z
pauli_array_bulk = jnp.repeat(jnp.array([1, 3, 3, 3, 3])[None], (Ny - 2) * (Nx - 2), axis=0)
pauli_array_edge = jnp.repeat(jnp.array([1, 3, 3, 3])[None], (Ny + Nx - 4) * 2, axis=0)
pauli_array_corner = jnp.repeat(jnp.array([1, 3, 3])[None], 4, axis=0)
# The location that each Hamiltonian term acts on
loc_array_bulk, loc_array_edge, loc_array_corner = loc_array(Ny, Nx)
# label_xxx[y, x] is a dict datatype and it is the location of loc_array_xxx such that pauli_array_bulk.at[label[i][:,0].astype(int), label[i][:,1].astype(int)] will
# show the pauli matrix that acts on lattice location
label_bulk, label_edge, label_corner = location_pauli_label(loc_array_bulk, loc_array_edge, loc_array_corner, Ny, Nx)
if (cmi_pattern == "no_decay"):
    print("no_decay")
    for i in label_bulk:
        pauli_array_bulk = pauli_array_bulk.at[label_bulk[i][:,0].astype(int), label_bulk[i][:,1].astype(int)].set(-pauli_array_bulk[label_bulk[i][:,0].astype(int), label_bulk[i][:,1].astype(int)]+4)
    for i in label_edge:
        pauli_array_edge = pauli_array_edge.at[label_edge[i][:,0].astype(int), label_edge[i][:,1].astype(int)].set(-pauli_array_edge[label_edge[i][:,0].astype(int), label_edge[i][:,1].astype(int)]+4)
    for i in label_corner:
        pauli_array_corner = pauli_array_corner.at[label_corner[i][:,0].astype(int), label_corner[i][:,1].astype(int)].set(-pauli_array_corner[label_corner[i][:,0].astype(int), label_corner[i][:,1].astype(int)]+4)
elif(cmi_pattern == "random"):
    print("random")
    for i in label_bulk:
        key, subkey = split(key, 2)
        p = jax.random.uniform(subkey, jnp.array([1]), float, 0 , 1)
        if p>0.5:
            pauli_array_bulk = pauli_array_bulk.at[label_bulk[i][:,0].astype(int), label_bulk[i][:,1].astype(int)].set(-pauli_array_bulk[label_bulk[i][:,0].astype(int), label_bulk[i][:,1].astype(int)]+4)
    for i in label_edge:
        key, subkey = split(key, 2)
        p = jax.random.uniform(subkey, jnp.array([1]), float, 0, 1)
        if p > 0.5:
            pauli_array_edge = pauli_array_edge.at[label_edge[i][:,0].astype(int), label_edge[i][:,1].astype(int)].set(-pauli_array_edge[label_edge[i][:,0].astype(int), label_edge[i][:,1].astype(int)]+4)
    for i in label_corner:
        key, subkey = split(key, 2)
        p = jax.random.uniform(subkey, jnp.array([1]), float, 0, 1)
        if p > 0.5:
            pauli_array_corner = pauli_array_corner.at[label_corner[i][:,0].astype(int), label_corner[i][:,1].astype(int)].set(-pauli_array_corner[label_corner[i][:,0].astype(int), label_corner[i][:,1].astype(int)]+4)

# We group the location that each Hamiltonian term acts on according to how many x,y,z they have in each term
xy_loc_bulk, yloc_bulk, zloc_bulk = local_element_indices_2d(5, pauli_array_bulk, loc_array_bulk)
xy_loc_edge, yloc_edge, zloc_edge = local_element_indices_2d(4, pauli_array_edge, loc_array_edge)
xy_loc_corner, yloc_corner, zloc_corner = local_element_indices_2d(3, pauli_array_corner, loc_array_corner)

batch_total_samples_2d = vmap(total_samples_2d, (0, None))
batch_new_coe_2d = vmap(new_coe_2d, (0, None, None, None))
T = T0

    # Your training step logic goes here

def training_step(carry, it, static_params):
    xy_loc_bulk, xy_loc_edge, xy_loc_corner, yloc_bulk, yloc_edge, yloc_corner, zloc_bulk, zloc_edge, zloc_corner, numsamples, fixed_params, basis_repeat, rnn_type  = static_params
    Ny, Nx, mag_fixed, magnetization, units = fixed_params
    params, optimizer_state, key = carry
    samples, sample_log_amp = sample_prob(numsamples, params, fixed_params, key)
    key, subkey1, subkey2 = split(key, 3)

    sigmas = jnp.concatenate((batch_total_samples_2d(samples, xy_loc_bulk),
                              batch_total_samples_2d(samples, xy_loc_edge)[:, 1:],
                              batch_total_samples_2d(samples, xy_loc_corner)[:, 1:]), axis=1).reshape(-1, Ny, Nx)
    matrixelements = jnp.concatenate((batch_new_coe_2d(samples, -jnp.ones((Ny - 2) * (Nx - 2)), yloc_bulk, zloc_bulk),
                                      batch_new_coe_2d(samples, -jnp.ones((Ny + Nx - 4) * 2), yloc_edge, zloc_edge)[:,
                                      1:],
                                      batch_new_coe_2d(samples, -jnp.ones(4), yloc_corner, zloc_corner)[:, 1:]),
                                     axis=1).reshape(numsamples, Nx * Ny + 1)

    log_all_amp = log_amp(sigmas, params, fixed_params)
    log_diag_amp = jnp.repeat(sample_log_amp, basis_repeat , axis=0)
    amp = jnp.exp(log_all_amp.ravel() - log_diag_amp).reshape(numsamples, Nx * Ny + 1)

    Eloc = jnp.sum((amp * matrixelements)[:, 1:], axis=1)
    meanE = jnp.mean(Eloc)
    varE = jnp.var(Eloc)

    # Calculate gradients with gradient clipping
    grads = jax.tree_map(clip_grad, grad_f(params, fixed_params, samples, Eloc, T))

    # Update parameters and optimizer state
    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, updates)

    # Update and return carry
    carry = (params, optimizer_state, key)
    lax.cond(it % 50 == 0, lambda x: jax.debug.print("steps : {}", x), lambda x: None, it)
    lax.cond(it % 50 == 0, lambda x: jax.debug.print("meanE : {}", meanE), lambda x: None, it)
    lax.cond(it % 50 == 0, lambda x: jax.debug.print("varE : {}", varE), lambda x: None, it)
    return carry, (meanE, varE)  # Return the updated carry and any other info you want to track

# Run the scan for a certain number of steps
numsteps = Nwarmup + (Nannealing + 1) * Ntrain + Nconvergence

# Set up initial values for carry

initial_key = PRNGKey(args.seed)
if (rnn_type == "vanilla"):
    init_params = init_vanilla_params(Nx, Ny, units, input_size, initial_key)
    # batch_rnn = vmap(vanilla_rnn_step, (0, 0, None))
elif (rnn_type == "multilayer_vanilla"):
    init_params = init_multilayer_vanilla_params(Nx, Ny, units, input_size, initial_key)
    # batch_rnn = vmap(multilayer_vanilla_rnn_step, (0, 0, None))
elif (rnn_type == "gru"):
    init_params = init_gru_params(Nx, Ny, units, input_size, initial_key)
elif (rnn_type == "tensor_gru"):
    init_params = init_tensor_gru_params(Nx, Ny, units, input_size, initial_key)

fixed_params = Ny, Nx, mag_fixed, magnetization, units
warmup_cosine_decay_scheduler = optax.warmup_cosine_decay_schedule(init_value=lr, peak_value=lrthreshold,
                                                                   warmup_steps=Nwarmup,
                                                                   decay_steps=numsteps, end_value=1e-5)
optimizer = optax.adamw(learning_rate=lr)
optimizer_state = optimizer.init(init_params)
initial_carry = (init_params, optimizer_state, initial_key)
basis_repeat = (jnp.ones(numsamples) * Nx * Ny + 1).astype(int)
static_params = xy_loc_bulk, xy_loc_edge, xy_loc_corner, yloc_bulk, yloc_edge, yloc_corner,  zloc_bulk,  zloc_edge, zloc_corner, numsamples, fixed_params, basis_repeat, rnn_type
grad_f = jax.jit(jax.grad(compute_cost), static_argnums=(1,))
t = time.time()
final_carry, (all_meanE, all_varE) = lax.scan(partial(training_step, static_params=static_params), initial_carry, jnp.arange(numsteps))
print(time.time()-t)
params, optimizer_state, key = final_carry
params_dict = jax.tree_util.tree_leaves(params)
with open(f"params/params_L{L}_numsamples{numsamples}_numunits{units}_rnntype_{rnn_type}.pkl", "wb") as f:
    pickle.dump(params_dict, f)
jnp.save("meanE_L"+str(L)+"cmi_pattern_"+cmi_pattern+".npy", all_meanE)
jnp.save("varE_L"+str(L)+"cmi_pattern_"+cmi_pattern+".npy", all_varE)
