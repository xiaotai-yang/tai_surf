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
#jax.config.update("jax_enable_x64", False)

parser = argparse.ArgumentParser()
parser.add_argument('--L', type = int, default=4)
parser.add_argument('--px', type = int, default=2)
parser.add_argument('--py', type = int, default=2)
parser.add_argument('--numunits', type = int, default=128)
parser.add_argument('--lr', type = float, default=5e-4)
parser.add_argument('--lrthreshold', type = float, default=5e-4)
parser.add_argument('--lrdecaytime', type = float, default=5000)
parser.add_argument('--mag_fixed', type = bool, default=False)
parser.add_argument('--Sz', type = int, default=0)
parser.add_argument('--spinparity_fixed', type = bool, default=False)
parser.add_argument('--spinparity_value', type = int, default=1)
parser.add_argument('--gradient_clip', type = bool, default=False)
parser.add_argument('--gradient_clipvalue', type = float, default=20.0)
parser.add_argument('--dotraining', type = bool, default=True)
parser.add_argument('--T0', type = float, default= 0.0)
parser.add_argument('--Nwarmup', type = int, default=0)
parser.add_argument('--Nannealing', type = int, default=0) #10000
parser.add_argument('--Ntrain', type = int, default=0)
parser.add_argument('--Nconvergence', type = int, default=5000)
parser.add_argument('--numsamples', type = int, default=512)
parser.add_argument('--testing_sample', type = int, default=5e+4)
parser.add_argument('--lrthreshold_convergence', type = float, default=5e-4)
parser.add_argument('--lrdecaytime_convergence', type = float, default=2500)
parser.add_argument('--seed', type = int, default=3)
parser.add_argument('--rnn_type', type = str, default="RWKV")
parser.add_argument('--cmi_pattern', type = str, default="no_decay")
parser.add_argument('--sparsity', type = int, default=0)
parser.add_argument('--basis_rotation', type = bool, default=True)
parser.add_argument('--train_state', type = bool, default=True)
parser.add_argument('--angle', type = float, default=0.000001)
parser.add_argument('--emb_size', type = int, default=8)
parser.add_argument('--ff_size', type = int, default=2048)
parser.add_argument('--num_layer', type = int, default=2)
parser.add_argument('--num_head', type = int, default=8)
parser.add_argument('--Omega', type = float, default=1)
parser.add_argument('--delta', type = float, default=0)
parser.add_argument('--Rb', type = float, default=7**(1/6))
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
ny_nx_indices = jnp.arange(Nx*Ny)
train_state = args.train_state
emb_size = args.emb_size
ff_size = args.ff_size
head = args.num_head
num_layer = args.num_layer
Omega = args.Omega
delta = args.delta
Rb = args.Rb

fixed_params = Ny, Nx, py, px, num_layer
batch_sample_prob = jax.jit(vmap(sample_prob, (None, None, None, 0)), static_argnames=['fixed_params'])
batch_log_amp = jax.jit(vmap(log_amp, (0, None, None, None)), static_argnames=['fixed_params'])
batch_flip_sample = jax.jit(vmap(vmap(vmap(flip_sample, (None, None, 0)), (None, 0, None)), (0, None, None)))
batch_int_E = jax.jit(vmap(int_E, (0, None, None)), static_argnames=['n'])

for delta in (jnp.array([1., ])):

    params = init_transformer_params(num_layer, ff_size, units, input_size, head, key)
    grad_f = jax.jit(jax.grad(compute_cost), static_argnums=(1,))

    # Assuming params are your model's parameters:
    warmup_cosine_decay_scheduler = optax.warmup_cosine_decay_schedule(init_value=lr/5, peak_value=lr,
                                                                       warmup_steps= 600,
                                                                       decay_steps= numsteps, end_value=lr/10)

    max_lr = lr
    min_lr = lr/4
    cycle_steps = 1000  # Adjust based on your training steps
    scheduler = lambda step: linear_cycling_with_hold(step, max_lr, min_lr, cycle_steps)
    optimizer = optax.adam(learning_rate=lr)
    optimizer_state = optimizer.init(params)
    T = T0
    for it in range(0, numsteps):
        start = time.time()
        key, subkey = split(key ,2)
        sample_key = split(subkey, numsamples)


        samples, sample_log_amp = batch_sample_prob(params, fixed_params, ny_nx_indices, sample_key)
        samples = jnp.transpose(samples.reshape(numsamples, Ny, Nx, py, px),(0, 1, 3, 2, 4)).reshape(numsamples, Ny*py, Nx*px)

        sigmas = batch_flip_sample(samples, jnp.arange(Ny * py), jnp.arange(Nx * px)).reshape(-1, Ny * py, Nx * px)
        sigmas = jnp.transpose(sigmas.reshape(-1, Ny, py, Nx, px), (0, 1, 3, 2, 4)).reshape(-1, Ny*Nx, py*px)
        log_all_amp = batch_log_amp(sigmas, params, fixed_params, ny_nx_indices)
        log_diag_amp = jnp.repeat(sample_log_amp, (jnp.ones(numsamples)*Ny*py*Nx*px).astype(int), axis=0)
        diag_part1 = -(jnp.sum((samples), axis = (1, 2)))*delta
        diag_part2 = batch_int_E(samples, L*px, Rb)

        Eloc = -Omega / 2 * jnp.sum(jnp.exp(log_all_amp.ravel() - log_diag_amp).reshape(numsamples, -1), axis=1) + diag_part1 + diag_part2
        meanE, varE = jnp.mean(Eloc), jnp.var(Eloc)
        samples = jnp.transpose(samples.reshape(numsamples, Ny, py, Nx, px), (0, 1, 3, 2, 4)).reshape(-1, Ny*Nx, py*px)
        meanEnergy.append(meanE)
        varEnergy.append(varE)

        if (T0!=0):
            if it+1<=Nwarmup:
                if (it+1)%100==0:
                    print("Pre-annealing, warmup phase:", (it+1), "/", Nwarmup)
                T = T0
            elif it+1 > Nwarmup and it+1<=Nwarmup+Nannealing*Ntrain:
                if (it+1)%100==0:
                    print("Pre-annealing, annealing phase:", (it+1-Nwarmup)//Ntrain, "/", Nannealing)
                T = T0*(1-((it+1-Nwarmup)//Ntrain)/Nannealing)
            else:
                T = 0.0

            if (it+1)%100 == 0:
                print("Temperature = ", T)
            meanF = jnp.mean(Eloc + T*jnp.real(2*(sample_log_amp)))
            varF = jnp.var(Eloc + T*jnp.real(2*(sample_log_amp)))
        if (it+1)%25==0 or it==0:
            print("learning_rate =", lr)
            print("Magnetization =", jnp.mean(jnp.sum(2*samples-1, axis = (1,2))))
            if T0 != 0:
                print('mean(E): {0}, varE: {1}, meanF: {2}, varF: {3}, #samples {4}, #Step {5} \n\n'.format(meanE,varE, meanF, varF, numsamples, it+1))
            elif T0 == 0.0:
                print('mean(E): {0}, varE: {1}, #samples {2}, #Step {3} \n\n'.format(meanE,varE,numsamples, it+1))

        grads = grad_f(params, fixed_params, samples, Eloc, T, ny_nx_indices)
        #print("Iteration:", it)
        '''
        if it%50 == 0:
            print(len(grads))
            for i in grads[:-1]:
                print("num:", i.ravel().shape[0], "norm:", jnp.linalg.norm(i))

            print("RWKV_params_size:", len(grads[-1]))
        
            for j in grads[-1]:
                print("num:", j.ravel().shape[0], "norm:", jnp.linalg.norm(j))        
        '''
        #print("grad_time:", time.time()-t)
        if gradient_clip == True:
            grads = jax.tree_map(clip_grad, grads)
        #print("clip_grads:", grads)
        # Update the optimizer state and the parameters
        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        if (it%500 == 0):
            params_dict = jax.tree_util.tree_leaves(params)
            with open(f"params/params_L{L}_numsamples{numsamples}_numunits{units}_rnntype_{rnn_type}_rotation_{basis_rotation}_angle{angle}.pkl", "wb") as f:
                pickle.dump(params_dict, f)
    
    jnp.save("result/meanE_L"+str(L)+"_units"+str(units)+"_cmi_pattern_"+cmi_pattern+"rotation"+str(basis_rotation)+"angle"+str(angle)+"_seed"+str(args.seed)+"_patch"+str(px*py)+".npy", jnp.array(meanEnergy))
    jnp.save("result/varE_L"+str(L)+"_units"+str(units)+"_cmi_pattern_"+cmi_pattern+"rotation"+str(basis_rotation)+"angle"+str(angle)+"_seed"+str(args.seed)+"_patch"+str(px*py)+".npy", jnp.array(varEnergy))
