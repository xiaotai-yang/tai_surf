import argparse
import itertools
import random

import numpy as np
import optax
import jax
from jax import numpy as jnp
from Helper_miscelluous import *
from patched_rnnfunction import *
import pickle
from jax import make_jaxpr
import jax.config
from functools import partial
#jax.config.update("jax_enable_x64", False)

parser = argparse.ArgumentParser()
parser.add_argument('--Ny', type = int, default=8)
parser.add_argument('--Nx', type = int, default=8)
parser.add_argument('--px', type = int, default=2)
parser.add_argument('--py', type = int, default=2)
parser.add_argument('--lr', type = float, default=2e-4)
parser.add_argument('--lrthreshold', type = float, default=5e-4)
parser.add_argument('--lrdecaytime', type = float, default=5000)
parser.add_argument('--gradient_clip', type = bool, default=True)
parser.add_argument('--gradient_clipvalue', type = float, default=10.0)
parser.add_argument('--dotraining', type = bool, default=True)
parser.add_argument('--T0', type = float, default= 0.0)
parser.add_argument('--Nwarmup', type = int, default=0)
parser.add_argument('--Nannealing', type = int, default=0) #10000
parser.add_argument('--Ntrain', type = int, default=0)
parser.add_argument('--Nconvergence', type = int, default=10000)
parser.add_argument('--numsamples', type = int, default=256)
parser.add_argument('--testing_sample', type = int, default=5e+4)
parser.add_argument('--lrthreshold_convergence', type = float, default=5e-4)
parser.add_argument('--lrdecaytime_convergence', type = float, default=2500)
parser.add_argument('--seed', type = int, default=3)
parser.add_argument('--rnn_type', type = str, default="RWKV")
parser.add_argument('--train_state', type = bool, default=True)
parser.add_argument('--emb_size', type = int, default=64)
parser.add_argument('--h_size', type = int, default=128)
parser.add_argument('--preout_size', type = int, default=1024)
parser.add_argument('--num_layer', type = int, default=2)
parser.add_argument('--Omega', type = float, default=1)
parser.add_argument('--delta', type = float, default=0)
parser.add_argument('--Rb', type = float, default=3**(1/6))
parser.add_argument('--units', type = int, default=16)
args = parser.parse_args()

numsamples = args.numsamples
lr=args.lr
units = args.units
lrdecaytime = args.lrdecaytime
lrdecaytime_conv = args.lrdecaytime_convergence
lrthreshold = args.lrthreshold
lrthreshold_conv = args.lrthreshold_convergence
T0 = args.T0
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

px = args.px
py = args.py
Nx = args.Nx
Ny = args.Ny
N = Ny*Nx*py*px
L = Ny*py
input_size = 2**(px*py)
key = PRNGKey(args.seed)
meanEnergy=[]
varEnergy=[]

ny_nx_indices = jnp.array([[(i, j) for j in range(Nx)] for i in range(Ny)])
train_state = args.train_state
emb_size = args.emb_size
h_size = args.h_size
preout_size = args.preout_size
num_layer = args.num_layer
Omega = args.Omega
delta = args.delta
Rb = args.Rb
T = T0

fixed_params = Ny, Nx, py, px, h_size, num_layer
if rnn_type == "RWKV":
    batch_sample_prob = jax.jit(vmap(sample_prob_RWKV, (None, None, None, 0)), static_argnames=['fixed_params'])
    batch_log_amp = jax.jit(vmap(log_amp_RWKV, (0, None, None, None)), static_argnames = ['fixed_params'])
elif rnn_type == "gru_rnn":
    batch_sample_prob = jax.jit(vmap(sample_prob_gru, (None, None, None, 0)), static_argnames=['fixed_params'])
    batch_log_amp = jax.jit(vmap(log_amp_gru, (0, None, None, None)), static_argnames=['fixed_params'])

batch_flip_sample = jax.jit(vmap(vmap(vmap(flip_sample, (None, None, 0)), (None, 0, None)), (0, None, None)))
batch_int_E = jax.jit(vmap(int_E, (0, None, None)), static_argnames=['n'])
batch_staggered_mag = jax.jit(vmap(staggered_magnetization, (0, None)), static_argnames=['L'])
grad_f = jax.jit(jax.grad(compute_cost), static_argnums=(1,))
key, subkey = split(key , 2)



for delta in ((jnp.arange(11))*0.3):
    params = params_init(rnn_type, Nx, Ny, units, input_size, emb_size, h_size, preout_size, num_layer, key)
    ny_nx_indices = jnp.array([[[i, j] for j in range(Nx)] for i in range(Ny)])

    # Assuming params are your model's parameters:
    warmup_cosine_decay_scheduler = optax.warmup_cosine_decay_schedule(init_value=lr/10, peak_value=lr,
                                                                       warmup_steps= 500,
                                                                       decay_steps= numsteps, end_value=lr/10)

    max_lr = lr
    min_lr = lr/4
    cycle_steps = 1000  # Adjust based on your training steps
    scheduler = lambda step: linear_cycling_with_hold(step, max_lr, min_lr, cycle_steps)
    optimizer = optax.adam(learning_rate=lr)
    optimizer_state = optimizer.init(params)
    t = time.time()

    for it in range(0, numsteps):

        start = time.time()
        key, subkey = split(key ,2)
        sample_key = split(key, numsamples)

        samples, sample_log_amp = batch_sample_prob(params, fixed_params, ny_nx_indices, sample_key)
        samples = jnp.transpose(samples.reshape(numsamples, Ny, Nx, py, px), (0, 1, 3, 2, 4)).reshape(numsamples, Ny*py, Nx*px)
        sigmas = batch_flip_sample(samples, jnp.arange(Ny*py), jnp.arange(Nx*px)).reshape(-1, Ny*py, Nx*px)
        sigmas = jnp.transpose(sigmas.reshape(-1, Ny, py, Nx, px), (0, 1, 3, 2, 4))

        log_all_amp = batch_log_amp(sigmas, params, fixed_params, ny_nx_indices)
        log_diag_amp = jnp.repeat(sample_log_amp, (jnp.ones(numsamples)*Ny*py*Nx*px).astype(int), axis=0)
        diag_part1 = -(jnp.sum((samples), axis = (1, 2)))*delta
        diag_part2 = batch_int_E(samples, L, Rb)
        Eloc = -Omega/2*jnp.sum(jnp.exp(log_all_amp.ravel()-log_diag_amp).reshape(numsamples, -1), axis=1)+diag_part1+diag_part2
        meanE, varE = jnp.mean(Eloc), jnp.var(Eloc)
        samples = jnp.transpose(samples.reshape(numsamples, Ny, py, Nx, px), (0, 1, 3, 2, 4))
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
                print('mean(E): {0}, varE: {1}, #samples {2}, #Step {3} \n\n'.format(meanE, varE, numsamples, it+1))

        grads = grad_f(params, fixed_params, samples, Eloc, T, ny_nx_indices)

        if (rnn_type == "RWKV"):
            if it%50 == 0:
                print(len(grads))
                for i in grads[:-1]:
                    print("num:", i.ravel().shape[0], "norm:", jnp.linalg.norm(i))

                print("RWKV_params_size:", len(grads[-1]))

                for j in grads[-1]:
                    print("num:", j.ravel().shape[0], "norm:", jnp.linalg.norm(j))
        elif(rnn_type == "gru_rnn"):
            if it%50 == 0:
                print(len(grads))
                for i in grads[:-1]:
                    print("num:", i.ravel().shape[0], "norm:", jnp.linalg.norm(i))

        #print("grad_time:", time.time()-t)
        if gradient_clip == True:
            grads = jax.tree_map(clip_grad, grads)
        #print("clip_grads:", grads)
        # Update the optimizer state and the parameters
        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        if (it%500 == 0):
            params_dict = jax.tree_util.tree_leaves(params)
            with open(f"params/params_L{L}_delta{delta}_numsamples{numsamples}_hiddensize{h_size}_embsize{emb_size}_seed{args.seed}_lr{lr}_patch{px*py}_rnntype_{rnn_type}.pkl", "wb") as f:
                pickle.dump(params_dict, f)
    print(time.time()-t)

    key = split(key, int(testing_sample))
    test_sample = batch_sample_prob(params, fixed_params, ny_nx_indices, key)
    stagger_mag = batch_staggered_mag(test_sample, L)
    jnp.save("result/meanE_L"+str(L)+"_delta"+str(delta)+"_hidden_size"+str(h_size)+"_emb_size"+str(emb_size)+"_seed"+str(args.seed)+"_lr"+str(lr)+"_patch"+str(px*py)+".npy", jnp.array(meanEnergy))
    jnp.save("result/varE_L"+str(L)+"_delta"+str(delta)+"_hidden_size"+str(h_size)+"_emb_size"+str(emb_size)+"_seed"+str(args.seed)+"_lr"+str(lr)+"_patch"+str(px*py)+".npy", jnp.array(varEnergy))
    jnp.save("result/staggered_mag_L"+str(L)+"_delta"+str(delta)+"_hidden_size"+str(h_size)+"_emb_size"+str(emb_size)+"_seed"+str(args.seed)+"_lr"+str(lr)+"_patch"+str(px*py)+".npy", stagger_mag)
