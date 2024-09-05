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
jax.config.update("jax_enable_x64", False)

parser = argparse.ArgumentParser()
parser.add_argument('--L', type = int, default=4)
parser.add_argument('--px', type = int, default=1)
parser.add_argument('--py', type = int, default=1)
parser.add_argument('--numunits', type = int, default=4)
parser.add_argument('--lr', type = float, default=4e-4)
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
parser.add_argument('--Nconvergence', type = int, default=3)
parser.add_argument('--numsamples', type = int, default=1)
parser.add_argument('--testing_sample', type = int, default=5e+4)
parser.add_argument('--lrthreshold_convergence', type = float, default=5e-4)
parser.add_argument('--lrdecaytime_convergence', type = float, default=2500)
parser.add_argument('--seed', type = int, default=3)
parser.add_argument('--rnn_type', type = str, default="RWKV")
parser.add_argument('--cmi_pattern', type = str, default="decay")
parser.add_argument('--sparsity', type = int, default=0)
parser.add_argument('--basis_rotation', type = bool, default=True)
parser.add_argument('--train_state', type = bool, default=True)
parser.add_argument('--angle', type = float, default=0.000001)
parser.add_argument('--emb_size', type = int, default=8)
parser.add_argument('--h_size', type = int, default=5)
parser.add_argument('--num_layer', type = int, default=2)
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
emb_size = args.emb_size
h_size = args.h_size
num_layer = args.num_layer
for angle in (0., ):
    print(sparsity)
    '''
    Initialization
    '''

    fixed_params = Ny, Nx, py, px, h_size, num_layer
    batch_sample_prob_rwkv = jax.jit(vmap(sample_prob, (None, None, None, 0)), static_argnames=['fixed_params'])
    batch_flip_sample = jax.jit(vmap(vmap(vmap(flip_sample, (None, None, 0)), (None, 0, None)), (0, None, None)))
    batch_log_amp_RWKV = jax.jit(vmap(log_amp, (0, None, None, None, None)), static_argnames=['fixed_params'])
    batch_int_E = jax.jit(vmap(int_E, (0, None, None)), static_argnames=['n'])
    grad_f = jax.jit(jax.grad(compute_cost), static_argnums=(1,))
    key, subkey = split(key, 2)
    wemb = jax.random.orthogonal(subkey, emb_size)[:input_size]

    x, y = jnp.cos(angle), jnp.sin(angle)
    params = params_init(rnn_type, Nx, Ny, units, input_size, emb_size, h_size, num_layer, key)
    grad_f = jax.jit(jax.grad(compute_cost), static_argnums=(1,))
    fixed_params = Ny, Nx, py, px, mag_fixed, magnetization
    #winit_emb_x, winit_emb_y = jnp.zeros((Nx, input_size)), jnp.zeros((Ny, input_size))
    #state_init_x, state_init_y = jnp.zeros((Nx, units)), jnp.zeros((Ny, units))
    ny_nx_indices = jnp.array([[(i, j) for j in range(Nx)] for i in range(Ny)])
    batch_total_samples_2d = vmap(total_samples_2d, (0, None), 0)
    batch_new_coe_2d = vmap(new_coe_2d, (0, None, None, None, None))
    batch_diag_coe = vmap(diag_coe, (0, None, None, None, None, None, None))

    # Assuming params are your model's parameters:
    max_lr = lr
    min_lr = lr/4
    cycle_steps = 1000  # Adjust based on your training steps
    scheduler = lambda step: linear_cycling_with_hold(step, max_lr, min_lr, cycle_steps)
    optimizer = optax.adam(learning_rate=scheduler)
    optimizer_state = optimizer.init(params)

    for it in range(0, numsteps):

        start = time.time()
        key, subkey = split(key ,2)
        sample_key = split(key, numsamples)
        '''
        if it+1<=Nwarmup+Nannealing*Ntrain: #Before the end of annealing
            lr_adap = max(lrthreshold, lr/(1+it/lrdecaytime))
        elif it+1>Nwarmup+Nannealing*Ntrain: #After annealing -> finetuning the model during convergence
            lr_adap = lrthreshold_conv/(1+(it-(Nwarmup+Nannealing*Ntrain))/lrdecaytime_conv)
        '''
        t0 = time.time()
        samples, sample_log_amp = vmap(sample_prob_RWKV, (None, None, None, 0))(params, fixed_params, ny_nx_indices, sample_key)
        #print("sample_time:", time.time()-t0)
        samples = jnp.transpose(samples.reshape(numsamples, Ny, Nx, py, px), (0, 1, 3, 2, 4)).reshape(numsamples, Ny*py, Nx*px)

        sigmas = jnp.concatenate((batch_total_samples_2d(samples, xy_loc_bulk),
                                 batch_total_samples_2d(samples, xy_loc_edge),
                                 batch_total_samples_2d(samples, xy_loc_corner)), axis=1).reshape(-1, Ny*py, Nx*px)

        t0 = time.time()
        sigmas = jnp.transpose(sigmas.reshape(total_samples, Ny, py, Nx, px), (0, 1, 3, 2, 4))
        log_all_amp = vmap(log_amp_RWKV, (0, None, None, None))(sigmas, params, fixed_params, ny_nx_indices)
        #print("log_amp_time:", time.time()-t0)
        log_diag_amp = jnp.repeat(sample_log_amp, (jnp.ones(numsamples)*(matrixelements.shape[1])).astype(int), axis=0)
        amp = jnp.exp(log_all_amp.ravel()-log_diag_amp).reshape(numsamples, -1)
        #print("log_diag_amp_shape:", log_diag_amp.shape)
        Eloc = jnp.sum((amp*matrixelements), axis=1) + batch_diag_coe(samples, zloc_bulk_diag, zloc_edge_diag, zloc_corner_diag, coe_bulk_diag, coe_edge_diag, coe_corner_diag)
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
        if (it+1)%1==0 or it==0:
            print("learning_rate =", lr)
            print("Magnetization =", jnp.mean(jnp.sum(2*samples-1, axis = (1,2))))
            if T0 != 0:
                print('mean(E): {0}, varE: {1}, meanF: {2}, varF: {3}, #samples {4}, #Step {5} \n\n'.format(meanE,varE, meanF, varF, numsamples, it+1))
            elif T0 == 0.0:
                print('mean(E): {0}, varE: {1}, #samples {2}, #Step {3} \n\n'.format(meanE,varE,numsamples, it+1))

        grads = grad_f(params, fixed_params, samples, Eloc, T, ny_nx_indices)
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
    print(time.time()-t)
    jnp.save("result/meanE_L"+str(L)+"_units"+str(units)+"_cmi_pattern_"+cmi_pattern+"rotation"+str(basis_rotation)+"angle"+str(angle)+"_seed"+str(args.seed)+".npy", jnp.array(meanEnergy))
    jnp.save("result/varE_L"+str(L)+"_units"+str(units)+"_cmi_pattern_"+cmi_pattern+"rotation"+str(basis_rotation)+"angle"+str(angle)+"_seed"+str(args.seed)+".npy", jnp.array(varEnergy))
