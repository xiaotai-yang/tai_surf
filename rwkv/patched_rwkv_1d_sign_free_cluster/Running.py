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

parser = argparse.ArgumentParser()
parser.add_argument('--L', type = int, default=16)
parser.add_argument('--p', type = int, default=4)
parser.add_argument('--lr', type = float, default=2e-4)
parser.add_argument('--lrthreshold', type = float, default=5e-4)
parser.add_argument('--lrdecaytime', type = float, default=5000)
parser.add_argument('--mag_fixed', type = bool, default=False)
parser.add_argument('--Sz', type = int, default=0)
parser.add_argument('--spinparity_fixed', type = bool, default=False)
parser.add_argument('--spinparity_value', type = int, default=1)
parser.add_argument('--gradient_clip', type = bool, default=True)
parser.add_argument('--gradient_clipvalue', type = float, default=20.0)
parser.add_argument('--dotraining', type = bool, default=True)
parser.add_argument('--T0', type = float, default= 0.0)
parser.add_argument('--Nwarmup', type = int, default=0)
parser.add_argument('--Nannealing', type = int, default=0) #10000
parser.add_argument('--Ntrain', type = int, default=0)
parser.add_argument('--Nconvergence', type = int, default=2400)
parser.add_argument('--numsamples', type = int, default=160)
parser.add_argument('--testing_sample', type = int, default=5e+4)
parser.add_argument('--lrthreshold_convergence', type = float, default=5e-4)
parser.add_argument('--lrdecaytime_convergence', type = float, default=2500)
parser.add_argument('--seed', type = int, default=3)
parser.add_argument('--cmi_pattern', type = str, default="no_decay")
parser.add_argument('--sparsity', type = int, default=0)
parser.add_argument('--basis_rotation', type = bool, default=True)
parser.add_argument('--train_state', type = bool, default=True)
parser.add_argument('--angle', type = float, default=0.000001)
parser.add_argument('--emb_size', type = int, default=16)
parser.add_argument('--h_size', type = int, default=32)
parser.add_argument('--num_layer', type = int, default=3)
parser.add_argument('--out_h_size', type = int, default=128)
args = parser.parse_args()

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
cmi_pattern = args.cmi_pattern
sparsity = args.sparsity
basis_rotation = args.basis_rotation
angle = args.angle
L = args.L
p = args.p
N = L
input_size = 2**(p)
key = PRNGKey(args.seed)
diag_bulk, diag_edge, diag_corner =False, False, False
meanEnergy=[]
varEnergy=[]
n_indices = jnp.array([i for i in range(N)])
train_state = args.train_state
emb_size = args.emb_size
h_size = args.h_size
num_layer = args.num_layer
out_h_size = args.out_h_size


params = params_init(N, input_size, emb_size, h_size, out_h_size, num_layer, key)
key, subkey = split(key, 2)
wemb = random.orthogonal(key,  emb_size)
grad_f = jax.jit(jax.grad(compute_cost), static_argnums=(2,))
fixed_params = N, p, h_size, num_layer

batch_sample_prob = jax.jit(vmap(sample_prob_RWKV, (None, None, None, None, 0)), static_argnames=['fixed_params'])
batch_log_amp = jax.jit(vmap(log_amp_RWKV, (0, None, None, None, None)), static_argnames=['fixed_params'])
batch_total_samples_1d = vmap(total_samples_1d, (0, None), 0)
batch_new_coe_1d = vmap(new_coe_1d, (0, None, None, None, None))
batch_diag_coe = vmap(diag_coe, (0, None, None, None, None, None, None))
batch_flip_site = jax.jit(vmap(vmap(flip_sample, (None, 0)),(0, None)))
base_site = jnp.arange(0, N*p, 2)
flip_site = jnp.transpose(jnp.array([base_site%(N*p), (base_site+1)%(N*p), (base_site+2)%(N*p)]), (1, 0))
z_site = (flip_site+1)%(N*p)
# Assuming params are your model's parameters:
warmup_cosine_decay_scheduler = optax.warmup_cosine_decay_schedule(init_value=lr/5, peak_value=lr,
                                                                   warmup_steps= 600,
                                                                   decay_steps= numsteps, end_value=lr/10)

max_lr = lr
min_lr = lr/4
cycle_steps = 1000  # Adjust based on your training steps
scheduler = lambda step: linear_cycling_with_hold(step, max_lr, min_lr, cycle_steps)
optimizer = optax.adam(learning_rate=warmup_cosine_decay_scheduler)
optimizer_state = optimizer.init(params)
T = T0
t = time.time()

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
    samples, sample_log_amp = batch_sample_prob(params, wemb, fixed_params, n_indices, sample_key)

    sigmas = batch_flip_site(samples, flip_site).reshape(-1, N, p)
    log_all_amp = batch_log_amp(sigmas, params, wemb, fixed_params, n_indices)
    log_diag_amp = jnp.repeat(sample_log_amp, (jnp.ones(numsamples) * int(N * p / 2)).astype(int), axis=0)
    E_z = jnp.sum(2 * (jnp.sum(samples[:, z_site], axis=2) % 2) - 1, axis=1)
    amp = jnp.exp(log_all_amp.ravel() - log_diag_amp).reshape(numsamples, -1)
    Eloc = jnp.sum(-amp, axis=1) + E_z
    meanE, varE = jnp.mean(Eloc), jnp.var(Eloc)
    samples = samples.reshape(numsamples, N, p)
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

    grads = grad_f(params, wemb, fixed_params, samples, Eloc, T, n_indices)
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
        with open(f"params/params_L{L}_patch{p}_numsamples{numsamples}_embsize{emb_size}_hiddensize{h_size}_rotation_{basis_rotation}_angle{angle}.pkl", "wb") as f:
            pickle.dump(params_dict, f)
print(time.time()-t)
jnp.save("result/meanE_L"+str(L)+"patch_"+str(p)+"_emb_size"+str(emb_size)+"_hidden_size"+str(h_size)+"rotation"+str(basis_rotation)+"angle"+str(angle)+"_seed"+str(args.seed)+"numsamples"+str(numsamples)+".npy", jnp.array(meanEnergy))
jnp.save("result/varE_L"+str(L)+"patch_"+str(p)+"_emb_size"+str(emb_size)+"hidden_size"+str(h_size)+"rotation"+str(basis_rotation)+"angle"+str(angle)+"_seed"+str(args.seed)+"numsamples"+str(numsamples)+".npy", jnp.array(varEnergy))
