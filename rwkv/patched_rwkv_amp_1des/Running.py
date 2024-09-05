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
#import jax.config

parser = argparse.ArgumentParser()
parser.add_argument('--L', type = int, default=64)
parser.add_argument('--p', type = int, default=1)
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
parser.add_argument('--Nconvergence', type = int, default=5000)
parser.add_argument('--numsamples', type = int, default=256)
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
ang_array = [1.571, 0.0, 0.157, 0.314, 0.471, 0.628, 0.785, 0.942, 1.1, 1.257, 1.414, 1.571]

for L, p  in  zip([6, 32, 64], [1, 2, 1]):
    N = L
    input_size = 2**p
    a = 0

    n_indices = jnp.arange(N)
    for angle in (jnp.arange(11)*jnp.pi/20):
        print(sparsity)
        angle = 0.5*jnp.pi
        '''
        Initialization
        '''
        x, y = jnp.cos(angle), jnp.sin(angle)
        params = params_init(N, input_size, emb_size, h_size, out_h_size, num_layer, key)
        key, subkey = split(key, 2)
        wemb = random.orthogonal(key,  emb_size)
        grad_f = jax.jit(jax.grad(compute_cost), static_argnums=(2,))
        fixed_params = N, p, h_size, num_layer
        (xy_loc_bulk, xy_loc_fl, xy_loc_xzz, yloc_bulk, yloc_fl, yloc_xzz, zloc_bulk, zloc_fl,
         zloc_xzz, off_diag_bulk_coe, off_diag_fl_coe, off_diag_xzz_coe, zloc_bulk_diag, zloc_fl_diag,
         zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag) = vmc_off_diag_es(N, p, angle, basis_rotation)

        batch_sample_prob = jax.jit(vmap(sample_prob_RWKV, (None, None, None, None, 0)), static_argnames=['fixed_params'])
        batch_log_amp = jax.jit(vmap(log_amp_RWKV, (0, None, None, None, None)), static_argnames=['fixed_params'])
        batch_total_samples_1d = vmap(total_samples_1d, (0, None), 0)
        batch_new_coe_1d = vmap(new_coe_1d, (0, None, None, None, None))
        batch_diag_coe = vmap(diag_coe, (0, None, None, None, None, None, None))
        batch_log_phase_dmrg = jax.jit(vmap(log_phase_dmrg, (0, None, None, None)))

        ang = ang_array[a]
        a += 1
        M0 = jnp.load("../entanglement_swapping/DMRG/mps_tensors/tensor_init_" + str(L * p) + "_angle_" + str(ang) + ".npy")
        M = jnp.load("../entanglement_swapping/DMRG/mps_tensors/tensor_" + str(L * p) + "_angle_" + str(ang) + ".npy")
        Mlast = jnp.load("../entanglement_swapping/DMRG/mps_tensors/tensor_last_" + str(L * p) + "_angle_" + str(ang) + ".npy")

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
            sample_log_amp += batch_log_phase_dmrg(samples, M0, M, Mlast)
            sigmas = jnp.concatenate((batch_total_samples_1d(samples, xy_loc_bulk),
                                      batch_total_samples_1d(samples, xy_loc_fl),
                                      batch_total_samples_1d(samples, xy_loc_xzz)), axis=1).reshape(-1, N, p)
            # minus sign account for the minus sign of each term in the Hamiltonian
            matrixelements = (jnp.concatenate((batch_new_coe_1d(samples, off_diag_bulk_coe, yloc_bulk, zloc_bulk, basis_rotation),
                                 batch_new_coe_1d(samples, off_diag_fl_coe, yloc_fl, zloc_fl, basis_rotation),
                                 batch_new_coe_1d(samples, off_diag_xzz_coe, yloc_xzz, zloc_xzz, basis_rotation)), axis=1)
                            .reshape(numsamples, -1))

            log_all_amp = batch_log_amp(sigmas, params, wemb, fixed_params, n_indices)+batch_log_phase_dmrg(sigmas.reshape(-1, L*p), M0, M, Mlast)
            log_diag_amp = jnp.repeat(sample_log_amp, (jnp.ones(numsamples)*(matrixelements.shape[1])).astype(int), axis=0)
            amp = jnp.exp(log_all_amp.ravel()-log_diag_amp).reshape(numsamples, -1)
            Eloc = jnp.sum((amp * matrixelements), axis=1) + batch_diag_coe(samples, zloc_bulk_diag, zloc_fl_diag, zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag)
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
                amp_exact = []
                for ind in range(2**N):
                    ind_array = jnp.array([ind >> i & 1 for i in range(N - 1, -1, -1)])
                    amp_exact.append(jnp.exp(log_amp_RWKV(ind_array, params, wemb, fixed_params, n_indices)+log_phase_dmrg(ind_array, M0, M, Mlast)))
                print("Exact amp:", amp_exact)
            grads = grad_f(params, wemb, fixed_params, samples, Eloc, T, n_indices, M0, M, Mlast)
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
                grads = jax.tree.map(clip_grad, grads)
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
