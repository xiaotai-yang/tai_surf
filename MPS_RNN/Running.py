import argparse
import itertools
import random
import os
import numpy as np
import optax
import jax
from jax import numpy as jnp
from Helperfunction import *
from Helper_miscelluous import *
from RNNfunction import *
import pickle
from jax import make_jaxpr
import jax.config
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", False)

parser = argparse.ArgumentParser()
parser.add_argument('--L', type = int, default=64)
parser.add_argument('--p', type = int, default=1)
parser.add_argument('--num_bond_dim', type = int, default=16)
parser.add_argument('--lr', type = float, default=1e-3)
parser.add_argument('--lrthreshold', type = float, default=5e-4)
parser.add_argument('--lrdecaytime', type = float, default=5000)
parser.add_argument('--mag_fixed', type = bool, default=False)
parser.add_argument('--Sz', type = int, default=0)
parser.add_argument('--gradient_clip', type = bool, default=False)
parser.add_argument('--gradient_clipvalue', type = float, default=10.0)
parser.add_argument('--dotraining', type = bool, default=True)
parser.add_argument('--T0', type = float, default= 0.0)
parser.add_argument('--Nwarmup', type = int, default=0)
parser.add_argument('--Nannealing', type = int, default=0) #10000
parser.add_argument('--Ntrain', type = int, default=0)
parser.add_argument('--Nconvergence', type = int, default=10000)
parser.add_argument('--numsamples', type = int, default=128)
parser.add_argument('--testing_sample', type = int, default=5e+4)
parser.add_argument('--lrthreshold_convergence', type = float, default=5e-4)
parser.add_argument('--lrdecaytime_convergence', type = float, default=2500)
parser.add_argument('--seed', type = int, default=3)
parser.add_argument('--cmi_pattern', type = str, default="no_decay")
parser.add_argument('--sparsity', type = int, default=0)
parser.add_argument('--basis_rotation', type = bool, default=True)
parser.add_argument('--angle', type = float, default=0.000001)
parser.add_argument('--sr', type = bool, default=False)
args = parser.parse_args()

bond_dim = args.num_bond_dim
numsamples = args.numsamples
lr=args.lr
lrdecaytime = args.lrdecaytime
lrdecaytime_conv = args.lrdecaytime_convergence
lrthreshold = args.lrthreshold
lrthreshold_conv = args.lrthreshold_convergence
T0 = args.T0
mag_fixed = args.mag_fixed
magnetization = 2*args.Sz
gradient_clip = args.gradient_clip
gradient_clipvalue = args.gradient_clipvalue
dotraining = args.dotraining
Nwarmup = args.Nwarmup
Nannealing = args.Nannealing
Ntrain = args.Ntrain
Nconvergence = args.Nconvergence
numsteps = Nwarmup + (Nannealing+1)*Ntrain + Nconvergence
cmi_pattern = args.cmi_pattern
sparsity = args.sparsity
basis_rotation = args.basis_rotation
angle = args.angle
sr = args.sr
L = args.L
p = args.p
input_size = 2 ** p
key = PRNGKey(args.seed)
diag_bulk, diag_fl, diag_xzz = False, False, False
meanEnergy=[]
varEnergy=[]
N = L
rnn_type = "mps_rnn"

for angle in (jnp.arange(11)*jnp.pi/20):
    # x and y are the cosine and sine of the rotation angle
    x, y = jnp.cos(angle), jnp.sin(angle)
    params =  init_mps_rnn_params(N, bond_dim, input_size, key)
    key, subkey = split(key, 2)
    # get the gradient function and compute cost is imported from Helper_miscelluous.py
    grad_f = jax.jit(jax.grad(compute_cost), static_argnums=(1,))
    if sr == True:
        grad_real_log_amp = jax.jit(jax.grad(mean_real_log_amp), static_argnums=(1,))
        grad_imag_log_amp = jax.jit(jax.grad(mean_imag_log_amp), static_argnums=(1,))
        jac_real_log_amp = jax.jit(jax.jacfwd(real_log_amp), static_argnums=(1,))
        jac_imag_log_amp = jax.jit(jax.jacfwd(imag_log_amp), static_argnums=(1,))

    if (sr == False):
        optimizer = optax.adam(learning_rate=lr)
    else:
        optimizer = optax.sgd(learning_rate=lr)

    optimizer_state = optimizer.init(params)
    fixed_params = N, p, bond_dim
    (xy_loc_bulk, xy_loc_fl, xy_loc_xzz, yloc_bulk, yloc_fl, yloc_xzz, zloc_bulk, zloc_fl,
     zloc_xzz, off_diag_bulk_coe, off_diag_fl_coe, off_diag_xzz_coe, zloc_bulk_diag, zloc_fl_diag,
     zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag) = vmc_off_diag(N, p, angle, basis_rotation)
    batch_sample_prob = jax.jit(vmap(sample_prob, (None, None, 0)), static_argnames=['fixed_params'])
    batch_log_amp = jax.jit(vmap(log_amp, (0, None, None)), static_argnames=['fixed_params'])
    batch_total_samples_1d = vmap(total_samples_1d, (0, None), 0)
    batch_new_coe_1d = vmap(new_coe_1d, (0, None, None, None, None))
    batch_diag_coe = vmap(diag_coe, (0, None, None, None, None, None, None))

    T = T0
    t = time.time()
    for it in range(0, numsteps):

        '''
        if it+1<=Nwarmup+Nannealing*Ntrain: #Before the end of annealing
            lr_adap = max(lrthreshold, lr/(1+it/lrdecaytime))
        elif it+1>Nwarmup+Nannealing*Ntrain: #After annealing -> finetuning the model during convergence
            lr_adap = lrthreshold_conv/(1+(it-(Nwarmup+Nannealing*Ntrain))/lrdecaytime_conv)
        '''
        key_ = split(key, numsamples)
        samples, sample_log_amp = batch_sample_prob(params, fixed_params, key_)
        key, subkey1, subkey2 = split(key_[0], 3)
        #print("samples:", samples)
        sigmas = jnp.concatenate((batch_total_samples_1d(samples, xy_loc_bulk),
                                 batch_total_samples_1d(samples, xy_loc_fl),
                                 batch_total_samples_1d(samples, xy_loc_xzz)), axis=1).reshape(-1, N, p)
        #print("sigmas:", sigmas)
        #minus sign account for the minus sign of each term in the Hamiltonian
        matrixelements = jnp.concatenate((batch_new_coe_1d(samples, off_diag_bulk_coe, yloc_bulk, zloc_bulk, basis_rotation),
                                         batch_new_coe_1d(samples, off_diag_fl_coe, yloc_fl, zloc_fl, basis_rotation),
                                         batch_new_coe_1d(samples, off_diag_xzz_coe, yloc_xzz, zloc_xzz, basis_rotation)), axis=1).reshape(numsamples, -1)
        #print("matrixelements:", matrixelements)
        log_all_amp = batch_log_amp(sigmas, params, fixed_params)
        log_diag_amp = jnp.repeat(sample_log_amp, (jnp.ones(numsamples)*(matrixelements.shape[1])).astype(int), axis=0)
        amp = jnp.exp(log_all_amp.ravel()-log_diag_amp).reshape(numsamples, -1)
        #print("log_all_amp_shape:", log_all_amp.shape)
        #print("log_diag_amp_shape:", log_diag_amp.shape)
        Eloc = jnp.sum((amp*matrixelements), axis=1) + batch_diag_coe(samples, zloc_bulk_diag, zloc_fl_diag, zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag)
        #print(batch_diag_coe(samples, zloc_bulk_diag, zloc_fl_diag, zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag))
        meanE = jnp.mean(Eloc)
        varE = jnp.var(Eloc)

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

        if (it+1)%50==0 or it==0:
            print("learning_rate =", lr)
            print("Magnetization =", jnp.mean(jnp.sum(2*samples-1, axis = (1))))
            if T0 != 0:
                print('mean(E): {0}, varE: {1}, meanF: {2}, varF: {3}, #samples {4}, #Step {5} \n\n'.format(meanE,varE, meanF, varF, numsamples, it+1))
            elif T0 == 0.0:
                print('mean(E): {0}, varE: {1}, #samples {2}, #Step {3} \n\n'.format(meanE,varE,numsamples, it+1))

        grads = grad_f(params, fixed_params, samples.reshape(-1, N, p), Eloc, T)
        #print("grad_time:", time.time()-t)
        if sr == True:
            o_prod = lambda x, y: jnp.outer(x, y)
            ravel_grads, unflatten = ravel_pytree(grads)
            samples = samples.reshape(-1, N, p)
            imag_log_amp_grads = grad_imag_log_amp(params, fixed_params, samples)
            real_log_amp_jac = jac_real_log_amp(params, fixed_params, samples)
            imag_log_amp_jac = jac_imag_log_amp(params, fixed_params, samples)

            imag_log_amp_grads, unflatten_imag_log_amp_grads = ravel_pytree(imag_log_amp_grads)
            real_log_amp_jac, unflatten_real_log_amp_jac = ravel_pytree(real_log_amp_jac)
            imag_log_amp_jac, unflatten_imag_log_amp_jac = ravel_pytree(imag_log_amp_jac)

            q_fisher = jnp.mean(vmap(o_prod, (0, 0), 0)(real_log_amp_jac, real_log_amp_jac), axis = 0) + jnp.mean(vmap(o_prod, (0, 0), 0)(imag_log_amp_jac, imag_log_amp_jac), axis = 0) - jnp.outer(imag_log_amp_grads, imag_log_amp_grads)
            q_fisher += jnp.eye(q_fisher.shape[0])*1e-3
            grads = unflatten(jnp.real(jax.scipy.sparse.linalg.cg(q_fisher, ravel_grads)[0]))
        '''
        count = 0
        for i in grads:
            count += 1
            print(count, ":", jnp.linalg.norm(i))
        '''
        if gradient_clip == True:
            grads = jax.tree_map(clip_grad, grads)


        #print("clip_grads:", grads)
        # Update the optimizer state and the parameters
        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        if (it%500 == 0):
            params_dict = jax.tree_util.tree_leaves(params)
            with open(f"params/params_L{L}_numsamples{numsamples}_bonddim{bond_dim}_rnntype_{rnn_type}_rotation_{basis_rotation}_angle{angle}.pkl", "wb") as f:
                pickle.dump(params_dict, f)
    print(time.time()-t)
    if not os.path.exists('./result/'):
        os.mkdir('./result/')
    jnp.save("result/meanE_L"+str(L)+"_bonddim"+str(bond_dim)+"_cmi_pattern_"+cmi_pattern+"rotation"+str(basis_rotation)+"angle"+str(angle)+"_seed"+str(args.seed)+".npy", jnp.array(meanEnergy))
    jnp.save("result/varE_L"+str(L)+"_bonddim"+str(bond_dim)+"_cmi_pattern_"+cmi_pattern+"rotation"+str(basis_rotation)+"angle"+str(angle)+"_seed"+str(args.seed)+".npy", jnp.array(varEnergy))