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
jax.config.update("jax_enable_x64", False)

parser = argparse.ArgumentParser()
parser.add_argument('--L', type = int, default=16)
parser.add_argument('--numunits', type = int, default=32)
parser.add_argument('--lr', type = float, default=5e-4)
parser.add_argument('--lrthreshold', type = float, default=5e-4)
parser.add_argument('--lrdecaytime', type = float, default=5000)
parser.add_argument('--mag_fixed', type = bool, default=False)
parser.add_argument('--Sz', type = int, default=0)
parser.add_argument('--gradient_clip', type = bool, default=True)
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
parser.add_argument('--rnn_type', type = str, default="tensor_gru")
parser.add_argument('--cmi_pattern', type = str, default="no_decay")
parser.add_argument('--sparsity', type = int, default=0)
parser.add_argument('--basis_rotation', type = bool, default=True)
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
input_size = 2
L = args.L
key = PRNGKey(args.seed)
diag_bulk, diag_fl, diag_xzz =False, False, False
meanEnergy=[]
varEnergy=[]
N = L


for angle in (0.0*jnp.pi, 0.05*jnp.pi, 0.1*jnp.pi, 0.15*jnp.pi, 0.20*jnp.pi, 0.25*jnp.pi, 0.3*jnp.pi, 0.35*jnp.pi, 0.4*jnp.pi, 0.45*jnp.pi, 0.5*jnp.pi):
    # x and y are the cosine and sine of the rotation angle
    x, y = jnp.cos(angle), jnp.sin(angle)
    if (rnn_type == "vanilla"):
        params = init_vanilla_params(N, units, input_size, key)
    elif (rnn_type == "multilayer_vanilla"):
        params = init_multilayer_vanilla_params(N, units, input_size, key)
    elif (rnn_type == "gru"):
        params = init_gru_params(N, units, input_size, key)
    # Only tensor_gru type of rnn is working right now
    elif (rnn_type == "tensor_gru"):
        params = init_tensor_gru_params(N, units, input_size, key)

    # get the gradient function and compute cost is imported from Helper_miscelluous.py
    grad_f = jax.jit(jax.grad(compute_cost), static_argnums=(1,))

    # fixed params is made static (constant) in jax environement
    fixed_params = N, mag_fixed, magnetization, units

    '''
    Learning rate schuling, but it doesn't show substantial improvement
    warmup_cosine_decay_scheduler = optax.warmup_cosine_decay_schedule(init_value=lr, peak_value=lrthreshold,
                                                                       warmup_steps=Nwarmup,
                                                                       decay_steps=numsteps, end_value=1e-5)
    
    max_lr = 0.0005
    min_lr = 0.00005
    cycle_steps = 1000  
    scheduler = lambda step: linear_cycling_with_hold(step, max_lr, min_lr, cycle_steps)
    '''
    optimizer = optax.adam(learning_rate=lr)
    optimizer_state = optimizer.init(params)
    if (basis_rotation == False):
    # create pauli matrices, 1 stands for pauli x and 3 stands for pauli z, fl means first and last, l means the left one
    # bulk means the XZX terms for the bulk. There are (N-3) terms of them.
    # fl means the first and last term which is ZX and XX acting on the first two sites and last two sites respectively
    # xzz is the term XZZ acting on the last three sites
        pauli_array_bulk, pauli_array_fl, pauli_array_xzz  = jnp.repeat(jnp.array([1,3,1])[None], (N-3), axis=0),jnp.array([[3, 1],[1, 1]]), jnp.array([[1,3,3]])
        loc_array_bulk, loc_array_fl, loc_array_xzz = loc_array_es(N)
    else :
        '''
        First repeat for each location then iterate over the combinations
        [[1,1,1]...,[1,1,1],[1,1,3], [1,1,3]..., [3,3,3],[3,3,3]]
        '''
        pauli_array_bulk, pauli_array_fl, pauli_array_xzz = jnp.repeat(generate_combinations(3), (N-3), axis=0), jnp.repeat(generate_combinations(2), 2, axis=0), jnp.repeat(generate_combinations(3), 1, axis=0)

        # The location that each Hamiltonian term acts on
        loc_array_bulk, loc_array_fl, loc_array_xzz = loc_array_es(N)
        loc_array_bulk, loc_array_fl, loc_array_xzz  = jnp.tile(loc_array_bulk, (8, 1)), jnp.tile(loc_array_fl, (4, 1)), jnp.tile(loc_array_xzz, (8, 1))

    '''
    label_xxx[y, x] is a dict datatype and it is the location of loc_array_xxx 
    such that pauli_array_bulk.at[label[i][:,0].astype(int), label[i][:,1].astype(int)] will
    show the pauli matrix that acts on lattice location (y, x). This function coupled with pauli_cmi_pattern
    are used previously to change the measurement basis  with different density. We actually don't need this function here 
    '''
    label_bulk, label_fl, label_xzz = location_pauli_label(loc_array_bulk, loc_array_fl, loc_array_xzz, N)
    pauli_array_bulk, pauli_array_fl, pauli_array_xzz = pauli_cmi_pattern(pauli_array_bulk, pauli_array_fl, pauli_array_xzz, label_bulk, label_fl, label_xzz, cmi_pattern, key, sparsity, L)

    '''
    We group the location that each Hamiltonian term acts on according to how many x,y,z they have in each term
    XX_loc_YYY is a dict datatype and its key is the number of Z-term and X-term (Z, X) and its value is the location
    of corresponding XX type of interaction acting on the lattice
    
    And off_diag_count is to count how many off-diagonal terms are there when we do VMC. It's just the total number of terms involving X and Y.
    off-diag_coe is the corresponding coeffiecient for each off-diagonal term. 
    '''
    if (basis_rotation == False):
        xy_loc_bulk, yloc_bulk, zloc_bulk = local_element_indices_1d(3, pauli_array_bulk, loc_array_bulk)
        xy_loc_fl, yloc_fl, zloc_fl = local_element_indices_1d(2, pauli_array_fl, loc_array_fl)
        xy_loc_xzz, yloc_xzz, zloc_xzz = local_element_indices_1d(3, pauli_array_xzz, loc_array_xzz)
        off_diag_bulk_count, off_diag_fl_count, off_diag_xzz_count = off_diag_count(xy_loc_bulk, xy_loc_fl, xy_loc_xzz)
        off_diag_bulk_coe, off_diag_fl_coe, off_diag_xzz_coe = -jnp.ones(off_diag_bulk_count), -jnp.ones(off_diag_fl_count), -jnp.ones(off_diag_xzz_count)
    else :
        xy_loc_bulk, yloc_bulk, zloc_bulk, off_diag_bulk_coe = local_element_indices_1d(3, pauli_array_bulk, loc_array_bulk, rotation = True, angle = angle)
        xy_loc_fl, yloc_fl, zloc_fl, off_diag_fl_coe = local_element_indices_1d(2, pauli_array_fl, loc_array_fl, rotation = True, angle = angle)
        xy_loc_xzz, yloc_xzz, zloc_xzz, off_diag_xzz_coe = local_element_indices_1d(3, pauli_array_xzz, loc_array_xzz, rotation = True, angle = angle)

    zloc_bulk_diag, zloc_fl_diag, zloc_xzz_diag = jnp.array([]), jnp.array([]), jnp.array([])
    coe_bulk_diag, coe_fl_diag, coe_xzz_diag = jnp.array([]), jnp.array([]), jnp.array([])

    # Here we get the diagonal term and its coefficient of the Hamiltonian
    if (3, 0) in xy_loc_bulk:
        if zloc_bulk[(3, 0)].size!=0:
            zloc_bulk_diag = zloc_bulk[(3, 0)]     #label the diagonal term by zloc_bulk_diag
            if (basis_rotation == False):
                # it's fine here since no ZZ term exist in the original Hamiltonian
                coe_bulk_diag = -jnp.ones(zloc_bulk_diag.shape[0])
            else:
                # For xy_loc_bulk, the original term is XZX, for it rotate to ZZZ, it will obtain a cos(\theta)*sin^2(\theta) coeffiecient
                coe_bulk_diag = -jnp.ones(zloc_bulk_diag.shape[0])*x*y**2
        del xy_loc_bulk[(3, 0)]
        del yloc_bulk[(3, 0)]
        del zloc_bulk[(3, 0)]
    if (2, 0) in xy_loc_fl:
        if zloc_fl[(2, 0)].size!=0:
            zloc_fl_diag = zloc_fl[(2, 0)]
            if (basis_rotation == False):
                # it's fine here since no ZZ term exist in the original Hamiltonian
                coe_fl_diag = jnp.ones(zloc_fl_diag.shape[0])
            else:
                # ZX term rotate to ZZ term, it will obtain a -cos(\theta)*sin(\theta) coeffiecient
                # XX term rotate to ZZ term, it will obtain a sin^2(\theta) coeffiecient
                coe_fl_diag = jnp.concatenate((jnp.ones(int(zloc_fl_diag.shape[0]/2))*x*y, -jnp.ones(int(zloc_fl_diag.shape[0]/2))*y**2))
                #print(coe_fl_diag)
        del xy_loc_fl[(2, 0)]
        del yloc_fl[(2, 0)]
        del zloc_fl[(2, 0)]
    if (3, 0) in xy_loc_xzz:
        if zloc_xzz[(3, 0)].size!=0:
            zloc_xzz_diag = zloc_xzz[(3, 0)]
            if (basis_rotation == False):
            # it's fine here since no ZZ term exist in the original Hamiltonian
                coe_xzz_diag = jnp.ones(zloc_xzz_diag.shape[0])
            else:
                # XZZ term rotate to ZZZ term, it will obtain a -cos^2(\theta)*sin(\theta) coeffiecient
                coe_xzz_diag = jnp.ones(zloc_xzz_diag.shape[0])*x**2*y
        del xy_loc_xzz[(3, 0)]
        del yloc_xzz[(3, 0)]
        del zloc_xzz[(3, 0)]

    batch_total_samples_1d = vmap(total_samples_1d, (0, None), 0)
    batch_new_coe_1d = vmap(new_coe_1d, (0, None, None, None, None))
    batch_diag_coe = vmap(diag_coe, (0, None, None, None, None, None, None))
    T = T0
    t = time.time()
    for it in range(0, numsteps):

        start = time.time()
        '''
        if it+1<=Nwarmup+Nannealing*Ntrain: #Before the end of annealing
            lr_adap = max(lrthreshold, lr/(1+it/lrdecaytime))
        elif it+1>Nwarmup+Nannealing*Ntrain: #After annealing -> finetuning the model during convergence
            lr_adap = lrthreshold_conv/(1+(it-(Nwarmup+Nannealing*Ntrain))/lrdecaytime_conv)
        '''
        samples, sample_log_amp = sample_prob(numsamples, params, fixed_params, key)

        key, subkey1, subkey2 = split(key, 3)
        #print("samples:", samples)
        sigmas = jnp.concatenate((batch_total_samples_1d(samples, xy_loc_bulk),
                                 batch_total_samples_1d(samples, xy_loc_fl),
                                 batch_total_samples_1d(samples, xy_loc_xzz)), axis=1).reshape(-1, N)
        #print("sigmas:", sigmas)
        #minus sign account for the minus sign of each term in the Hamiltonian
        matrixelements = jnp.concatenate((batch_new_coe_1d(samples, off_diag_bulk_coe, yloc_bulk, zloc_bulk, basis_rotation),
                                         batch_new_coe_1d(samples, off_diag_fl_coe, yloc_fl, zloc_fl, basis_rotation),
                                         batch_new_coe_1d(samples, off_diag_xzz_coe, yloc_xzz, zloc_xzz, basis_rotation)), axis=1).reshape(numsamples, -1)
        #print("matrixelements:", matrixelements)
        log_all_amp = log_amp(sigmas, params, fixed_params)
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

        grads = grad_f(params, fixed_params, samples, Eloc, T)
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
    if not os.path.exists('./result/'):
        os.mkdir('./result/')
    jnp.save("result/meanE_L"+str(L)+"_units"+str(units)+"_cmi_pattern_"+cmi_pattern+"rotation"+str(basis_rotation)+"angle"+str(angle)+"_seed"+str(args.seed)+".npy", jnp.array(meanEnergy))
    jnp.save("result/varE_L"+str(L)+"_units"+str(units)+"_cmi_pattern_"+cmi_pattern+"rotation"+str(basis_rotation)+"angle"+str(angle)+"_seed"+str(args.seed)+".npy", jnp.array(varEnergy))
