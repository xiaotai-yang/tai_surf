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
#jax.config.update("jax_enable_x64", False)

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
parser.add_argument('--rnn_type', type = str, default="RWKV")
parser.add_argument('--cmi_pattern', type = str, default="decay")
parser.add_argument('--sparsity', type = int, default=0)
parser.add_argument('--basis_rotation', type = bool, default=False)
parser.add_argument('--train_state', type = bool, default=True)
parser.add_argument('--angle', type = float, default=0.000001)
parser.add_argument('--emb_size', type = int, default=8)
parser.add_argument('--h_size', type = int, default=32)
parser.add_argument('--num_layer', type = int, default=3)
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
for angle in ((jnp.arange(9)+1)*0.05*jnp.pi):
    print(sparsity)
    '''
    Initialization
    '''
    x, y = jnp.cos(angle), jnp.sin(angle)
    params = params_init(rnn_type, Nx, Ny, units, input_size, emb_size, h_size, num_layer, key)
    grad_f = jax.jit(jax.grad(compute_cost), static_argnums=(1,))
    fixed_params = Ny, Nx, py, px, mag_fixed, magnetization, num_layer
    #winit_emb_x, winit_emb_y = jnp.zeros((Nx, input_size)), jnp.zeros((Ny, input_size))
    #state_init_x, state_init_y = jnp.zeros((Nx, units)), jnp.zeros((Ny, units))
    ny_nx_indices = jnp.array([[[i, j] for j in range(Nx)] for i in range(Ny)])
    batch_total_samples_2d = vmap(total_samples_2d, (0, None), 0)
    batch_new_coe_2d = vmap(new_coe_2d, (0, None, None, None, None))
    batch_diag_coe = vmap(diag_coe, (0, None, None, None, None, None, None))

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
    if (basis_rotation == False):
    # create pauli matrices, 1 stands for pauli x and 3 stands for pauli z
        total_samples = numsamples*Nx*px*Ny*py
        pauli_array_bulk, pauli_array_edge, pauli_array_corner  = jnp.repeat(jnp.array([1,3,3,3,3])[None], (Ny*py-2)*(Nx*px-2), axis=0), jnp.repeat(jnp.array([1,3,3,3])[None], (Ny*py+Nx*px-4)*2, axis=0), jnp.repeat(jnp.array([1,3,3])[None], 4, axis=0)
        loc_array_bulk, loc_array_edge, loc_array_corner = loc_array_gf(Ny*py, Nx*px)
    else :
        total_samples = numsamples * ((Nx * px - 2) * (Ny * py - 2) * (2 ** (5) - 1) + (Nx * px - 2) * 2 * (2 ** (4) - 1) + (Ny * py - 2) * 2 * (2 ** (4) - 1) + 4 * (2 ** (3) - 1))
        '''
        First repeat for each location then iterate over the combinations
        [[1,1,1,1,1]...,[1,1,1,1,1],[1,1,1,1,3]...[3,3,3,3,3]]
        '''
        pauli_array_bulk, pauli_array_edge, pauli_array_corner = jnp.repeat(generate_combinations(5), (Ny*py-2)*(Nx*px-2), axis=0), jnp.repeat(generate_combinations(4),(Ny*py+Nx*px-4)*2, axis=0), jnp.repeat(generate_combinations(3), 4, axis=0)

        # The location that each Hamiltonian term acts on
        loc_array_bulk, loc_array_edge, loc_array_corner = loc_array_gf(Ny*py, Nx*px)
        loc_array_bulk, loc_array_edge, loc_array_corner  = jnp.tile(loc_array_bulk, (32, 1, 1)), jnp.tile(loc_array_edge, (16, 1, 1)), jnp.tile(loc_array_corner, (8, 1, 1 ))


    '''
    label_xxx[y, x] is a dict datatype and it is the location of loc_array_xxx 
    such that pauli_array_bulk.at[label[i][:,0].astype(int), label[i][:,1].astype(int)] will
    show the pauli matrix that acts on lattice location
    '''
    label_bulk, label_edge, label_corner = location_pauli_label(loc_array_bulk, loc_array_edge, loc_array_corner, Ny*py, Nx*px)
    pauli_array_bulk, pauli_array_edge, pauli_array_corner = pauli_cmi_pattern(pauli_array_bulk, pauli_array_edge, pauli_array_corner, label_bulk, label_edge, label_corner, cmi_pattern, key, sparsity, L*py)

    '''
    We group the location that each Hamiltonian term acts on according to how many x,y,z they have in each term
    XX_loc_YYY is a dict datatype and its key is the number of Z-term and X-term (Z, X) and its value is the location
    of corresponding XX type of interaction acting on the lattice 
    '''
    if (basis_rotation == False):
        xy_loc_bulk, yloc_bulk, zloc_bulk = local_element_indices_2d(5, pauli_array_bulk, loc_array_bulk)
        xy_loc_edge, yloc_edge, zloc_edge = local_element_indices_2d(4, pauli_array_edge, loc_array_edge)
        xy_loc_corner, yloc_corner, zloc_corner = local_element_indices_2d(3, pauli_array_corner, loc_array_corner)
        off_diag_bulk_count, off_diag_edge_count, off_diag_corner_count = off_diag_count(xy_loc_bulk, xy_loc_edge, xy_loc_corner)
        off_diag_bulk_coe, off_diag_edge_coe, off_diag_corner_coe = -jnp.ones(off_diag_bulk_count), -jnp.ones(off_diag_edge_count), -jnp.ones(off_diag_corner_count)
    else :
        xy_loc_bulk, yloc_bulk, zloc_bulk, off_diag_bulk_coe = local_element_indices_2d(5, pauli_array_bulk, loc_array_bulk, rotation = True, angle = angle)
        xy_loc_edge, yloc_edge, zloc_edge, off_diag_edge_coe = local_element_indices_2d(4, pauli_array_edge, loc_array_edge, rotation = True, angle = angle)
        xy_loc_corner, yloc_corner, zloc_corner, off_diag_corner_coe = local_element_indices_2d(3, pauli_array_corner, loc_array_corner, rotation = True, angle = angle)

    zloc_bulk_diag, zloc_edge_diag, zloc_corner_diag = jnp.array([]), jnp.array([]), jnp.array([])
    coe_bulk_diag, coe_edge_diag, coe_corner_diag = jnp.array([]), jnp.array([]), jnp.array([])

    if (5, 0) in xy_loc_bulk:
        if zloc_bulk[(5, 0)].size!=0:
            zloc_bulk_diag = zloc_bulk[(5, 0)]     #label the diagonal term by zloc_bulk_diag
            if (basis_rotation == False):
                coe_bulk_diag = -jnp.ones(zloc_bulk_diag.shape[0])
            else:
                coe_bulk_diag = -jnp.ones(zloc_bulk_diag.shape[0])*x**4*y #Here is the coefficient for the diagonal term. We can change it later if we want
        del xy_loc_bulk[(5, 0)]
        del yloc_bulk[(5, 0)]
        del zloc_bulk[(5, 0)]
    if (4, 0) in xy_loc_edge:
        if zloc_edge[(4, 0)].size!=0:
            zloc_edge_diag = zloc_edge[(4, 0)]
            if (basis_rotation == False):
                coe_edge_diag = -jnp.ones(zloc_edge_diag.shape[0])
            else:
                coe_edge_diag = -jnp.ones(zloc_edge_diag.shape[0])*x**3*y
        del xy_loc_edge[(4, 0)]
        del yloc_edge[(4, 0)]
        del zloc_edge[(4, 0)]
    if (3, 0) in xy_loc_corner:
        if zloc_corner[(3, 0)].size!=0:
            zloc_corner_diag = zloc_corner[(3, 0)]
            if (basis_rotation == False):
                coe_corner_diag = -jnp.ones(zloc_corner_diag.shape[0])
            else:
                coe_corner_diag = -jnp.ones(zloc_corner_diag.shape[0])*x**2*y
        del xy_loc_corner[(3, 0)]
        del yloc_corner[(3, 0)]
        del zloc_corner[(3, 0)]

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
        samples, sample_log_amp = vmap(sample_prob_RWKV, (None, None, None, 0))(params, fixed_params, ny_nx_indices, sample_key)
        #print("sample_time:", time.time()-t0)
        samples = jnp.transpose(samples.reshape(numsamples, Ny, Nx, py, px), (0, 1, 3, 2, 4)).reshape(numsamples, Ny*py, Nx*px)

        sigmas = jnp.concatenate((batch_total_samples_2d(samples, xy_loc_bulk),
                                 batch_total_samples_2d(samples, xy_loc_edge),
                                 batch_total_samples_2d(samples, xy_loc_corner)), axis=1).reshape(-1, Ny*py, Nx*px)
        #minus sign account for the minus sign of each term in the Hamiltonian
        matrixelements = jnp.concatenate((batch_new_coe_2d(samples, off_diag_bulk_coe, yloc_bulk, zloc_bulk, basis_rotation),
                                         batch_new_coe_2d(samples, off_diag_edge_coe, yloc_edge, zloc_edge, basis_rotation),
                                         batch_new_coe_2d(samples, off_diag_corner_coe, yloc_corner, zloc_corner, basis_rotation)), axis=1).reshape(numsamples, -1)

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
            grads = jax.tree.map(clip_grad, grads)
        #print("clip_grads:", grads)
        # Update the optimizer state and the parameters
        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        if (it%500 == 0):
            params_dict = jax.tree_util.tree_leaves(params)
            with open(f"params/params_L{L}_numsamples{numsamples}_numunits{units}_rnntype_{rnn_type}_rotation_{basis_rotation}_angle{angle}.pkl", "wb") as f:
                pickle.dump(params_dict, f)
    print(time.time()-t)
    jnp.save("result/meanE_L"+str(L)+"_units"+str(units)+"_cmi_pattern_"+cmi_pattern+"rotation"+str(basis_rotation)+"angle"+str(angle)+"_seed"+str(args.seed)+"_patch"+str(px*py)+".npy", jnp.array(meanEnergy))
    jnp.save("result/varE_L"+str(L)+"_units"+str(units)+"_cmi_pattern_"+cmi_pattern+"rotation"+str(basis_rotation)+"angle"+str(angle)+"_seed"+str(args.seed)+"_patch"+str(px*py)+".npy", jnp.array(varEnergy))
