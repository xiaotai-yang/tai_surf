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
parser.add_argument('--L', type = int, default = 16)
parser.add_argument('--p', type = int, default = 1)
parser.add_argument('--numunits', type = int, default=32)
parser.add_argument('--lr', type = float, default=1e-3)
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
parser.add_argument('--sr', type = bool, default=False)
args = parser.parse_args()

units = args.numunits
numsamples = args.numsamples
p = args.p
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
sr = args.sr
input_size = 2 ** p
L = args.L
key = PRNGKey(args.seed)
diag_bulk, diag_fl, diag_xzz =False, False, False
meanEnergy=[]
varEnergy=[]
meanEnergy_dmrg=[]
varEnergy_dmrg=[]
N = L


for angle in (0.0*jnp.pi,  0.05*jnp.pi, 0.1*jnp.pi, 0.15*jnp.pi, 0.20*jnp.pi, 0.25*jnp.pi, 0.3*jnp.pi, 0.35*jnp.pi, 0.4*jnp.pi, 0.45*jnp.pi, 0.5*jnp.pi):
    # x and y are the cosine and sine of the rotation angle
    ang = round(angle, 3)
    with open('params/params_L16_numsamples128_numunits32_rnntype_tensor_gru_rotation_True_angle1.5707963267948966.pkl', 'rb') as f:
        params = pickle.load(f)
    fixed_params = N, p, units
    (xy_loc_bulk, xy_loc_fl, xy_loc_xzz, yloc_bulk, yloc_fl, yloc_xzz, zloc_bulk, zloc_fl,
    zloc_xzz, off_diag_bulk_coe, off_diag_fl_coe, off_diag_xzz_coe, zloc_bulk_diag, zloc_fl_diag,
    zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag) = vmc_off_diag_es(N, p, angle, basis_rotation)
    batch_sample_prob = jax.jit(vmap(sample_prob, (None, None, None, 0)), static_argnames=['fixed_params'])
    batch_log_amp = jax.jit(vmap(log_amp, (0, None, None, None)), static_argnames=['fixed_params'])
    batch_total_samples_1d = vmap(total_samples_1d, (0, None), 0)
    batch_new_coe_1d = vmap(new_coe_1d, (0, None, None, None, None))
    batch_diag_coe = vmap(diag_coe, (0, None, None, None, None, None, None))
    batch_log_amp_dmrg = jax.jit(vmap(log_amp_dmrg, (0, None, None, None)))
    wemb = jnp.eye(2**p)
    M0 = jnp.load("../entanglement_swapping/DMRG/mps_tensors/tensor_init_" + str(L * p) + "_angle_" + str(ang) + ".npy")
    M = jnp.load("../entanglement_swapping/DMRG/mps_tensors/tensor_" + str(L * p) + "_angle_" + str(ang) + ".npy")
    Mlast = jnp.load("../entanglement_swapping/DMRG/mps_tensors/tensor_last_" + str(L * p) + "_angle_" + str(ang) + ".npy")
    dmrg_samples = jnp.load("../entanglement_swapping/DMRG/mps_tensors/sample_" + str(L * p) + "_angle_" + str(ang) + ".npy")-1
    num_dmrg_samples = dmrg_samples.shape[0]
    print(M.shape)
    T = T0
    t = time.time()
    for it in range(0, numsteps):

        start = time.time()
        key_ = split(key, numsamples)
        samples, sample_log_amp = batch_sample_prob(params, wemb, fixed_params, key_)
        samples_log_amp_dmrg = batch_log_amp_dmrg(samples, M0, M, Mlast)
        dmrg_samples_log_amp = batch_log_amp_dmrg(dmrg_samples, M0, M, Mlast)
        key, subkey1, subkey2 = split(key_[0], 3)
        sigmas = jnp.concatenate((batch_total_samples_1d(samples, xy_loc_bulk),
                                 batch_total_samples_1d(samples, xy_loc_fl),
                                 batch_total_samples_1d(samples, xy_loc_xzz)), axis=1).reshape(-1, N, p)
        dmrg_sigmas = jnp.concatenate((batch_total_samples_1d(dmrg_samples, xy_loc_bulk),
                                        batch_total_samples_1d(dmrg_samples, xy_loc_fl),
                                        batch_total_samples_1d(dmrg_samples, xy_loc_xzz)), axis=1).reshape(-1, N, p)

        matrixelements = (jnp.concatenate((batch_new_coe_1d(samples, off_diag_bulk_coe, yloc_bulk, zloc_bulk, basis_rotation),
                                         batch_new_coe_1d(samples, off_diag_fl_coe, yloc_fl, zloc_fl, basis_rotation),
                                         batch_new_coe_1d(samples, off_diag_xzz_coe, yloc_xzz, zloc_xzz, basis_rotation)), axis=1)
                          .reshape(numsamples, -1))
        dmrg_matrixelements = (jnp.concatenate((batch_new_coe_1d(dmrg_samples, off_diag_bulk_coe, yloc_bulk, zloc_bulk, basis_rotation),
                                            batch_new_coe_1d(dmrg_samples, off_diag_fl_coe, yloc_fl, zloc_fl, basis_rotation),
                                            batch_new_coe_1d(dmrg_samples, off_diag_xzz_coe, yloc_xzz, zloc_xzz, basis_rotation)), axis=1)
                            .reshape(num_dmrg_samples, -1))
        #print("matrixelements:", matrixelements)
        log_all_amp = batch_log_amp(sigmas, params, wemb, fixed_params)
        log_all_amp_dmrg = batch_log_amp_dmrg(sigmas.reshape(-1, L*p), M0, M, Mlast)
        log_all_amp_dmrg_samples = batch_log_amp_dmrg(dmrg_sigmas.reshape(-1, L*p), M0, M, Mlast)

        log_diag_amp = jnp.repeat(sample_log_amp, (jnp.ones(numsamples)*(matrixelements.shape[1])).astype(int), axis=0)
        log_diag_amp_dmrg = jnp.repeat(samples_log_amp_dmrg, (jnp.ones(numsamples)*(matrixelements.shape[1])).astype(int), axis=0)
        log_diag_amp_dmrg_samples = jnp.repeat(dmrg_samples_log_amp, (jnp.ones(num_dmrg_samples)*(dmrg_matrixelements.shape[1])).astype(int), axis=0)

        amp = jnp.exp(log_all_amp.ravel()-log_diag_amp).reshape(numsamples, -1)
        amp_dmrg = jnp.exp(log_all_amp_dmrg.ravel()-log_diag_amp_dmrg).reshape(numsamples, -1)
        amp_dmrg_samples = jnp.exp(log_all_amp_dmrg_samples.ravel()-log_diag_amp_dmrg_samples).reshape(num_dmrg_samples, -1)

        Eloc = jnp.sum((amp*matrixelements), axis=1) + batch_diag_coe(samples, zloc_bulk_diag, zloc_fl_diag, zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag)
        Eloc_dmrg = jnp.sum((amp_dmrg*(matrixelements)), axis=1) + batch_diag_coe(samples, zloc_bulk_diag, zloc_fl_diag, zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag)
        Eloc_dmrg_samples = jnp.sum((amp_dmrg_samples*(dmrg_matrixelements)), axis=1) + batch_diag_coe(dmrg_samples, zloc_bulk_diag, zloc_fl_diag, zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag)

        meanE,  varE = jnp.mean(Eloc), jnp.var(Eloc)
        meanE_dmrg,  varE_dmrg = jnp.mean(Eloc_dmrg), jnp.var(Eloc_dmrg)
        meanE_dmrg_samples,  varE_dmrg_samples = jnp.mean(Eloc_dmrg_samples), jnp.var(Eloc_dmrg_samples)

        meanEnergy.append(meanE)
        varEnergy.append(varE)
        meanEnergy_dmrg.append(meanE_dmrg)
        varEnergy_dmrg.append(varE_dmrg)

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
        if (it+1)%5==0 or it==0:
            if T0 != 0:
                print('mean(E): {0}, varE: {1}, meanF: {2}, varF: {3}, #samples {4}, #Step {5} \n\n'.format(meanE,varE, meanF, varF, numsamples, it+1))
            elif T0 == 0.0:
                print('mean(E): {0}, varE: {1}, #samples {2}, #Step {3} \n\n'.format(meanE,varE,numsamples, it+1))
                print('mean(E_dmrg): {0}, varE_dmrg: {1}, #samples {2}, #Step {3} \n\n'.format(meanE_dmrg,varE_dmrg,numsamples, it+1))
                print('mean(E_dmrg_samples): {0}, varE_dmrg_samples: {1}, #samples {2}, #Step {3} \n\n'.format(meanE_dmrg_samples,varE_dmrg_samples,num_dmrg_samples, it+1))
