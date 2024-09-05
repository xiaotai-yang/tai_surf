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



N = 64
L = N
p = 1
basis_rotation = True
for angle in (0.0*jnp.pi, 0.05*jnp.pi, 0.1*jnp.pi, 0.15*jnp.pi, 0.20*jnp.pi, 0.25*jnp.pi, 0.3*jnp.pi, 0.35*jnp.pi, 0.4*jnp.pi, 0.45*jnp.pi, 0.5*jnp.pi):
    # x and y are the cosine and sine of the rotation angle
    ang = round(angle, 3)

    fixed_params = N, p
    (xy_loc_bulk, xy_loc_fl, xy_loc_xzz, yloc_bulk, yloc_fl, yloc_xzz, zloc_bulk, zloc_fl,
    zloc_xzz, off_diag_bulk_coe, off_diag_fl_coe, off_diag_xzz_coe, zloc_bulk_diag, zloc_fl_diag,
    zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag) = vmc_off_diag_es(N, p, angle, basis_rotation)
    batch_total_samples_1d = vmap(total_samples_1d, (0, None), 0)
    batch_new_coe_1d = vmap(new_coe_1d, (0, None, None, None, None))
    batch_diag_coe = vmap(diag_coe, (0, None, None, None, None, None, None))
    batch_log_amp_dmrg = jax.jit(vmap(log_amp_dmrg, (0, None, None, None)))
    wemb = jnp.eye(2**p)
    M0 = jnp.load("../entanglement_swapping/DMRG/mps_tensors/tensor_init_" + str(L * p) + "_angle_" + str(ang) + ".npy")
    M = jnp.load("../entanglement_swapping/DMRG/mps_tensors/tensor_" + str(L * p) + "_angle_" + str(ang) + ".npy")
    Mlast = jnp.load("../entanglement_swapping/DMRG/mps_tensors/tensor_last_" + str(L * p) + "_angle_" + str(ang) + ".npy")
    dmrg_samples = jnp.load("../entanglement_swapping/DMRG/mps_tensors/sample_" + str(L * p) + "_angle_" + str(ang) + ".npy")-1

    #dmrg_samples = dmrg_samples[0:1, :]
    num_dmrg_samples = dmrg_samples.shape[0]

    dmrg_samples_log_amp = batch_log_amp_dmrg(dmrg_samples, M0, M, Mlast)
    dmrg_sigmas = jnp.concatenate((batch_total_samples_1d(dmrg_samples, xy_loc_bulk),
                                    batch_total_samples_1d(dmrg_samples, xy_loc_fl),
                                    batch_total_samples_1d(dmrg_samples, xy_loc_xzz)), axis=1).reshape(-1, N, p)

    dmrg_matrixelements = (jnp.concatenate((batch_new_coe_1d(dmrg_samples, off_diag_bulk_coe, yloc_bulk, zloc_bulk, basis_rotation),
                                        batch_new_coe_1d(dmrg_samples, off_diag_fl_coe, yloc_fl, zloc_fl, basis_rotation),
                                        batch_new_coe_1d(dmrg_samples, off_diag_xzz_coe, yloc_xzz, zloc_xzz, basis_rotation)), axis=1)
                        .reshape(num_dmrg_samples, -1))
    #print("matrixelements:", matrixelements)
    log_all_amp_dmrg_samples = batch_log_amp_dmrg(dmrg_sigmas.reshape(-1, L*p), M0, M, Mlast)
    log_diag_amp_dmrg_samples = jnp.repeat(dmrg_samples_log_amp, (jnp.ones(num_dmrg_samples)*(dmrg_matrixelements.shape[1])).astype(int), axis=0)

    amp_dmrg_samples = jnp.exp(log_all_amp_dmrg_samples.ravel()-log_diag_amp_dmrg_samples).reshape(num_dmrg_samples, -1)

    Eloc_dmrg_samples = jnp.sum((amp_dmrg_samples*(dmrg_matrixelements)), axis=1) + batch_diag_coe(dmrg_samples, zloc_bulk_diag, zloc_fl_diag, zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag)

    meanE_dmrg_samples,  varE_dmrg_samples = jnp.mean(Eloc_dmrg_samples), jnp.var(Eloc_dmrg_samples)
    print("meanE_dmrg_samples:", meanE_dmrg_samples, "varE_dmrg_samples:", varE_dmrg_samples)
    #print(off_diag_bulk_coe)


