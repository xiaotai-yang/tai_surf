import netket as nk
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import jax.random as random
from jax.random import PRNGKey, split, categorical
import jax.lax as lax
from jax.lax import scan
import jax.nn as nn
import time
from tqdm import tqdm
from functools import partial
import time
from math import ceil
from RNNfunction import *
import numpy as np

def J1J2J3_MatrixElements(sigmap,diag_amp,J1,J2,J3):    #sigmap is (num_samples ,input basis)

    #the diagonal part is simply the sum of all Sz-Sz interactions

    diag = (jnp.sum(jnp.abs(sigmap[:,:-1]-sigmap[:,1:])*2-1, axis = (1,2))+jnp.sum(jnp.abs(sigmap[:,:,:-1]-sigmap[:,:,1:])*2-1, axis=(1,2)))*(-0.25)*J1  #J1 interactions
    diag += (jnp.sum(jnp.abs(sigmap[:,:-1,:-1]-sigmap[:,1:,1:])*2-1, axis=(1,2))+jnp.sum(jnp.abs(sigmap[:,1:,:-1]-sigmap[:,:-1,1:])*2-1, axis=(1,2)))*(-0.25)*J2  #J2 interactions
    diag += (jnp.sum(jnp.abs(sigmap[:,:-2]-sigmap[:,2:])-1, axis=(1,2))+jnp.sum(jnp.abs(sigmap[:,:,:-2]-sigmap[:,:,2:])*2-1, axis=(1,2)))*(-0.25)*J3  #J3 interactions

    matrixelements = jnp.array(diag)/diag_amp  
    sigmas = sigmap
    basis_where = jnp.zeros(1)

    #off-diagonal part
    #J1 ↓

    mask = jnp.array(sigmap[:,:-1, :] != sigmap[:, 1:, :])
    num_avail_basis = jnp.sum(mask, axis=(1, 2))
    
    #t = time.time()
    ini = jnp.repeat(diag_amp, num_avail_basis)*-0.5*J1
    #print("repeat_time:",time.time()-t)
    matrixelements = jnp.concatenate((matrixelements, ini), axis = 0)
    nonzero_loc = jnp.array(jnp.nonzero(mask))
    num_nonzero = nonzero_loc.shape[1]

    ind1 = nonzero_loc.at[0].set(jnp.arange(num_nonzero))

    ind2 = ind1+jnp.array([[0],[1],[0]])
    #t = time.time()
    sig = jnp.repeat(sigmap, num_avail_basis, axis=0)
    #print("sig_repeat_time:", time.time()-t)
    temp = sig[ind1[0] , ind1[1], ind1[2]]
    
    sig = sig.at[ind1[0] , ind1[1], ind1[2]].set(sig[ind2[0] , ind2[1], ind2[2]])
    sig = sig.at[ind2[0] , ind2[1], ind2[2]].set(temp)
   
    sigmas = jnp.concatenate((sigmas, sig), axis = 0)
    basis_where = jnp.concatenate((basis_where, jnp.cumsum(num_avail_basis)), axis=0)
                
    #J1 →
    mask = (sigmap[:, :, :-1] != sigmap[:, :, 1:])           
    num_avail_basis = jnp.sum(mask, axis=(1, 2))
    matrixelements = jnp.concatenate((matrixelements, jnp.repeat(1/diag_amp, num_avail_basis, axis=0)*-0.5*J1), axis = 0)
    
    nonzero_loc = jnp.array(jnp.nonzero(mask))
    num_nonzero = nonzero_loc.shape[1]
    ind1 = nonzero_loc.at[0].set(jnp.arange(num_nonzero))
    ind2 = ind1+jnp.array([[0],[0],[1]]) 
    sig = jnp.repeat(sigmap, num_avail_basis, axis=0)
    temp = sig[ind1[0] , ind1[1], ind1[2]]
    sig = sig.at[ind1[0] , ind1[1], ind1[2]].set(sig[ind2[0] , ind2[1], ind2[2]])
    sig = sig.at[ind2[0] , ind2[1], ind2[2]].set(temp)
    sigmas = jnp.concatenate((sigmas, sig), axis = 0)
    basis_where = jnp.append(basis_where, basis_where[-1])
    basis_where = jnp.concatenate((basis_where, jnp.cumsum(num_avail_basis)+basis_where[-1]), axis = 0)                           
    if (J2!=0):                           
        #J2 →↓
        mask = (sigmap[:, :-1, :-1] != sigmap[:, 1:, 1:])           
        num_avail_basis = jnp.sum(mask, axis=(1, 2))
        matrixelements = jnp.concatenate((matrixelements, jnp.repeat(1/diag_amp, num_avail_basis, axis=0)*-0.5*J2), axis = 0)

        nonzero_loc = jnp.array(jnp.nonzero(mask))
        num_nonzero = nonzero_loc.shape[1]
        ind1 = nonzero_loc.at[0].set(jnp.arange(num_nonzero))
        ind2 = ind1+jnp.array([[0],[1],[1]]) 
        sig = jnp.repeat(sigmap, num_avail_basis, axis=0)
        temp = sig[ind1[0] , ind1[1], ind1[2]]

        sig = sig.at[ind1[0] , ind1[1], ind1[2]].set(sig[ind2[0] , ind2[1], ind2[2]])
        sig = sig.at[ind2[0] , ind2[1], ind2[2]].set(temp)

        sigmas = jnp.concatenate((sigmas, sig), axis = 0)
        basis_where = jnp.append(basis_where, basis_where[-1])
        basis_where = jnp.concatenate((basis_where, jnp.cumsum(num_avail_basis)+basis_where[-1]), axis = 0)   
        
        #J2 ←↓
        mask = (sigmap[:, :-1, 1:] != sigmap[:, 1:, :-1])           
        num_avail_basis = jnp.sum(mask, axis=(1, 2))
        matrixelements = jnp.concatenate((matrixelements, jnp.repeat(1/diag_amp, num_avail_basis, axis=0)*-0.5*J2), axis = 0)

        nonzero_loc = jnp.array(jnp.nonzero(mask))
        num_nonzero = nonzero_loc.shape[1]
        ind1 = nonzero_loc.at[0].set(jnp.arange(num_nonzero))
        ind2 = ind1+jnp.array([[0],[1],[-1]]) 
        sig = jnp.repeat(sigmap, num_avail_basis, axis=0)
        temp = sig[ind1[0] , ind1[1], ind1[2]]
        sig = sig.at[ind1[0] , ind1[1], ind1[2]].set(sig[ind2[0] , ind2[1], ind2[2]])
        sig = sig.at[ind2[0] , ind2[1], ind2[2]].set(temp)
        sigmas = jnp.concatenate((sigmas, sig), axis = 0)
        basis_where = jnp.append(basis_where, basis_where[-1])
        basis_where = jnp.concatenate((basis_where, jnp.cumsum(num_avail_basis)+basis_where[-1]), axis = 0) 
    
    if (J3!=0):
    #J3 ↓↓
        mask = (sigmap[:-2, :] != sigmap[2:, :])           
        num_avail_basis = jnp.sum(mask, axis=(1, 2))
        matrixelements = jnp.concatenate((matrixelements, jnp.repeat(1/diag_amp, num_avail_basis, axis=0)*-0.5*J3), axis = 0)

        nonzero_loc = jnp.array(jnp.nonzero(mask))
        num_nonzero = nonzero_loc.shape[1]
        ind1 = nonzero_loc.at[0].set(jnp.arange(num_nonzero))
        ind2 = ind1+jnp.array([[0],[2],[0]]) 
        sig = jnp.repeat(sigmap, num_avail_basis, axis=0)
        temp = sig[ind1[0] , ind1[1], ind1[2]]
        sig = sig.at[ind1[0] , ind1[1], ind1[2]].set(sig[ind2[0] , ind2[1], ind2[2]])
        sig = sig.at[ind2[0] , ind2[1], ind2[2]].set(temp)
        sigmas = jnp.concatenate((sigmas, sig), axis = 0)
        basis_where = jnp.append(basis_where, basis_where[-1])
        basis_where = jnp.concatenate((basis_where, jnp.cumsum(num_avail_basis)+basis_where[-1]), axis = 0) 
    #J3 →→
        mask = (sigmap[:, :-2] != sigmap[:, 2:])           
        num_avail_basis = jnp.sum(mask, axis=(1, 2))
        matrixelements = jnp.concatenate((matrixelements, jnp.repeat(1/diag_amp, num_avail_basis, axis=0)*-0.5*J3), axis = 0)

        nonzero_loc = jnp.array(jnp.nonzero(mask))
        num_nonzero = nonzero_loc.shape[1]
        ind1 = nonzero_loc.at[0].set(jnp.arange(num_nonzero))
        ind2 = ind1+jnp.array([[0],[0],[2]]) 
        sig = jnp.repeat(sigmap, num_avail_basis, axis=0)
        temp = sig[ind1[0] , ind1[1], ind1[2]]
        sig = sig.at[ind1[0] , ind1[1], ind1[2]].set(sig[ind2[0] , ind2[1], ind2[2]])
        sig = sig.at[ind2[0] , ind2[1], ind2[2]].set(temp)
        sigmas = jnp.concatenate((sigmas, sig), axis = 0)
        basis_where = jnp.append(basis_where, basis_where[-1])
        basis_where = jnp.concatenate((basis_where, jnp.cumsum(num_avail_basis)+basis_where[-1]), axis = 0) 

    return matrixelements, sigmas, basis_where

def J1J2J3_MatrixElements_numpy(sigmap,diag_amp,J1,J2,J3,Nx,Ny):    #sigmap is (num_samples ,input basis)

    #the diagonal part is simply the sum of all Sz-Sz interactions

    diag = (np.sum(np.abs(sigmap[:,:-1]-sigmap[:,1:])*2-1, axis = (1,2))+np.sum(np.abs(sigmap[:,:,:-1]-sigmap[:,:,1:])*2-1, axis=(1,2)))*(-0.25)*J1  #J1 interactions
    diag += (np.sum(np.abs(sigmap[:,:-1,:-1]-sigmap[:,1:,1:])*2-1, axis=(1,2))+np.sum(np.abs(sigmap[:,1:,:-1]-sigmap[:,:-1,1:])*2-1, axis=(1,2)))*(-0.25)*J2  #J2 interactions
    diag += (np.sum(np.abs(sigmap[:,:-2]-sigmap[:,2:])-1, axis=(1,2))+np.sum(np.abs(sigmap[:,:,:-2]-sigmap[:,:,2:])*2-1, axis=(1,2)))*(-0.25)*J3  #J3 interactions

    matrixelements = np.ones(sigmap.shape[0])*diag
    log_diag_amp = np.log(diag_amp)  
    sigmas = sigmap
    basis_where = np.zeros(1)

    #off-diagonal part
    #J1 ↓

    mask = np.array(sigmap[:,:-1, :] != sigmap[:, 1:, :])
    num_avail_basis = np.sum(mask, axis=(1, 2))
    
    #print("repeat_time:",time.time()-t)
    matrixelements = np.concatenate((matrixelements, np.repeat(-0.5*J1, num_avail_basis.sum(), axis=0)), axis = 0)
    log_diag_amp_array = np.concatenate((log_diag_amp, np.repeat(log_diag_amp, num_avail_basis, axis=0)), axis = 0)
    nonzero_loc = np.array(np.nonzero(mask))
    num_nonzero = nonzero_loc.shape[1]

    nonzero_loc[0] = np.arange(num_nonzero)
    ind1 = nonzero_loc
    ind2 = ind1+np.array([[0],[1],[0]])
    #t = time.time()
    sig = np.repeat(sigmap, num_avail_basis, axis=0)
    #print("sig_repeat_time:", time.time()-t)
    temp = sig[ind1[0] , ind1[1], ind1[2]]
    
    sig[ind1[0] , ind1[1], ind1[2]] = sig[ind2[0] , ind2[1], ind2[2]]
    sig[ind2[0] , ind2[1], ind2[2]] = temp
   
    sigmas = np.concatenate((sigmas, sig), axis = 0)
    basis_where = np.concatenate((basis_where, np.cumsum(num_avail_basis)), axis=0)
                
    #J1 →
    mask = (sigmap[:, :, :-1] != sigmap[:, :, 1:])           
    num_avail_basis = np.sum(mask, axis=(1, 2))
    matrixelements = np.concatenate((matrixelements, np.repeat(-0.5*J1, num_avail_basis.sum(), axis=0)), axis = 0)
    log_diag_amp_array = np.concatenate((log_diag_amp_array, np.repeat(log_diag_amp, num_avail_basis, axis=0)), axis = 0)
    nonzero_loc = np.array(np.nonzero(mask))
    num_nonzero = nonzero_loc.shape[1]
    nonzero_loc[0] = np.arange(num_nonzero)
    ind1 = nonzero_loc
    ind2 = ind1+np.array([[0],[0],[1]]) 
    sig = np.repeat(sigmap, num_avail_basis, axis=0)
    temp = sig[ind1[0] , ind1[1], ind1[2]]
    sig[ind1[0] , ind1[1], ind1[2]] = sig[ind2[0] , ind2[1], ind2[2]]
    sig[ind2[0] , ind2[1], ind2[2]] = temp
    sigmas = np.concatenate((sigmas, sig), axis = 0)
    basis_where = np.append(basis_where, basis_where[-1])
    basis_where = np.concatenate((basis_where, np.cumsum(num_avail_basis)+basis_where[-1]), axis = 0)                           
    if (J2!=0):                           
        #J2 →↓
        mask = (sigmap[:, :-1, :-1] != sigmap[:, 1:, 1:])           
        num_avail_basis = np.sum(mask, axis=(1, 2))
        matrixelements = np.concatenate((matrixelements, np.repeat(0.5*J2, num_avail_basis.sum(), axis=0)), axis = 0)
        log_diag_amp_array = np.concatenate((log_diag_amp_array, np.repeat(log_diag_amp, num_avail_basis, axis=0)), axis = 0)
        nonzero_loc = np.array(np.nonzero(mask))
        num_nonzero = nonzero_loc.shape[1]
        nonzero_loc[0] = np.arange(num_nonzero)
        ind1 = nonzero_loc
        ind2 = ind1+np.array([[0],[1],[1]]) 
        sig = np.repeat(sigmap, num_avail_basis, axis=0)
        temp = sig[ind1[0] , ind1[1], ind1[2]]

        sig[ind1[0] , ind1[1], ind1[2]] = sig[ind2[0] , ind2[1], ind2[2]]
        sig[ind2[0] , ind2[1], ind2[2]] = temp

        sigmas = np.concatenate((sigmas, sig), axis = 0)
        basis_where = np.append(basis_where, basis_where[-1])
        basis_where = np.concatenate((basis_where, np.cumsum(num_avail_basis)+basis_where[-1]), axis = 0)   
        
        #J2 ←↓
        mask = (sigmap[:, :-1, 1:] != sigmap[:, 1:, :-1])           
        num_avail_basis = np.sum(mask, axis=(1, 2))
        matrixelements = np.concatenate((matrixelements, np.repeat(0.5*J2, num_avail_basis.sum(), axis=0)), axis = 0)
        log_diag_amp_array = np.concatenate((log_diag_amp_array, np.repeat(log_diag_amp, num_avail_basis, axis=0)), axis = 0)
        nonzero_loc = np.array(np.nonzero(mask))
        num_nonzero = nonzero_loc.shape[1]
        nonzero_loc[0] = np.arange(num_nonzero)
        ind1 = nonzero_loc
        ind2 = ind1+np.array([[0],[1],[-1]]) 
        sig = np.repeat(sigmap, num_avail_basis, axis=0)
        temp = sig[ind1[0] , ind1[1], ind1[2]]
        sig[ind1[0] , ind1[1], ind1[2]] = sig[ind2[0] , ind2[1], ind2[2]]
        sig[ind2[0] , ind2[1], ind2[2]] = temp
        sigmas = np.concatenate((sigmas, sig), axis = 0)
        basis_where = np.append(basis_where, basis_where[-1])
        basis_where = np.concatenate((basis_where, np.cumsum(num_avail_basis)+basis_where[-1]), axis = 0) 
    
    if (J3!=0):
    #J3 ↓↓
        mask = (sigmap[:, -2] != sigmap[:, 2:])
        num_avail_basis = np.sum(mask, axis=(1, 2))
        matrixelements = np.concatenate((matrixelements, np.repeat(0.5*J3, num_avail_basis.sum(), axis=0)), axis = 0)
        log_diag_amp_array = np.concatenate((log_diag_amp_array, np.repeat(log_diag_amp, num_avail_basis, axis=0)), axis = 0)
        nonzero_loc = np.array(np.nonzero(mask))
        num_nonzero = nonzero_loc.shape[1]
        nonzero_loc[0] = np.arange(num_nonzero)
        ind1 = nonzero_loc
        ind2 = ind1+np.array([[0],[2],[0]]) 
        sig = np.repeat(sigmap, num_avail_basis, axis=0)
        temp = sig[ind1[0] , ind1[1], ind1[2]]
        sig[ind1[0] , ind1[1], ind1[2]] = sig[ind2[0] , ind2[1], ind2[2]]
        sig[ind2[0] , ind2[1], ind2[2]] = temp
        sigmas = np.concatenate((sigmas, sig), axis = 0)
        basis_where = np.append(basis_where, basis_where[-1])
        basis_where = np.concatenate((basis_where, np.cumsum(num_avail_basis)+basis_where[-1]), axis = 0) 
    #J3 →→
        mask = (sigmap[:, :, :-2] != sigmap[:, :, 2:])
        num_avail_basis = np.sum(mask, axis=(1, 2))
        matrixelements = np.concatenate((matrixelements, np.repeat(0.5*J3, num_avail_basis.sum(), axis=0)), axis = 0)
        log_diag_amp_array = np.concatenate((log_diag_amp_array, np.repeat(log_diag_amp, num_avail_basis, axis=0)), axis = 0)
        nonzero_loc = np.array(np.nonzero(mask))
        num_nonzero = nonzero_loc.shape[1]
        nonzero_loc[0] = np.arange(num_nonzero)
        ind1 = nonzero_loc
        ind2 = ind1+np.array([[0],[0],[2]]) 
        sig = np.repeat(sigmap, num_avail_basis, axis=0)
        temp = sig[ind1[0] , ind1[1], ind1[2]]
        sig[ind1[0] , ind1[1], ind1[2]] = sig[ind2[0] , ind2[1], ind2[2]]
        sig[ind2[0] , ind2[1], ind2[2]] = temp
        sigmas = np.concatenate((sigmas, sig), axis = 0)
        basis_where = np.append(basis_where, basis_where[-1])
        basis_where = np.concatenate((basis_where, np.cumsum(num_avail_basis)+basis_where[-1]), axis = 0) 

    return matrixelements, log_diag_amp_array ,sigmas, basis_where

@partial(jax.jit, static_argnames=['fixed_params','J1','J2','J3','num_samples'])                                     
def Get_Elocs(J1,J2,J3, num_samples,params, sample_amp, fixed_params):
    Ny, Nx, mag_fixed, magnetization, units = fixed_params      
    matrixelements, sigmas, basis_where = J1J2J3_MatrixElements(samples, J1, J2, J3, Nx, Ny, params, sample_amp)
    left_basis = (Nx*(Ny-1)+(Nx-1)*Ny)*numsamples-matrixelements.shape[0]
    if left_basis>0:
        sigmas = jnp.concatenate((sigmas, jnp.zeros((left_basis, Ny, Nx))), axis=0).astype(int)
        matrixelements = jnp.concatenate((matrixelements, jnp.zeros(left_basis)), axis=0)
        
    amp = jnp.exp(log_amp(sigmas, params, fixed_params))
    diag_local_E = matrixelements[:numsamples]*amp[:numsamples]
    matrixelements_off_diag, amp_off_diag = matrixelements[numsamples:-left_basis], amp[numsamples:-left_basis]  
    basis_where = basis_where.reshape(-1, numsamples+1)
    ind1, ind2 = basis_where[:, :-1].astype(jnp.int32), basis_where[:, 1:].astype(jnp.int32)
    block_sum = compute_block_sum(amp_off_diag, matrixelements_off_diag, ind1, ind2)

    local_E = jnp.sum(block_sum, axis=0)+diag_local_E
    #print("local_E", local_E)
    return local_E

@jax.jit
def compute_block_sum(amp_off_diag, matrixelements_off_diag, ind1, ind2):
    # Get the shape parameters
    N, M = ind1.shape

    # Create broadcastable indices for amp_off_diag and matrixelements_off_diag
    broadcast_indices = jnp.arange(amp_off_diag.shape[0]).reshape(1, 1, -1)

    # Create masks based on ind1 and ind2
    mask = (broadcast_indices >= ind1[:,:,None]) & (broadcast_indices < ind2[:,:,None])

    # Use mask to compute the sum
    result = jnp.sum(mask * amp_off_diag[None, None, :] * matrixelements_off_diag[None, None, :], axis=-1)

    return result


