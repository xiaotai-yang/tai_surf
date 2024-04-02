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
from params_initialization import *

def softmax (x):
    return jnp.exp(x)/jnp.sum(jnp.exp(x))     
def heavyside(inputs):
    sign = jnp.sign(jnp.sign(inputs) + 0.1 ) #tf.sign(0) = 0, this is a way to cure that since I want tf.sign(0) = 1
    return 0.5*(sign+1.0)


def one_hot_encoding(x, num_classes):
    """Converts batched integer labels to one-hot encoded arrays."""
    return jnp.eye(num_classes)[x]
def sample_discrete(key, probabilities, size=None):
    """Sample from a discrete distribution defined by probabilities."""
    logits = jnp.log(probabilities)
    return categorical(key, logits, shape=size)

def normalization(probs, num_up, num_generated_spins, magnetization, num_samples, Ny, Nx):
    num_down = num_generated_spins - num_up
    activations_up = heavyside(((Ny*Nx+magnetization)//2-1) - num_up)
    activations_down = heavyside(((Ny*Nx-magnetization)//2-1) - num_down)
    probs_ = probs*jnp.stack([activations_up,activations_down], axis = 1)
    probs__ = probs_/(jnp.expand_dims(jnp.linalg.norm(probs_, axis=1, ord=1), axis = 1)) #l1 normalizing
    
    return probs__

@partial(jax.jit, static_argnames=['num_samples','fixed_params'])    
def sample_prob(num_samples, params, fixed_params, key):

    def scan_fun_1d(carry_1d, indices):
        '''
        rnn_state_x_1d, inputs_x_1d : ↓↓↓...↓
        rnn_state_yi_1d, inputs_yi_1d : → or ←
        mag_fixed : To apply U(1) symmetry
        num_1d : count the indices of rnn_state_yi
        key : for random number generation
        num_samples
        params_1d: rnn_parameters on that row
        '''
        ny, nx = indices
        rnn_states_x_1d, rnn_states_yi_1d,  num_spin, num_up, key, inputs_x_1d, inputs_yi_1d, mag_fixed, magnetization, num_samples,Ny,Nx, params_1d = carry_1d
        params_point = tuple(p[nx] for p in params_1d)
        rnn_states = jnp.concatenate((rnn_states_yi_1d, rnn_states_x_1d[:, nx]), axis = 1)
        rnn_inputs = jnp.concatenate((inputs_yi_1d, inputs_x_1d[:, nx]), axis = 1)
        new_state, new_prob, new_phase = batch_rnn(rnn_inputs, rnn_states, params_point)
        print("rnn_states:", rnn_states)
        print("rnn_inputs:", rnn_inputs)
        #jax.debug.print("new_state: {}", new_state)
        #jax.debug.print("new_prob: {}", new_prob)
        # new_state will be stacked so that it will be the new input of rnn_state_x_1d of the next row 
        rnn_states_yi_1d = new_state
        key, subkey = split(key)
        samples_output =  categorical(subkey, jnp.log(new_prob))#sampling
        inputs_yi_1d = one_hot_encoding(samples_output, num_classes=2) # one_hot_encoding of the sample
        num_up += 1-samples_output 
        num_spin += 1

        return (rnn_states_x_1d, rnn_states_yi_1d, num_spin, num_up, key, inputs_x_1d, inputs_yi_1d, mag_fixed, magnetization, num_samples, Ny,Nx, params_1d), (samples_output, new_prob, new_phase, new_state)

    def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]

        rnn_states_x, rnn_states_y, num_spin, num_up, key, inputs_x, inputs_y, mag_fixed, magnetization, num_samples, Ny,Nx, params_2d = carry_2d

        index = indices[0,0]
        params_1d = tuple(p[index] for p in params_2d)
        #rnn_states_x and rnn_states_y are of shape [Nx] and [Ny] 

        carry_1d = rnn_states_x, rnn_states_y[:,index],  num_spin, num_up, key, inputs_x, inputs_y[:,index], mag_fixed, magnetization, num_samples,Ny,Nx, params_1d
        _, y = scan(scan_fun_1d, carry_1d, indices)
        
        rnn_states_x, dummy_state_y,  num_spin, num_up, key, inputs_x, dummy_inputs_y, mag_fixed, magnetization, num_samples,Ny,Nx, params_1d = _
        row_samples, row_prob, row_phase, rnn_states_x = y
        rnn_states_x = jnp.transpose(rnn_states_x, (1, 0, 2))
        rnn_states_x = jnp.flip(rnn_states_x, 1) # reverse the direction of input of for the next line
        inputs_x = one_hot_encoding(row_samples, 2)
        inputs_x = jnp.transpose(inputs_x, (1, 0, 2))
        inputs_x = jnp.flip(inputs_x, 1)
        row_samples = lax.cond(index%2,  lambda x: jnp.flip(x, 0), lambda x: x, row_samples)
        row_prob = lax.cond(index%2, lambda x: jnp.flip(x, 0) , lambda x: x, row_prob)
        row_phase = lax.cond(index%2, lambda x: jnp.flip(x, 0), lambda x: x, row_phase)
        return (rnn_states_x, rnn_states_y,  num_spin, num_up, key, inputs_x, inputs_y, mag_fixed, magnetization, num_samples,Ny, Nx, params_2d), (row_samples, row_prob, row_phase)
    #initialization
    Ny, Nx, mag_fixed, magnetization,  units= fixed_params
    ny_nx_indices = jnp.array([[(i, j) for j in range(Nx)] for i in range(Ny)])
    batch_rnn = vmap(tensor_gru_rnn_step, (0, 0, None))
    batch_rnn_states_init_x, batch_rnn_states_init_y = jnp.zeros((num_samples, Nx, units)), jnp.zeros((num_samples, Ny, units))
    batch_inputs_init_x, batch_inputs_init_y = jnp.zeros((num_samples, Nx, 2 )), jnp.zeros((num_samples, Nx, 2))

    init = batch_rnn_states_init_x, batch_rnn_states_init_y, 0, jnp.zeros(num_samples), key, batch_inputs_init_x, batch_inputs_init_y, mag_fixed, magnetization, num_samples, Ny, Nx, params

    __, (samples, probs, phase) = scan(scan_fun_2d, init, ny_nx_indices)

    probs, phase, samples = jnp.transpose(probs, (2,0,1,3)),jnp.transpose(phase, (2,0,1,3)), jnp.transpose(samples, (2,0,1))
    probs, phase = jnp.take_along_axis(probs, samples[..., jnp.newaxis], axis=-1).squeeze(-1), jnp.take_along_axis(phase, samples[..., jnp.newaxis], axis=-1).squeeze(-1)
    
    #jax.debug.print("scan_rnn_params: {}", scan_rnn_params)
    sample_amp = jnp.sum(jnp.log(probs), axis=(1,2))*0.5+jnp.sum(phase, axis=(1,2))*1j

    return samples, sample_amp    

@partial(jax.jit, static_argnames=['fixed_params'])
def log_amp(samples, params, fixed_params):
        # samples : (num_samples, Ny, Nx)   
    def scan_fun_1d(carry_1d, indices):
        '''
        rnn_state_x_1d, inputs_x_1d : ↓↓↓...↓
        rnn_state_yi_1d, inputs_yi_1d : → or ←
        mag_fixed : To apply U(1) symmetry
        num_1d : count the indices of rnn_state_yi
        num_samples
        params_1d: rnn_parameters on that row
        '''
        ny, nx = indices
        rnn_states_x_1d, rnn_states_yi_1d,  num_spin, num_up, inputs_x_1d, inputs_yi_1d, mag_fixed, magnetization, samples, num_samples,Ny,Nx,  params_1d = carry_1d 

        params_point = tuple(p[nx] for p in params_1d)
        rnn_states = jnp.concatenate((rnn_states_yi_1d, rnn_states_x_1d[:,nx]), axis=1)
        rnn_inputs = jnp.concatenate((inputs_yi_1d, inputs_x_1d[:,nx]), axis=1)

        new_state, new_prob, new_phase = batch_rnn(rnn_inputs, rnn_states, params_point)
        #jax.debug.print("new_phase: {}", new_phase)
        # new_state will be stacked so that it will be the new input of rnn_state_x_1d of the next row 
        rnn_states_yi_1d = new_state
        #new_prob = normalization(new_prob , num_up, num_spin, magnetization, num_samples, Ny,Nx)*(mag_fixed)+new_prob*(1-mag_fixed)
        #jax.debug.print("new_prob_normalized:{}", new_prob)
        samples_output =  lax.cond(ny%2, lambda x:samples[:, ny, -nx-1], lambda x:samples[:, ny, nx], None)
        inputs_yi_1d = one_hot_encoding(samples_output, num_classes=2) # one_hot_encoding of the sample
        num_up += 1-samples_output 
        num_spin += 1
        
        
        return (rnn_states_x_1d, rnn_states_yi_1d, num_spin, num_up, inputs_x_1d, inputs_yi_1d, mag_fixed, magnetization, samples, num_samples,Ny,Nx, params_1d), (samples_output, new_prob, new_phase, new_state)


    def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]

        rnn_states_x, rnn_states_y, num_spin, num_up, inputs_x, inputs_y, mag_fixed, magnetization, samples, num_samples, Ny,Nx, params = carry_2d
        index = indices[0,0]
        params_1d = tuple(p[index] for p in params)
        carry_1d = rnn_states_x, rnn_states_y[:,index],  num_spin, num_up, inputs_x, inputs_y[:,index], mag_fixed, magnetization, samples, num_samples, Ny, Nx, params_1d

        _, y = scan(scan_fun_1d, carry_1d, indices)
        rnn_states_x, dummy_state_y,  num_spin, num_up, inputs_x, dummy_inputs_y, mag_fixed, magnetization, samples, num_samples, Ny, Nx, params_1d = _
        row_samples, row_prob, row_phase, rnn_states_x = y
        rnn_states_x = jnp.transpose(rnn_states_x, (1,0,2))
        rnn_states_x = jnp.flip(rnn_states_x, 1) # reverse the direction of input of for the next line
        inputs_x = one_hot_encoding(row_samples, num_classes=2)
        inputs_x = jnp.transpose(inputs_x, (1,0,2))
        inputs_x = jnp.flip(inputs_x, 1)
        row_samples = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_samples)
        row_prob = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_prob)
        row_phase = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_phase)
        return (rnn_states_x, rnn_states_y,  num_spin, num_up,  inputs_x, inputs_y, mag_fixed, magnetization, samples, num_samples,Ny,Nx, params), (row_samples, row_prob, row_phase)

    #initialization

    Ny, Nx, mag_fixed, magnetization, units= fixed_params
    ny_nx_indices = jnp.array([[(i, j) for j in range(Nx)] for i in range(Ny)])
    batch_rnn = vmap(tensor_gru_rnn_step, (0, 0, None))
    num_samples = samples.shape[0]
    batch_rnn_states_init_x, batch_rnn_states_init_y = jnp.zeros((num_samples, Nx, units)), jnp.zeros((num_samples, Ny, units))
    batch_inputs_init_x, batch_inputs_init_y = jnp.zeros((num_samples, Nx, 2)), jnp.zeros((num_samples, Nx, 2))
    init = batch_rnn_states_init_x, batch_rnn_states_init_y, 0, jnp.zeros(num_samples), batch_inputs_init_x, batch_inputs_init_y, mag_fixed, magnetization, samples, num_samples, Ny, Nx, params

     
    #print("samples_eval0:", samples)
    
    __, (samples, probs, phase) = scan(scan_fun_2d, init, ny_nx_indices)

    #print("samples_eval1:", samples)
    probs, phase, samples = jnp.transpose(probs, (2,0,1,3)),jnp.transpose(phase, (2,0,1,3)), jnp.transpose(samples, (2,0,1))
    #jax.debug.print("phase:{}", phase)
    #print("probs_original:", probs)
    #print("samples:", samples)#
    probs, phase = jnp.take_along_axis(probs, samples[..., jnp.newaxis], axis=-1).squeeze(-1), jnp.take_along_axis(phase, samples[..., jnp.newaxis], axis=-1).squeeze(-1)
    #jax.debug.print("probs_choice: {}", probs)
    log_probs, phase = jnp.sum(jnp.log(probs), axis=(1,2)), jnp.sum(phase, axis=(1,2))  
    log_amp = log_probs/2 + phase*1j
   
    return log_amp


