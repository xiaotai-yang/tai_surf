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


def softmax(x):
    return jnp.exp(x) / jnp.sum(jnp.exp(x))

def heavyside(inputs):
    sign = jnp.sign(jnp.sign(inputs) + 0.1)  # tf.sign(0) = 0, this is a way to cure that since I want tf.sign(0) = 1
    return 0.5 * (sign + 1.0)

#Rewrite this part for multiple qubit
def one_hot_encoding(x, num_classes=2):
    """Converts batched integer labels to one-hot encoded arrays."""
    return jnp.eye(num_classes)[x]
def sample_discrete(key, probabilities, size=None):
    """Sample from a discrete distribution defined by probabilities."""
    logits = jnp.log(probabilities)
    return categorical(key, logits, shape=size)
def normalization(probs, num_up, num_generated_spins, magnetization, num_samples, Ny, Nx):
    num_down = num_generated_spins - num_up
    activations_up = heavyside(((Ny * Nx + magnetization) // 2 - 1) - num_up)
    activations_down = heavyside(((Ny * Nx - magnetization) // 2 - 1) - num_down)
    probs_ = probs * jnp.stack([activations_up, activations_down], axis=1)
    probs__ = probs_ / (jnp.expand_dims(jnp.linalg.norm(probs_, axis=1, ord=1), axis=1))  # l1 normalizing

    return probs__

class rnn_function:
    def __init__(self, Lx, Ly, px, py, mag_fixed, magnetization, units, key, rnn, params):
        ### lattice setting
        self.Lx, self.Ly = Lx, Ly
        self.px, self.py = px, py
        self.Nx, self.Ny = Lx//px,  Ly//py
        self.ny_nx_indices = jnp.array([[(i, j) for j in range(self.Nx)] for i in range(self.Ny)])
        ### rnn setting
        self.params = params
        self.units = units
        self.batch_rnn = vmap(rnn, (0, 0, None))
        ### additional property
        self.mag_fixed, self.magnetization = mag_fixed, magnetization
        self.key = key

    def rnn_ini(self, num_samples, samples, params, mode):
        input_init_x, input_init_y = jnp.zeros((num_samples, self.Nx, 2 ** (self.px * self.py))), jnp.zeros((num_samples, self.Nx, 2 ** (self.px * self.py)))
        rnn_states_init_x, rnn_states_init_y = jnp.zeros((num_samples, self.Nx, self.units)), jnp.zeros((num_samples, self.Ny, self.units))
        rnn_init = rnn_states_init_x, rnn_states_init_y, 0, jnp.zeros(num_samples), self.key, input_init_x, input_init_y, self.mag_fixed, self.magnetization, samples, num_samples, self.Ny, self.Nx, mode, params
        return rnn_init

    def sample_prob(self, num_samples, params, mode=0):
        jax.debug.print("num_samples: {}", num_samples)
        rnn_init = self.rnn_ini(num_samples, jnp.zeros((num_samples, self.Ly, self.Lx)).astype(jnp.int32), mode, params)
        __, (samples, probs, phase) = scan(self.scan_fun_2d, rnn_init, self.ny_nx_indices)
        probs, phase, samples = jnp.transpose(probs, (2, 0, 1, 3)), jnp.transpose(phase, (2, 0, 1, 3)), jnp.transpose(samples, (2, 0, 1))
        #print("sample_probs:", probs)
        #print("samples:", samples)
        probs = jnp.take_along_axis(probs, samples[..., jnp.newaxis], axis=-1).squeeze(-1)
        phase = jnp.take_along_axis(phase, samples[..., jnp.newaxis], axis=-1).squeeze(-1)

        # jax.debug.print("scan_rnn_params: {}", scan_rnn_params)
        sample_amp = jnp.sum(jnp.log(probs), axis=(1, 2)) * 0.5 + jnp.sum(phase, axis=(1, 2)) * 1j
        return samples, sample_amp

    def log_amp(self, samples, params, mode = 1):
        num_samples = samples.shape[0]
        rnn_init = self.rnn_ini(num_samples, samples, mode, params)

        __, (samples, probs, phase) = scan(self.scan_fun_2d, rnn_init, self.ny_nx_indices)

        # print("samples_eval1:", samples)
        probs, phase, samples = jnp.transpose(probs, (2, 0, 1, 3)), jnp.transpose(phase, (2, 0, 1, 3)), jnp.transpose(samples, (2, 0, 1))
        # print("probs_original:", probs)
        # print("samples:", samples)#
        probs = jnp.take_along_axis(probs, samples[..., jnp.newaxis], axis=-1).squeeze(-1)
        phase = jnp.take_along_axis(phase, samples[..., jnp.newaxis], axis=-1).squeeze(-1)
        # jax.debug.print("probs_choice: {}", probs)
        log_probs, phase = jnp.sum(jnp.log(probs), axis=(1, 2)), jnp.sum(phase, axis=(1, 2))
        log_amp = log_probs / 2 + phase * 1j

        return log_amp
    def scan_func_1d(self, carry_1d, indices):
        return _scan_fun_1d(carry_1d, indices)

    def scan_fun_2d(self, carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]

        rnn_states_x, rnn_states_y, num_spin, num_up, key, inputs_x, inputs_y, mag_fixed, magnetization, samples, num_samples, Ny, Nx, mode, params_2d = carry_2d

        index = indices[0, 0]
        params_1d = tuple(p[index] for p in params_2d)
        # rnn_states_x and rnn_states_y are of shape [Nx] and [Ny]

        carry_1d = rnn_states_x, rnn_states_y[:, index], num_spin, num_up, key, inputs_x, inputs_y[:,
                                                                                          index], mag_fixed, magnetization, samples, num_samples, Ny, Nx, mode, params_1d
        _, y = scan(self.scan_fun_1d, carry_1d, indices)

        rnn_states_x, dummy_state_y, num_spin, num_up, key, inputs_x, dummy_inputs_y, mag_fixed, magnetization, samples, num_samples, Ny, Nx, mode, params_1d = _
        row_samples, row_prob, row_phase, rnn_states_x = y
        rnn_states_x = jnp.transpose(rnn_states_x, (1, 0, 2))
        rnn_states_x = jnp.flip(rnn_states_x, 1)  # reverse the direction of input of for the next line
        inputs_x = one_hot_encoding(row_samples)
        inputs_x = jnp.transpose(inputs_x, (1, 0, 2))
        inputs_x = jnp.flip(inputs_x, 1)
        row_samples = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_samples)
        row_prob = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_prob)
        row_phase = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_phase)
        return (rnn_states_x, rnn_states_y, num_spin, num_up, key, inputs_x, inputs_y, mag_fixed, magnetization, samples,
            num_samples, Ny, Nx, mode, params_2d), (row_samples, row_prob, row_phase)
def _scan_fun_1d(batch_rnn, carry_1d, indices):
    '''
    rnn_state_x_1d, inputs_x_1d : ↓↓↓...↓
    rnn_state_yi_1d, inputs_yi_1d : → or ←
    mag_fixed : To apply U(1) symmetry
    num_1d : count the indices of rnn_state_yi
    key : for random number generation
    num_samples: number of samples
    params_1d: rnn_parameters on that row
    '''
    ny, nx = indices
    rnn_states_x_1d, rnn_states_yi_1d, num_spin, num_up, key, inputs_x_1d, inputs_yi_1d, mag_fixed, magnetization, samples, num_samples, Ny, Nx, mode, params_1d = carry_1d
    params_point = tuple(p[nx] for p in params_1d)
    rnn_states = jnp.concatenate((rnn_states_yi_1d, rnn_states_x_1d[:, nx]), axis=1)
    rnn_inputs = jnp.concatenate((inputs_yi_1d, inputs_x_1d[:, nx]), axis=1)
    new_state, new_prob, new_phase = batch_rnn(rnn_inputs, rnn_states, params_point)
    # jax.debug.print("new_state: {}", new_state)
    # jax.debug.print("new_prob: {}", new_prob)
    # new_state will be stacked so that it will be the new input of rnn_state_x_1d of the next row
    rnn_states_yi_1d = new_state
    key, subkey = split(key)

    print(samples)
    samples_output = lax.cond(mode == 0,
                              lambda x: categorical(subkey, jnp.log(new_prob)),
                              lambda x: lax.cond(ny%2, lambda y:samples[:, ny, -nx-1],
                              lambda y: samples[:, ny, nx],
                              None),
                              None)  # sampling

    inputs_yi_1d = one_hot_encoding(samples_output)  # one_hot_encoding of the sample
    num_up += 1 - samples_output
    num_spin += 1

    return (rnn_states_x_1d, rnn_states_yi_1d, num_spin, num_up, key, inputs_x_1d, inputs_yi_1d, mag_fixed,
        magnetization, samples, num_samples, Ny, Nx, mode, params_1d), (samples_output, new_prob, new_phase, new_state)

def _scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]

    rnn_states_x, rnn_states_y, num_spin, num_up, key, inputs_x, inputs_y, mag_fixed, magnetization, samples, num_samples, Ny, Nx, mode, params_2d = carry_2d

    index = indices[0, 0]
    params_1d = tuple(p[index] for p in params_2d)
    # rnn_states_x and rnn_states_y are of shape [Nx] and [Ny]

    carry_1d = rnn_states_x, rnn_states_y[:, index], num_spin, num_up, key, inputs_x, inputs_y[:,
                                                                                      index], mag_fixed, magnetization, samples, num_samples, Ny, Nx, mode, params_1d
    _, y = scan(_scan_fun_1d, carry_1d, indices)

    rnn_states_x, dummy_state_y, num_spin, num_up, key, inputs_x, dummy_inputs_y, mag_fixed, magnetization, samples, num_samples, Ny, Nx, mode, params_1d = _
    row_samples, row_prob, row_phase, rnn_states_x = y
    rnn_states_x = jnp.transpose(rnn_states_x, (1, 0, 2))
    rnn_states_x = jnp.flip(rnn_states_x, 1)  # reverse the direction of input of for the next line
    inputs_x = one_hot_encoding(row_samples)
    inputs_x = jnp.transpose(inputs_x, (1, 0, 2))
    inputs_x = jnp.flip(inputs_x, 1)
    row_samples = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_samples)
    row_prob = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_prob)
    row_phase = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_phase)
    return (
    rnn_states_x, rnn_states_y, num_spin, num_up, key, inputs_x, inputs_y, mag_fixed, magnetization, samples,
    num_samples, Ny, Nx, mode, params_2d), (row_samples, row_prob, row_phase)

@partial(jit, static_argnums=(0, 2, 3, 4))
def compute_cost(self, params, samples, Eloc, Temperature):
    samples = jax.lax.stop_gradient(samples)
    Eloc = jax.lax.stop_gradient(Eloc)

    # First term
    log_amps_tensor = self.log_amp(samples, params)
    term1 = 2 * jnp.real(jnp.mean(log_amps_tensor.conjugate() * (Eloc - jnp.mean(Eloc))))
    # Second term
    term2 = 4 * Temperature * (
                jnp.mean(jnp.real(log_amps_tensor) * jax.lax.stop_gradient(jnp.real(log_amps_tensor)))
                - jnp.mean(jnp.real(log_amps_tensor)) * jnp.mean(
            jax.lax.stop_gradient(jnp.real(log_amps_tensor))))

    cost = term1 + term2
    return cost


