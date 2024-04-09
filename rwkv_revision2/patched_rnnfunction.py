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
from line_profiler import LineProfiler
from params_initialization import *


def softmax(x):
    return jnp.exp(x) / jnp.sum(jnp.exp(x))


def heavyside(inputs):
    sign = jnp.sign(jnp.sign(inputs) + 0.1)  # tf.sign(0) = 0, this is a way to cure that since I want tf.sign(0) = 1
    return 0.5 * (sign + 1.0)


def one_hot_encoding(x, num_classes):
    """Converts batched integer labels to one-hot encoded arrays."""
    return jnp.eye(num_classes)[x]


def sample_discrete(key, probabilities, size=None):
    """Sample from a discrete distribution defined by probabilities."""
    logits = jnp.log(probabilities)
    return categorical(key, logits, shape=size)


def int_to_binary_array(x, num_bits):
    """
    Converts an array of integers to their binary representation arrays with a fixed number of bits.
    This function is designed to be compatible with Jax's vmap for vectorization over an array of integers.

    Parameters:
    - x: An array of integers, the numbers to convert.
    - num_bits: Integer, the fixed number of bits for the binary representation.

    Returns:
    - A 2D Jax array where each row is the binary representation of an integer in 'x'.
    """
    # Create an array of bit positions: [2^(num_bits-1), 2^(num_bits-2), ..., 1]
    powers_of_two = 2 ** jnp.arange(num_bits - 1, -1, -1)

    # Expand dims of x and powers_of_two for broadcasting
    x_expanded = x[:, None]
    powers_of_two_expanded = powers_of_two[None, :]

    # Perform bitwise AND between each number and each power of two, then right shift to get the bit value
    binary_matrix = (x_expanded & powers_of_two_expanded) >> jnp.arange(num_bits - 1, -1, -1)

    return binary_matrix.astype(jnp.int32)  # Ensure the result is integer


def binary_array_to_int(binary_array, num_bits):
    """
    Converts a 2D array of binary representations to their decimal equivalents.

    Parameters:
    - binary_array: A 2D Jax array where each row represents a binary number.

    Returns:
    - A 1D Jax array of integers, the decimal equivalents of the binary representations.
    """
    powers_of_two = 2 ** jnp.arange(num_bits - 1, -1, -1)
    # Multiply each bit by its corresponding power of two and sum the results
    decimals = jnp.dot(binary_array, powers_of_two)
    return decimals

def normalization(probs, num_up, num_generated_spins, magnetization, num_samples, Ny, Nx):
    num_down = num_generated_spins - num_up
    activations_up = heavyside(((Ny * Nx + magnetization) // 2 - 1) - num_up)
    activations_down = heavyside(((Ny * Nx - magnetization) // 2 - 1) - num_down)
    probs_ = probs * jnp.stack([activations_up, activations_down], axis=1)
    probs__ = probs_ / (jnp.expand_dims(jnp.linalg.norm(probs_, axis=1, ord=1), axis=1))  # l1 normalizing

    return probs__


def sample_prob_RWKV(params, wemb, fixed_params, ny_nx_indices, key):
    Ny, Nx, py, px, h_size, num_layer = fixed_params
    int_to_binary = partial(int_to_binary_array, num_bits=px * py)

    def scan_fun_1d(carry_1d, indices):
        ny, nx = indices

        input_x, input_yi, t_last_x1, t_last_x1s, t_last_x1e, t_last_x2, t_last_y1i, t_last_y2i, t_alpha_state_x1, t_beta_state_x1, t_alpha_state_yi, t_beta_state_yi, c_last_x1, c_last_x1s, c_last_x1e, c_last_x2, c_last_y1i, c_last_y2i, key = carry_1d

        rnn_input = jnp.concatenate((input_yi, input_x[nx]), axis=0)
        last_t_state = jnp.concatenate((t_last_y2i, t_last_x1s[nx], t_last_x2[nx], t_last_x1e[nx]), axis=1)
        last_c_state = jnp.concatenate((c_last_y2i, c_last_x1s[nx], c_last_x2[nx], c_last_x1e[nx]), axis=1)
        t_alpha_state = jnp.concatenate((t_alpha_state_yi, t_alpha_state_x1[nx]), axis=1)
        t_beta_state = jnp.concatenate((t_beta_state_yi, t_beta_state_x1[nx]), axis=1)
        x, t_states, c_states, prob, phase = RWKV_step(rnn_input, (last_t_state, t_alpha_state, t_beta_state),
        (last_c_state), num_layer, RWKV_net_params, jnp.array([ny, nx]))

        key, subkey = split(key)
        block_sample = categorical(subkey, jnp.log(prob))
        probs, phase = prob[block_sample], phase[block_sample]
        t_last_y2i = t_last_y1i
        c_last_y2i = c_last_y1i
        t_last_y1i, t_alpha_state_yi, t_beta_state_yi = t_states
        c_last_y1i = c_states
        input_yi = wemb[block_sample]

        return (input_x, input_yi, t_last_x1, t_last_x1s, t_last_x1e, t_last_x2, t_last_y1i, t_last_y2i, t_alpha_state_x1, t_beta_state_x1,
        t_alpha_state_yi, t_beta_state_yi, c_last_x1, c_last_x1s, c_last_x1e, c_last_x2, c_last_y1i, c_last_y2i, key), (
        t_last_y1i, t_alpha_state_yi, t_beta_state_yi, c_last_y1i, block_sample, probs, phase)

    def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]

        (input_x, input_y, t_last_x1, t_last_x2, t_last_y1s, t_last_y1e, t_last_y2, t_alpha_state_x, t_beta_state_x,
         t_alpha_state_y, t_beta_state_y, c_last_x1, c_last_x2, c_last_y1s, c_last_y1e, c_last_y2, key) = carry_2d
        ## The shape of last_y1s, last_y1e are (Ny+1)
        index = indices[0, 0]
        t_last_x1s = jnp.concatenate((t_last_y1s[index][None, ...], t_last_x1[:-1]), axis=0)
        c_last_x1s = jnp.concatenate((c_last_y1s[index][None, ...], c_last_x1[:-1]), axis=0)
        t_last_x1e = jnp.concatenate((t_last_x1[1:], t_last_y1e[index][None, ...]), axis=0)
        c_last_x1e = jnp.concatenate((c_last_x1[1:], c_last_y1e[index][None, ...]), axis=0)
        carry_1d = (input_x, input_y[index], t_last_x1, t_last_x1s, t_last_x1e, t_last_x2, t_last_y1e[index + 1], t_last_y2[index],
        t_alpha_state_x, t_beta_state_x, t_alpha_state_y[index], t_beta_state_y[index], c_last_x1, c_last_x1s, c_last_x1e, c_last_x2, c_last_y1e[index + 1],
        c_last_y2[index], key)
        _, y = scan(scan_fun_1d, carry_1d, indices)
        '''
        The stacked y1i becomes the x1 in the next row
        The stacked y2i becomes the x2e in the next row
        '''
        t_last_x2 = t_last_x1  # x2 for the next row
        c_last_x2 = c_last_x1  # x2 for the next row
        t_last_x1, t_alpha_state_x1, t_beta_state_x1, c_last_x1, row_block_sample, row_prob, row_phase = y
        key = _[-1]
        t_last_x2 = jnp.flip(t_last_x2, 0)
        t_last_x1 = jnp.flip(t_last_x1, 0)
        c_last_x2 = jnp.flip(c_last_x2, 0)
        c_last_x1 = jnp.flip(c_last_x1, 0)
        t_alpha_state_x1 = jnp.flip(t_alpha_state_x1, 0)
        t_beta_state_x1 = jnp.flip(t_beta_state_x1, 0)
        input_x = wemb[jnp.flip(row_block_sample)]
        row_block_sample = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_block_sample)
        row_prob = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_prob)
        row_phase = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_phase)

        return (input_x, input_y, t_last_x1, t_last_x2, t_last_y1s, t_last_y1e, t_last_y2, t_alpha_state_x1, t_beta_state_x1,
                t_alpha_state_y, t_beta_state_y, c_last_x1, c_last_x2, c_last_y1s, c_last_y1e, c_last_y2, key), (row_block_sample, row_prob, row_phase)

    # initialization

    init = (*params[:7], jnp.zeros((Nx, num_layer, h_size)), jnp.zeros((Ny, num_layer, h_size)),
            jnp.zeros((Nx, num_layer, h_size)), jnp.zeros((Ny, num_layer, h_size)) ,*params[7:12], key)
    RWKV_net_params = params[12:]
    __, (samples, probs, phase) = scan(scan_fun_2d, init, ny_nx_indices)

    samples = vmap(int_to_binary, 0)(samples).reshape(Ny * py, Nx * px)
    log_probs, phase = jnp.sum(jnp.log(probs)), jnp.sum(phase)
    samples_log_amp = log_probs / 2 + phase * 1j

    return samples, samples_log_amp



def log_amp_RWKV(samples, params, wemb, fixed_params, ny_nx_indices):
    Ny, Nx, py, px, h_size, num_layer = fixed_params
    binary_to_int = partial(binary_array_to_int, num_bits=px * py)
    def scan_fun_1d(carry_1d, indices):
        ny, nx = indices

        (input_x, input_yi, t_last_x1, t_last_x1s, t_last_x1e, t_last_x2, t_last_y1i, t_last_y2i, t_alpha_state_x1, t_beta_state_x1,
         t_alpha_state_yi, t_beta_state_yi, c_last_x1, c_last_x1s, c_last_x1e, c_last_x2, c_last_y1i, c_last_y2i) = carry_1d

        rnn_input = jnp.concatenate((input_yi, input_x[nx]), axis=0)
        last_t_state = jnp.concatenate((t_last_y2i, t_last_x1s[nx], t_last_x2[nx], t_last_x1e[nx]), axis=1)
        last_c_state = jnp.concatenate((c_last_y2i, c_last_x1s[nx], c_last_x2[nx], c_last_x1e[nx]), axis=1)
        t_alpha_state = jnp.concatenate((t_alpha_state_yi, t_alpha_state_x1[nx]), axis=1)
        t_beta_state = jnp.concatenate((t_beta_state_yi, t_beta_state_x1[nx]), axis=1)
        x, t_states, c_states, prob, phase = RWKV_step(rnn_input, (last_t_state, t_alpha_state, t_beta_state),
                                                       (last_c_state), num_layer, RWKV_net_params, jnp.array([ny, nx]))
        block_sample = binary_to_int(lax.cond(ny%2, lambda x: x[ny, -nx-1], lambda x: x[ny, nx], samples).ravel())
        probs, phase = prob[block_sample], phase[block_sample]

        t_last_y2i = t_last_y1i
        c_last_y2i = c_last_y1i
        t_last_y1i, t_alpha_state_yi, t_beta_state_yi = t_states
        c_last_y1i = c_states
        input_yi = wemb[block_sample]

        return ((input_x, input_yi, t_last_x1, t_last_x1s, t_last_x1e, t_last_x2, t_last_y1i, t_last_y2i, t_alpha_state_x1,
            t_beta_state_x1, t_alpha_state_yi, t_beta_state_yi, c_last_x1, c_last_x1s, c_last_x1e, c_last_x2, c_last_y1i, c_last_y2i),
            (t_last_y1i, t_alpha_state_yi, t_beta_state_yi, c_last_y1i, block_sample, probs, phase))

    def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]

        (input_x, input_y, t_last_x1, t_last_x2, t_last_y1s, t_last_y1e, t_last_y2, t_alpha_state_x, t_beta_state_x, t_alpha_state_y,
         t_beta_state_y, c_last_x1, c_last_x2, c_last_y1s, c_last_y1e, c_last_y2) = carry_2d
        index = indices[0, 0]

        t_last_x1s = jnp.concatenate((t_last_y1s[index][None, ...], t_last_x1[:-1]), axis=0)
        c_last_x1s = jnp.concatenate((c_last_y1s[index][None, ...], c_last_x1[:-1]), axis=0)
        t_last_x1e = jnp.concatenate((t_last_x1[1:], t_last_y1e[index][None, ...]), axis=0)
        c_last_x1e = jnp.concatenate((c_last_x1[1:], c_last_y1e[index][None, ...]), axis=0)

        carry_1d = (input_x, input_y[index], t_last_x1, t_last_x1s, t_last_x1e, t_last_x2, t_last_y1e[index + 1], \
        t_last_y2[index], t_alpha_state_x, t_beta_state_x, t_alpha_state_y[index], t_beta_state_y[index], c_last_x1,
        c_last_x1s, c_last_x1e, c_last_x2, c_last_y1e[index + 1], c_last_y2[index])
        _, y = scan(scan_fun_1d, carry_1d, indices)

        '''
        The stacked y1i becomes the x1 in the next row
        The stacked y2i becomes the x2e in the next row
        '''

        t_last_x2 = t_last_x1  # x2 for the next row
        c_last_x2 = c_last_x1  # x2 for the next row
        t_last_x1, t_alpha_state_x1, t_beta_state_x1, c_last_x1, row_block_sample, row_prob, row_phase = y
        t_last_x2 = jnp.flip(t_last_x2, 0)
        t_last_x1 = jnp.flip(t_last_x1, 0)
        c_last_x2 = jnp.flip(c_last_x2, 0)
        c_last_x1 = jnp.flip(c_last_x1, 0)

        t_alpha_state_x1 = jnp.flip(t_alpha_state_x1, 0)
        t_beta_state_x1 = jnp.flip(t_beta_state_x1, 0)
        input_x = wemb[jnp.flip(row_block_sample)]
        row_prob = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_prob)
        row_phase = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_phase)


        return (input_x, input_y, t_last_x1, t_last_x2, t_last_y1s, t_last_y1e, t_last_y2, t_alpha_state_x1, t_beta_state_x1,
                t_alpha_state_y, t_beta_state_y, c_last_x1, c_last_x2, c_last_y1s, c_last_y1e, c_last_y2), (row_prob, row_phase)

    # initialization

    init = (*params[:7], jnp.zeros((Nx, num_layer, h_size)), jnp.zeros((Ny, num_layer, h_size)),
    jnp.zeros((Nx, num_layer, h_size)), jnp.zeros((Ny, num_layer, h_size)), *params[7:12])
    RWKV_net_params = params[12:]
    __, (probs, phase) = scan(scan_fun_2d, init, ny_nx_indices)
    log_probs, phase = jnp.sum(jnp.log(probs)), jnp.sum(phase)
    log_amp = log_probs / 2 + phase * 1j

    return log_amp
