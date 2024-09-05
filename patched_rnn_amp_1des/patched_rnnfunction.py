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


def sample_prob_RWKV(params, wemb, fixed_params, n_indices, key):
    N, p, h_size, num_layer = fixed_params
    int_to_binary = partial(int_to_binary_array, num_bits=p)

    def scan_fun(carry, n):
        input, last_t, t_alpha, t_beta, last_c, key = carry
        x, t_states, c_states, out_prob = RWKV_step(input, (last_t, t_alpha, t_beta),
        last_c, num_layer, RWKV_net_params, n)

        key, subkey = split(key)
        block_sample = categorical(subkey, jnp.log(out_prob))
        prob = out_prob[block_sample]
        last_t, t_alpha, t_beta = t_states
        last_c= c_states
        input = wemb[block_sample]

        return (input, last_t, t_alpha, t_beta, last_c, key), (block_sample, prob)

    init = (params[0], jnp.zeros((num_layer, h_size)), jnp.zeros((num_layer, h_size)), jnp.zeros((num_layer, h_size)), params[2], key)
    RWKV_net_params = params[3:]
    __, (samples, probs) = scan(scan_fun, init, n_indices)
    samples = int_to_binary(samples).reshape(N*p)
    log_probs = jnp.sum(jnp.log(probs))
    samples_log_amp = log_probs / 2

    return samples, samples_log_amp



def log_amp_RWKV(samples, params, wemb, fixed_params, ny_nx_indices):
    N, p, h_size, num_layer = fixed_params
    binary_to_int = partial(binary_array_to_int, num_bits=p)
    def scan_fun(carry, n):
        input, last_t, t_alpha, t_beta, last_c = carry
        x, t_states, c_states, out_prob = RWKV_step(input, (last_t, t_alpha, t_beta),
last_c, num_layer, RWKV_net_params, n)
        block_sample = binary_to_int(samples[n])
        prob= out_prob[block_sample]
        last_t, t_alpha, t_beta = t_states
        last_c = c_states
        input = wemb[block_sample]

        return (input, last_t, t_alpha, t_beta, last_c), (block_sample, prob)

    init = (params[0], params[1], jnp.zeros((num_layer, h_size)), jnp.zeros((num_layer, h_size)), params[2])
    RWKV_net_params = params[3:]
    __, (samples, probs) = scan(scan_fun, init, ny_nx_indices)

    log_probs = jnp.sum(jnp.log(probs))
    log_amp = log_probs / 2
    return log_amp

def log_phase_dmrg(samples, M0, M, Mlast):

    def scan_fun(vec, indices):
        n = indices
        vec = M[samples[n+1],:,:,n] @ vec
        return vec, None

    vec_init = M0[samples[0]]
    vec_last = Mlast[samples[-1]]
    N = samples.shape[0]
    n_indices = jnp.arange(N-2)
    amp_last, _ = scan(scan_fun, vec_init, n_indices)
    amp = jnp.dot(amp_last, vec_last)
    sign = amp / jnp.abs(amp)
    log_phase = lax.cond(jnp.abs(amp)>1e-12, lambda x:(-sign+1)/2*jnp.pi*1j, lambda x: 0.+0.*1j, None)
    return log_phase