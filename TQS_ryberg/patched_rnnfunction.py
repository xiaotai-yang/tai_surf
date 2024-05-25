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

def pos(loc, units):
    odd_f = jnp.repeat(jnp.array([1, 0]), units // 2)
    even_f = jnp.repeat(jnp.array([0, 1]), units // 2)
    p = jnp.arange(units)/units
    return jnp.sin(loc/10000**(p))*odd_f+ jnp.cos(loc/10000**(p))*even_f

def pos_2d(Ny, Nx, units):
    x_odd_f = jnp.repeat(jnp.array([1, 0, 0, 0]), units // 4)
    x_even_f = jnp.repeat(jnp.array([0, 1, 0, 0]), units // 4)
    y_odd_f = jnp.repeat(jnp.array([0, 0, 1, 0]), units // 4)
    y_even_f = jnp.repeat(jnp.array([0, 0, 0, 1]), units // 4)
    p = jnp.arange(units)/units
    x = jnp.arange(Ny*Nx+1) %  Nx
    y = jnp.arange(Ny*Nx+1) // Nx
    return jnp.sin(jnp.outer(x, 1/10000**(p)))*x_odd_f + jnp.cos(jnp.outer(x, 1/10000**(p)))*x_even_f + jnp.sin(jnp.outer(y, 1/10000**(p)))*y_odd_f + jnp.cos(jnp.outer(y, 1/10000**(p)))*y_even_f
@partial(jax.jit, static_argnames=['fixed_params'])
def sample_prob(params, fixed_params, ny_nx_indices, key):

    Ny, Nx, py, px, num_layer = fixed_params
    int_to_binary = partial(int_to_binary_array, num_bits=px * py)
    wemb, Wi, bi = params[0], params[1], params[2]
    wemb = jnp.concatenate((wemb, jnp.zeros((1, wemb.shape[1]))), axis=0)
    units = wemb.shape[1]
    def scan_fun(carry_1d, loc):
        input_, x, key = carry_1d
        x = x.at[0].set(nn.tanh(vmap(linear, (0, None, None))(wemb[input_] + pos_2d(Ny, Nx, units), Wi, bi)))
        x, new_prob = TF_step(x, loc, num_layer, params)
        key, subkey = split(key)
        block_sample = categorical(subkey, jnp.log(new_prob))
        probs = new_prob[block_sample]
        output = input_.at[loc+1].set(block_sample)

        return (output, x, key), (block_sample, probs)

    # initialization
    init = -jnp.ones(Ny*Nx+1, dtype=int), jnp.zeros((num_layer+1, Ny*Nx+1, units)), key
    __, (samples, probs) = scan(scan_fun, init, ny_nx_indices)
    print(samples.shape)
    samples = int_to_binary(samples)
    log_probs = jnp.sum(jnp.log(probs))
    samples_log_amp = log_probs / 2 

    return samples, samples_log_amp

@partial(jax.jit, static_argnames=['fixed_params'])
def log_amp(samples, params, fixed_params, ny_nx_indices):

    Ny, Nx, py, px, num_layer = fixed_params
    binary_to_int = partial(binary_array_to_int, num_bits=px * py)
    wemb, Wi, bi = params[0], params[1], params[2]
    wemb = jnp.concatenate((wemb, jnp.zeros((1, wemb.shape[1]))), axis=0)
    units = wemb.shape[1]
    def scan_fun(carry_1d, loc):
        '''
        rnn_state_x_1d, inputs_x_1d : ↓↓↓...↓
        rnn_state_yi_1d, inputs_yi_1d : → or ←
        mag_fixed : To apply U(1) symmetry
        num_1d : count the indices of rnn_state_yi
        num_samples
        params_1d: rnn_parameters on that row
        '''
        input_ , x  = carry_1d
        x = x.at[0].set(nn.tanh(vmap(linear, (0, None, None))(wemb[input_] + pos_2d(Ny, Nx, units), Wi, bi)))
        x, new_prob = TF_step(x, loc, num_layer, params)
        block_sample = binary_to_int(samples[loc])
        probs = new_prob[block_sample]
        output = input_.at[loc+1].set(block_sample)

        return (output, x), (probs,)

    # initialization
    init = -jnp.ones(Ny*Nx+1, dtype=int), jnp.zeros((num_layer+1, Ny*Nx+1, units))
    __, (probs,) = scan(scan_fun, init, ny_nx_indices)

    # jax.debug.print("probs_choice: {}", probs)
    log_probs = jnp.sum(jnp.log(probs))
    log_amp = log_probs / 2 

    return log_amp
