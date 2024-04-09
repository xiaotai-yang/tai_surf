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

def one_hot_encoding(x, num_classes = 2):
    """Converts batched integer labels to one-hot encoded arrays."""
    return jnp.eye(num_classes)[x]
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

def unitary(key, n):
    a, b = random.normal(key, (2, n, n))
    z = a + b * 1j
    q, r = jnp.linalg.qr(z)
    d = jnp.diag(r)
    return q * d / abs(d)
def init_mps_rnn_params(N, bond_dim, input_size, key):
    key, subkey = split(key, 2)
    # M = (vmap(unitary, (0, None), 0)(random.split(key, input_size*(N)), bond_dim)/jnp.sqrt(input_size)).reshape(N, input_size, bond_dim, bond_dim)
    M = random.uniform(key, shape = (N, input_size, bond_dim, bond_dim)) * 0.001
    v = jnp.zeros((N, input_size, bond_dim))
    #M = jax.random.orthogonal(key, bond_dim, shape = (N, input_size))/jnp.sqrt(input_size)
    eta = jnp.zeros((N, input_size, bond_dim))
    wphase ,bphase = jnp.ones((N, input_size, bond_dim)), jnp.zeros((N, input_size))
    return M, v, eta, wphase, bphase

def mps_rnn_step(local_states, params):
    M, v, eta = params
    local_states = jnp.matmul(M, local_states) + v
    output_states = local_states/jnp.linalg.norm(local_states)
    prob = jnp.sum(jnp.exp(eta)*jnp.abs(output_states)**2, axis = 1)
    prob = prob/jnp.sum(prob)
    return output_states, prob


def sample_prob(params, fixed_params, key):
    def scan_fun(carry, indices):
        n = indices
        rnn_states, key, params = carry
        M, v, eta, wphase, bphase = tuple(p[n] for p in params)
        out_states, prob = mps_rnn_step(rnn_states, (M, v, eta))
        sample = categorical(key, jnp.log(prob))
        out_prob = prob[sample]
        rnn_states = out_states[sample]
        out_phase = jnp.angle(jnp.dot(wphase[sample].T, rnn_states) + bphase[sample])
        key, subkey = split(key)
        return (rnn_states, key, params), (sample, out_prob, out_phase)

    N, patch, bond_dim = fixed_params
    n_indices = jnp.array([i for i in range(N)])
    int_to_binary = partial(int_to_binary_array, num_bits=patch)
    h_init = jnp.ones((bond_dim))
    init = h_init, key, params
    __, (block_samples, probs, phase) = scan(scan_fun, init, n_indices)
    samples = int_to_binary(block_samples).reshape(N * patch)
    sample_log_amp = jnp.sum(jnp.log(probs))*0.5 + jnp.sum(phase)*1j
    return samples, sample_log_amp

def log_amp(samples, params, fixed_params):
        # samples : (num_samples, Ny, Nx)
    def scan_fun(carry, indices):
        n = indices
        rnn_states, params = carry
        M, v, eta, wphase, bphase  = tuple(p[n] for p in params)
        out_states, prob = mps_rnn_step(rnn_states, (M, v, eta))
        block_sample = samples[n]
        sample = binary_to_int(block_sample)
        out_prob = prob[sample]
        rnn_states = out_states[sample]
        out_phase = jnp.angle(jnp.dot(wphase[sample].T, rnn_states) + bphase[sample])

        return (rnn_states, params), (out_prob, out_phase)

    #initialization
    N, patch, bond_dim = fixed_params
    binary_to_int = partial(binary_array_to_int, num_bits=patch)
    n_indices = jnp.array([i for i in range(N)])
    h_init = jnp.ones((bond_dim), dtype=complex)
    init = h_init, params
    __, (probs, phase) = scan(scan_fun, init, n_indices)

    log_amp = jnp.sum(jnp.log(probs)) * 0.5 + jnp.sum(phase)*1j

    return log_amp

