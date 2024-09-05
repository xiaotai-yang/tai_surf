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

def softmax (x):
    return jnp.exp(x)/jnp.sum(jnp.exp(x))     
def heavyside(inputs):
    sign = jnp.sign(jnp.sign(inputs) + 0.1 ) #tf.sign(0) = 0, this is a way to cure that since I want tf.sign(0) = 1
    return 0.5*(sign+1.0)

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

def one_hot_encoding(x, num_classes = 2):
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

def random_layer_params(N, m, n, key):
    w_key, b_key = random.split(key)
    #outkey1, outkey2 = random.split(w_key)
    return  (2*random.uniform(w_key, (N, m, n))-1)/jnp.sqrt((m+n)/2),  (2*random.normal(b_key, (N, m))-1)/jnp.sqrt((m+n)/2)

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, N,  key):
    keys = random.split(key, len(sizes))
    outkey = keys[0]
    return outkey, [random_layer_params(N, m, n, k) for m, n, k in zip(sizes[1:], sizes[:-1], keys[1:])]

def init_tensor_gru_params(N, units, input_size, key):
    # input is already concantenated
    key, u_params = init_network_params([(units * input_size), units], N, key)
    key, r_params = init_network_params([ (units * input_size), units], N, key)
    key, s_params = init_network_params([(units * input_size), units], N, key)
    key, amp_params = init_network_params([units, input_size], N, key)
    key, phase_params = init_network_params([units, input_size], N, key)

    Wu, bu, Wr, br, Ws, bs, = u_params[0][0], u_params[0][1], r_params[0][0], r_params[0][1], s_params[0][0], s_params[0][1],
    Wamp, bamp = amp_params[0][0] * 0, amp_params[0][1] * 0

    return (Wu, bu, Wr, br, Ws, bs, Wamp, bamp)

def tensor_gru_rnn_step(local_inputs, local_states, params):  # local_input is already concantenated
    Wu, bu, Wr, br, Ws, bs, Wamp, bamp = params
    rnn_inputs = jnp.outer(local_inputs, local_states).ravel()
    u = nn.sigmoid(jnp.dot(Wu, rnn_inputs) + bu)
    r = nn.tanh(jnp.dot(Wr, rnn_inputs) + br)
    s = jnp.dot(Ws, rnn_inputs) + bs
    new_state = u * r + (1 - u) * s
    prob = nn.softmax(jnp.dot(Wamp, new_state) + bamp)

    return new_state, prob


def sample_prob(params, wemb, fixed_params, key):

    def scan_fun(carry, indices):
        '''
        mag_fixed : To apply U(1) symmetry
        num_1d : count the indices of rnn_state_yi
        key : for random number generation
        num_samples
        params_1d: rnn_parameters on that row
        '''
        n = indices
        state, input, key, params = carry
        params_point = tuple(p[n] for p in params)
        state, new_prob = tensor_gru_rnn_step(input, state, params_point)
        key, subkey = split(key)
        block_sample = categorical(subkey, jnp.log(new_prob)) #sampling
        prob = new_prob[block_sample]
        input = wemb[block_sample]
        return (state, input, key, params), (block_sample, prob)

    N, p, units = fixed_params
    int_to_binary = partial(int_to_binary_array, num_bits=p)
    n_indices = jnp.array([i for i in range(N)])
    input_init, state_init = jnp.zeros(2**p), jnp.zeros((units))
    init = state_init, input_init, key, params

    __, (block_samples, probs) = scan(scan_fun, init, n_indices)
    samples = int_to_binary(block_samples).reshape(N*p)
    log_probs = jnp.sum(jnp.log(probs))
    sample_amp = log_probs / 2
    return samples, sample_amp

def log_amp(sample, params, wemb, fixed_params):
        # samples : (num_samples, Ny, Nx)   
    def scan_fun(carry, indices):
        '''
        mag_fixed : To apply U(1) symmetry
        num_1d : count the indices of rnn_state_yi
        num_samples
        params_1d: rnn_parameters on that row
        '''
        n = indices
        state, input, samples, params = carry
        params_point = tuple(p[n] for p in params)
        state, new_prob = tensor_gru_rnn_step(input, state, params_point)
        samples_output = binary_to_int(sample[n])
        prob = new_prob[samples_output]
        input = wemb[samples_output] # one_hot_encoding of the sample

        return (state, input, samples, params), (prob, )

    #initialization
    N, p, units = fixed_params
    binary_to_int = partial(binary_array_to_int, num_bits=p)
    n_indices = jnp.array([i for i in range(N)])
    state_init, input_init = jnp.zeros((units)), jnp.zeros(2**p)
    init = state_init, input_init, sample, params

    __, (probs, ) = scan(scan_fun, init, n_indices)
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
    sign = lax.cond(jnp.abs(amp)>1e-12, lambda x: amp / jnp.abs(amp), lambda x: 1. , None)
    log_phase = (-sign+1)/2*jnp.pi*1j
    return log_phase
