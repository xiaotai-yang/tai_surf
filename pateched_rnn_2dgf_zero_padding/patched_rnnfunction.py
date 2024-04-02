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

@partial(jax.jit, static_argnames=['fixed_params'])
def sample_prob(params, fixed_params, ny_nx_indices, key):

    Ny, Nx, py, px, mag_fixed, magnetization = fixed_params
    int_to_binary = partial(int_to_binary_array, num_bits=px * py)
    wemb = params[0]
    net_params = params[5:-4]
    out_params = params[-4:]
    def scan_fun_1d(carry_1d, indices):
        ny, nx = indices
        rnn_states_x_1d, rnn_states_yi_1d, inputs_x_1d, inputs_yi_1d, key = carry_1d
        rnn_states = jnp.concatenate((rnn_states_yi_1d, rnn_states_x_1d[nx]), axis=0)
        rnn_inputs = jnp.concatenate((inputs_yi_1d, inputs_x_1d[nx]), axis=0)

        new_state, new_prob, new_phase = tensor_gru_rnn_step(rnn_inputs, rnn_states,  tuple(px[nx] for px in tuple(py[ny] for py in net_params)), tuple(px[nx] for px in tuple(py[ny] for py in out_params)))
        key, subkey = split(key)
        block_sample = categorical(subkey, jnp.log(new_prob))
        probs, phase = new_prob[block_sample], new_phase[block_sample]
        inputs_yi_1d = wemb[block_sample]

        return (rnn_states_x_1d, new_state, inputs_x_1d, inputs_yi_1d, key), (block_sample, probs, phase, new_state)

    def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]
        rnn_states_x, rnn_states_y, inputs_x, inputs_y, key = carry_2d
        index = indices[0, 0]
        carry_1d = rnn_states_x, rnn_states_y[index], inputs_x, inputs_y[index], key
        _, y = scan(scan_fun_1d, carry_1d, indices)
        key = _[-1]
        row_block_sample, row_prob, row_phase, rnn_states_x = y
        rnn_states_x = jnp.flip(rnn_states_x, 0)  # reverse the direction of input of for the next line
        inputs_x = wemb[jnp.flip(row_block_sample)]
        row_block_sample = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_block_sample)
        row_prob = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_prob)
        row_phase = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_phase)

        return (rnn_states_x, rnn_states_y, inputs_x, inputs_y, key), (row_block_sample, row_prob, row_phase)


    # initialization
    init = params[3], params[4], params[1], params[2], key
    __, (samples, probs, phase) = scan(scan_fun_2d, init, ny_nx_indices)
    samples = vmap(int_to_binary, 0)(samples).reshape(Ny*py, Nx*px)
    log_probs, phase = jnp.sum(jnp.log(probs)), jnp.sum(phase)
    samples_log_amp = log_probs / 2 + phase * 1j

    return samples, samples_log_amp
'''
@partial(jax.jit, static_argnames=['fixed_params'])
def sample_prob(params, fixed_params, key, ny_nx_indices):
    def scan_fun_1d(carry_1d, indices):
        '''
'''
        rnn_state_x_1d, inputs_x_1d : ↓↓↓...↓
        rnn_state_yi_1d, inputs_yi_1d : → or ←
        mag_fixed : To apply U(1) symmetry
        num_1d : count the indices of rnn_state_yi
        key : for random number generation
        num_samples
        params_1d: rnn_parameters on that row
        '''
'''
        ny, nx = indices
        rnn_states_x_1d, rnn_states_yi_1d, num_spin, num_up, key, inputs_x_1d, inputs_yi_1d, params_1d = carry_1d
        params_point = tuple(p[nx] for p in params_1d)
        rnn_states = jnp.concatenate((rnn_states_yi_1d, rnn_states_x_1d[:, nx]), axis=1)
        rnn_inputs = jnp.concatenate((inputs_yi_1d, inputs_x_1d[:, nx]), axis=1)
        print("rnn_states:", rnn_states)
        print("rnn_inputs:", rnn_inputs)
        new_state, new_prob, new_phase = batch_rnn(rnn_inputs, rnn_states, params_point)

        #jax.debug.print("new_prob: {}", new_prob)
        #jax.debug.print("new_phase: {}", new_phase)
        # new_state will be stacked so that it will be the new input of rnn_state_x_1d of the next row
        rnn_states_yi_1d = new_state
        key, subkey = split(key)
        samples_output_digit = categorical(subkey, jnp.log(new_prob))  # sampling
        # jax.debug.print("samples_output_digit: {}", samples_output_digit)
        inputs_yi_1d = one_hot(samples_output_digit)  # one_hot_encoding of the sample
        samples_output_binary = int_to_binary(samples_output_digit)
        #jax.debug.print("samples_output: {}", samples_output_digit)
        #jax.debug.print("samples_output_binary: {}", samples_output_binary)
        #jax.debug.print("inputs_yi_1d: {}", inputs_yi_1d)
        num_up += jnp.sum(1 - samples_output_binary, axis = -1)
        num_spin += 4

        return (rnn_states_x_1d, rnn_states_yi_1d, num_spin, num_up, key, inputs_x_1d, inputs_yi_1d, params_1d), (samples_output_binary, new_prob, new_phase, new_state)

    def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]

        rnn_states_x, rnn_states_y, num_spin, num_up, key, inputs_x, inputs_y, params_2d = carry_2d

        index = indices[0, 0]
        params_1d = tuple(p[index] for p in params_2d)
        # rnn_states_x and rnn_states_y are of shape [Nx] and [Ny]

        carry_1d = rnn_states_x, rnn_states_y[:, index], num_spin, num_up, key, inputs_x, inputs_y[:,index], params_1d
        _, y = scan(scan_fun_1d, carry_1d, indices)

        rnn_states_x, dummy_state_y, num_spin, num_up, key, inputs_x, dummy_inputs_y, params_1d = _
        row_samples, row_prob, row_phase, rnn_states_x = y
        rnn_states_x = jnp.transpose(rnn_states_x, (1, 0, 2))
        rnn_states_x = jnp.flip(rnn_states_x, 1)  # reverse the direction of input of for the next line
        inputs_x = one_hot(vmap(binary_to_int, 0)(row_samples))
        inputs_x = jnp.transpose(inputs_x, (1, 0, 2))
        inputs_x = jnp.flip(inputs_x, 1)
        row_samples = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_samples)
        row_prob = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_prob)
        row_phase = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_phase)
        return (rnn_states_x, rnn_states_y, num_spin, num_up, key, inputs_x, inputs_y, params_2d), (row_samples, row_prob, row_phase)

    # initialization
    Ny, Nx, py, px, mag_fixed, magnetization, num_samples, units = fixed_params # N indicates how many patches in a row or a column and p is the qubit length in a patch

    one_hot = partial(one_hot_encoding, num_classes=2 ** (px * py))
    binary_to_int = partial(binary_array_to_int, num_bits=px * py)
    int_to_binary = partial(int_to_binary_array, num_bits=px * py)

    batch_rnn = vmap(tensor_gru_rnn_step, (0, 0, None))
    batch_rnn_states_init_x, batch_rnn_states_init_y = jnp.zeros((num_samples, Nx, units)), jnp.zeros((num_samples, Ny, units))
    batch_inputs_init_x, batch_inputs_init_y = jnp.zeros((num_samples, Nx, 2**(px*py))), jnp.zeros((num_samples, Ny, 2**(px*py)))
    init = batch_rnn_states_init_x, batch_rnn_states_init_y, 0, jnp.zeros(num_samples), key, batch_inputs_init_x, batch_inputs_init_y, params
    __, (samples, probs, phase) = scan(scan_fun_2d, init, ny_nx_indices)

    probs, phase, samples = jnp.transpose(probs, (2, 0, 1, 3)), jnp.transpose(phase, (2, 0, 1, 3)), jnp.transpose(
        samples, (2, 0, 1, 3))
    probs = jnp.take_along_axis(probs, vmap(vmap(binary_to_int,1, 1), 1, 1)(samples)[..., jnp.newaxis], axis=-1).squeeze(-1)
    phase = jnp.take_along_axis(phase, vmap(vmap(binary_to_int,1, 1), 1, 1)(samples)[..., jnp.newaxis], axis=-1).squeeze(-1)

    # jax.debug.print("scan_rnn_params: {}", scan_rnn_params)
    sample_amp = jnp.sum(jnp.log(probs), axis=(1, 2)) * 0.5 + jnp.sum(phase, axis=(1, 2)) * 1j

    return samples, sample_amp
'''


@partial(jax.jit, static_argnames=['fixed_params'])
def log_amp(samples, params, fixed_params, ny_nx_indices):

    Ny, Nx, py, px, mag_fixed, magnetization = fixed_params
    one_hot = partial(one_hot_encoding, num_classes=2 ** (px * py))
    binary_to_int = partial(binary_array_to_int, num_bits=px * py)
    wemb = params[0]
    net_params = params[5:-4]
    out_params = params[-4:]
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
        rnn_states_x_1d, rnn_states_yi_1d, inputs_x_1d, inputs_yi_1d = carry_1d
        rnn_states = jnp.concatenate((rnn_states_yi_1d, rnn_states_x_1d[nx]), axis=0)
        rnn_inputs = jnp.concatenate((inputs_yi_1d, inputs_x_1d[nx]), axis=0)
        new_state, new_prob, new_phase = tensor_gru_rnn_step(rnn_inputs, rnn_states,  tuple(px[nx] for px in tuple(py[ny] for py in net_params)), tuple(px[nx] for px in tuple(py[ny] for py in out_params)))
        block_sample = binary_to_int(lax.cond(ny%2, lambda x: x[ny, -nx-1], lambda x: x[ny, nx], samples).ravel())
        probs, phase = new_prob[block_sample], new_phase[block_sample]
        inputs_yi_1d = wemb[block_sample]
        return (rnn_states_x_1d, new_state, inputs_x_1d, inputs_yi_1d), (probs, phase, new_state)

    def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]
        rnn_states_x, rnn_states_y, inputs_x, inputs_y= carry_2d
        index = indices[0, 0]
        carry_1d = rnn_states_x, rnn_states_y[index], inputs_x, inputs_y[index]
        _, y = scan(scan_fun_1d, carry_1d, indices)
        row_prob, row_phase, rnn_states_x = y
        rnn_states_x = jnp.flip(rnn_states_x, 0)  # reverse the direction of input of for the next line
        row_input = lax.cond(index%2, lambda x: x[index], lambda x: jnp.flip(x[index], 0), samples)
        inputs_x = wemb[binary_to_int(row_input.reshape(Nx, py*px))]

        return (rnn_states_x, rnn_states_y, inputs_x, inputs_y), (row_prob, row_phase)

    # initialization
    init = params[3], params[4], params[1], params[2]
    __, (probs, phase) = scan(scan_fun_2d, init, ny_nx_indices)

    # jax.debug.print("probs_choice: {}", probs)
    log_probs, phase = jnp.sum(jnp.log(probs)), jnp.sum(phase)
    log_amp = log_probs / 2 + phase * 1j

    return log_amp


@partial(jax.jit, static_argnames=['fixed_params'])
def sample_prob_RWKV(all_params, fixed_params, ny_nx_indices, key):

    (wemb, init_emb_x, init_emb_y, wln_in, bln_in, wln_out, bln_out, whead, bhead, wprob, bprob, wphase, bphase, RWKV_cell_params), (t_alpha_init_x, t_alpha_init_y, t_beta_init_x, t_beta_init_y) = all_params
    Ny, Nx, py, px, num_layer = fixed_params
    one_hot = partial(one_hot_encoding, num_classes=2 ** (px * py))
    int_to_binary = partial(int_to_binary_array, num_bits=px * py)
    binary_to_int = partial(binary_array_to_int, num_bits=px * py)
    def scan_fun_1d(carry_1d, indices):

        ny, nx = indices
        t_alpha_x_1d, t_alpha_yi_1d,t_beta_x_1d, t_beta_yi_1d, inputs_x_1d, inputs_yi_1d, key = carry_1d
        alpha_state = jnp.concatenate((t_alpha_yi_1d, t_alpha_x_1d[nx]), axis=1) #state is of shape (num_layer, h_size) so we connect it on the axis 1
        beta_state = jnp.concatenate((t_beta_yi_1d, t_beta_x_1d[nx]), axis=1)
        inputs = jnp.concatenate((inputs_yi_1d, inputs_x_1d[nx]), axis=0)
        last_inputs = jnp
        x = wemb[binary_to_int(inputs)]
        x = layer_norm(x, wln_in[ny, nx], bln_in[ny, nx])
        _ , y = lax.scan(partial(RWKV_cell, params = RWKV_cell_params), (x, (x, alpha_state, beta_state), x), jnp.arange(num_layer))
        x = _[0]
        t_states, c_states = y
        x = whead @ layer_norm(x, w_out[indices], b_out[indices]) + bhead[indices]
        prob = nn.softmax(wprob @ x + bprob)
        phase = 2 * jnp.pi * nn.soft_sign(wphase @ x + bphase)
        key, subkey = split(key)
        block_sample = categorical(subkey, jnp.log(new_prob))
        probs, phase = new_prob[block_sample], new_phase[block_sample]
        inputs_yi_1d = wemb[block_sample]

        return (t_alpha_x_1d, t_alpha_yi_1d, t_beta_x_1d, t_beta_yi_1d, new_state, inputs_x_1d, inputs_yi_1d, key), (block_sample, probs, phase, new_state)

    def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]
        rnn_states_x, rnn_states_y, inputs_x, inputs_y, key = carry_2d
        index = indices[0, 0]
        carry_1d = rnn_states_x, rnn_states_y[index], inputs_x, inputs_y[index], key
        _, y = scan(scan_fun_1d, carry_1d, indices)
        key = _[-1]
        row_block_sample, row_prob, row_phase, rnn_states_x = y
        rnn_states_x = jnp.flip(rnn_states_x, 0)  # reverse the direction of input of for the next line
        inputs_x = one_hot(jnp.flip(row_block_sample))
        row_block_sample = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_block_sample)
        row_prob = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_prob)
        row_phase = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_phase)

        return (rnn_states_x, rnn_states_y, inputs_x, inputs_y, key), (row_block_sample, row_prob, row_phase)

    # initialization
    init = t_alpha_init_x, t_alpha_init_y, t_beta_init_x, t_beta_init_y, init_emb_x, init_emb_y, key
    __, (samples, probs, phase) = scan(scan_fun_2d, init, ny_nx_indices)
    samples = vmap(int_to_binary, 0)(samples).reshape(Ny*py, Nx*px)
    log_probs, phase = jnp.sum(jnp.log(probs)), jnp.sum(phase)
    samples_log_amp = log_probs / 2 + phase * 1j

    return samples, samples_log_amp

@partial(jax.jit, static_argnames=['fixed_params'])
def log_amp_RWKV(samples, params, fixed_params, fixed_init, ny_nx_indices):

    Ny, Nx, py, px, mag_fixed, magnetization = fixed_params
    rnn_states_init_x, rnn_states_init_y, inputs_init_x, inputs_init_y = fixed_init
    one_hot = partial(one_hot_encoding, num_classes=2 ** (px * py))
    binary_to_int = partial(binary_array_to_int, num_bits=px * py)
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
        rnn_states_x_1d, rnn_states_yi_1d, inputs_x_1d, inputs_yi_1d = carry_1d
        rnn_states = jnp.concatenate((rnn_states_yi_1d, rnn_states_x_1d[nx]), axis=0)
        rnn_inputs = jnp.concatenate((inputs_yi_1d, inputs_x_1d[nx]), axis=0)
        new_state, new_prob, new_phase = tensor_gru_rnn_step(rnn_inputs, rnn_states,  tuple(px[nx] for px in tuple(py[ny] for py in params)))
        block_sample = binary_to_int(lax.cond(ny%2, lambda x: x[ny, -nx-1], lambda x: x[ny, nx], samples).ravel())
        probs, phase = new_prob[block_sample], new_phase[block_sample]
        inputs_yi_1d = one_hot(block_sample)
        return (rnn_states_x_1d, new_state, inputs_x_1d, inputs_yi_1d), (probs, phase, new_state)

    def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]
        rnn_states_x, rnn_states_y, inputs_x, inputs_y= carry_2d
        index = indices[0, 0]
        carry_1d = rnn_states_x, rnn_states_y[index], inputs_x, inputs_y[index]
        _, y = scan(scan_fun_1d, carry_1d, indices)
        row_prob, row_phase, rnn_states_x = y
        rnn_states_x = jnp.flip(rnn_states_x, 0)  # reverse the direction of input of for the next line
        row_input = lax.cond(index%2, lambda x: x[index], lambda x: jnp.flip(x[index], 0), samples)
        inputs_x = one_hot(binary_to_int(row_input.reshape(Nx, py*px)))

        return (rnn_states_x, rnn_states_y, inputs_x, inputs_y), (row_prob, row_phase)

    # initialization
    init = rnn_states_init_x, rnn_states_init_y, inputs_init_x, inputs_init_y
    __, (probs, phase) = scan(scan_fun_2d, init, ny_nx_indices)

    # jax.debug.print("probs_choice: {}", probs)
    log_probs, phase = jnp.sum(jnp.log(probs)), jnp.sum(phase)
    log_amp = log_probs / 2 + phase * 1j

    return log_amp