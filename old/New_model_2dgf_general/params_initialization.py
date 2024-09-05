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
from jax.nn.initializers import he_normal

def random_layer_params(ny, nx, m, n, key):
    w_key, b_key = random.split(key)
    # outkey1, outkey2 = random.split(w_key)
    return (2 * random.uniform(w_key, (ny, nx, m, n)) - 1) / jnp.sqrt(3 * (m + n) / 2), (
                2 * random.normal(b_key, (ny, nx, m)) - 1) / jnp.sqrt(3 * (m + n) / 2)

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, ny, nx, key):
    keys = random.split(key, len(sizes))
    outkey = keys[0]
    return outkey, [random_layer_params(ny, nx, m, n, k) for m, n, k in zip(sizes[1:], sizes[:-1], keys[1:])]

def init_vanilla_params(Nx, Ny, units, input_size, key):
    key, rnn_params = init_network_params([units + input_size, units], Ny, Nx, hidden_size, key)
    key, amp_params = init_network_params([units + input_size, input_size], Ny, Nx, hidden_size, key)
    key, phase_params = init_network_params([units + input_size, input_size], Ny, Nx, hidden_size, key)
    Wrnn, brnn = rnn_params[0][0], rnn_params[0][1]
    Wamp, bamp = amp_params[0][0], amp_params[0][1]
    Wphase, bphase = phase_params[0][0], phase_params[0][1]  # 2*units → output(amplitude)
    rnn_states_init_x = random.normal(random.split(key)[0], (Nx, 2 * units))  # states at vertical direction
    rnn_states_init_y = random.normal(random.split(key)[1], (Ny, 2 * units))

    return rnn_states_init_x, rnn_states_init_y, (
    Winput, binput, Wrnn1, brnn1, Wrnn2, brnn2, Wrnn3, brnn3, Wamp, bamp, Wphase, bphase)

def vanilla_rnn_step(local_inputs, local_states, params):  # local_input is already concantenated
    Wrnn, brnn, Wamp, bamp, Wphase, bphase = params

    new_state = jnp.arcsinh(jnp.dot(Wrnn, jnp.concatenate((local_states, local_inputs), axis=0)) + brnn)
    prob = nn.softmax(jnp.dot(Wamp, new_state) + bamp)
    phase = jnp.pi * nn.soft_sign(jnp.dot(Wphase, new_state) + bphase)

    return new_state, prob, phase

def init_multilayer_vanilla_params(Nx, Ny, units, input_size, key):
    hidden_size = 5 * units
    key, Winput_params = init_network_params([2 * input_size, units], Ny, Nx, hidden_size,
                                             key)  # augment input dimension
    key, rnn_params = init_network_params([5 * units, hidden_size, hidden_size, 2 * units], Ny, Nx, hidden_size, key)
    # 2*units+augmen_input → hidden_layer → hidden layer → 2*rnn_state
    key, amp_params = init_network_params([2 * units, input_size], Ny, Nx, hidden_size, key)
    key, phase_params = init_network_params([2 * units, input_size], Ny, Nx, hidden_size, key)

    Winput, binput = Winput_params[0][0], Winput_params[0][1]
    Wrnn1, Wrnn2, Wrnn3, brnn1, brnn2, brnn3 = rnn_params[0][0], rnn_params[1][0], rnn_params[2][0], rnn_params[0][1], \
    rnn_params[1][1], rnn_params[2][1]
    Wamp, bamp = amp_params[0][0], amp_params[0][1]
    Wphase, bphase = phase_params[0][0], phase_params[0][1]  # 2*units → output(amplitude)
    rnn_states_init_x = random.normal(random.split(key)[0], (Nx, 2 * units))  # states at vertical direction
    rnn_states_init_y = random.normal(random.split(key)[1], (Ny, 2 * units))

    return rnn_states_init_x, rnn_states_init_y, (
    Winput, binput, Wrnn1, brnn1, Wrnn2, brnn2, Wrnn3, brnn3, Wamp, bamp, Wphase, bphase)

def multilayer_vanilla_rnn_step(local_inputs, local_states, params):  # local_input is already concantenated
    Winput, binput, Wrnn1, brnn1, Wrnn2, brnn2, Wrnn3, brnn3, Wamp, bamp, Wphase, bphase = params
    #local_inputs = 2*local_inputs-1
    encode_input = nn.relu(jnp.dot(Winput, local_inputs) + binput)
    layer1 = nn.relu(jnp.dot(Wrnn1, jnp.concatenate((encode_input, local_states))) + brnn1)
    layer2 = nn.relu(jnp.dot(Wrnn2, layer1) + brnn2)
    new_state = jnp.arcsinh(jnp.dot(Wrnn3, layer2) + brnn3)
    prob = nn.softmax(jnp.dot(Wamp, new_state) + bamp)
    phase = jnp.pi * nn.soft_sign(jnp.dot(Wphase, new_state) + bphase)

    return new_state, prob, phase

def init_gru_params(Nx, Ny, units, input_size, key):
    # input is already concantenated
    key, u_params = init_network_params([2 * (units + input_size), units], Ny, Nx, key)
    key, r_params = init_network_params([2 * (units + input_size), units], Ny, Nx, key)
    key, s_params = init_network_params([2 * (units + input_size), units], Ny, Nx, key)
    key, c1_params = init_network_params([2 * units, units], Ny, Nx, key)
    key, c2_params = init_network_params([2 * input_size, units], Ny, Nx, key)
    key, amp_params = init_network_params([units, input_size], Ny, Nx, key)
    key, phase_params = init_network_params([units, input_size], Ny, Nx, key)

    Wu, bu, Wr, br, Ws, bs, Wc1, bc1, Wc2, bc2 = u_params[0][0], u_params[0][1], r_params[0][0], r_params[0][1], \
    s_params[0][0], s_params[0][1], c1_params[0][0], c1_params[0][1], c2_params[0][0], c2_params[0][1]
    Wamp, bamp, Wphase, bphase = amp_params[0][0] * 0, amp_params[0][1] * 0, phase_params[0][0], phase_params[0][1]

    return (Wu, bu, Wr, br, Ws, bs, Wc1, bc1, Wc2, bc2, Wamp, bamp, Wphase, bphase)

def gru_rnn_step(local_inputs, local_states, params):  # local_input is already concantenated
    Wu, bu, Wr, br, Ws, bs, Wc1, bc1, Wc2, bc2, Wamp, bamp, Wphase, bphase = params
    rnn_inputs = jnp.concatenate((local_states, local_inputs), axis=0)
    u = nn.sigmoid(jnp.dot(Wu, rnn_inputs) + bu)
    r = nn.sigmoid(jnp.dot(Wr, rnn_inputs) + br)
    s = jnp.dot(Ws, rnn_inputs) + bs
    htilde = jnp.tanh(jnp.dot(Wc2, local_inputs) + r * (jnp.dot(Wc1, local_states) + bc1) + bc2)
    new_state = (1 - u) * s + u * htilde
    prob = nn.softmax(jnp.dot(Wamp, new_state) + bamp)
    phase = jnp.pi * nn.soft_sign(jnp.dot(Wphase, new_state) + bphase)

    return new_state, prob, phase

def init_tensor_gru_params(Nx, Ny, units, input_size, key):
    # input is already concantenated
    key, u_params = init_network_params( [ 4*(units * input_size), units], Ny, Nx, key)
    key, r_params = init_network_params( [ 4*(units * input_size), units], Ny, Nx, key)
    key, s_params = init_network_params( [ 4*(units * input_size), units], Ny, Nx, key)
    key, amp_params = init_network_params([units, input_size], Ny, Nx, key)
    key, phase_params = init_network_params([units, input_size], Ny, Nx, key)

    Wu, bu, Wr, br, Ws, bs = u_params[0][0], u_params[0][1], r_params[0][0], r_params[0][1], s_params[0][0], s_params[0][1]
    Wamp, bamp, Wphase, bphase = amp_params[0][0] * 0, amp_params[0][1] * 0, phase_params[0][0], phase_params[0][1]

    return (Wu, bu, Wr, br, Ws, bs, Wamp, bamp, Wphase, bphase)

def tensor_gru_rnn_step(local_inputs, local_states, params):  # local_input is already concantenated
    Wu, bu, Wr, br, Ws, bs, Wamp, bamp, Wphase, bphase = params
    #local_inputs = 2*local_inputs-1
    rnn_inputs = jnp.outer(local_inputs, local_states).ravel()
    u = nn.sigmoid(jnp.dot(Wu, rnn_inputs) + bu)
    r = nn.tanh(jnp.dot(Wr, rnn_inputs) + br)
    s = jnp.dot(Ws, rnn_inputs) + bs
    new_state = u * r + (1 - u) * s
    prob = nn.softmax(jnp.dot(Wamp, new_state) + bamp)
    phase = jnp.pi * nn.soft_sign(jnp.dot(Wphase, new_state) + bphase)

    return new_state, prob, phase

