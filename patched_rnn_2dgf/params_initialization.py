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

    encode_input = nn.relu(jnp.dot(Winput, local_inputs) + binput)
    layer1 = nn.relu(jnp.dot(Wrnn1, jnp.concatenate((encode_input, local_states))) + brnn1)
    layer2 = nn.relu(jnp.dot(Wrnn2, layer1) + brnn2)
    new_state = jnp.arcsinh(jnp.dot(Wrnn3, layer2) + brnn3)
    prob = nn.softmax(jnp.dot(Wamp, new_state) + bamp)
    phase = jnp.pi * nn.soft_sign(jnp.dot(Wphase, new_state) + bphase)

    return new_state, prob, phase

def init_gru_params(Nx, Ny, units, input_size, key):
    # input is already concantenated
    key, emb_key, init_emb_key_x, init_emb_key_y, state_emb_key_x, state_emb_key_y, out_key = random.split(key, 7)
    wemb = random.uniform(emb_key, (input_size, units), minval=-1e-2, maxval=1e-2)
    winit_emb_x, winit_emb_y = random.uniform(init_emb_key_x, (Nx, units), minval=-1e-2, maxval=1e-2), random.uniform(init_emb_key_y, (Ny, units), minval=-1e-2, maxval=1e-2)
    state_init_x, state_init_y = jnp.zeros((Nx, units)), jnp.zeros((Ny, units))
    Wout = jnp.zeros((Ny, Nx, units, 2*units))
    Wamp, bamp = jnp.zeros((input_size, units)), jnp.zeros((input_size))
    Wphase, bphase = jnp.zeros((input_size, units)), jnp.zeros((input_size))
    units = 2* units
    Wu1, Wr1, Ws1 = jnp.zeros((Ny, Nx, units, units)), jnp.zeros((Ny, Nx, units, units)), jnp.zeros((Ny, Nx, units, units))
    Wu2, Wr2, Ws2 = jnp.zeros((Ny, Nx, units, units)), jnp.zeros((Ny, Nx, units, units)), jnp.zeros((Ny, Nx, units, units))
    bu1, br1, bs1 = jnp.zeros((Ny, Nx, units)), jnp.zeros((Ny, Nx, units)), jnp.zeros((Ny, Nx, units))
    bu2, br2, bs2 = jnp.zeros((Ny, Nx, units)), jnp.zeros((Ny, Nx, units)), jnp.zeros((Ny, Nx, units))
    wln_in, bln_in, wln_out, bln_out = jnp.ones((Ny, Nx, units)), jnp.zeros((Ny, Nx, units)), jnp.ones((Ny, Nx, units)), jnp.zeros((Ny, Nx, units))
    wln_u, wln_r = jnp.ones((Ny, Nx, units)), jnp.ones((Ny, Nx, units))
    bln_u, bln_r = jnp.zeros((Ny, Nx, units)), jnp.zeros((Ny, Nx, units))


    return (wemb, winit_emb_x, winit_emb_y, state_init_x, state_init_y, wln_in, bln_in, wln_out, bln_out, wln_u, bln_u, wln_r, bln_r, Wu1, Wr1, Ws1, Wu2, Wr2, Ws2, bu1, br1, bs1, bu2, br2, bs2, Wout, Wamp, bamp, Wphase, bphase)


def gru_rnn_step(local_inputs, local_states, params, output_params, ln = True):  # local_input is already concantenated
    wln_in, bln_in, wln_u, bln_u, wln_r, bln_r, Wu1, Wr1, Ws1, Wu2, Wr2, Ws2, bu1, br1, bs1, bu2, br2, bs2, Wout = params
    Wamp, bamp, Wphase, bphase = output_params
    u = nn.sigmoid(lax.cond(ln == True, lambda x,y :layer_norm(Wu1 @ x + bu1 + Wu2 @ y + bu2 , wln_u, bln_u), lambda x, y: Wu1 @ x + bu1 + Wu2 @ y + bu2, local_inputs, local_states))
    r = nn.sigmoid(lax.cond(ln == True, lambda x,y :layer_norm(Wr1 @ x + br1 + Wr2 @ y + br2, wln_r, bln_r), lambda x, y: Wr1 @ x + br1 + Wr2 @ y + br2, local_inputs, local_states))
    s = nn.tanh(lax.cond(ln == True, lambda x: layer_norm(Ws1 @ x + bs1, wln_in, bln_in), lambda x: Ws1 @ x + bs1, local_inputs))
    new_state = Wout @ (u * s + (1 - u) * local_states)
    prob = nn.softmax(jnp.dot(Wamp, new_state) + bamp)
    phase = jnp.arctan(jnp.dot(Wphase, new_state) + bphase)

    return new_state, prob, phase

'''
def init_tensor_gru_params(Nx, Ny, units, input_size, key, ln = False, generalized_output=False, state_init = False):
    # input is already concantenated
    key, emb_key, init_emb_key_x, init_emb_key_y, state_emb_key_x, state_emb_key_y, u_key, r_key, s_key, out_key = random.split(key, 10)
    tensor_size = 4 * (units * input_size)
    wemb = random.orthogonal(emb_key, input_size)
    if state_init == True:
        winit_emb_x, winit_emb_y = random.uniform(init_emb_key_x, (Nx, input_size), minval=-1e-4, maxval=1e-4), random.uniform(init_emb_key_y, (Ny, input_size), minval=-1e-4, maxval=1e-4)
        state_init_x, state_init_y = jnp.tile(jnp.arange(units)/units, (Nx, 1)), jnp.tile(jnp.arange(units)/units, (Ny, 1))
    Wu, Wr, Ws = random.uniform(u_key, (Ny, Nx, units, tensor_size), minval=-1e-4, maxval=1e-4), random.uniform(r_key, (Ny, Nx, units , tensor_size), minval=-1e-4, maxval=1e-4), random.uniform(s_key, (Ny, Nx, units, tensor_size), minval=-1e-4, maxval=1e-4)
    Wout = random.normal(out_key, (Ny, Nx, units, units))
    if generalized_output == False:
        Wamp, bamp = jnp.zeros((Ny, Nx, input_size, units)), jnp.zeros((Ny, Nx, input_size))
        Wphase, bphase = jnp.zeros((Ny, Nx, input_size, units)), jnp.zeros((Ny, Nx, input_size))
    else:
        Wamp, bamp = jnp.zeros((input_size, units)), jnp.zeros((input_size))
        Wphase, bphase = jnp.zeros((input_size, units)), jnp.zeros((input_size))

    wln_in, bln_in = jnp.ones((Ny, Nx, tensor_size)), jnp.zeros((Ny, Nx, tensor_size))
    wln_u, wln_r, wln_out = jnp.ones((Ny, Nx, units)), jnp.ones((Ny, Nx, units)), jnp.ones((Ny, Nx, units))
    bln_u, bln_r, bln_out = jnp.zeros((Ny, Nx, units)), jnp.zeros((Ny, Nx, units)), jnp.zeros((Ny, Nx, units))
    return (wemb, winit_emb_x, winit_emb_y, state_init_x, state_init_y, wln_in, bln_in, wln_u, bln_u, wln_r, bln_r, wln_out, bln_out, Wu, Wr, Ws, Wout, Wamp, bamp, Wphase, bphase)

@partial(jax.jit, static_argnames=("ln",))
def tensor_gru_rnn_step(local_inputs, local_states, params, output_params, ln = False):  # local_input is already concantenated
    wln_in, bln_in, wln_u, bln_u, wln_r, bln_r, wln_out, bln_out, Wu, Wr, Ws, Wout = params
    Wamp, bamp, Wphase, bphase = output_params

    rnn_inputs = lax.cond(ln == True, lambda x, y:layer_norm(jnp.outer(x, y).ravel(), wln_in, bln_in), lambda x,y: jnp.outer(x, y).ravel(), local_inputs, local_states)
    u = nn.sigmoid(lax.cond(ln == True, lambda x: layer_norm(Wu @ x, wln_u, bln_u), lambda x: Wu @ x, rnn_inputs))
    r = nn.tanh(lax.cond(ln == True, lambda x: layer_norm(Wr @ x, wln_r, bln_r), lambda x: Wr @ x, rnn_inputs))
    s = Ws @ rnn_inputs
    gru_state = u * r + (1 - u) * s
    new_state = lax.cond(ln == True, lambda x: layer_norm(Wout @ x, wln_out, bln_out), lambda x: Wout @ x, gru_state)
    prob = nn.softmax(jnp.dot(Wamp, new_state) + bamp)
    phase = jnp.arctan(jnp.dot(Wphase, new_state) + bphase)

    return new_state, prob, phase

def layer_norm(x, w, b):
    mean = jnp.mean(x)
    std =  jnp.sqrt((jnp.sum((x - mean)**2) + 1e-10)/(x.size-1))
    return (x - mean )/ std * w + b
'''
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
