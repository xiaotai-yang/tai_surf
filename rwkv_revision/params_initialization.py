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


def init_tensor_gru_params(Nx, Ny, units, input_size, key, ln = False, generalized_output=False, state_init = True):
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
def tensor_gru_rnn_step(local_inputs, local_states, params, output_params, ln = True):  # local_input is already concantenated
    print("begin")
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


def init_RWKV_params(input_size, emb_size, h_size,  num_layer, out_size, Ny, Nx, key):
    (key, emb_key, init_x_key, init_y_key, t_last_x1_key, t_last_x2_key, t_last_y1s_key, t_last_y1e_key, t_last_y2_key,  key_tout, key_txout, key_talpha_out,
     c_last_x1_key, c_last_x2_key, c_last_y1s_key, c_last_y1e_key, c_last_y2_key, key_tbeta_out, key_tlast_x, key_c_wv, key_clast_x, key_cxout) = split(key, 22)
    wemb = random.uniform(emb_key, (input_size, emb_size), minval=-1e-4, maxval=1e-4) #input hasn't been connected
    x_init = random.uniform(init_x_key, (Nx, emb_size), minval=-1e-4, maxval=1e-4)
    y_init = random.uniform(init_y_key, (Ny, emb_size), minval=-1e-4, maxval=1e-4)
    t_last_x1_init = random.uniform(t_last_x1_key, (Nx, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    t_last_x2_init = random.uniform(t_last_x2_key, (Nx, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    t_last_y1s_init = random.uniform(t_last_y1s_key, (Ny+1, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    t_last_y1e_init = random.uniform(t_last_y1e_key, (Ny+1, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    t_last_y2_init = random.uniform(t_last_y2_key, (Ny, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    c_last_x1_init = random.uniform(c_last_x1_key, (Nx, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    c_last_x2_init = random.uniform(c_last_x2_key, (Nx, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    c_last_y1s_init = random.uniform(c_last_y1s_key, (Ny + 1, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    c_last_y1e_init = random.uniform(c_last_y1e_key, (Ny + 1, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    c_last_y2_init = random.uniform(c_last_y2_key, (Ny, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    t_alpha_init_x, t_alpha_init_y, t_beta_init_x, t_beta_init_y = jnp.zeros((Nx, num_layer, h_size)), jnp.zeros((Ny, num_layer, h_size)), jnp.zeros((Nx, num_layer, h_size)), jnp.zeros((Ny, num_layer, h_size))
    t_xout = random.normal(key_tout, (Ny, Nx, num_layer, emb_size, 2*emb_size))
    t_alphaout = 0.5*jnp.tile(jnp.eye(h_size), (Ny, Nx, num_layer, 1, 2))
    t_betaout = 0.5*jnp.tile(jnp.eye(h_size), (Ny, Nx, num_layer, 1, 2))
    #t_betaout = random.uniform(key_tbeta_out, (Ny, Nx, num_layer, h_size, 2*h_size), minval = 0.3/h_size, maxval = 0.6/h_size)
    c_xout = random.normal(key_cxout, (Ny, Nx, num_layer, emb_size, 2*emb_size))/emb_size
    emb_size, h_size = 2*emb_size, 2*h_size #tensor product the input from two directions and concantenate the hidden state
    wln_in, bln_in, wln_out, bln_out = jnp.ones((Ny, Nx, emb_size)), jnp.zeros((Ny, Nx, emb_size)), jnp.ones((Ny, Nx, emb_size)), jnp.zeros((Ny, Nx, emb_size))  #in&out layer_norm params
    wln, bln = jnp.ones((2, Ny, Nx, num_layer, emb_size)), jnp.zeros((2, Ny, Nx, num_layer, emb_size))  #time&channel layer_norm params

    # time mixing params
    decay = jnp.tile(-5 + jnp.array([8*(jnp.arange(h_size)/(h_size-1))**(0.7 + 1.3*i/(num_layer-1)) for i in range(num_layer)]), (Ny, Nx, 1, 1))
    bonus = jnp.tile(0.5*(jnp.arange(h_size)%3-1)+jnp.log(0.3), (Ny, Nx, num_layer, 1))
    t_mix_k = jnp.tile(jnp.array([(jnp.arange(emb_size) / emb_size) ** (1 - i / num_layer) for i in range(num_layer)]), (Ny, Nx, 1, 1))
    t_mix_v = t_mix_k + jnp.transpose(jnp.tile(jnp.arange(num_layer) * 0.3 / (num_layer - 1), (Ny, Nx, emb_size, 1)), (0, 1, 3, 2))
    t_mix_r = 0.5 * t_mix_k
    t_wk, t_wv, t_wr = jnp.zeros((Ny, Nx, num_layer, h_size, emb_size)), jnp.zeros((Ny, Nx, num_layer, h_size, emb_size)), jnp.zeros((Ny, Nx, num_layer, h_size, emb_size))
    t_wout = jnp.sqrt(h_size/emb_size)*random.normal(key_tout, (Ny, Nx, num_layer, emb_size, h_size))
    t_wlast_x = random.normal(key_tlast_x, (Ny, Nx, num_layer, emb_size, 2*emb_size)) #since last_x is twice larger than x

    # channel mixing params
    c_mix_k =  jnp.tile(jnp.array([(jnp.arange(emb_size) / emb_size) ** (1 - i / num_layer) for i in range(num_layer)]), (Ny, Nx, 1, 1))
    c_mix_r =  jnp.tile(jnp.array([(jnp.arange(emb_size) / emb_size) ** (1 - i / num_layer) for i in range(num_layer)]), (Ny, Nx, 1, 1))
    c_wr, c_wv, c_wk = jnp.zeros((Ny, Nx, num_layer, emb_size, emb_size)), jnp.sqrt(h_size/emb_size)*random.normal(key_c_wv, (Ny, Nx, num_layer, emb_size, emb_size)), jnp.zeros((Ny, Nx, num_layer, emb_size, emb_size))
    c_wlast_x = random.normal(key_clast_x, (Ny, Nx, num_layer, emb_size, 2*emb_size)) #since last_x is twice larger than x
    # output params
    whead, bhead = jnp.tile(jnp.eye(emb_size), (Ny, Nx, 1, 1)), jnp.zeros((Ny, Nx, emb_size))
    wprob, bprob, wphase, bphase = jnp.zeros((out_size, emb_size)), jnp.zeros((out_size)), jnp.zeros((out_size, emb_size)), jnp.zeros((out_size))
    RWKV_cell_params = wln[0], bln[0], wln[1], bln[1], decay, bonus, t_mix_k, t_mix_v, t_mix_r, t_wk, t_wv, t_wr, t_wout, t_wlast_x, c_mix_k, c_mix_r, c_wk, c_wv, c_wr, c_wlast_x, t_xout, t_alphaout, t_betaout, c_xout

    return (wemb, x_init, y_init, t_last_x1_init, t_last_x2_init, t_last_y1s_init, t_last_y1e_init, t_last_y2_init, t_alpha_init_x, t_alpha_init_y,
            t_beta_init_x, t_beta_init_y, c_last_x1_init, c_last_x2_init, c_last_y1s_init, c_last_y1e_init, c_last_y2_init,  wln_in, bln_in, wln_out, bln_out, whead, bhead, wprob, bprob, wphase, bphase, RWKV_cell_params)
def time_mixing(x, t_state, decay, bonus, t_mix_k, t_mix_v, t_mix_r, t_wlast_x, Wk, Wv, Wr, Wout, t_xout, t_alphaout, t_betaout):
    last_x, last_alpha, last_beta = t_state
    last_x = layer_norm(t_wlast_x @ last_x, 1, 0)
    k = Wk @ (x * t_mix_k + last_x * (1 - t_mix_k))
    v = Wv @ (x * t_mix_v + last_x * (1 - t_mix_v))
    r = Wr @ (x * t_mix_r + last_x * (1 - t_mix_r))
    wkv = (last_alpha + jnp.exp(bonus + k) * v) / \
          (last_beta + jnp.exp(bonus + k))
    #jax.debug.print("t_x:{}", x)
    #jax.debug.print("t_last_x:{}", last_x)
    #jax.debug.print("wkv_numerator:{}", last_alpha+jnp.exp(bonus+k)*v)
    #jax.debug.print("wkv_denominator:{}", last_beta+jnp.exp(bonus+k))
    rwkv = nn.sigmoid(r) * wkv
    #jax.debug.print("decay:{}", decay)
    #jax.debug.print("exp(k):{}", jnp.exp(k))
    alpha = jnp.exp(-jnp.exp(decay)) * last_alpha + jnp.exp(k) * v
    beta = jnp.exp(-jnp.exp(decay)) * last_beta + jnp.exp(k)
    #jax.debug.print("alpha:{}", alpha)
    #jax.debug.print("beta:{}", beta)
    alpha = t_alphaout @ alpha
    beta = t_betaout @ beta
    #jax.debug.print("alpha_out:{}", alpha)
    #jax.debug.print("beta_out:{}", beta)
    x = t_xout @ x
    return Wout @ rwkv, (x, alpha, beta)

def channel_mixing(x, c_states, c_mix_k, c_mix_r, c_wlast_x, Wk, Wv, Wr, c_xout):
    last_x = c_states #change tuple into array
    last_x = layer_norm(c_wlast_x @ last_x, 1, 0)

    k = Wk @ (x * c_mix_k + last_x * (1 - c_mix_k))
    r = Wr @ (x * c_mix_r + last_x * (1 - c_mix_r))
    #jax.debug.print("Wk:{}", Wk)
    #jax.debug.print("Wv:{}", Wv)
    #jax.debug.print("c_mix_k:{}", c_mix_k)
    #jax.debug.print("cK:{}", k)
    vk = Wv @ nn.selu(k)
    x = c_xout @ x

    return nn.sigmoid(r) * vk, x

def layer_norm(x, w, b):
    mean = jnp.mean(x)
    std =  jnp.sqrt((jnp.sum((x - mean)**2) + 1e-10)/(x.size-1))
    return (x - mean)/ std * w + b

def RWKV_step(x, t_states, c_states, num_layer, RWKV_net_params, indices):
    ny, nx = indices
    w_in, b_in, whead, bhead, w_out, b_out, wprob, bprob, wphase, bphase, RWKV_cell_params = RWKV_net_params
    x = layer_norm(x, w_in[ny, nx], b_in[ny, nx])

    x , y = lax.scan(partial(RWKV_cell, params = tuple(px[nx] for px in tuple(py[ny] for py in RWKV_cell_params)), cell_t_states = t_states, cell_c_states = c_states), x, jnp.arange(num_layer))
    #x = _[0]
    #jax.debug.print("x:{}", x)
    t_states, c_states = y
    #x = whead[ny, nx] @ x + bhead[ny, nx]
    x = whead[ny, nx] @ layer_norm(x, w_out[ny, nx], b_out[ny, nx]) + bhead[ny, nx]
    prob = nn.softmax(wprob @ x + bprob)
    phase = 2*jnp.pi*nn.soft_sign(wphase @ x + bphase)
    return x, t_states, c_states, prob, phase

def RWKV_cell(carry, i, params, cell_t_states, cell_c_states): # carry = (x, t_states, c_states)
    #modify it for different layer of t_state and c_state and x .
    x = carry
    wln_i, bln_i, wln_m, bln_m, decay, bonus, t_mix_k, t_mix_v, t_mix_r, t_wk, t_wv, t_wr, t_wout, t_wlast_x, c_mix_k, c_mix_r, c_wk, c_wv, c_wr, c_wlast_x, t_xout, t_alphaout, t_betaout, c_xout = tuple(p[i] for p in params)
    layer_t_states = tuple(t[i] for t in cell_t_states)
    layer_c_states = cell_c_states[i]

    x_ = layer_norm(x, wln_i, bln_i)
    dx, output_t_states = time_mixing(x_, layer_t_states, decay, bonus, t_mix_k, t_mix_v, t_mix_r, t_wlast_x, t_wk, t_wv, t_wr, t_wout,  t_xout, t_alphaout, t_betaout)
    x = x + dx
    x_ = layer_norm(x, wln_m, bln_m)
    dx, output_c_states = channel_mixing(x_, layer_c_states, c_mix_k, c_mix_r, c_wlast_x, c_wk, c_wv, c_wr, c_xout)
    x = x + dx
    # carry need to be modified
    return x, (output_t_states, output_c_states)



def init_transformer_params(T, ff_size, units, input_size, head, key):
    # input is already concantenated
    initializer = he_normal()
    key_encode, keyq, keyk, keyv, keyo, keyfh, keyhf, keyhh, keyho = random.split(key, 9)
    We, be = initializer(key_encode, (units, input_size)), jnp.zeros((units))
    Wq, bq = initializer(keyq,(T, head, int(units/head) , units))*T*head, jnp.zeros((T, head, int(units/head)))
    Wk, bk = initializer(keyk,(T, head, int(units/head) , units))*T*head, jnp.zeros((T, head, int(units/head)))
    Wv, bv = initializer(keyv,(T, head, int(units/head) , units))*T*head, jnp.zeros((T, head, int(units/head)))
    Wo, bo = initializer(keyo,(T, units, units)), jnp.zeros((T, units))
    a, b = jnp.ones((T, 2, units)), jnp.zeros((T, 2, units))
    Wfh, bfh = initializer(keyfh,(T, ff_size, units)), jnp.zeros((T, ff_size))
    Whf, bhf = initializer(keyhf,(T, units, ff_size)), jnp.zeros((T, units))
    Whh, bhh = initializer(keyhh,(units, units)), jnp.zeros((units))
    Who, bho = initializer(keyho,(input_size, units)), jnp.zeros((input_size))

    return (We, be, Wq, bq, Wk, bk, Wv, bv, Wo, bo, a, b, Wfh, bfh, Whf, bhf, Whh, bhh, Who, bho)
