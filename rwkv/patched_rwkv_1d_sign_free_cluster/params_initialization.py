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

def init_RWKV_params(emb_size, h_size, num_layer, out_h_size, out_size, N, key):
    (key, emb_key, init_x_key, init_y_key, t_last_x1_key,
     c_last_x1_key, key_tlast_x, key_c_wv,  prob1_key, phase1_key,
     prob2_key, phase2_key, prob3_key, phase3_key) = split(key, 14)

    x_init = random.uniform(init_x_key, (emb_size,), minval=-1e-4, maxval=1e-4)
    t_init = random.uniform(t_last_x1_key, (num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    c_init = random.uniform(c_last_x1_key, (num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    wln_in, bln_in, wln_out, bln_out = jnp.ones((N, emb_size)), jnp.zeros((N, emb_size)), jnp.ones((N, emb_size)), jnp.zeros((N, emb_size))  #in&out layer_norm params
    wln, bln = jnp.ones((2, N, num_layer, emb_size)), jnp.zeros((2, N, num_layer, emb_size))  #time&channel layer_norm params

    # time mixing params
    decay = jnp.tile(-5 + jnp.array([8*(jnp.arange(h_size)/(h_size-1))**(0.7 + 1.3*i/(num_layer-1)) for i in range(num_layer)]), (N, 1, 1))
    bonus = jnp.tile(0.5*(jnp.arange(h_size)%3-1)+jnp.log(0.3), (N, num_layer, 1))
    t_mix_k = jnp.tile(jnp.array([(jnp.arange(emb_size) / emb_size) ** (1 - i / num_layer) for i in range(num_layer)]), (N, 1, 1))
    t_mix_v = t_mix_k + jnp.transpose(jnp.tile(jnp.arange(num_layer) * 0.3 / (num_layer - 1), (N, emb_size, 1)), (0, 2, 1))
    t_mix_r = 0.5 * t_mix_k
    t_wk, t_wv, t_wr = jnp.zeros((N, num_layer, h_size, emb_size)), jnp.zeros((N, num_layer, h_size, emb_size)), jnp.zeros((N, num_layer, h_size, emb_size))
    t_wout = random.normal(key_tlast_x, (N, num_layer, emb_size, h_size))*jnp.sqrt(h_size/emb_size) #since last_x is twice larger than x

    # channel mixing params
    c_mix_k =  jnp.tile(jnp.array([(jnp.arange(emb_size) / emb_size) ** (1 - i / num_layer) for i in range(num_layer)]), (N, 1, 1))
    c_mix_r =  jnp.tile(jnp.array([(jnp.arange(emb_size) / emb_size) ** (1 - i / num_layer) for i in range(num_layer)]), (N, 1, 1))
    c_wr, c_wv, c_wk = jnp.zeros((N, num_layer, emb_size, emb_size)), jnp.sqrt(h_size/emb_size)*random.normal(key_c_wv, (N, num_layer, emb_size, emb_size)), jnp.zeros((N, num_layer, emb_size, emb_size))

    # output params
    whead, bhead = jnp.tile(jnp.eye(emb_size), (N, 1, 1)), jnp.zeros((N, emb_size))
    wprob1, bprob1  = random.uniform(prob1_key, (out_h_size, emb_size))*jnp.sqrt(6/(emb_size)), jnp.zeros((out_h_size))
    wphase1, bphase1 = random.uniform(phase1_key, (out_h_size, emb_size))*jnp.sqrt(6/(emb_size)), jnp.zeros((out_h_size))
    wprob2, bprob2 = jnp.zeros((out_size, out_h_size)), jnp.zeros((out_size))
    wphase2, bphase2 = jnp.zeros((out_size, out_h_size)), jnp.zeros((out_size))
    RWKV_cell_params = wln[0], bln[0], wln[1], bln[1], decay, bonus, t_mix_k, t_mix_v, t_mix_r, t_wk, t_wv, t_wr, t_wout, c_mix_k, c_mix_r, c_wk, c_wv, c_wr

    return (x_init, t_init, c_init, wln_in, bln_in, wln_out, bln_out, whead, bhead, wprob1, bprob1, wphase1, bphase1, wprob2, bprob2, wphase2, bphase2, RWKV_cell_params)
def time_mixing(x, t_state, decay, bonus, t_mix_k, t_mix_v, t_mix_r, Wk, Wv, Wr, Wout):
    last_x, last_alpha, last_beta = t_state

    k = Wk @ (x * t_mix_k + last_x * (1 - t_mix_k))
    v = Wv @ (x * t_mix_v + last_x * (1 - t_mix_v))
    r = Wr @ (x * t_mix_r + last_x * (1 - t_mix_r))
    wkv = (last_alpha + jnp.exp(bonus + k) * v) / \
          (last_beta + jnp.exp(bonus + k))

    rwkv = nn.sigmoid(r) * wkv
    alpha = jnp.exp(-jnp.exp(decay)) * last_alpha + jnp.exp(k) * v
    beta = jnp.exp(-jnp.exp(decay)) * last_beta + jnp.exp(k)

    return Wout @ rwkv, (x, alpha, beta)

def channel_mixing(x, c_states, c_mix_k, c_mix_r, Wk, Wv, Wr):
    last_x = c_states #change tuple into array

    k = Wk @ (x * c_mix_k + last_x * (1 - c_mix_k))
    r = Wr @ (x * c_mix_r + last_x * (1 - c_mix_r))
    vk = Wv @ nn.selu(k)

    return nn.sigmoid(r) * vk, x

def rms_norm(x, w, b):
    return x/(jnp.sqrt(jnp.sum(x**2) + 1e-10)) * w + b
def layer_norm(x, w, b):
    mean = jnp.mean(x)
    std =  jnp.sqrt((jnp.sum((x - mean)**2) + 1e-10)/(x.size-1))
    return (x - mean)/ std * w + b

def RWKV_step(x, t_states, c_states, num_layer, RWKV_net_params, indices):
    n = indices
    w_in, b_in, whead, bhead, wln_out, bln_out, wprob1, bprob1, wphase1, bphase1, wprob2, bprob2, wphase2, bphase2, RWKV_cell_params = RWKV_net_params
    x = rms_norm(x, w_in[n], b_in[n])
    x , y = lax.scan(partial(RWKV_cell, params = tuple(p[n] for p in RWKV_cell_params), cell_t_states = t_states, cell_c_states = c_states), x, jnp.arange(num_layer))
    t_states, c_states = y
    x = whead[n] @ rms_norm(x, wln_out[n], bln_out[n]) + bhead[n]
    prob = nn.softmax(wprob2 @ nn.relu(wprob1 @ x + bprob1) + bprob2)
    phase = 2*jnp.pi*nn.soft_sign(wphase2 @ nn.relu(wphase1 @ x + bphase1) + bphase2)
    return x, t_states, c_states, prob, phase

def RWKV_cell(carry, i, params, cell_t_states, cell_c_states): # carry = (x, t_states, c_states)
    #modify it for different layer of t_state and c_state and x .
    x = carry
    wln_i, bln_i, wln_m, bln_m, decay, bonus, t_mix_k, t_mix_v, t_mix_r, t_wk, t_wv, t_wr, t_wout, c_mix_k, c_mix_r, c_wk, c_wv, c_wr = tuple(p[i] for p in params)
    layer_t_states = tuple(t[i] for t in cell_t_states)
    layer_c_states = cell_c_states[i]

    x_ = rms_norm(x, wln_i, bln_i)
    dx, output_t_states = time_mixing(x_, layer_t_states, decay, bonus, t_mix_k, t_mix_v, t_mix_r, t_wk, t_wv, t_wr, t_wout)
    x = x + dx
    x_ = rms_norm(x, wln_m, bln_m)
    dx, output_c_states = channel_mixing(x_, layer_c_states, c_mix_k, c_mix_r, c_wk, c_wv, c_wr)
    x = x + dx
    # carry need to be modified
    return x, (output_t_states, output_c_states)

