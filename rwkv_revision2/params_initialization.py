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

def init_RWKV_params(input_size, emb_size, h_size,  num_layer, out_size, Ny, Nx, key):
    (key, emb_key, init_x_key, init_y_key, t_last_x1_key, t_last_x2_key, t_last_y1s_key, t_last_y1e_key, t_last_y2_key,  key_tout, key_txout, key_talpha_out,
     c_last_x1_key, c_last_x2_key, c_last_y1s_key, c_last_y1e_key, c_last_y2_key, key_tbeta_out, key_tlast_x, key_c_wv, key_clast_x, key_cxout) = split(key, 22)

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
    t_xout = 0.5*jnp.tile(jnp.eye(emb_size), (Ny, Nx, num_layer, 1, 2))
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

    return (x_init, y_init, t_last_x1_init, t_last_x2_init, t_last_y1s_init, t_last_y1e_init, t_last_y2_init,  c_last_x1_init, c_last_x2_init, c_last_y1s_init,
            c_last_y1e_init, c_last_y2_init,  wln_in, bln_in, wln_out, bln_out, whead, bhead, wprob, bprob, wphase, bphase, RWKV_cell_params)
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
    alpha = jnp.exp(-jnp.exp(decay)) * last_alpha + jnp.exp(k) * k
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

