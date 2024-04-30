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

def init_tensor_gru_params(Nx, Ny, units, input_size, key, ln = False, generalized_output=False):
    # input is already concantenated
    key,  state_emb_key_x, state_emb_key_y, u_key, r_key, s_key, out_key = random.split(key, 7)
    state_init_x, state_init_y = jnp.tile(jnp.arange(units)/units, (Nx, 1)), jnp.tile(jnp.arange(units)/units, (Ny, 1))
    Wu, Wr = jnp.zeros((Ny, Nx, input_size**2, units, 2*units)), random.uniform(r_key, (Ny, Nx, input_size**2, units , 2*units), minval=-1e-4, maxval=1e-4)
    Wout = random.normal(out_key, (Ny, Nx, 8*units, units))*jnp.sqrt(1/(8*units))
    if generalized_output == False:
        Wamp, bamp = jnp.zeros((Ny, Nx, input_size, 8*units)), jnp.zeros((Ny, Nx, input_size))
        Wphase, bphase = jnp.zeros((Ny, Nx, input_size, 8*units)), jnp.zeros((Ny, Nx, input_size))
    else:
        Wamp, bamp = jnp.zeros((input_size, 8*units)), jnp.zeros((input_size))
        Wphase, bphase = jnp.zeros((input_size, 8*units)), jnp.zeros((input_size))

    wln_out = jnp.ones((Ny, Nx, 8*units))
    bln_out = jnp.zeros((Ny, Nx, 8*units))
    return (state_init_x, state_init_y, wln_out, bln_out, Wu, Wr, Wout, Wamp, bamp, Wphase, bphase)
@partial(jax.jit, static_argnames=("ln",))
def tensor_gru_rnn_step(input_y, input_x, input_size, local_states, params, output_params, ln = True):  # local_input is already concantenated

    wln_out, bln_out, Wu, Wr, Wout = params
    Wamp, bamp, Wphase, bphase = output_params
    local_input = jnp.array(input_y*input_size + input_x, int)
    u = nn.sigmoid(Wu[local_input] @ local_states)
    r = nn.tanh(Wr[local_input] @ local_states)
    gru_state = u * r + (1 - u) * jnp.mean(local_states.reshape(2, -1), axis = 0)
    new_state = nn.elu(lax.cond(ln == True, lambda x: layer_norm(Wout @ x, wln_out, bln_out), lambda x: Wout @ x, gru_state))
    prob = nn.softmax(jnp.dot(Wamp, new_state) + bamp)
    phase = jnp.arctan(jnp.dot(Wphase, new_state) + bphase)

    return gru_state, prob, phase

def log_cosh(x):
    """
    Logarithm of the hyperbolic cosine, implemented in a more stable way.
    """
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)

def init_RWKV_params(input_size, emb_size, h_size, preout_size, num_layer, out_size, Ny, Nx, key):
    (key, emb_key, init_x_key, init_y_key, t_last_x1_key, t_last_x2_key, t_last_y1s_key, t_last_y1e_key, t_last_y2_key, key_tout,
     c_last_x1_key, c_last_x2_key, c_last_y1s_key, c_last_y1e_key, c_last_y2_key, key_c_wk, key_c_wv, key_head) = split(key, 18)
    wemb = random.uniform(emb_key, (Ny, Nx, input_size, emb_size), minval=-1e-4, maxval=1e-4)
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

    emb_size = 2*emb_size #tensor product the input from two directions and concantenate the hidden state
    wln_in, wln_out = jnp.ones((Ny, Nx, emb_size)), jnp.ones((Ny, Nx, emb_size))  #in&out layer_norm params
    wln = jnp.ones((2, Ny, Nx, num_layer, emb_size))   #time&channel layer_norm params

    # time mixing params
    decay = jnp.tile(-4 + jnp.array([7*(jnp.arange(h_size)/(h_size-1))**(0.7 + 1.3*i/(num_layer-1)) for i in range(num_layer)]), (Ny, Nx, 1, 1))
    bonus = jnp.tile(0.5*(jnp.arange(h_size)%3-1)+jnp.log(0.3), (Ny, Nx, num_layer, 1))
    t_mix_k = jnp.tile(jnp.array([(jnp.arange(emb_size) / emb_size) ** (1 - i / num_layer) for i in range(num_layer)]), (Ny, Nx, 1, 1))
    t_mix_v = t_mix_k + jnp.transpose(jnp.tile(jnp.arange(num_layer) * 0.3 / (num_layer - 1), (Ny, Nx, emb_size, 1)), (0, 1, 3, 2))
    t_mix_r = 0.5 * t_mix_k

    t_wk1, t_wv1, t_wr1 = jnp.zeros((Ny, Nx, num_layer, h_size, emb_size)), jnp.zeros((Ny, Nx, num_layer, h_size, emb_size)), jnp.zeros((Ny, Nx, num_layer, h_size, emb_size))
    t_wk2, t_wv2, t_wr2 = jnp.ones((Ny, Nx, num_layer, emb_size, 2*emb_size)), jnp.ones((Ny, Nx, num_layer, emb_size, 2*emb_size)), jnp.ones((Ny, Nx, num_layer, emb_size, 2*emb_size))
    t_wout = jnp.sqrt(h_size/emb_size)*random.normal(key_tout, (Ny, Nx, num_layer, emb_size, h_size))

    # channel mixing params
    c_mix_k =  jnp.tile(jnp.array([(jnp.arange(emb_size) / emb_size) ** (1 - i / num_layer) for i in range(num_layer)]), (Ny, Nx, 1, 1))
    c_mix_r =  jnp.tile(jnp.array([(jnp.arange(emb_size) / emb_size) ** (1 - i / num_layer) for i in range(num_layer)]), (Ny, Nx, 1, 1))
    c_wr1, c_wv1  = jnp.zeros((Ny, Nx, num_layer, emb_size, emb_size)), jnp.sqrt(h_size/emb_size)*random.normal(key_c_wv, (Ny, Nx, num_layer, emb_size, emb_size))
    c_wk1 = random.uniform(key_c_wk, (Ny, Nx, num_layer, emb_size, emb_size), minval= 1e-4, maxval=1e-4)
    c_wr2, c_wk2 =  jnp.tile(jnp.eye(emb_size), (Ny, Nx, num_layer, 1, 2)), jnp.ones((Ny, Nx, num_layer, emb_size, 2*emb_size))
    # output params
    whead, bhead = random.uniform(key_head, (Ny, Nx, preout_size, emb_size), minval = -jnp.sqrt(3/preout_size)  , maxval = jnp.sqrt(3/preout_size)), jnp.zeros((Ny, Nx, preout_size))
    wprob, bprob = jnp.zeros((Ny, Nx, out_size, preout_size)), jnp.zeros((Ny, Nx, out_size))
    RWKV_cell_params = (wln[0], wln[1], decay, bonus, t_mix_k, t_mix_v, t_mix_r, t_wk1, t_wv1, t_wr1,
                        t_wk2, t_wv2, t_wr2, t_wout, c_mix_k, c_mix_r, c_wk1, c_wv1, c_wr1, c_wk2, c_wr2)

    return (wemb, x_init, y_init, t_last_x1_init, t_last_x2_init, t_last_y1s_init, t_last_y1e_init, t_last_y2_init,
            c_last_x1_init, c_last_x2_init, c_last_y1s_init, c_last_y1e_init, c_last_y2_init,  wln_in,
            wln_out, whead, bhead, wprob, bprob, RWKV_cell_params)
def logaddexp(a, b):
    max_tensor = jnp.maximum(a, b)
    return max_tensor + jnp.log(jnp.exp(a - max_tensor) + jnp.exp(b - max_tensor))
def lerp(x1, x2, mu):
    return x1 + mu * (x2 - x1)

def lora(x, lam, A, B):
    return lam + B @ nn.tanh(A @ x)
def ddlerp (x1, x2, mu, lam, A, B):
    return x1 + lora(lerp(x1, x2, mu), lam, A, B)*(x2 - x1)
def time_mixing(x, t_state, decay, bonus, t_mix_k, t_mix_v, t_mix_r, Wk, Wv, Wr, Wk1, Wv1, Wr1, Wk2, Wv2, Wr2, Wout):
    last_x, alpha, beta= t_state
    k = (Wk1 @ x) * t_mix_k + (Wk2 @ last_x) * (1 - t_mix_k)
    v = (Wv1 @ x) * t_mix_v + (Wv2 @ last_x) * (1 - t_mix_v)
    r = (Wr1 @ x) * t_mix_r + (Wr2 @ last_x) * (1 - t_mix_r)

    last_alpha = jnp.mean(alpha.reshape(2, -1), 0)
    last_beta = jnp.mean(beta.reshape(2, -1), 0)
    wkv = (last_alpha + jnp.exp(bonus + k) * v) / (last_beta + jnp.exp(bonus + k))
    rwkv = nn.sigmoid(r) * wkv
    alpha = jnp.exp(-jnp.exp(decay)) * last_alpha + jnp.exp(k) * v
    beta = jnp.exp(-jnp.exp(decay)) * last_beta + jnp.exp(k)
    x = jnp.mean(x.reshape(2, -1), 0)

    return Wout @ rwkv, (x, alpha, beta)

def channel_mixing(x, c_states, c_mix_k, c_mix_r, Wk1, Wv, Wr1, Wk2, Wr2):
    last_x = c_states #change tuple into array

    k = (Wk1 @ x) * c_mix_k + (Wk2 @ last_x) * (1 - c_mix_k)
    r = (Wr1 @ x) * c_mix_r + (Wr2 @ last_x) * (1 - c_mix_r)
    #jax.debug.print("Wk:{}", Wk)
    #jax.debug.print("Wv:{}", Wv)
    #jax.debug.print("c_mix_k:{}", c_mix_k)
    #jax.debug.print("cK:{}", k)
    vk = Wv @ nn.elu(k)

    return nn.sigmoid(r) * vk, jnp.mean(x.reshape(2, -1), axis = 0)

def RWKV_step(x, t_states, c_states, num_layer, RWKV_net_params, indices):
    ny, nx = indices
    w_in, w_out, whead, bhead, wprob, bprob, RWKV_cell_params = RWKV_net_params
    x = rms_norm(x, w_in[ny, nx])
    x , y = lax.scan(partial(RWKV_cell, params = tuple(px[nx] for px in tuple(py[ny] for py in RWKV_cell_params)), cell_t_states = t_states, cell_c_states = c_states), x, jnp.arange(num_layer))
    #x = _[0]
    #jax.debug.print("x:{}", x)
    t_states, c_states = y
    #x = whead[ny, nx] @ x + bhead[ny, nx]
    prob = nn.softmax(wprob[ny, nx] @ log_cosh(whead[ny, nx] @ rms_norm(x, w_out[ny, nx]) + bhead[ny, nx]) + bprob[ny, nx])

    return x, t_states, c_states, prob

def RWKV_cell(carry, i, params, cell_t_states, cell_c_states): # carry = (x, t_states, c_states)
    #modify it for different layer of t_state and c_state and x .
    x = carry
    (wln_i, wln_m, decay, bonus, t_mix_k, t_mix_v, t_mix_r, t_wk1, t_wv1, t_wr1, t_wout,
     c_mix_k, c_mix_r, c_wk1, c_wv1, c_wr1) = tuple(p[i] for p in params)
    layer_t_states = tuple(t[i] for t in cell_t_states)
    layer_c_states = cell_c_states[i]

    x_ = rms_norm(x, wln_i)
    dx, output_t_states = time_mixing(x_, layer_t_states, decay, bonus, t_mix_k, t_mix_v, t_mix_r, t_wk1, t_wv1, t_wr1, t_wout)
    x = x + dx
    x_ = rms_norm(x, wln_m)
    dx, output_c_states = channel_mixing(x_, layer_c_states, c_mix_k, c_mix_r, c_wk1, c_wv1, c_wr1)
    x = x + dx
    # carry need to be modified
    return x, (output_t_states, output_c_states)

def layer_norm(x, w, b):
    mean = jnp.mean(x)
    std =  jnp.sqrt((jnp.sum((x - mean)**2) + 1e-10)/(x.size-1))
    return (x - mean)/ std * w + b

def rms_norm(x, w):
    return w * x/jnp.sqrt(jnp.mean(x**2) + 1e-10)
