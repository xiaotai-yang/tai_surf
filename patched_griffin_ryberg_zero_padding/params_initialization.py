import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import jax.random as random
from jax.random import uniform, PRNGKey, split, categorical
import jax.lax as lax
import jax.nn as nn
import time
from tqdm import tqdm
from functools import partial
from jax.nn.initializers import he_normal
from typing import NamedTuple

def rms_norm(x, w):
    return w * x/jnp.sqrt(jnp.mean(x**2) + 1e-10)

def lerp(x1, x2, mu):
    return x1 + mu * (x2 - x1)

class Params(NamedTuple):
    # time mixing params
    Ws_t: jnp.ndarray
    Wu_t: jnp.ndarray
    bu_t: jnp.ndarray
    Wr_t: jnp.ndarray
    br_t: jnp.ndarray
    Wd_t: jnp.ndarray
    bd_t: jnp.ndarray
    Wout_t: jnp.ndarray
    mu: jnp.ndarray
    c: jnp.ndarray
    nt: jnp.ndarray
    # channel mixing params
    Wu_c: jnp.ndarray
    bu_c: jnp.ndarray
    Wr_c: jnp.ndarray
    br_c: jnp.ndarray
    Wout_c: jnp.ndarray
    nc: jnp.ndarray
    # other params

    state_init_x: jnp.ndarray
    state_init_y: jnp.ndarray
    Wprob: jnp.ndarray
    bprob: jnp.ndarray
    Whead: jnp.ndarray
    bhead: jnp.ndarray

def init_params(Nx, Ny, units, num_layer, input_size, p_key) -> Params:
    key = split(p_key, 11)

    # Embedding and initial state parameters

    state_init_x = jnp.zeros((Nx, num_layer, units**2))
    state_init_y = jnp.zeros((Ny, num_layer, units**2))

    # Time-related parameters (using he_normal for weight matrices in recurrent settings)
    Ws_t = jnp.tile(jnp.eye(units**2), (Ny, Nx, num_layer, 1, 1))
    Wu_t, bu_t = jnp.zeros((Ny, Nx, num_layer, units**2, units**2)), jnp.zeros((Ny, Nx, num_layer, units**2))
    Wr_t, br_t = he_normal()(key[3], (Ny, Nx, num_layer, units**2, units**2)), jnp.zeros((Ny, Nx, num_layer, units**2))
    Wd_t, bd_t = uniform(key[4], (Ny, Nx, num_layer, units**2, units**2), minval= -1e-4, maxval=1e-4), jnp.sqrt(jnp.tan(uniform(key[3],(Ny, Nx, num_layer, units**2), minval=0.0001, maxval=0.1)))
    Wout_t = he_normal()(key[5], (Ny, Nx, num_layer, units**2, units**2))
    nt = jnp.ones((Ny, Nx, num_layer, units**2))
    mu = jnp.tile((jnp.arange(0, 1,  units**2) + 0.2) * 0.8, (Ny, Nx, num_layer, 1))
    c = uniform(key[5], (Ny, Nx, num_layer, units ** 2), minval=0.95, maxval=1.05)

    # Channel-related parameters
    Wu_c, bu_c = he_normal()(key[6],(Ny, Nx, num_layer, units**2, units**2)), jnp.zeros((Ny, Nx, units**2))
    Wr_c, br_c = he_normal()(key[7], (Ny, Nx, num_layer, units**2, units**2)), jnp.zeros((Ny, Nx, units**2))
    Wout_c = he_normal()(key[8], (Ny, Nx, num_layer, units**2, units**2))
    nc = jnp.ones((Ny, Nx, num_layer, units**2))
    Whead, bhead = he_normal()(key[9], (Ny, Nx, units**2, units**2)), jnp.zeros((Ny, Nx, units**2))
    Wprob, bprob = jnp.zeros((Ny, Nx, input_size, units**2)), jnp.zeros((Ny, Nx, input_size))

    return Params(Ws_t, Wu_t, bu_t, Wr_t, br_t, Wd_t, bd_t, Wout_t, mu, c, nt
                  , Wu_c, bu_c, Wr_c, br_c, Wout_c, nc
                  , state_init_x, state_init_y, Wprob, bprob, Whead, bhead)

def griffin_step(x, state_x, state_y, params: Params, num_layer, indices):
    ny, nx = indices
    _ , new_state = lax.scan(partial(griffin_cell, params = params, indices = indices) ,
(x, state_x, state_y), jnp.arange(num_layer))
    x, state_x, state_y = _
    prob = nn.softmax(params.Wprob[ny, nx] @ nn.gelu(params.Whead[ny, nx] @ x + params.bhead[ny, nx]) + params.bprob[ny, nx])
    return prob, new_state

def griffin_cell(carry, layer, params: Params, indices):  # local_input is already concantenated
    x, state_x , state_y = carry
    dx, new_state = griffin_time(x, state_x, state_y, indices, layer, params)
    x = x + dx
    x += griffin_channel(x, indices, layer, params)
    return (x, state_x, state_y), new_state
def griffin_time(x, state_x, state_y, indices, layer, tp: Params):
    ny, nx = indices
    x = rms_norm(x, tp.nt[ny, nx, layer])
    state = tp.Ws_t[ny, nx, layer] @ lerp(state_x[layer], state_y[layer], tp.mu[ny, nx, layer])
    u = nn.sigmoid(tp.Wu_t[ny, nx, layer] @ x + tp.bu_t[ny, nx, layer])
    r = nn.gelu(tp.Wr_t[ny, nx, layer] @ x + tp.br_t[ny, nx, layer])
    d = (1 / (1 + (tp.Wd_t[ny, nx, layer] @ x + tp.bd_t[ny, nx, layer]) ** 2))**(tp.c[ny, nx, layer])
    #jax.debug.print("d:{}", d)
    new_state = d * state + jnp.sqrt(1 - d ** 2 + 1e-7) * u * x

    return tp.Wout_t[ny, nx, layer] @ (new_state * r) , new_state

def griffin_channel(x, indices, layer, cp: Params):
    ny , nx = indices
    x = rms_norm(x, cp.nc[ny, nx, layer])
    u = nn.gelu(cp.Wu_c[ny, nx, layer] @ x + cp.bu_c[ny, nx, layer])
    r = cp.Wr_c[ny, nx, layer] @ x + cp.br_c[ny, nx, layer]
    return cp.Wout_c[ny, nx, layer] @ (u * r)
