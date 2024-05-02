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
from typing import NamedTuple

def rms_norm(x, w):
    return w * x/jnp.sqrt(jnp.mean(x**2) + 1e-10)

def lerp(x1, x2, mu):
    return x1 + mu * (x2 - x1)
class t_params(NamedTuple):
    Ws: jnp.ndarray
    Wu: jnp.ndarray
    bu: jnp.ndarray
    Wr: jnp.ndarray
    br: jnp.ndarray
    Wd: jnp.ndarray
    bd: jnp.ndarray
    Wout: jnp.ndarray
    mu: jnp.ndarray
    c: jnp.ndarray
    nt: jnp.ndarray
class c_params(NamedTuple):
    Wu: jnp.ndarray
    bu: jnp.ndarray
    Wr: jnp.ndarray
    br: jnp.ndarray
    Wout: jnp.ndarray
    nc: jnp.ndarray

class o_params(NamedTuple):
    wemb: jnp.ndarray
    emb_x: jnp.ndarray
    emb_y: jnp.ndarray
    state_init_x: jnp.ndarray
    state_init_y: jnp.ndarray
    Wprob: jnp.ndarray
    bprob: jnp.ndarray
    Whead: jnp.ndarray
    bhead: jnp.ndarray
class Params(NamedTuple):
    tp: t_params
    cp: c_params
    op: o_params
def init_params(Nx, Ny, units, input_size, p_key) -> Params:
    key = random.split(p_key, 11)
    wemb = random.uniform(key[0], (Nx, Ny, input_size, units), minval=-1e-4, maxval=1e-4)
    emb_x = random.uniform(key[1], (Nx, input_size), minval=-1e-4, maxval=1e-4)
    emb_y = random.uniform(key[2], (Ny, input_size), minval=-1e-4, maxval=1e-4)
    state_init_x = jnp.zeros((Nx, units**2))
    state_init_y = jnp.zeros((Ny, units**2))
    Ws_t = jnp.tile(jnp.eye(units**2), (Ny, Nx, 1, 1))
    Wu_t, bu_t = jnp.zeros(Ny, Nx, units**2, units**2), jnp.zeros(Ny, Nx, units**2)
    Wr_t, br_t = he_normal(key[3], (Ny, Nx, units**2, units**2)), jnp.zeros(Ny, Nx, units**2)
    Wd_t, bd_t = random.uniform(key[4], (Ny, Nx, units**2, units**2), minval= -1e-4, maxval=1e-4), jnp.sqrt(jnp.tan(random.uniform(key[3],(Ny, Nx, units**2), minval=0.0001, maxval=0.1)))
    Wout_t = he_normal(key[5], (Ny, Nx, units**2, units**2))
    nt = jnp.ones((Ny, Nx, units**2))
    mu = jnp.tile((jnp.arange(0, 1,  units**2) + 0.2) * 0.8, (Ny, Nx, 1))
    c = random.uniform(key[5], (Ny, Nx, units ** 2), minval=0.95, maxval=1.05)
    Wu_c, bu_c = he_normal(key[6],(Ny, Nx, units**2, units**2)), jnp.zeros(Ny, Nx, units)
    Wr_c, br_c = he_normal(key[7], (Ny, Nx, units**2, units**2)), jnp.zeros(Ny, Nx, units)
    Wout_c = he_normal(key[8], (Ny, Nx, units**2, units**2))
    nc = jnp.ones((Ny, Nx, units))
    Whead, bhead = he_normal(key[9], (Ny, Nx, units**2, units**2)), jnp.zeros(Ny, Nx, units**2)
    Wprob, bprob = he_normal(key[10], (Ny, Nx, units**2, input_size)), jnp.zeros(Ny, Nx, input_size)
    return Params(tp =  t_params(Ws_t, Wu_t, bu_t, Wr_t, br_t, Wd_t, bd_t, Wout_t, mu, c, nt)
                  , cp = c_params(Wu_c, bu_c, Wr_c, br_c, Wout_c, nc)
                  , op = o_params(wemb, emb_x, emb_y, state_init_x, state_init_y, Wprob, bprob, Whead, bhead))

def griffin_step(x, state_x, state_y, params: Params, num_layer, indices):
    ny, nx = indices
    x, new_state = lax.scan(partial(griffin_cell, tp = tuple(tx[nx] for tx in tuple(ty[ny] for ty in params.tp)),
    cp = tuple(cx[nx] for cx in tuple(cy[ny] for cy in params.cp)), state_x = state_x, state_y = state_y)
    , x, jnp.arange(num_layer))
    prob = nn.softmax(params.op.Wprob @ nn.gelu(params.op.Whead @ x + params.op.bhead) + params.op.bprob)

    return prob, new_state

def griffin_cell(carry, tp, cp, layer):  # local_input is already concantenated
    x, state_x , state_y = carry
    dx, new_state = griffin_time(x, state_x, state_y, tp)
    x = x + dx
    x += griffin_channel(x, cp)
    return x, new_state
def griffin_time(x, state_x, state_y, tp):
    x = rms_norm(x, tp.nt)
    state = tp.Ws @ lerp(state_x, state_y, tp.mu)
    u = nn.sigmoid(tp.Wu @ x + tp.bu)
    r = nn.gelu(tp.Wr @ x + tp.br)
    d = 1 / (1 + (tp.Wd @ x + tp.bd) ** 2) ** tp.c
    new_state = d * state + jnp.sqrt(1 - d ** 2) * u * x

    return tp.Wout @ (new_state * r) , new_state

def griffin_channel(x, cp):
    x = rms_norm(x, cp.nc)
    u = nn.gelu(cp.Wu @ x + cp.bu)
    r = cp.Wr @ x + br
    return cp.Wout @ (u * r)
