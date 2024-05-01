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

def rms_norm(x, w):
    return w * x/jnp.sqrt(jnp.mean(x**2) + 1e-10)

def lerp(x1, x2, mu):
    return x1 + mu * (x2 - x1)

def init_params(Nx, Ny, units, input_size, key):
    # input is already concantenated
    k = random.split(key, 10)
    initializer = jax.nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1))
    wemb = random.uniform(k[0], (Nx, Ny, input_size, units), minval=-1e-4, maxval=1e-4)
    Ws_t = jnp.tile(jnp.eye(units**2), (Ny, Nx, 1, 1))
    Wu_t, bu_t = jnp.zeros(Ny, Nx, units**2, units**2), jnp.zeros(Ny, Nx, units**2)
    Wr_t, br_t = initializer(k[1], (Ny, Nx, units**2, units**2)), jnp.zeros(Ny, Nx, units**2)
    Wd_t, bd_t = random.uniform(k[2], (Ny, Nx, units**2, units**2), minval= -1e-4, maxval=1e-4), jnp.sqrt(jnp.tan(random.uniform(k[3],(Ny, Nx, units**2), minval=0.0001, maxval=0.1)))
    Wout_t = initializer(k[5], (Ny, Nx, units**2, units**2))
    mu = jnp.tile((jnp.arange(0, 1,  units**2) + 0.2) * 0.8, (Ny, Nx, 1))
    c = random.uniform(k[4], (Ny, Nx, units ** 2), minval=0.95, maxval=1.05)
    Wu_c, bu_c = initializer(k[6],(Ny, Nx, units**2, units**2)), jnp.zeros(Ny, Nx, units)
    Wr_c, br_c = initializer(k[7], (Ny, Nx, units**2, units**2)), jnp.zeros(Ny, Nx, units)
    Wout_c = initializer(k[8], (Ny, Nx, units**2, units**2))
    return (wemb), (Ws_t, Wu_t, bu_t, Wr_t, br_t, Wd_t, bd_t, Wout_t, mu, c), (Wu_c, bu_c, Wr_c, br_c, Wout_c)

def griffin_step(x, state_x, state_y, tp, cp, np, num_layer, indices):
    ny, nx = indices
    x, new_state = lax.scan(partial(griffin_cell, tp = tuple(tx[nx] for tx in tuple(ty[ny] for ty in tp)),
    cp = tuple(cx[nx] for cx in tuple(cy[ny] for cy in cp)), cell_c_states = c_states), x, jnp.arange(num_layer))
 ,jnp.arange(num_layer))
    prob = nn.softmax(Wprob @ nn.gelu(Whead @ x + bhead) + bprob)

    return prob, new_state
def griffin_cell(x, state_x, state_y, tp, cp, np):  # local_input is already concantenated
    nt, nc = np
    x_ = rms_norm(x, nt)
    dx, new_state = griffin_time(x_, state_x, state_y, tp)
    x = x + dx
    x_ = rms_norm(x, nc)
    dx = griffin_channel(x_, cp)
    x = x + dx
    return x, new_state
def griffin_time(x, state_x, state_y, params):
    Ws, Wu, bu, Wr, br, Wd, bd, Wout, mu, c = params
    state = Ws @ lerp(state_x, state_y, mu)
    u = nn.sigmoid(Wu @ x + bu)
    r = nn.gelu(Wr @ x + br)
    d = 1 / (1 + (Wd @ x + bd) ** 2) ** c
    new_state = d * state + jnp.sqrt(1 - d ** 2) * u * x

    return Wout @ (new_state * r) , new_state

def griffin_channel(x, params):
    Wu, bu, Wr, br, Wout = params
    u = nn.gelu(Wu @ x + bu)
    r = Wr @ x + br
    return Wout @ (u * r)
