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
from jax.nn.initializers import he_normal, he_uniform, glorot_normal, glorot_uniform
@jax.jit
def rms_norm(x, w, b):
    return w * x/jnp.sqrt(jnp.mean(x**2) + 1e-8) + b

def layer_norm(x, w, b):
    mean = jnp.mean(x)
    std =  jnp.sqrt((jnp.sum((x - mean)**2) + 1e-8)/(x.size-1))
    return (x - mean)/ std * w + b

@jax.jit
def linear(x, w, b):
    return w@x+b
@jax.jit
def attention(q, k, v, loc):
    a = q.shape[0]
    N = k.shape[0]
    m =  jnp.tril(-(jnp.ones((N, N)) - jnp.eye(N))*1e9).T
    QKV = nn.softmax(jnp.matmul(q, k.T)/jnp.sqrt(a) + m[loc]) @ v
    return QKV

def init_transformer_params(T, ff_size, units, input_size, head, key):
    # input is already concantenated
    i1, i2 = (he_normal(), glorot_normal())

    key_encode, key_i, keyq, keyk, keyv, keyo, keyfh, keyhf, keyhh1, keyhh2, keyho = random.split(key, 11)
    Wemb = random.uniform(key_encode, (input_size, units), minval=-1e-4, maxval=1e-4)
    Wi, bi = i2(key_i, (units, units)), jnp.zeros((units))
    Wq, bq = i2(keyq,(T, head, int(units/head) , units)), jnp.zeros((T, head, int(units/head)))
    Wk, bk = i2(keyk,(T, head, int(units/head) , units)), jnp.zeros((T, head, int(units/head)))
    Wv, bv = i2(keyv,(T, head, int(units/head) , units)), jnp.zeros((T, head, int(units/head)))
    Wo, bo = i2(keyo,(T, units, units)), jnp.zeros((T, units))
    a1, a2, b1, b2  = jnp.ones((T, units)), jnp.ones((T, units)), jnp.zeros((T, units)), jnp.zeros((T, units))
    #a, b = jnp.ones((T, units)), jnp.ones((T, units))
    Wfh, bfh = i1(keyfh,(T, ff_size, units)), jnp.zeros((T, ff_size))
    Whf, bhf = i1(keyhf,(T, units, ff_size)), jnp.zeros((T, units))
    Whh1, bhh1 = i1(keyhh1,(units, units)), jnp.zeros((units))
    Whh2, bhh2 = i1(keyhh2,(units, units)), jnp.zeros((units))
    Who1, bho1 = jnp.zeros((input_size, units)), jnp.zeros((input_size))
    Who2, bho2 = jnp.zeros((input_size, units)), jnp.zeros((input_size))
    #return (Wemb, Wq, bq, Wk, bk, Wv, bv, Wo, bo, a, b, Wfh, bfh, Whf, bhf, Whh, bhh, Who1, bho1)
    return (Wemb, Wi, bi, Wq, bq, Wk, bk, Wv, bv, Wo, bo, a1, a2, b1, b2, Wfh, bfh, Whf, bhf, Whh1, bhh1, Whh2, bhh2, Who1, bho1, Who2, bho2)
@partial(jax.jit, static_argnames=("num_layer"))
def TF_step(x, loc, num_layer, params):
    #(Wemb, Wq, bq, Wk, bk, Wv, bv, Wo, bo, a, b, Wfh, bfh, Whf, bhf, Whh, bhh, Who1, bho1) = params
    (Wemb, Wi, bi, Wq, bq, Wk, bk, Wv, bv, Wo, bo, a1, a2, b1, b2, Wfh, bfh, Whf, bhf, Whh1, bhh1, Whh2, bhh2, Who1, bho1, Who2, bho2) = params
    _, y_ = scan(partial(TF_cell, cell_params = (Wq, bq, Wk, bk, Wv, bv, Wo, bo, a1, a2, b1, b2, Wfh, bfh, Whf, bhf), loc = loc)
    , (x, x[:, loc]) , jnp.arange(num_layer))
    state1 = nn.relu(Whh1 @ _[1][-1] + bhh1)
    prob = nn.softmax(Who1 @ state1 + bho1)
    state2 = nn.relu(Whh2 @ _[1][-1] + bhh2)
    phase = jnp.arctan(Who2 @ state2 + bho2)
    return _[0], prob, phase

@jax.jit
def TF_cell(x_, l, cell_params, loc):
    Wq, bq, Wk, bk, Wv, bv, Wo, bo, a1, a2, b1, b2, Wfh, bfh, Whf, bhf = cell_params
    x, y = x_
    Q = linear (y[l], Wq[l], bq[l])
    K = vmap(linear, (0, None, None), 1)(x[l], Wk[l], bk[l])
    V = vmap(linear, (0, None, None), 1)(x[l], Wv[l], bv[l])

    # Now Q is of shape (head, L, units/head)
    out = linear(vmap(attention,(0, 0, 0, None))(Q, K, V, loc).ravel(), Wo[l], bo[l])
    z = layer_norm(y[l] + out, a1[l], b1[l])
    y = y.at[l+1].set(layer_norm(z + linear(nn.relu(linear(z, Wfh[l], bfh[l])), Whf[l], bhf[l]), a2[l], b2[l]))
    x = x.at[l+1, loc].set(y[l+1])
    return (x, y), None

