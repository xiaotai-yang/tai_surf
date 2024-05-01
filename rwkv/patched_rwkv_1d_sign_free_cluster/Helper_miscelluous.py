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
import time
from math import ceil
import itertools
from patched_rnnfunction import *
import numpy as np
import optax

def generate_combinations(length):
    set1 = [1] * length
    set2 = [3] * length
    combinations = list(itertools.product(*zip(set1, set2)))
    return jnp.array(combinations)


def all_coe(array, reference, x, y):
    # Function to calculate x^n * y^m for a single element
    def calculate_product(element):
        # Count the number of flips compared to the reference
        flips = jnp.sum(element != reference)
        same = jnp.sum(element == reference)
        return x ** same * y ** flips

    coe = jnp.apply_along_axis(calculate_product, 1, array)
    coe_len = coe.shape[0]
    comb_num = 2 ** (array.shape[1])
    return coe[:-int(coe_len/comb_num)]


def linear_cycling_with_hold(schedule_step, max_lr, min_lr, cycle_steps):
    if schedule_step <= 6 * cycle_steps:
        cycle = jnp.floor(1 + schedule_step / (2 * cycle_steps))
        x = jnp.abs(schedule_step / cycle_steps - 2 * cycle + 1)
        lr = min_lr + (max_lr - min_lr) * jnp.maximum(0, (1 - x))
    else:
        lr = min_lr
    return lr




def clip_grad(g, clip_norm=20.0):
    norm = jnp.linalg.norm(g)
    scale = jnp.minimum(1.0, clip_norm / (norm + 1e-6))
    return g * scale

def schedule(step: float, min_lr: float, max_lr: float, period: float) -> float:
    """Compute a learning rate that oscillates sinusoidally between min_lr and max_lr."""
    oscillation = (jnp.sin(jnp.pi * step / period) + 1) / 2  # Will be between 0 and 1
    return min_lr + (max_lr - min_lr) * oscillation

def warmup_cosine_decay_scheduler(lr, lrthreshold, Nwarmup, numsteps):
    warmup_cosine_decay_scheduler = optax.warmup_cosine_decay_schedule(init_value=lr, peak_value=lrthreshold,
                                                                       warmup_steps=Nwarmup,
                                                                       decay_steps=numsteps, end_value=5e-4)

    max_lr = 0.0005
    min_lr = 0.00005
    cycle_steps = 1000  # Adjust based on your training steps

    return lambda step: linear_cycling_with_hold(step, max_lr, min_lr, cycle_steps)

@partial(jax.jit, static_argnames=['fixed_parameters',])
def compute_cost(parameters, wemb, fixed_parameters, samples, Eloc, Temperature, ny_nx_indices):
    samples = jax.lax.stop_gradient(samples)
    Eloc = jax.lax.stop_gradient(Eloc)

    # First term

    log_amps_tensor = (jax.jit(vmap(log_amp_RWKV, (0, None, None, None, None)), static_argnames=['fixed_params'])
                       (samples, parameters, wemb, fixed_parameters, ny_nx_indices))
    term1 = 2 * jnp.real(jnp.mean(log_amps_tensor.conjugate() * (Eloc - jnp.mean(Eloc))))
    # Second term

    term2 = 4 * Temperature * (jnp.mean(jnp.real(log_amps_tensor) * jax.lax.stop_gradient(jnp.real(log_amps_tensor)))
                               - jnp.mean(jnp.real(log_amps_tensor)) * jnp.mean(
                jax.lax.stop_gradient(jnp.real(log_amps_tensor))))

    cost = term1 + term2

    return cost
def params_init(N, input_size, emb_size, h_size, out_h_size, num_layer, key):
    out_size = input_size
    params = init_RWKV_params(emb_size, h_size, num_layer, out_h_size, out_size, N, key)
    return params

def flip_sample(sample, x):
    return sample.at[x].set(1-sample[x])
