
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
from RNNfunction import *
import numpy as np

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
    if schedule_step <= 8 * cycle_steps:
        cycle = jnp.floor(1 + schedule_step / (2 * cycle_steps))
        x = jnp.abs(schedule_step / cycle_steps - 2 * cycle + 1)
        lr = min_lr + (max_lr - min_lr) * jnp.maximum(0, (1 - x))
    else:
        lr = 2 * min_lr
    return lr


@partial(jax.jit, static_argnames=['fixed_parameters'])
def compute_cost(parameters, wemb, fixed_parameters, samples, Eloc, Temperature):
    samples = jax.lax.stop_gradient(samples)
    Eloc = jax.lax.stop_gradient(Eloc)

    # First term

    log_amps_tensor = jax.jit(vmap(log_amp,(0, None, None, None)), static_argnames=['fixed_params'])(samples, parameters, wemb, fixed_parameters)
    term1 = 2 * jnp.real(jnp.mean(log_amps_tensor.conjugate() * (Eloc - jnp.mean(Eloc))))
    # Second term

    term2 = 4 * Temperature * (jnp.mean(jnp.real(log_amps_tensor) * jax.lax.stop_gradient(jnp.real(log_amps_tensor)))
                               - jnp.mean(jnp.real(log_amps_tensor)) * jnp.mean(
                jax.lax.stop_gradient(jnp.real(log_amps_tensor))))

    cost = term1 + term2

    return cost

def real_log_amp(parameters, fixed_parameters, samples):
    samples = jax.lax.stop_gradient(samples)
    log_amps_tensor = log_amp(samples, parameters, fixed_parameters)

    return jnp.real(log_amps_tensor)

def imag_log_amp(parameters, fixed_parameters, samples):
    samples = jax.lax.stop_gradient(samples)
    log_amps_tensor = log_amp(samples, parameters, fixed_parameters)

    return jnp.imag(log_amps_tensor)
def mean_real_log_amp(parameters, fixed_parameters, samples):
    samples = jax.lax.stop_gradient(samples)
    log_amps_tensor = log_amp(samples, parameters, fixed_parameters)

    return jnp.mean(jnp.real(log_amps_tensor))

def mean_imag_log_amp(parameters, fixed_parameters, samples):
    samples = jax.lax.stop_gradient(samples)
    log_amps_tensor = log_amp(samples, parameters, fixed_parameters)

    return jnp.mean(jnp.imag(log_amps_tensor))

def clip_grad(g, clip_norm=5.0):
    norm = jnp.linalg.norm(g)
    scale = jnp.minimum(1.0, clip_norm / (norm + 1e-6))
    return g * scale

def schedule(step: float, min_lr: float, max_lr: float, period: float) -> float:
    """Compute a learning rate that oscillates sinusoidally between min_lr and max_lr."""
    oscillation = (jnp.sin(jnp.pi * step / period) + 1) / 2  # Will be between 0 and 1
    return min_lr + (max_lr - min_lr) * oscillation

def flip_sample(sample, x):
    return sample.at[x].set(1-sample[x])

def create_tensor(n):
    # Create a 4D meshgrid to generate the indices for the n*n*n*n tensor
    a, b, i, j = jnp.meshgrid(jnp.arange(n), jnp.arange(n), jnp.arange(n), jnp.arange(n), indexing='ij')
    # Calculate the values according to the given condition
    M = jnp.where((a == i) & (b == j), 0, 1/ ((a - i) ** 2 + (b - j) ** 2)**(3))

    return M


