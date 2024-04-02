import jax
import netket as nk
import numpy as np
from netket.operator.spin import sigmax, sigmaz
import time
import netket.nn as nknn
import flax.linen as nn
import jax.numpy as jnp
from scipy.sparse.linalg import eigsh

jax.config.update("jax_enable_x64", False)
import optax
import itertools

N = 64
hi = nk.hilbert.Spin(s=1 / 2, N=N)
g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
angle_list = [0, 0.05 * jnp.pi, 0.10 * jnp.pi, 0.15 * jnp.pi, 0.2 * jnp.pi, 0.25 * jnp.pi, 0.3 * jnp.pi, 0.35 * jnp.pi,
              0.4 * jnp.pi, 0.45 * jnp.pi, 0.5 * jnp.pi]

a = 0
for angle in angle_list:

    os = -jnp.cos(angle) ** 2 * sigmaz(hi, 0) * sigmax(hi, 1)
    os += jnp.cos(angle) * jnp.sin(angle) * sigmaz(hi, 0) * sigmaz(hi, 1)
    os += jnp.sin(angle) ** 2 * sigmax(hi, 0) * sigmaz(hi, 1)
    os -= jnp.cos(angle) * jnp.sin(angle) * sigmax(hi, 0) * sigmax(hi, 1)

    for j in range(N - 3):
        os -= jnp.cos(angle) ** 3 * sigmax(hi, j) * sigmaz(hi, j + 1) * sigmax(hi, j + 2)
        os += jnp.cos(angle) ** 2 * jnp.sin(angle) * sigmaz(hi, j) * sigmaz(hi, j + 1) * sigmax(hi, j + 2)
        os -= jnp.cos(angle) ** 2 * jnp.sin(angle) * sigmax(hi, j) * sigmax(hi, j + 1) * sigmax(hi, j + 2)
        os += jnp.cos(angle) ** 2 * jnp.sin(angle) * sigmax(hi, j) * sigmaz(hi, j + 1) * sigmaz(hi, j + 2)
        os += jnp.cos(angle) * jnp.sin(angle) ** 2 * sigmaz(hi, j) * sigmax(hi, j + 1) * sigmax(hi, j + 2)
        os += jnp.cos(angle) * jnp.sin(angle) ** 2 * sigmax(hi, j) * sigmax(hi, j + 1) * sigmaz(hi, j + 2)
        os -= jnp.cos(angle) * jnp.sin(angle) ** 2 * sigmaz(hi, j) * sigmaz(hi, j + 1) * sigmaz(hi, j + 2)
        os -= jnp.sin(angle) ** 3 * sigmaz(hi, j) * sigmax(hi, j + 1) * sigmaz(hi, j + 2)

    # Additional lines outside the loop
    os -= jnp.cos(angle) ** 3 * sigmax(hi, N - 3) * sigmaz(hi, N - 2) * sigmaz(hi, N - 1)
    os += jnp.cos(angle) ** 2 * jnp.sin(angle) * sigmaz(hi, N - 3) * sigmaz(hi, N - 2) * sigmaz(hi, N - 1)
    os -= jnp.cos(angle) ** 2 * jnp.sin(angle) * sigmax(hi, N - 3) * sigmax(hi, N - 2) * sigmaz(hi, N - 1)
    os -= jnp.cos(angle) ** 2 * jnp.sin(angle) * sigmax(hi, N - 3) * sigmaz(hi, N - 2) * sigmax(hi, N - 1)
    os += jnp.cos(angle) * jnp.sin(angle) ** 2 * sigmaz(hi, N - 3) * sigmax(hi, N - 2) * sigmaz(hi, N - 1)
    os += jnp.cos(angle) * jnp.sin(angle) ** 2 * sigmaz(hi, N - 3) * sigmaz(hi, N - 2) * sigmax(hi, N - 1)
    os -= jnp.cos(angle) * jnp.sin(angle) ** 2 * sigmax(hi, N - 3) * sigmax(hi, N - 2) * sigmax(hi, N - 1)
    os += jnp.sin(angle) ** 3 * sigmaz(hi, N - 3) * sigmax(hi, N - 2) * sigmax(hi, N - 1)

    os -= jnp.cos(angle) ** 2 * sigmax(hi, N - 2) * sigmax(hi, N - 1)
    os += jnp.cos(angle) * jnp.sin(angle) * sigmax(hi, N - 2) * sigmaz(hi, N - 1)
    os -= jnp.sin(angle) ** 2 * sigmaz(hi, N - 2) * sigmaz(hi, N - 1)
    os += jnp.cos(angle) * jnp.sin(angle) * sigmaz(hi, N - 2) * sigmax(hi, N - 1)
    evals = nk.exact.lanczos_ed(os, compute_eigenvectors=False)
    print(evals[0])

    class FFNN(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=4*x.shape[-1],
                         use_bias=True,
                         param_dtype=jnp.complex64,
                        )(x)
            x = nknn.log_cosh(x)
            x = jnp.sum(x, axis=-1)
            return x
    # RBM ansatz with alpha=1
    ma = FFNN()
    # Metropolis Exchange Sampling
    # Notice that this sampler exchanges two neighboring sites
    # thus preservers the total magnetization
    sa = nk.sampler.MetropolisLocal(hilbert=hi)
    schedule = optax.piecewise_constant_schedule(0.01, {1000: 0.5})
    # Optimizer
    op = nk.optimizer.Sgd(learning_rate=schedule)

    # Stochastic Reconfiguration
    sr = nk.optimizer.SR(diag_shift=optax.linear_schedule(0.03, 0.005, 1000))

    # The variational state
    vs = nk.vqs.MCState(sa, ma, n_samples=4096)

    # The ground-state optimization loop
    gs = nk.VMC(
        hamiltonian=os,
        optimizer=op,
        preconditioner=sr,
        variational_state=vs)

    start = time.time()
    gs.run(out='FFNN' + str(a), n_iter=10000)
    combinations = np.array(list(itertools.product([1, -1], repeat=N)))
    np.save("FFNN" + str(a) + "_amp.npy", vs.log_value(combinations))
    end = time.time()
    a += 1
    print('### Symmetric FFNN calculation')
    print('Has', vs.n_parameters, 'parameters')
    print('The Symmetric FFNN calculation took', end - start, 'seconds')
