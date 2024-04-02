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

N = 20
numsamples = 4096
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
    #evals = nk.exact.lanczos_ed(os, compute_eigenvectors=False)
    #print(evals[0])
    # RBM ansatz with alpha=1
    ma = nk.models.RBM(alpha=8, param_dtype=complex)
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
    vs = nk.vqs.MCState(sa, ma, n_samples=numsamples)

    # The ground-state optimization loop
    gs = nk.VMC(
        hamiltonian=os,
        optimizer=op,
        preconditioner=sr,
        variational_state=vs)

    start = time.time()
    gs.run(out='RBM' + str(a)+ "L=" +str(N)+"_numsample="+str(numsamples), n_iter=5000)

    if N<= 20:
        combinations = np.array(list(itertools.product([-1, 1], repeat=N)))
        np.save("RBM" + str(a)+"L="+str(N) + "_numsample4096_amp.npy", vs.log_value(combinations))

    end = time.time()
    a += 1
    print('### Symmetric RBM calculation')
    print('Has', vs.n_parameters, 'parameters')
    print('The Symmetric RBM calculation took', end - start, 'seconds')
