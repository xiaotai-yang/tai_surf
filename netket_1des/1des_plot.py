import jax
import netket as nk
import numpy as np
from netket.operator.spin import sigmax, sigmaz
import time
import netket.nn as nknn
import flax.linen as nn
import jax.numpy as jnp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", False)
import optax
import itertools
import json
N = 20
hi = nk.hilbert.Spin(s=1 / 2, N=N)
g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
angle_list = [0, 0.05 * jnp.pi, 0.10 * jnp.pi, 0.15 * jnp.pi, 0.2 * jnp.pi, 0.25 * jnp.pi, 0.3 * jnp.pi, 0.35 * jnp.pi,
              0.4 * jnp.pi, 0.45 * jnp.pi, 0.5 * jnp.pi]
numsamples = 2048
a = 0
trace_dis = []
E_diff = []
for angle in angle_list:
    '''
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
    evals = nk.exact.lanczos_ed(os, compute_eigenvectors=True)
    np.save("RBM"+str(a)+"_exact_L=20.npy", evals[1][:, 0])
    '''
    evals = np.load("RBM"+str(a)+"_exact_L=20.npy")
    nqs = np.load("RBM" + str(a) + "L=20_amp.npy")
    nqs_norm = np.linalg.norm(np.exp(nqs).astype(np.complex128))
    diff = (np.exp(nqs) / nqs_norm - evals)

    plt.scatter(np.arange(evals.shape[0]), np.abs(evals), s = 0.001, alpha = 0.2)
    plt.scatter(np.arange(diff.shape[0]), np.abs(np.exp(nqs) / nqs_norm), s = 0.001, alpha = 0.2)
    plt.ylim(-0.001, 0.01)
    legend = plt.legend(("Exact", "RBM"),
           scatterpoints=1,
           loc='upper right',
           ncol=3,
           fontsize=10)
    for handle in legend.legendHandles:
        handle._sizes = [20]  # Increase marker size in the legend
    plt.ylabel('Amp_value')
    plt.xlabel('Basis')
    plt.title("Amp_value: exact v.s. nqs")
    plt.savefig("RBM" + str(a) + "L=20_amp_value_exact_nqs.png", dpi = 150)
    plt.clf()
    
    plt.scatter(np.arange(evals.shape[0]), np.angle(evals), s = 0.01, alpha = 0.2)
    plt.scatter(np.arange(diff.shape[0]), np.angle(np.exp(nqs) / nqs_norm), s = 0.01, alpha = 0.2)
    plt.ylim(-np.pi-0.1, np.pi+0.1)
    legend = plt.legend(("Exact", "RBM"),
           scatterpoints=1,
           loc='upper right',
           ncol=3,
           fontsize=10)
    for handle in legend.legendHandles:
        handle._sizes = [20]  # Increase marker size in the legend
    plt.ylabel('Amp_phase')
    plt.xlabel('Basis')
    plt.title("Amp_phase: exact v.s. nqs")
    plt.savefig("RBM" + str(a) + "L=20_amp_phase_exact_nqs.png", dpi = 150)
    plt.clf()

    plt.scatter(np.arange(diff.shape[0]), np.abs(diff), s = 0.001, alpha = 0.2, color='darkred')
    plt.ylim(-0.001, 0.01)
    plt.ylabel('Amp_value')
    plt.xlabel('Basis')
    plt.title("Amp_value_diff")
    plt.savefig("RBM" + str(a) + "L=20_amp_value_diff.png", dpi = 150)
    plt.clf()

    plt.scatter(np.arange(diff.shape[0]), np.angle(np.exp(nqs)/nqs_norm)-np.angle(evals), s = 0.01, alpha = 0.2, color = 'darkgreen')
    plt.ylim(-np.pi-0.1, np.pi+0.1)
    plt.ylabel('Abs_phase')
    plt.xlabel('Basis')
    plt.title("Amp_phase_diff")
    plt.savefig("RBM" + str(a) + "L=20_amp_phase_diff.png", dpi = 150)
    plt.clf()
    
    trace_dis.append(np.sqrt(1-np.abs(np.sum(np.conj(np.exp(nqs)/nqs_norm)*evals))**2))
    E_diff.append(json.load(open("RBM"+str(a)+"L=20"+".log"))['Energy']['Mean']['real'][-1]+20)
    print("nps_norm:", nqs_norm)
    print("diff_sum:", np.sum(diff))
    print("gs_sum", np.sum(np.abs(evals)))
    a += 1

plt.scatter(E_diff, trace_dis)
plt.ylabel("trace_distance")
plt.xlabel("Energy_difference")
plt.title("Trace_distance v.s. Energy_difference")
plt.savefig("RBM_L=20_tdiff_Ediff_numsample"+str(numsamples)+".png", dpi = 150)
plt.clf()

plt.scatter(np.arange(11)*np.pi*0.05, trace_dis)
plt.ylabel("trace_distance")
plt.xlabel("Rotation_angle")
plt.title("Trace_distance v.s. Rotation_angle L = 20 numsample 2048")
plt.savefig("RBM_L=20_tdiff_angle_numsample"+str(numsamples)+".png", dpi = 150)