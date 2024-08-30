import jax
import jax.numpy as jnp
from jax import grad, hessian, jit
from jax.scipy.linalg import solve

jax.config.update("jax_enable_x64", True)


# Function to evaluate the objective, gradient, and Hessian
def objective_with_log_barrier(x, A, b, c, t):
    f0 = jnp.dot(c.T, x)  # Linear term c^T x
    # Log barrier term
    phi = -jnp.sum(jnp.log(b - jnp.dot(A, x)))
    obj = t * f0 + phi
    return obj


# Compute the gradient and Hessian using JAX automatic differentiation
grad_f = jit(grad(objective_with_log_barrier))
hess_f = jit(hessian(objective_with_log_barrier))


# Newton step calculation
def newton_step(x, A, b, c, t):
    g = grad_f(x, A, b, c, t)
    H = hess_f(x, A, b, c, t)
    delta_x_nt = -solve(H, g)
    lam = jnp.sqrt(-jnp.dot(g.T, delta_x_nt))
    return delta_x_nt, lam


# Backtracking line search
def line_search(x, A, b, c, t, delta_x_nt, lam, alpha=0.1, beta=0.7):
    s = 1.0
    while True:
        new_x = x + s * delta_x_nt
        print(objective_with_log_barrier(new_x, A, b, c, t),
              objective_with_log_barrier(x, A, b, c, t) - alpha * s * lam ** 2)
        if (objective_with_log_barrier(new_x, A, b, c, t) <= objective_with_log_barrier(x, A, b, c,
                                                                                        t) - alpha * s * lam ** 2):
            break
        s *= beta

    return new_x, lam, s


def out_loop(x, A, b, c, t, eps=1e-5, eps_out=1e-6):
    history = {
        'x': [],
        'lam': [],
        'lam2': [],
        's': [],
        'fx_k': []
    }
    while True:
        inner_x, inner_lam, inner_lam2, inner_s, inner_fx_k = [], [], [], [], []
        while True:
            delta_x_nt, lam = newton_step(x, A, b, c, t)
            print("lam_square:", lam ** 2)
            if lam ** 2 / 2 < eps:
                x, lam, s = line_search(x, A, b, c, t, delta_x_nt, lam)
                fx_k = objective_with_log_barrier(x, A, b, c, t)
                inner_x.append(x.copy())
                inner_lam.append(lam.item())
                inner_lam2.append(lam.item() ** 2 / 2)
                inner_s.append(s)
                inner_fx_k.append(fx_k.item())
                print("inner_loop_finished")
                break
            x, lam, s = line_search(x, A, b, c, t, delta_x_nt, lam)
            fx_k = objective_with_log_barrier(x, A, b, c, t)

            # Record the values in each inner iteration
            inner_x.append(x.copy())
            inner_lam.append(lam.item())
            inner_lam2.append(lam.item() ** 2 / 2)
            inner_s.append(s)
            inner_fx_k.append(fx_k.item())

        # Record the values in each outer iteration
        history['x'].append(inner_x)
        history['lam'].append(inner_lam)
        history['lam2'].append(inner_lam2)
        history['s'].append(inner_s)
        history['fx_k'].append(inner_fx_k)

        t *= 10
        if b.shape[0] / t < eps_out:
            break
    return history


A = jnp.array([
    [1.3048, 0.0590, -0.7024, -0.8736, -0.2313],
    [-0.9301, -1.2304, 0.7595, -1.6002, -1.4855],
    [0.3382, -0.8725, 1.7097, -0.6783, 0.5803],
    [-0.5309, -0.8027, 0.4854, -0.9312, -0.3640],
    [-0.7574, 1.1024, 0.0972, 1.1002, -0.6000],
    [-2.6316, -0.4804, 1.0733, -0.6100, -0.8518],
    [0.7602, 0.0914, 0.7280, -1.2760, 0.7939],
    [1.8749, -1.3331, -2.4489, -0.4163, -0.1089],
    [0.4996, -0.1831, -1.7916, -0.6850, -1.5528],
    [1.3763, -1.6259, -0.8334, 0.2622, 2.1708]
], dtype=jnp.float64)

b = jnp.array([2.7085, 2.3553, 2.7741, 7.5326, 7.1057, 2.2556, 1.7183, 7.7532, 1.7955, 9.6114], dtype=jnp.float64)

c = jnp.array([-0.0161, 1.1670, 1.2126, 0.2647, -1.6935], dtype=jnp.float64)

x0 = jnp.array([1.0964, -0.0574, 0.1919, 1.2813, 0.4028], dtype=jnp.float64)

ans = out_loop(x0, A, b, c, 1.0, eps=1e-5)

ans['lam']
import numpy as np
import matplotlib.pyplot as plt

k = 1
for i in ans['lam2']:
    plt.semilogy(i, label=" k = " + str(k))
    plt.xlabel('Iteration')
    plt.ylabel('lam^2/2')
    k += 1
plt.legend()
plt.show()

# flattened_ans = {key: [item for sublist in ans[key] for item in sublist] for key in ans}


for i in ans['fx_k']:
    round_i = [round(elem, 4) for elem in i]
    print(round_i)

for i in ans['lam']:
    round_i = [round(elem, 5) for elem in i]
    print(round_i)

for i in ans['s']:
    round_i = [round(elem, 5) for elem in i]
    print(round_i)
final_f = []

for i in ans['fx_k']:
    final_f.append(i[-1])
print(final_f)
print(ans['x'][-1][-1])
import numpy as np

A_big = np.genfromtxt('A_big.csv', delimiter=',', skip_header=0)  # adjust parameters as necessary
b_big = np.genfromtxt('b_big.csv', delimiter=',', skip_header=0)  # adjust parameters as necessary
c_big = np.genfromtxt('c_big.csv', delimiter=',', skip_header=0)  # adjust parameters as necessary
x0_big = np.genfromtxt('x0_big.csv', delimiter=',', skip_header=0)  # adjust parameters as necessary

ans_big = out_loop(x0_big, A_big, b_big, c_big, 1.0, eps=1e-5)
import numpy as np
import matplotlib.pyplot as plt

k = 1
for i in ans_big['lam2']:
    plt.semilogy(i, label=" k = " + str(k))
    plt.xlabel('Iteration')
    plt.ylabel('lam^2/2')
    k += 1
plt.legend()
plt.show()

# flattened_ans = {key: [item for sublist in ans[key] for item in sublist] for key in ans}

for i in ans_big['fx_k']:
    round_i = [round(elem, 4) for elem in i]
    print(round_i)
for i in ans_big['s']:
    round_i = [round(elem, 5) for elem in i]
    print(round_i)
for i in ans_big['lam']:
    round_i = [round(elem, 5) for elem in i]
    print(round_i)
final_f = []

for i in ans_big['fx_k']:
    final_f.append(i[-1])
print(final_f)
print(ans_big['x'][-1][-1]) 