import jax
import optax
import netket as nk
import numpy as np
from netket.operator.spin import sigmax,sigmaz
import time
import netket.nn as nknn
import flax.linen as nn
import jax.numpy as jnp
from scipy.sparse.linalg import eigsh
L = 4
N = L*L
hi = nk.hilbert.Spin(s=1 / 2, N =  N)

H = -1*sigmax(hi, 0)*sigmaz(hi, 1)*sigmaz(hi, L)
H += -1*sigmax(hi, L-1)*sigmaz(hi, L-2)*sigmaz(hi, 2*L-1)
H += -1*sigmax(hi, L*(L-1))*sigmaz(hi, L*(L-1)+1)*sigmaz(hi, L*(L-2))
H += -1*sigmax(hi, L*L-1)*sigmaz(hi, L*L-2)*sigmaz(hi, L*(L-1)-1)
H += sum([-1*sigmax(hi,i)*sigmaz(hi,(i+1))*sigmaz(hi,i+L)*sigmaz(hi,(i-L)) for i in (np.arange(L,L*(L-1), L))])  #left
H += sum([-1*sigmax(hi,i)*sigmaz(hi,(i-1))*sigmaz(hi,i+L)*sigmaz(hi,(i-L)) for i in (np.arange(2*L-1, L*L-1, L ))])  #right
H += sum([-1*sigmax(hi,i)*sigmaz(hi,(i+1))*sigmaz(hi,i-1)*sigmaz(hi,(i+L)) for i in range(1, L-1)])  #top
H += sum([-1*sigmax(hi,i)*sigmaz(hi,(i+1))*sigmaz(hi,i-1)*sigmaz(hi,(i-L)) for i in range(L*(L-1)+1, L*L-1)])  #bottom
H += sum([-1*sigmax(hi,i)*sigmaz(hi,(i+1))*sigmaz(hi,i-1)*sigmaz(hi,(i+L))*sigmaz(hi,i-L) for i in ((np.meshgrid(np.arange(L-2)+1, np.arange(L-2)+1)[1]*L+np.meshgrid(np.arange(L-2)+1, np.arange(L-2)+1)[0]).ravel())])
#%%
evals = nk.exact.lanczos_ed(H,  compute_eigenvectors=False)
print(evals)
class FFNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=2*x.shape[-1],
                     use_bias=True,
                     param_dtype=jnp.complex64,
                     kernel_init=nn.initializers.normal(stddev=0.01),
                     bias_init=nn.initializers.normal(stddev=0.01)
                    )(x)
        x = nn.Dense(features=2*x.shape[-1],
                     use_bias=True,
                     param_dtype=jnp.complex64,
                     kernel_init=nn.initializers.normal(stddev=0.01),
                     bias_init=nn.initializers.normal(stddev=0.01)
                    )(x)
        x = nknn.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x

model = FFNN()
n_iter=1000
ma = nk.models.RBM(alpha=2)
sa = nk.sampler.MetropolisLocal(hilbert=hi)

# Optimizer
warmup_cosine_decay_scheduler = optax.warmup_cosine_decay_schedule(init_value=0.05, peak_value=0.10,
                                                                   warmup_steps=100,
                                                                   decay_steps= n_iter,
                                                                   end_value=0.04)
op = nk.optimizer.Sgd(learning_rate = warmup_cosine_decay_scheduler)
# Stochastic Reconfiguration
sr = nk.optimizer.SR(diag_shift=optax.exponential_decay(init_value = 0.01, transition_steps = 2000, decay_rate = 0.5, end_value=0.0001))

# The variational state
vs = nk.vqs.MCState(sa, model, n_samples=1024)

# The ground-state optimization loop
gs = nk.VMC(
    hamiltonian=H,
    optimizer=op,
    preconditioner=sr,
    variational_state=vs)

start = time.time()
gs.run(out='RBM', n_iter=10000)
end = time.time()

print('### RBM calculation')
print('Has',vs.n_parameters,'parameters')
print('The RBM calculation took',end-start,'seconds')
