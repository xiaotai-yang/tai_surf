import jax
import netket as nk
from netket.operator.spin import sigmax,sigmaz
import time

N = 16
hi = nk.hilbert.Spin(s=1 / 2, N =  N)

H = -1*sigmaz(hi,0)*sigmax(hi, 1)
H += -1*sigmax(hi,N-3)*sigmaz(hi, N-2)*sigmaz(hi, N-1)
H += -1*sigmax(hi,N-2)*sigmax(hi, N-1)
H += sum([-1*sigmax(hi,i)*sigmaz(hi,(i+1))*sigmax(hi,i+2) for i in range(N-3)])

HX = -1 * sigmax(hi, 0) * sigmaz(hi, 1)
HX += -1 * sigmaz(hi, N - 3) * sigmax(hi, N - 2) * sigmax(hi, N - 1)
HX += -1 * sigmaz(hi, N - 2) * sigmaz(hi, N - 1)
HX += sum([-1 * sigmaz(hi, i) * sigmax(hi, (i + 1)) * sigmaz(hi, i + 2) for i in range(N - 3)])

graph = nk.graph.Chain(length=N, pbc=True)
g = nk.graph.Hypercube(length=N, n_dim=1, pbc=True)

ma = nk.models.RBM(alpha=2)
sa = nk.sampler.MetropolisLocal(hilbert=hi)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.05)
# Stochastic Reconfiguration
sr = nk.optimizer.SR(diag_shift=0.1)

# The variational state
vs = nk.vqs.MCState(sa, ma, n_samples=2048)

# The ground-state optimization loop
gs = nk.VMC(
    hamiltonian=HX,
    optimizer=op,
    preconditioner=sr,
    variational_state=vs)

start = time.time()
gs.run(out='RBM', n_iter=25000)
end = time.time()

print('### RBM calculation')
print('Has',vs.n_parameters,'parameters')
print('The RBM calculation took',end-start,'seconds')
