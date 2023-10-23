import argparse
import itertools
import numpy as np
import optax
import jax
from jax import numpy as jnp
from Helperfunction import *
from RNNfunction import *
import pickle
from jax import make_jaxpr
import jax.config
jax.config.update("jax_enable_x64", False)


def clip_grad(g, clip_norm=10.0):
    norm = jnp.linalg.norm(g)
    scale = jnp.minimum(1.0, clip_norm / (norm + 1e-6))
    return g * scale

@partial(jax.jit, static_argnames=['fixed_parameters'])    
def compute_cost(parameters, fixed_parameters, samples, Eloc, Temperature):
    
    samples = jax.lax.stop_gradient(samples)
    Eloc = jax.lax.stop_gradient(Eloc)
    
    # First term

    log_amps_tensor = log_amp(samples, parameters, fixed_parameters)
    term1 = 2 * jnp.real(jnp.mean(log_amps_tensor.conjugate() * (Eloc - jnp.mean(Eloc))))
    # Second term
    
    term2 = 4 * Temperature * (jnp.mean(jnp.real(log_amps_tensor) * jax.lax.stop_gradient(jnp.real(log_amps_tensor)))
               - jnp.mean(jnp.real(log_amps_tensor)) * jnp.mean(jax.lax.stop_gradient(jnp.real(log_amps_tensor))))

    cost = term1 + term2
    
    return cost

def schedule(step: float, min_lr: float, max_lr: float, period: float) -> float:
    """Compute a learning rate that oscillates sinusoidally between min_lr and max_lr."""
    oscillation = (jnp.sin(jnp.pi * step / period) + 1) / 2  # Will be between 0 and 1
    return min_lr + (max_lr - min_lr) * oscillation

parser = argparse.ArgumentParser()
parser.add_argument('--L', type = int, default=8)
parser.add_argument('--numunits', type = int, default=48)
parser.add_argument('--lr', type = float, default=5e-4)
parser.add_argument('--J1', type = float, default=1.0) 
parser.add_argument('--J2', type = float, default=0.2)
parser.add_argument('--J3', type = float, default=0.0)
parser.add_argument('--lrthreshold', type = float, default=5e-4)
parser.add_argument('--lrdecaytime', type = float, default=5000)
parser.add_argument('--mag_fixed', type = bool, default=True)
parser.add_argument('--Sz', type = int, default=0)
parser.add_argument('--spinparity_fixed', type = bool, default=False)
parser.add_argument('--spinparity_value', type = int, default=1)
parser.add_argument('--gradient_clip', type = bool, default=True)
parser.add_argument('--gradient_clipvalue', type = float, default=20.0)
parser.add_argument('--dotraining', type = bool, default=True)
parser.add_argument('--T0', type = float, default= 0.5)
parser.add_argument('--Nwarmup', type = int, default=0)
parser.add_argument('--Nannealing', type = int, default=1) #10000
parser.add_argument('--Ntrain', type = int, default=1)
parser.add_argument('--Nconvergence', type = int, default=5000)
parser.add_argument('--numsamples', type = int, default=128)
parser.add_argument('--testing_sample', type = int, default=5e+4)
parser.add_argument('--lrthreshold_convergence', type = float, default=5e-4)
parser.add_argument('--lrdecaytime_convergence', type = float, default=2500)
parser.add_argument('--seed', type = int, default=1)
parser.add_argument('--rnn_type', type = str, default="gru")
args = parser.parse_args()

units = args.numunits
numsamples = args.numsamples
lr=args.lr
lrdecaytime = args.lrdecaytime
lrdecaytime_conv = args.lrdecaytime_convergence
lrthreshold = args.lrthreshold
lrthreshold_conv = args.lrthreshold_convergence
T0 = args.T0
mag_fixed = args.mag_fixed
magnetization = 2*args.Sz
spinparity_fixed = args.spinparity_fixed
spinparity_value = args.spinparity_value
J1 = args.J1
J2 = args.J2
J3 = args.J3
gradient_clip = args.gradient_clip
gradient_clipvalue = args.gradient_clipvalue
dotraining = args.dotraining
Nwarmup = args.Nwarmup
Nannealing = args.Nannealing
Ntrain = args.Ntrain
Nconvergence = args.Nconvergence
numsteps = Nwarmup + (Nannealing+1)*Ntrain + Nconvergence
testing_sample = args.testing_sample
rnn_type = args.rnn_type
input_size = 2
L = args.L
Nx = L
Ny = L
key = PRNGKey(111)

meanEnergy=[]
varEnergy=[]

optimizer = optax.adam(lr)

N = Nx*Ny
adam_optimizer = optax.adam(lr)
if (rnn_type == "vanilla"):
    params = init_vanilla_params(Nx, Ny, units, input_size, key)
    #batch_rnn = vmap(vanilla_rnn_step, (0, 0, None)) 
elif (rnn_type == "multilayer_vanilla"):
    params = init_multilayer_vanilla_params(Nx, Ny, units, input_size, key)
    #batch_rnn = vmap(multilayer_vanilla_rnn_step, (0, 0, None))
elif (rnn_type == "gru"):
    params = init_gru_params(Nx, Ny, units, input_size, key)    
    

grad_f = jax.jit(jax.grad(compute_cost), static_argnums=(1,))
fixed_params = Ny, Nx, mag_fixed, magnetization, units
# Assuming params are your model's parameters:
optimizer_state = optimizer.init(params)

T = T0

for it in range(0, numsteps):

    start = time.time()
    '''
    if it+1<=Nwarmup+Nannealing*Ntrain: #Before the end of annealing
        lr_adap = max(lrthreshold, lr/(1+it/lrdecaytime))
    elif it+1>Nwarmup+Nannealing*Ntrain: #After annealing -> finetuning the model during convergence
        lr_adap = lrthreshold_conv/(1+(it-(Nwarmup+Nannealing*Ntrain))/lrdecaytime_conv)
    ''' 
    t0 = time.time()
    samples, sample_amp = sample_prob(numsamples, params, fixed_params)
    t = time.time()
    matrixelements, sigmas, basis_where = J1J2J3_MatrixElements(samples, J1, J2, J3, Nx, Ny, params,sample_amp)
    print("matrixelement_t:", time.time()-t)
    left_basis = (Nx*(Ny-1)+(Nx-1)*Ny)*numsamples-matrixelements.shape[0]
    if left_basis>0:
        sigmas = jnp.concatenate((sigmas, jnp.zeros((left_basis, Ny, Nx))), axis=0).astype(int)
        matrixelements = jnp.concatenate((matrixelements, jnp.zeros(left_basis)), axis=0)
    t = time.time()
    amp = jnp.exp(log_amp(sigmas, params, fixed_params))
    jax.device_get(amp)
    print("calculation_t:", time.time()-t)
    diag_local_E = matrixelements[:numsamples]*amp[:numsamples]
    matrixelements_off_diag, amp_off_diag = matrixelements[numsamples:-left_basis], amp[numsamples:-left_basis]  
    basis_where = basis_where.reshape(-1, numsamples+1)
    ind1, ind2 = basis_where[:, :-1].astype(jnp.int32), basis_where[:, 1:].astype(jnp.int32)
    block_sum = compute_block_sum(amp_off_diag, matrixelements_off_diag, ind1, ind2)
   
    

    Eloc = jnp.sum(block_sum, axis=0)+diag_local_E
    #Eloc = jax.lax.stop_gradient(Get_Elocs(J1,J2,J3, Nx, Ny, samples, params, fixed_params))
    print("total_t:", time.time()-t0)
    meanE = jnp.mean(Eloc)
    varE = jnp.var(Eloc)

    meanEnergy.append(meanE)
    varEnergy.append(varE)
        
    if (T0!=0): 
        if it+1<=Nwarmup:
            if (it+1)%50==0:
                print("Pre-annealing, warmup phase:", (it+1), "/", Nwarmup)
            T = T0
        elif it+1 > Nwarmup and it+1<=Nwarmup+Nannealing*Ntrain:
            if (it+1)%50==0:
                print("Pre-annealing, annealing phase:", (it+1-Nwarmup)//Ntrain, "/", Nannealing)
            T = T0*(1-((it+1-Nwarmup)//Ntrain)/Nannealing)
        else:
            T = 0.0

        if (it+1)%2 == 0:
            print("Temperature = ", T)
        meanF = jnp.mean(Eloc + T*jnp.real(jnp.log(sample_amp*sample_amp.conjugate())))
        varF = jnp.var(Eloc + T*jnp.real(jnp.log(sample_amp*sample_amp.conjugate())))
    if (it+1)%1==0 or it==0:
        print("learning_rate =", lr)
        print("Magnetization =", jnp.mean(jnp.sum(2*samples-1, axis = (1,2))))
        if T0 != 0:
            print('mean(E): {0}, varE: {1}, meanF: {2}, varF: {3}, #samples {4}, #Step {5} \n\n'.format(meanE,varE, meanF, varF, numsamples, it+1))
        elif T0 == 0.0:
            print('mean(E): {0}, varE: {1}, #samples {2}, #Step {3} \n\n'.format(meanE,varE,numsamples, it+1))
    
    t = time.time()
    grads = grad_f(params, fixed_params, samples, Eloc, T)
    print("grad_time:", time.time()-t)
    
    if gradient_clip == True:
        grads = jax.tree_map(clip_grad, grads)
    #print("clip_grads:", grads)
    # Update the optimizer state and the parameters
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)

    if (it%500 == 0):
        params_dict = jax.tree_util.tree_leaves(params)
        with open(f"params/params_L{L}.pkl", "wb") as f:
            pickle.dump(params_dict, f)
    
