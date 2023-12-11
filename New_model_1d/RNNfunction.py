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

def softmax (x):
    return jnp.exp(x)/jnp.sum(jnp.exp(x))     
def heavyside(inputs):
    sign = jnp.sign(jnp.sign(inputs) + 0.1 ) #tf.sign(0) = 0, this is a way to cure that since I want tf.sign(0) = 1
    return 0.5*(sign+1.0)


def one_hot_encoding(x, num_classes = 2):
    """Converts batched integer labels to one-hot encoded arrays."""
    return jnp.eye(num_classes)[x]
def sample_discrete(key, probabilities, size=None):
    """Sample from a discrete distribution defined by probabilities."""
    logits = jnp.log(probabilities)
    return categorical(key, logits, shape=size)

def normalization(probs, num_up, num_generated_spins, magnetization, num_samples, N):
    num_down = num_generated_spins - num_up
    activations_up = heavyside(((N+magnetization)//2-1) - num_up)
    activations_down = heavyside(((N-magnetization)//2-1) - num_down)
    probs_ = probs*jnp.stack([activations_up,activations_down], axis = 1)
    probs__ = probs_/(jnp.expand_dims(jnp.linalg.norm(probs_, axis=1, ord=1), axis = 1)) #l1 normalizing
    
    return probs__ 

def random_layer_params(N, m, n, units, key, scale=1e-2):
    w_key, b_key = random.split(key)
    #outkey1, outkey2 = random.split(w_key)
    return  (2*random.uniform(w_key, (N, m, n))-1)/jnp.sqrt(units),  (2*random.normal(b_key, (N, m))-1)/jnp.sqrt(units)

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, N, units,  key):
    keys = random.split(key, len(sizes))
    outkey = keys[0]
    return outkey, [random_layer_params(N, m, n, units, k) for m, n, k in zip(sizes[1:], sizes[:-1], keys[1:])]

def init_vanilla_params(N, units, input_size, key):

    key, rnn_params = init_network_params([units+input_size, units], N, hidden_size, key)
    key, amp_params = init_network_params([units+input_size ,input_size], N, hidden_size, key)
    key, phase_params = init_network_params([units+input_size, input_size], N, hidden_size, key)
    Wrnn, brnn = rnn_params[0][0], rnn_params[0][1] 
    Wamp, bamp = amp_params[0][0], amp_params[0][1]
    Wphase, bphase = phase_params[0][0], phase_params[0][1]                # 2*units → output(amplitude)
    rnn_states_init_x = random.normal(random.split(key)[0], (Nx, 2*units)) #states at vertical direction
    rnn_states_init_y = random.normal(random.split(key)[1], (Ny, 2*units))
    
    return rnn_states_init_x, rnn_states_init_y, (Winput, binput, Wrnn1, brnn1, Wrnn2, brnn2, Wrnn3, brnn3, Wamp, bamp, Wphase, bphase) 

def vanilla_rnn_step(local_inputs, local_states, params):   #local_input is already concantenated       
    Wrnn, brnn, Wamp, bamp, Wphase, bphase = params

    new_state = jnp.arcsinh(jnp.dot(Wrnn, jnp.concatenate((local_states, local_inputs),axis=0))+brnn)
    prob = nn.softmax(jnp.dot(Wamp, new_state)+bamp)
    phase = jnp.pi*nn.soft_sign(jnp.dot(Wphase, new_state)+bphase)
    
    return new_state, prob, phase

def init_multilayer_vanilla_params(N, units, input_size, key):
    hidden_size = 5*units
    key, Winput_params = init_network_params([ 2*input_size, units], N, hidden_size, key) #augment input dimension
    key, rnn_params = init_network_params([5*units, hidden_size, hidden_size, 2*units], N, hidden_size, key)
      # 2*units+augmen_input → hidden_layer → hidden layer → 2*rnn_state
    key, amp_params = init_network_params([2*units ,input_size], N, hidden_size, key)
    key, phase_params = init_network_params([2*units, input_size], N, hidden_size, key)
        
    Winput, binput = Winput_params[0][0],  Winput_params[0][1]
    Wrnn1, Wrnn2, Wrnn3, brnn1, brnn2, brnn3 = rnn_params[0][0], rnn_params[1][0], rnn_params[2][0], rnn_params[0][1], rnn_params[1][1], rnn_params[2][1] 
    Wamp, bamp = amp_params[0][0], amp_params[0][1]
    Wphase, bphase = phase_params[0][0], phase_params[0][1]                # 2*units → output(amplitude)
    rnn_states_init = random.normal(random.split(key)[0], (N, 2*units)) #states at vertical direction
    
    return rnn_states_init, (Winput, binput, Wrnn1, brnn1, Wrnn2, brnn2, Wrnn3, brnn3, Wamp, bamp, Wphase, bphase)


def multilayer_vanilla_rnn_step(local_inputs, local_states, params):   #local_input is already concantenated       
    Winput, binput, Wrnn1, brnn1, Wrnn2, brnn2, Wrnn3, brnn3, Wamp, bamp, Wphase, bphase = params

    encode_input = nn.relu(jnp.dot(Winput, local_inputs)+binput)  
    layer1 = nn.relu(jnp.dot(Wrnn1, jnp.concatenate((encode_input, local_states)))+brnn1)
    layer2 = nn.relu(jnp.dot(Wrnn2, layer1)+brnn2)
    new_state = jnp.arcsinh(jnp.dot(Wrnn3, layer2)+brnn3)
    prob = nn.softmax(jnp.dot(Wamp, new_state)+bamp)
    phase = jnp.pi*nn.soft_sign(jnp.dot(Wphase, new_state)+bphase)

    return new_state, prob, phase

def init_gru_params(N, units, input_size, key):
    #input is already concantenated      
    key, u_params = init_network_params([(units+input_size), units], N, units, key)
    key, r_params = init_network_params([(units+input_size), units], N, units, key)
    key, s_params = init_network_params([(units+input_size), units], N, units, key)
    key, c1_params = init_network_params([units, units], N, units, key)
    key, c2_params = init_network_params([input_size, units], N, units, key)
    key, amp_params = init_network_params([units ,input_size], N, units, key)
    key, phase_params = init_network_params([units, input_size], N, units, key)
        
    Wu, bu, Wr, br, Ws, bs, Wc1, bc1, Wc2, bc2 = u_params[0][0], u_params[0][1], r_params[0][0], r_params[0][1], s_params[0][0], s_params[0][1],c1_params[0][0], c1_params[0][1], c2_params[0][0], c2_params[0][1]
    Wamp, bamp, Wphase, bphase = amp_params[0][0], amp_params[0][1], phase_params[0][0], phase_params[0][1]
    
    return (Wu, bu, Wr, br,Ws,bs, Wc1, bc1, Wc2, bc2, Wamp, bamp, Wphase, bphase) 

def gru_rnn_step(local_inputs, local_states, params):   #local_input is already concantenated       
    Wu, bu, Wr, br,Ws,bs, Wc1, bc1, Wc2, bc2, Wamp, bamp, Wphase, bphase = params
    rnn_inputs = jnp.concatenate((local_states, local_inputs), axis=0)
    u = nn.sigmoid(jnp.dot(Wu, rnn_inputs)+bu)
    r = nn.sigmoid(jnp.dot(Wr, rnn_inputs)+br)
    s = jnp.dot(Ws, rnn_inputs)+bs
    htilde = jnp.tanh(jnp.dot(Wc2, local_inputs)+r*(jnp.dot(Wc1, local_states)+bc1)+bc2)
    new_state = (1-u)*s+u*htilde
    prob = nn.softmax(jnp.dot(Wamp, new_state)+bamp)
    phase = jnp.pi*nn.soft_sign(jnp.dot(Wphase, new_state)+bphase)

    
    return new_state, prob, phase

@partial(jax.jit, static_argnames=['num_samples','fixed_params'])    
def sample_prob(num_samples, params, fixed_params, key):
    N, mag_fixed, magnetization,  units= fixed_params
    n_indices = jnp.array([i for i in range(N)])
    inputs_init = jnp.array([1.,0.])
    rnn_states_init = jnp.zeros(units)
    batch_rnn = vmap(gru_rnn_step, (0, 0, None))

    def scan_fun_1d(carry_1d, indices):
        '''
        rnn_state_x_1d, inputs_x_1d : ↓↓↓...↓
        rnn_state_yi_1d, inputs_yi_1d : → or ←
        mag_fixed : To apply U(1) symmetry
        key : for random number generation
        num_samples
        params: rnn_parameters
        '''
        rnn_states,  num_spin, num_up, key, rnn_inputs, mag_fixed, magnetization, num_samples,N, params = carry_1d
        
        params_point = tuple(p[indices] for p in params)

        new_state, new_prob, new_phase = batch_rnn(rnn_inputs, rnn_states, params_point)
        #jax.debug.print("new_state: {}", new_state)
        #jax.debug.print("new_prob: {}", new_prob)
        new_prob = normalization(new_prob , num_up, num_spin, magnetization, num_samples, N)*(mag_fixed)+new_prob*(1-mag_fixed)

        key, subkey = split(key)
        samples_output =  categorical(subkey, jnp.log(new_prob))#sampling
        inputs_ = one_hot_encoding(samples_output) # one_hot_encoding of the sample
        num_up += 1-samples_output 
        num_spin += 1

        return (new_state, num_spin, num_up, key, inputs_, mag_fixed, magnetization, num_samples, N, params), (samples_output, new_prob, new_phase)

    batch_rnn_states_init = jnp.repeat(jnp.expand_dims(rnn_states_init, axis=0), num_samples, axis=0)
    batch_inputs_init = jnp.repeat(jnp.expand_dims(inputs_init, axis=0), num_samples, axis = 0)

    init = batch_rnn_states_init, 0, jnp.zeros(num_samples), key, batch_inputs_init, mag_fixed, magnetization, num_samples, N, params

    __, (samples, probs, phase) = scan(scan_fun_1d, init, n_indices)

    probs, phase, samples = jnp.transpose(probs, (1,0,2)),jnp.transpose(phase, (1,0,2)), jnp.transpose(samples, (1,0))
    #print("sample_probs:", probs)
    #print("samples:", samples)
    probs, phase = jnp.take_along_axis(probs, samples[..., jnp.newaxis], axis=-1).squeeze(-1), jnp.take_along_axis(phase, samples[..., jnp.newaxis], axis=-1).squeeze(-1)
    
    #jax.debug.print("scan_rnn_params: {}", scan_rnn_params)
    sample_amp = jnp.sqrt(jnp.prod(probs, axis=(1)))*jnp.exp(jnp.sum(phase, axis=(1))*1j)

    return samples, sample_amp    

@partial(jax.jit, static_argnames=['fixed_params'])
def log_amp(samples, params, fixed_params):
        # samples : (num_samples, Ny, Nx)   
    def scan_fun_1d(carry_1d, indices):

        rnn_states, num_spin, num_up, rnn_inputs, mag_fixed, magnetization, samples, num_samples, N, params = carry_1d
        params_point = tuple(p[indices] for p in params)
        new_state, new_prob, new_phase = batch_rnn(rnn_inputs, rnn_states, params_point)
        new_prob = normalization(new_prob , num_up, num_spin, magnetization, num_samples, N)*(mag_fixed)+new_prob*(1-mag_fixed)
        samples_output =  samples[:, indices]
        num_up += 1-samples_output 
        num_spin += 1

        return (new_state, num_spin, num_up, rnn_inputs, mag_fixed, magnetization, samples, num_samples,N, params), (samples_output, new_prob, new_phase)
    #initialization

    N, mag_fixed, magnetization, units= fixed_params
    n_indices = jnp.array([i for i in range(N)])
    rnn_states_init = jnp.zeros(units)
    batch_rnn = vmap(gru_rnn_step, (0, 0, None))
    num_samples = samples.shape[0]
    batch_rnn_states_init = jnp.repeat(jnp.expand_dims(rnn_states_init, axis=0), num_samples, axis = 0)
    batch_inputs_init = jnp.repeat(jnp.expand_dims(jnp.array([1.,0.]), axis=0), num_samples, axis = 0)
    init = batch_rnn_states_init, 0, jnp.zeros(num_samples), batch_inputs_init, mag_fixed, magnetization, samples, num_samples, N, params

     
    #print("samples_eval0:", samples)
    
    __, (samples, probs, phase) = scan(scan_fun_1d, init, n_indices)
     
    #print("samples_eval1:", samples)
    probs, phase, samples = jnp.transpose(probs, (1,0,2)),jnp.transpose(phase, (1,0,2)), jnp.transpose(samples, (1,0))
    #print("probs_original:", probs)
    
    #print("samples:", samples)#
    probs, phase = jnp.take_along_axis(probs, samples[..., jnp.newaxis], axis=-1).squeeze(-1), jnp.take_along_axis(phase, samples[..., jnp.newaxis], axis=-1).squeeze(-1)
    #print("probs_choice:", probs)
    log_probs, phase = jnp.sum(jnp.log(probs), axis=(1)), jnp.sum(phase, axis=(1))
    log_amp = log_probs/2 + phase*1j
   
    return log_amp

            