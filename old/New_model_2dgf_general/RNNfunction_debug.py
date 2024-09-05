import netket as nk
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

def random_layer_params(ny, nx, m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    #outkey1, outkey2 = random.split(w_key)
    return  random.normal(w_key, (ny, nx, m, n)),  random.normal(b_key, (ny, nx, m))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, ny ,nx,  key):
    keys = random.split(key, len(sizes))
    outkey = keys[0]
    return outkey, [random_layer_params(ny, nx, m, n, k) for m, n, k in zip(sizes[1:], sizes[:-1], keys[1:])]

def one_hot_encoding(x, num_classes = 2):
    """Converts batched integer labels to one-hot encoded arrays."""
    return jnp.eye(num_classes)[x]
def sample_discrete(key, probabilities, size=None):
    """Sample from a discrete distribution defined by probabilities."""
    logits = jnp.log(probabilities)
    return categorical(key, logits, shape=size)

def normalization(probs, num_up, num_generated_spins, magnetization, num_samples, Ny, Nx):
    num_down = num_generated_spins - num_up
    activations_up = heavyside(((Ny*Nx+magnetization)//2-1) - num_up)
    activations_down = heavyside(((Ny*Nx-magnetization)//2-1) - num_down)
    probs_ = probs*jnp.stack([activations_down,activations_up], axis = 1).astype(jnp.float32)
    probs__ = probs_/(jnp.expand_dims(jnp.linalg.norm(probs_, axis=1, ord=1), axis = 1)) #l1 normalizing
    
    return probs__  

def init_params(Nx, Ny, units, input_size, key):
    hidden_size = 5*units
    key, Winput_params = init_network_params([ 2*input_size, units], Ny, Nx, key) #augment input dimension
    
    Winput, binput = Winput_params[0][0],  Winput_params[0][1]
    key, rnn_params = init_network_params([5*units, hidden_size, hidden_size, 2*units], Ny, Nx, key)
      # 2*units+augmen_input → hidden_layer → hidden layer → 2*rnn_state

    key, amp_params = init_network_params([2*units ,input_size], Ny, Nx, key)
 
    Wamp, bamp = amp_params[0][0], amp_params[0][1]
    
    key, phase_params = init_network_params([2*units,input_size ], Ny, Nx, key)
    Wphase, bphase = phase_params[0][0], phase_params[0][1]                # 2*units → output(amplitude)
    rnn_states_init_x = random.normal(random.split(key)[0], (Nx, 2*units)) #states at vertical direction
    rnn_states_init_y = random.normal(random.split(key)[1], (Ny, 2*units))
    
    return Winput, binput, rnn_params, Wamp, bamp, Wphase, bphase, rnn_states_init_x, rnn_states_init_y

def rnn_step(local_inputs, local_states, params):   #local_input is already concantenated
        
        Winput, binput, Wrnn1, brnn1, Wrnn2, brnn2, Wrnn3, brnn3, Wamp, bamp, Wphase, bphase = params
        
        encode_input = nn.relu(jnp.dot(Winput, local_inputs)+binput)  
        layer1 = nn.relu(jnp.dot(Wrnn1, jnp.concatenate((encode_input, local_states)))+brnn1)
        layer2 = nn.relu(jnp.dot(Wrnn2, layer1)+brnn2)
        new_state = jnp.arcsinh(jnp.dot(Wrnn3, layer2)+brnn3)
        prob = nn.softmax(jnp.dot(Wamp, new_state)+bamp)
        phase = jnp.pi*nn.soft_sign(jnp.dot(Wphase, new_state)+bphase)
              
        return new_state, prob, phase
    
class RNNwavefunction(object):
    def __init__(self, systemsize_x, systemsize_y, units=10, mag_fixed = False, magnetization = 0):
        self.key = PRNGKey(0)
        self.Nx=systemsize_x  #number of sites in the 2d model
        self.Ny=systemsize_y
        self.N = self.Nx*self.Ny
        self.magnetization = magnetization
        self.mag_fixed = mag_fixed
        self.input_size = 2
        self.units = units
        self.params = init_params(self.Nx, self.Ny, self.units, self.input_size, self.key)
        
        self.Winput, self.binput, self.rnn_params, self.Wamp, self.bamp, self.Wphase, self.bphase, self.rnn_states_init_x, self.rnn_states_init_y = self.params
        
        #separate self.Wrnn to make it into jnp.array form instead of a list
        self.Wrnn1, self.Wrnn2, self.Wrnn3 = self.rnn_params[0][0], self.rnn_params[1][0], self.rnn_params[2][0]
        self.brnn1, self.brnn2, self.brnn3 = self.rnn_params[0][1], self.rnn_params[1][1], self.rnn_params[2][1]
        
        self.inputs_init_x = jnp.repeat(jnp.expand_dims(jnp.array([1.,0.]), axis=0), self.Nx, axis=0)
        self.inputs_init_y = jnp.repeat(jnp.expand_dims(jnp.array([1.,0.]), axis=0), self.Ny, axis=0)
        self.scan_rnn_params = self.Winput, self.binput, self.Wrnn1, self.brnn1, self.Wrnn2, self.brnn2, self.Wrnn3, self.brnn3, self.Wamp, self.bamp, self.Wphase, self.bphase
    #need to deal with in place assignment of rnn units
    def sample_amp(self, num_samples):
        '''
        for ny in range(-1,self.Ny+1): #Loop over the number of sites (# need modification)
            for nx in range(-1,self.Nx+1):
                rnn_states[str(ny)+str(nx)]= self.rnn.zero_state(numsamples, dtype=tf.float32)
                inputs[str(ny)+str(nx)] = jnp.zeros((self.num_samples, inputdim)) #Feed the table b in tf.
        '''
        @jax.jit
        def scan_fun_1d(carry_1d, indices):
            '''
            rnn_state_x_1d, inputs_x_1d : ↓↓↓...↓
            rnn_state_yi_1d, inputs_yi_1d : → or ←
            mag_fixed : To apply U(1) symmetry
            num_1d : count the indices of rnn_state_yi
            key : for random number generation
            num_samples
            params_1d: rnn_parameters on that row
            '''
            ny, nx = indices
            rnn_states_x_1d, rnn_states_yi_1d,  num_spin, num_up, key, inputs_x_1d, inputs_yi_1d, mag_fixed, magnetization, num_samples,Ny,Nx, params_1d = carry_1d 
            
            params_point = tuple(p[nx] for p in params_1d)
            rnn_states = jnp.concatenate((rnn_states_yi_1d, rnn_states_x_1d[:,nx]), axis=1)
            rnn_inputs = jnp.concatenate((inputs_yi_1d, inputs_x_1d[:,nx]), axis=1)
            #print("rnn_states.shape:",rnn_states.shape)
            #print("rnn_inputs.shape:",rnn_inputs.shape)
            #print("Winput_1d[nx].shape:",Winput_1d[nx].shape)
            new_state, new_prob, new_phase = batch_rnn(rnn_inputs, rnn_states, params_point) 
            # new_state will be stacked so that it will be the new input of rnn_state_x_1d of the next row 
            rnn_states_yi_1d = new_state
            #print("new_prob.shape:",new_prob.shape)
            #print("num_up.shape:", num_up.shape)
            new_prob = normalization(new_prob , num_up, num_spin, magnetization, num_samples,Ny,Nx)*(mag_fixed)+new_prob*(1-mag_fixed)
            key, subkey = split(key)
            samples_output =  categorical(subkey, jnp.log(new_prob))#sampling
            #print("samples_output_shape", samples_output.shape)
            inputs_yi_1d = one_hot_encoding(samples_output) # one_hot_encoding of the sample
            #print("inputs_yi_1d_shape:",inputs_yi_1d.shape)
            #print("inputs_yi_1d:",inputs_yi_1d)
            num_up += 1-samples_output 
            num_spin += 1
            print("num_spin", num_spin)
            return (rnn_states_x_1d, rnn_states_yi_1d, num_spin, num_up, key, inputs_x_1d, inputs_yi_1d, mag_fixed, magnetization, num_samples, Ny,Nx, params_1d), (samples_output, new_prob, new_state)
        
        @jax.jit
        def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]
            
            rnn_states_x, rnn_states_y, num_spin, num_up, key, inputs_x, inputs_y, mag_fixed, magnetization, num_samples, Ny,Nx, params = carry_2d
          
            index = indices[0,0]
            params_1d = tuple(p[index] for p in params)
            #rnn_states_x and rnn_states_y are of shape [Nx] and [Ny] 
            
            carry_1d = rnn_states_x, rnn_states_y[:,index],  num_spin, num_up, key, inputs_x, inputs_y[:,index], mag_fixed, magnetization, num_samples,Ny,Nx, params_1d
            print("num_up.shape:", num_up.shape)
            _, y = scan(scan_fun_1d, carry_1d, indices)
            
            row_samples, row_prob, rnn_states_x = y
            rnn_states_x = jnp.transpose(rnn_states_x,(1,0,2))
            rnn_states_x = jnp.flip(rnn_states_x, 1) # reverse the direction of input of for the next line
            inputs_x = one_hot_encoding(row_samples)
            inputs_x = jnp.transpose(inputs_x, (1,0,2))
            inputs_x = jnp.flip(inputs_x, 1)
            
            return (rnn_states_x, rnn_states_y,  num_spin, num_up, key, inputs_x, inputs_y, mag_fixed, magnetization, num_samples,Ny, Nx, params), (row_samples, row_prob)
        
        
        #initialization

        inputdim = 2
        
        self.batch_rnn_states_init_x = jnp.repeat(jnp.expand_dims(self.rnn_states_init_x, axis=0), num_samples, axis=0)
        self.batch_rnn_states_init_y = jnp.repeat(jnp.expand_dims(self.rnn_states_init_y, axis=0), num_samples, axis=0)
        
        self.batch_inputs_init_x = jnp.repeat(jnp.expand_dims(self.inputs_init_x, axis=0), num_samples, axis = 0) # the real sample is at [1:-1, 1:-1]
        self.batch_inputs_init_y = jnp.repeat(jnp.expand_dims(self.inputs_init_x, axis=0), num_samples, axis = 0)
        
        phase = jnp.zeros((num_samples, self.Ny, self.Nx))
        batch_rnn = vmap(rnn_step, (0, 0, None))
        
        init = self.batch_rnn_states_init_x, self.batch_rnn_states_init_y, 0, jnp.zeros(num_samples), PRNGKey(3), self.batch_inputs_init_x, self.batch_inputs_init_y, self.mag_fixed, self.magnetization, num_samples, self.Ny, self.Nx, self.scan_rnn_params
        
        ny_nx_indices = jnp.array([[(i, j) for i in range(self.Nx)] for j in range(self.Ny)])
        
        __, (samples, probs) = scan(scan_fun_2d, init, ny_nx_indices)
        
        probs, samples = jnp.transpose(probs, (2,0,1,3)), jnp.transpose(samples, (2,0,1))
        probs = jnp.take_along_axis(probs, samples[..., jnp.newaxis], axis=-1).squeeze(-1)
        probs = jnp.prod(probs, axis=(1,2))  
        
        return samples, probs
    

###########################################################################################################3             
    def log_amp(self, samples, get_samples):
        # samples : (num_samples, Ny, Nx)
        
        @jax.jit
        def scan_fun_1d(carry_1d, indices):
            '''
            rnn_state_x_1d, inputs_x_1d : ↓↓↓...↓
            rnn_state_yi_1d, inputs_yi_1d : → or ←
            mag_fixed : To apply U(1) symmetry
            num_1d : count the indices of rnn_state_yi
            key : for random number generation
            num_samples
            params_1d: rnn_parameters on that row
            '''
            ny, nx = indices
            rnn_states_x_1d, rnn_states_yi_1d,  num_spin, num_up, key, inputs_x_1d, inputs_yi_1d, mag_fixed, magnetization, samples, num_samples,Ny,Nx,  params_1d = carry_1d 
            
            params_point = tuple(p[nx] for p in params_1d)
            rnn_states = jnp.concatenate((rnn_states_yi_1d, rnn_states_x_1d[:,nx]), axis=1)
            rnn_inputs = jnp.concatenate((inputs_yi_1d, inputs_x_1d[:,nx]), axis=1)
            print("rnn_states.shape:",rnn_states.shape)
            print("rnn_inputs.shape:",rnn_inputs.shape)

            new_state, new_prob, new_phase = batch_rnn(rnn_inputs, rnn_states, params_point) 
            # new_state will be stacked so that it will be the new input of rnn_state_x_1d of the next row 
            rnn_states_yi_1d = new_state
           
            print("num_up.shape:", num_up.shape)
            new_prob = (normalization(new_prob , num_up, num_spin, magnetization, num_samples,Ny,Nx)*(mag_fixed)+new_prob*(1-mag_fixed))
            key, subkey = split(key)
            samples_output =  samples[:, ny, nx]
            print("new_prob.shape:",new_prob.shape)
            print("samples_output_shape", samples_output.shape)
            inputs_yi_1d = one_hot_encoding(samples_output) # one_hot_encoding of the sample
            print("inputs_yi_1d_shape:",inputs_yi_1d.shape)
            print("inputs_yi_1d:",inputs_yi_1d)
            num_up += 1-samples_output 
            num_spin += 1
            
            return (rnn_states_x_1d, rnn_states_yi_1d, num_spin, num_up, key, inputs_x_1d, inputs_yi_1d, mag_fixed, magnetization, samples, num_samples, Ny,Nx, params_1d), (samples_output, new_prob, new_state)
        
        @jax.jit
        def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]
            
            rnn_states_x, rnn_states_y, num_spin, num_up, key, inputs_x, inputs_y, mag_fixed, magnetization, samples, num_samples, Ny, Nx, params = carry_2d
            index = indices[0,0]
            params_1d = tuple(p[index] for p in params)
            carry_1d = rnn_states_x, rnn_states_y[:,index],  num_spin, num_up, key, inputs_x, inputs_y[:,index], mag_fixed, magnetization, samples, num_samples, Ny, Nx, params_1d
            
            _, y = scan(scan_fun_1d, carry_1d, indices)
            
            row_samples, row_prob, rnn_states_x = y
            #print("row_samples:", row_samples)
            rnn_states_x = jnp.transpose(rnn_states_x,(1,0,2))
            rnn_states_x = jnp.flip(rnn_states_x, 1) # reverse the direction of input of for the next line
            inputs_x = one_hot_encoding(row_samples)
            inputs_x = jnp.transpose(inputs_x, (1,0,2))
            inputs_x = jnp.flip(inputs_x, 1)
            
            return (rnn_states_x, rnn_states_y,  num_spin, num_up, key, inputs_x, inputs_y, mag_fixed, magnetization, samples, num_samples,Ny,Nx, params), (row_samples, row_prob)
        
        #initialization
        
        num_samples = samples.shape[0]
        inputdim = 2
        
        self.batch_rnn_states_init_x = jnp.repeat(jnp.expand_dims(self.rnn_states_init_x, axis=0), num_samples, axis=0)
        self.batch_rnn_states_init_y = jnp.repeat(jnp.expand_dims(self.rnn_states_init_y, axis=0), num_samples, axis=0)
        
        self.batch_inputs_init_x = jnp.repeat(jnp.expand_dims(self.inputs_init_x, axis=0), num_samples, axis = 0) #Important the real sample is at [1:-1, 1:-1]
        self.batch_inputs_init_y = jnp.repeat(jnp.expand_dims(self.inputs_init_x, axis=0), num_samples, axis = 0)
        
        phase = jnp.zeros((num_samples, self.Ny, self.Nx))
        batch_rnn = vmap(rnn_step, (0, 0, None))
        
        init = self.batch_rnn_states_init_x, self.batch_rnn_states_init_y, 0, jnp.zeros(num_samples), PRNGKey(3), self.batch_inputs_init_x, self.batch_inputs_init_y, self.mag_fixed, self.magnetization, samples, num_samples, self.Ny, self.Nx, self.scan_rnn_params

        ny_nx_indices = jnp.array([[(i, j) for i in range(self.Nx)] for j in range(self.Ny)])
        
        __, (samples, probs) = scan(scan_fun_2d, init, ny_nx_indices)
        probs, samples = jnp.transpose(probs, (2,0,1,3)), jnp.transpose(samples, (2,0,1))
        probs = jnp.take_along_axis(probs, samples[..., jnp.newaxis], axis=-1).squeeze(-1)
        probs = jnp.prod(probs, axis=(1,2))     
        if (get_samples == True):
            return samples, probs
        else:
            return probs
        
        
        ''' 
        sample_input = one_hot_encoding(samples) 
        batch_amp_rnn = vmap(self.rnn_step,((0,1,2),(0,1,2), None))
        params = (Winput, binput, Wrnn1, brnn1, Wrnn2, brnn2, Wrnn3, brnn3, Wamp, bamp, Wphase, bphase)

        def scan_ini_1d(carry_1d, indices):
            ny, nx = indices
            rnn_states_x_1d, rnn_states_yi_1d, num_1d, inputs_x_1d, inputs_yi_1d, num_samples = carry_1d
            num_1d += 1

            return jnp.concatenate(rnn_states_x_1d), jnp.concatenate(rnn_states_x_1d, )

        def scan_ini_2d(carry_2d, indices):
            rnn_states_x, rnn_states_y, num_2d, num_1d, inputs_x, inputs_y, num_samples = carry_2d
            carry_1d = rnn_states_x, rnn_states_y[num_2d], num_1d, inputs_x, inputs_y[num_2d] 

            _, (rnn_states_x, inputs_x) = scan(scan_ini_1d, carry_1d, indices)
            rnn_states_x = jnp.flip(rnn_states_x, 1)
            inputs_x = jnp.flip(inputs_x, 1)
            num_2d += 1
            num_1d = 0

            return _, (rnn_states_x, inputs_x)

        init = self.batch_rnn_states_init_x, self.batch_rnn_states_init_y, 0, 0, inputs_x, inputs_y, self.num_samples,
        ny_nx_indices = [[(i, j) for i in range(self.Nx)] for j in range(self.Ny)] 
        
        _ , (global_states, global_inputs) = scan(scan_ini_2d, init, ny_nx_indices)

        rnn_states, cond_prob, phase= batch_amp_rnn(global_inputs, global_states, params)
        log_prob = jnp.sum(jnp.log(cond_prob), axis = (1,2))

        return log_prob
        '''