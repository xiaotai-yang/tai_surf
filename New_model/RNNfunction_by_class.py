

    
    
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
        self.set_params(self.params)
        
    #need to deal with in place assignment of rnn units
    def set_params(self, params):
        self.Winput, self.binput, self.rnn_params, self.Wamp, self.bamp, self.Wphase, self.bphase, self.rnn_states_init_x, self.rnn_states_init_y = params
        
        #separate self.Wrnn to make it into jnp.array form instead of a list
        self.Wrnn1, self.Wrnn2, self.Wrnn3 = self.rnn_params[0][0], self.rnn_params[1][0], self.rnn_params[2][0]
        self.brnn1, self.brnn2, self.brnn3 = self.rnn_params[0][1], self.rnn_params[1][1], self.rnn_params[2][1]
        
        self.inputs_init_x = jnp.repeat(jnp.expand_dims(jnp.array([1.,0.]), axis=0), self.Nx, axis=0)
        self.inputs_init_y = jnp.repeat(jnp.expand_dims(jnp.array([1.,0.]), axis=0), self.Ny, axis=0)
        
        self.scan_rnn_params = self.Winput, self.binput, self.Wrnn1, self.brnn1, self.Wrnn2, self.brnn2, self.Wrnn3, self.brnn3, self.Wamp, self.bamp, self.Wphase, self.bphase
        
    def sample_prob(self, num_samples):
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
            new_state, new_prob, new_phase = batch_rnn(rnn_inputs, rnn_states, params_point) 
            # new_state will be stacked so that it will be the new input of rnn_state_x_1d of the next row 
            rnn_states_yi_1d = new_state
            new_prob = normalization(new_prob , num_up, num_spin, magnetization, num_samples,Ny,Nx)*(mag_fixed)+new_prob*(1-mag_fixed)
            key, subkey = split(key)
            samples_output =  categorical(subkey, jnp.log(new_prob))#sampling
            inputs_yi_1d = one_hot_encoding(samples_output) # one_hot_encoding of the sample
            num_up += 1-samples_output 
            num_spin += 1

            return (rnn_states_x_1d, rnn_states_yi_1d, num_spin, num_up, key, inputs_x_1d, inputs_yi_1d, mag_fixed, magnetization, num_samples, Ny,Nx, params_1d), (samples_output, new_prob, new_state)
        
        @jax.jit
        def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]
            
            rnn_states_x, rnn_states_y, num_spin, num_up, key, inputs_x, inputs_y, mag_fixed, magnetization, num_samples, Ny,Nx, params = carry_2d
          
            index = indices[0,0]
            params_1d = tuple(p[index] for p in params)
            #rnn_states_x and rnn_states_y are of shape [Nx] and [Ny] 
            
            carry_1d = rnn_states_x, rnn_states_y[:,index],  num_spin, num_up, key, inputs_x, inputs_y[:,index], mag_fixed, magnetization, num_samples,Ny,Nx, params_1d
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

    def log_amp(self, samples, params):
        # samples : (num_samples, Ny, Nx)
        
        @jax.jit
        def scan_fun_1d(carry_1d, indices):
            '''
            rnn_state_x_1d, inputs_x_1d : ↓↓↓...↓
            rnn_state_yi_1d, inputs_yi_1d : → or ←
            mag_fixed : To apply U(1) symmetry
            num_1d : count the indices of rnn_state_yi
            num_samples
            params_1d: rnn_parameters on that row
            '''
            ny, nx = indices
            rnn_states_x_1d, rnn_states_yi_1d,  num_spin, num_up, inputs_x_1d, inputs_yi_1d, mag_fixed, magnetization, samples, num_samples,Ny,Nx,  params_1d = carry_1d 
            
            params_point = tuple(p[nx] for p in params_1d)
            rnn_states = jnp.concatenate((rnn_states_yi_1d, rnn_states_x_1d[:,nx]), axis=1)
            rnn_inputs = jnp.concatenate((inputs_yi_1d, inputs_x_1d[:,nx]), axis=1)


            new_state, new_prob, new_phase = batch_rnn(rnn_inputs, rnn_states, params_point) 
            # new_state will be stacked so that it will be the new input of rnn_state_x_1d of the next row 
            rnn_states_yi_1d = new_state
            new_prob = (normalization(new_prob , num_up, num_spin, magnetization, num_samples,Ny,Nx)*(mag_fixed)+new_prob*(1-mag_fixed))

            samples_output =  samples[:, ny, nx]
            inputs_yi_1d = one_hot_encoding(samples_output) # one_hot_encoding of the sample
            num_up += 1-samples_output 
            num_spin += 1
            
            return (rnn_states_x_1d, rnn_states_yi_1d, num_spin, num_up, inputs_x_1d, inputs_yi_1d, mag_fixed, magnetization, samples, num_samples, Ny,Nx, params_1d), (samples_output, new_prob, new_phase, new_state)
        
        @jax.jit
        def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]
            
            rnn_states_x, rnn_states_y, num_spin, num_up, inputs_x, inputs_y, mag_fixed, magnetization, samples, num_samples, Ny, Nx, params = carry_2d
            index = indices[0,0]
            params_1d = tuple(p[index] for p in params)
            carry_1d = rnn_states_x, rnn_states_y[:,index],  num_spin, num_up, inputs_x, inputs_y[:,index], mag_fixed, magnetization, samples, num_samples, Ny, Nx, params_1d
            
            _, y = scan(scan_fun_1d, carry_1d, indices)
            row_samples, row_prob, row_phase, rnn_states_x = y
            rnn_states_x = jnp.transpose(rnn_states_x,(1,0,2))
            rnn_states_x = jnp.flip(rnn_states_x, 1) # reverse the direction of input of for the next line
            inputs_x = one_hot_encoding(row_samples)
            inputs_x = jnp.transpose(inputs_x, (1,0,2))
            inputs_x = jnp.flip(inputs_x, 1)
            
            return (rnn_states_x, rnn_states_y,  num_spin, num_up,  inputs_x, inputs_y, mag_fixed, magnetization, samples, num_samples,Ny,Nx, params), (row_samples, row_prob, row_phase)
        
        #initialization
        
        num_samples = samples.shape[0]
        inputdim = 2
        
        self.batch_rnn_states_init_x = jnp.repeat(jnp.expand_dims(self.rnn_states_init_x, axis=0), num_samples, axis=0)
        self.batch_rnn_states_init_y = jnp.repeat(jnp.expand_dims(self.rnn_states_init_y, axis=0), num_samples, axis=0)        
        self.batch_inputs_init_x = jnp.repeat(jnp.expand_dims(self.inputs_init_x, axis=0), num_samples, axis = 0) 
        self.batch_inputs_init_y = jnp.repeat(jnp.expand_dims(self.inputs_init_x, axis=0), num_samples, axis = 0)
        
        phase = jnp.zeros((num_samples, self.Ny, self.Nx, 2))
        batch_rnn = vmap(rnn_step, (0, 0, None))
        
        init = self.batch_rnn_states_init_x, self.batch_rnn_states_init_y, 0, jnp.zeros(num_samples), self.batch_inputs_init_x, self.batch_inputs_init_y, self.mag_fixed, self.magnetization, samples, num_samples, self.Ny, self.Nx, self.scan_rnn_params

        ny_nx_indices = jnp.array([[(i, j) for i in range(self.Nx)] for j in range(self.Ny)])
        
        __, (samples, probs, phase) = scan(scan_fun_2d, init, ny_nx_indices)
        probs, phase, samples = jnp.transpose(probs, (2,0,1,3)),jnp.transpose(probs, (2,0,1,3)), jnp.transpose(samples, (2,0,1))
        probs, phase = jnp.take_along_axis(probs, samples[..., jnp.newaxis], axis=-1).squeeze(-1), jnp.take_along_axis(phase, samples[..., jnp.newaxis], axis=-1).squeeze(-1)
        log_probs, phase = jnp.sum(jnp.log(probs), axis=(1,2)), jnp.sum(phase, axis=(1,2))  
        log_amp = log_probs/2 + phase*1j
        
        return log_amp
        