import tensorflow as tf
import numpy as np
import random
from tensorflow.python.client import device_lib

def get_numavailable_gpus():
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

def phase_softsign(inputs):
    return np.pi*tf.nn.softsign(inputs)

def phase_tanh(inputs):
    return np.pi*tf.nn.tanh(inputs)

def phase_atan(inputs):
    return tf.atan(inputs)

def heavyside(inputs):
    sign = tf.sign(tf.sign(inputs) + 0.1 ) #tf.sign(0) = 0, this is a way to cure that since I want tf.sign(0) = 1
    return 0.5*(sign+1.0)

def regularized_identity(inputs, epsilon = 1e-4):
    sign = tf.sign(tf.sign(inputs) + 0.1 ) #tf.sign(0) = 0, this is a way to cure that since I want tf.sign(0) = 1
    return tf.stop_gradient(sign)*tf.sqrt(inputs**2 + epsilon**2)

class RNNwavefunction(object):
    def __init__(self,systemsize_x, systemsize_y,cell=None,activation=tf.nn.relu,units=[10],scope='RNNwavefunction',seed = 111, mag_fixed = False, magnetization = 0):
        """
            systemsize_x:  int
                         size of x-dim
            systemsize_y: int
                          size of y_dim
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
            scope:       str
                         the name of the name-space scope
            activation:  activation for the RNN cell
            seed:        pseudo random generator
            mag_fixed:   bool to whether fix the magnetization or not
            magnetization: value of magnetization if mag_fixed = True
        """
        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN wavefunction
        self.Nx=systemsize_x #number of sites in the 2d model
        self.Ny=systemsize_y
        self.N = self.Nx*self.Ny
        self.magnetization = magnetization
        self.mag_fixed = mag_fixed

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator
        list_in = [2]+units[:-1]

        #Defining the neural network
        with self.graph.as_default():
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):

              tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator
              self.rnn=cell(num_units = units[0],num_in = list_in[0],name="rnn_"+str(0),dtype=tf.float32)

              self.dense = tf.compat.v1.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense', dtype = tf.float32)
              # self.dense_phase = tf.compat.v1.layers.Dense(2,activation=phase_atan,name='wf_dense_phase', dtype = tf.float32)
              self.dense_phase = tf.compat.v1.layers.Dense(2,activation=phase_softsign,name='wf_dense_phase', dtype = tf.float32)

################################################################################

    def normalization(self, probs, num_up, num_generated_spins, magnetization):
        num_down = num_generated_spins - num_up
        activations_up = heavyside(((self.N+magnetization)//2-1) - num_up)
        activations_down = heavyside(((self.N-magnetization)//2-1) - num_down)

        probs = probs*tf.cast(tf.stack([activations_down,activations_up], axis = 1), tf.float32)
        probs = probs/(tf.reshape(tf.norm(tensor=probs, axis = 1, ord=1), [self.numsamples,1])) #l1 normalizing
        return probs

################################################################################
    
    def sample(self,numsamples,inputdim):
        """
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            Parameters:

            numsamples:      int
                             number of samples to be produced
            inputdim:        int
                             hilbert space dimension

            ------------------------------------------------------------------------
            Returns:         a tensor

            samples:         tf.Tensor of shape (numsamples,systemsize)
                             the samples in integer encoding
        """

        with self.graph.as_default(): #Call the default graph, used if willing to create multiple graphs.
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):

                #Initial input to feed to the lstm

                self.inputdim=inputdim
                self.outputdim=self.inputdim
                self.numsamples=numsamples


                samples=[[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
                rnn_states = {}
                inputs = {}

                for ny in range(-2,self.Ny): #Loop over the number of sites
                    for nx in range(-2,self.Nx+2):
                        rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                        inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32) #Feed the table b in tf.

                num_up = tf.zeros(self.numsamples, dtype = tf.float32)
                num_generated_spins = 0

                #Begin Sampling
                for ny in range(self.Ny):

                    if ny%2 == 0:

                        for nx in range(self.Nx): #left to right

                            local_inputs = [inputs[str(nx-1)+str(ny)],inputs[str(nx)+str(ny-1)]]

                            local_states = [rnn_states[str(nx-1)+str(ny)], rnn_states[str(nx)+str(ny-1)]]

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn(local_inputs, local_states)

                            output=self.dense(rnn_output)

                            if self.mag_fixed:
                                output = self.normalization(output, num_up, num_generated_spins, self.magnetization)

                            with tf.device('/CPU:0'): #necessary otherwise you might get a tensorflow error
                                sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1,])

                            samples[nx][ny] = sample_temp
                            inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float32)

                            num_generated_spins += 1
                            num_up = tf.add(num_up,tf.cast(sample_temp, tf.float32))

                    if ny%2 == 1:

                        for nx in range(self.Nx-1,-1,-1): #right to left

                            local_inputs = [inputs[str(nx+1)+str(ny)],inputs[str(nx)+str(ny-1)]]

                            local_states = [rnn_states[str(nx+1)+str(ny)], rnn_states[str(nx)+str(ny-1)]]

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn(local_inputs, local_states)

                            output=self.dense(rnn_output)
                            if self.mag_fixed:
                                output = self.normalization(output, num_up, num_generated_spins, self.magnetization)

                            with tf.device('/CPU:0'):
                                sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1,])

                            samples[nx][ny] = sample_temp
                            inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float32)

                            num_generated_spins += 1
                            num_up = tf.add(num_up,tf.cast(sample_temp, tf.float32))


        self.samples=tf.transpose(a=tf.stack(values=samples,axis=0), perm = [2,0,1])

        return self.samples
    
    
    def label_sample(self, num_first_sample, num_second_sample, inputdim, label, label_value, position):
        """
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            Parameters:

            numsamples:      int
                             number of samples to be produced
            inputdim:        int
                             hilbert space dimension
            label : The location of |B|
            
            label_value: The value of each point in |B|
            
            position: The position of |A|

            ------------------------------------------------------------------------
            Returns:         a tensor

            samples:         tf.Tensor of shape (numsamples,systemsize)
                             the samples in integer encoding
            cond_prob_A: cond_prob of |A|
        """

        with self.graph.as_default(): #Call the default graph, used if willing to create multiple graphs.
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):

                #Initial input to feed to the lstm

                self.inputdim=inputdim
                self.outputdim=self.inputdim
                self.num_first_sample = num_first_sample
                self.num_second_sample = num_second_sample
                self.numsamples=num_first_sample*num_second_sample


                samples=[[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
                rnn_states = {}
                inputs = {}
                '''
                num_first_sample = label_value.shape[0]
                
                '''
                for ny in range(-2,self.Ny): #Loop over the number of sites
                    for nx in range(-2,self.Nx+2):
                        rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples, dtype=tf.float32)
                        inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,  inputdim), dtype = tf.float32) #Feed the table b in tf.

                num_up = tf.zeros(self.numsamples, dtype = tf.float32)
                num_generated_spins = 0

                #Begin Sampling
                for ny in range(self.Ny):

                    if ny%2 == 0:
                        
                        for nx in range(self.Nx): #left to right
                        
                            local_inputs = [inputs[str(nx-1)+str(ny)],inputs[str(nx)+str(ny-1)]]

                            local_states = [rnn_states[str(nx-1)+str(ny)], rnn_states[str(nx)+str(ny-1)]]

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn(local_inputs, local_states)

                            output=self.dense(rnn_output)
                            
                                              
                            if self.mag_fixed:
                                output = self.normalization(output, num_up, num_generated_spins, self.magnetization)

                            
                            with tf.device('/CPU:0'): #necessary otherwise you might get a tensorflow error
                                sample_temp_ = tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1,])

                                #print("undetected")
                                if len(np.where((np.array(label)==np.array([ny, nx])).all(axis=1))[0]) != 0:   #np.where return a tuple, ask for the zeroth one (the only one) 
                                    print("detected")
                                    sample_temp_ = tf.convert_to_tensor(np.repeat(label_value[:, np.where((np.array(label)==np.array([ny, nx])).all(axis=1))[0]], self.num_second_sample)) # modify here to make label_value an array to assign to each first_sample
                                

                            samples[nx][ny] = sample_temp_
                            inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp_,depth=self.outputdim, dtype = tf.float32)

                            if (ny == position[0] and nx ==position[1]):
                                return tf.reduce_sum(input_tensor=tf.multiply(output, inputs[str(nx)+str(ny)]),axis=1)
                                 
                            
                            num_generated_spins += 1
                            num_up = tf.add(num_up,tf.cast(sample_temp_, tf.float32))

                    if ny%2 == 1:

                        for nx in range(self.Nx-1,-1,-1): #right to left

                            local_inputs = [inputs[str(nx+1)+str(ny)],inputs[str(nx)+str(ny-1)]]

                            local_states = [rnn_states[str(nx+1)+str(ny)], rnn_states[str(nx)+str(ny-1)]]

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn(local_inputs, local_states)

                            output=self.dense(rnn_output)
                                            
                            if self.mag_fixed:
                                output = self.normalization(output, num_up, num_generated_spins, self.magnetization)

                       
                            with tf.device('/CPU:0'): #necessary otherwise you might get a tensorflow error
                                sample_temp_=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1,])
                                if len(np.where((np.array(label)==np.array([ny, nx])).all(axis=1))[0]) != 0:
                                    print("detected")
                                    sample_temp_ = tf.convert_to_tensor(np.repeat(label_value[:, np.where((np.array(label)==np.array([ny, nx])).all(axis=1))[0]], self.num_second_sample))
                                
                            samples[nx][ny] = sample_temp_
                            inputs[str(nx)+str(ny)] = tf.one_hot(sample_temp_,depth=self.outputdim, dtype = tf.float32)
                            
                           
                            if (ny == position[0] and nx ==position[1]):
                                return tf.reduce_sum(input_tensor=tf.multiply(output, inputs[str(nx)+str(ny)]),axis=1)
                                 
                            num_generated_spins += 1
                            num_up = tf.add(num_up,tf.cast(sample_temp_, tf.float32))

                                            
#############################################################################################
                                            
    def cond_label(position , L):
        '''
        Input
        position: 2d coordinate indicating the ith step
        L: the size of the lattice
        -----------------------
        Output
        The label of the lattice for each step (k)
        '''
        label = np.zeros((L, L))
        for i in range (position[0]):
            for j in range (L):
                label[i][j] = np.abs(i-position[0])+np.abs(j-position[1])
        if (position[0]%2 == 0):
            for j in range(position[1]):
                label[position[0]][j] = np.abs(j-position[1])
        else :
            for j in range(position[1], L):
                label[position[0]][j] = np.abs(j-position[1])
        return label

    def binary_to_decimal(tensor):
        ans = 0
        for i in range (tensor.shape[0]):
            ans += 2**i*tensor[i]
        return ans

    def distance_label(A, L):   #B is tuple of array with size (|A|,2), 1st denotes the numbers of  points, 2nd denotes the coordinate.
        A_distance = np.zeros((L,L))*L**2  # denotes the distance between each lattice point and |A|
        dis_label = np.zeros((L,L))

        for i in range (A[0]):
            for j in range (L):
                if ( A_distance[i,j] < np.sqrt((i-A[0])**2 + (j-A[1])**2)):
                    A_distance[i,j] = np.sqrt((i-A[0])**2 + (j-A[1])**2)
        if A[0]%2 == 0:
            for j in range(A[1]):
                if ( A_distance[A[0], j] < np.sqrt((j-A[1])**2)):
                    A_distance[A[0],j] = np.sqrt((j-A[1])**2)
        elif A[0]%2 == 1:
            for j in range(L-A[1]-1):
                if ( A_distance[A[0], -j-1] < np.sqrt((L-j-1-A[1])**2)):
                    A_distance[A[0],-j-1] = np.sqrt((L-j-1-A[1])**2)
        indices = np.squeeze(np.dstack(np.unravel_index(np.argsort(A_distance.ravel()), (L, L)))) 
        flat_indices = np.argsort(A_distance.ravel()) 
        # the 1st indice indicates the closet lattice points to B  
        temp = 0     # denotes the order
        count = []   # denote the number of lattice for each order
        temp_count = 1
        for i in range (L**2):
            if (i!=0):
                if (A_distance[indices[i][0],indices[i][1]]> A_distance[indices[i-1][0],indices[i-1][1]]):
                    temp += 1
                    count.append(temp_count)
                    temp_count = 1
                else :
                    temp_count +=1
            dis_label[indices[i][0],indices[i][1]] = temp
        count.append(temp_count)    
        temp = 0
        marginal_label = []
        for i in range (len(count)):
            temp1 = temp
            temp += count[i]
            marginal_label.append(flat_indices[temp1:temp])

        return dis_label, A_distance   # reverse 


                                                
    def log_amplitude_general(self,samples,inputdim, inference=False):
        """
            calculate the log-amplitudes of ```samples``
            ------------------------------------------------------------------------
            Parameters:

            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,system-size)
                             containing the input samples in integer encoding
            inputdim:        int
                             dimension of the input space

            ------------------------------------------------------------------------
            Returns:
            log-probs, log_phases        tf.Tensor of shape (number of samples,)
                             the log-probability of each sample and the log-phase of each sample
            """
        with self.graph.as_default():

            self.inputdim=inputdim
            self.outputdim=self.inputdim

            self.numsamples=tf.shape(input=samples)[0]

            #Initial input to feed to the lstm
            self.outputdim=self.inputdim

            samples_=tf.transpose(a=samples, perm = [1,2,0])
            rnn_states = {}
            inputs = {}

            for ny in range(-2,self.Ny): #Loop over the number of sites
                for nx in range(-2,self.Nx+2):
                    rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                    inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32) #Feed the table b in tf.

            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                probs = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
                log_phases = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]

                num_up = tf.zeros(self.numsamples, dtype = tf.float32)
                num_generated_spins = 0

                #Begin estimation of log amplitudes
                for ny in range(self.Ny):

                    if ny%2 == 0:

                        for nx in range(self.Nx): #left to right

                            local_inputs = [inputs[str(nx-1)+str(ny)],inputs[str(nx)+str(ny-1)]]

                            local_states = [rnn_states[str(nx-1)+str(ny)], rnn_states[str(nx)+str(ny-1)]]

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn(local_inputs, local_states)

                            output=self.dense(rnn_output)
                            output_phase = self.dense_phase(rnn_output)

                            if self.mag_fixed:
                                probs[nx][ny] = self.normalization(output, num_up, num_generated_spins, self.magnetization)
                            else:
                                probs[nx][ny] = output

                            log_phases[nx][ny] = output_phase

                            inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim,dtype = tf.float32)

                            num_generated_spins += 1
                            num_up = tf.add(num_up,tf.cast(samples_[nx,ny], tf.float32))


                    if ny%2 == 1:

                        for nx in range(self.Nx-1,-1,-1): #right to left

                            local_inputs = [inputs[str(nx+1)+str(ny)],inputs[str(nx)+str(ny-1)]]

                            local_states = [rnn_states[str(nx+1)+str(ny)], rnn_states[str(nx)+str(ny-1)]]

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn(local_inputs, local_states)

                            output=self.dense(rnn_output)
                            output_phase = self.dense_phase(rnn_output)

                            if self.mag_fixed:
                                probs[nx][ny] = self.normalization(output, num_up, num_generated_spins, self.magnetization)
                            else:
                                probs[nx][ny] = output

                            log_phases[nx][ny] = output_phase
                            inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim,dtype = tf.float32)

                            num_generated_spins += 1
                            num_up = tf.add(num_up,tf.cast(samples_[nx,ny], tf.float32))

            probs=tf.transpose(a=tf.stack(values=probs,axis=0),perm=[2,0,1,3])
            log_phases=tf.transpose(a=tf.stack(values=log_phases,axis=0),perm=[2,0,1,3])

            one_hot_samples=tf.one_hot(samples, depth=self.inputdim, dtype = tf.float32)
            cond_probs = tf.reduce_sum(input_tensor=tf.multiply(probs,one_hot_samples),axis=3)
            self.log_probs=tf.reduce_sum(input_tensor=tf.reduce_sum(input_tensor=tf.math.log(cond_probs),axis=2),axis=1)
            
            log_phase_ = tf.reduce_sum(tf.multiply(log_phases, one_hot_samples),axis=3)                                                     
            self.log_phases=tf.reduce_sum(input_tensor=tf.reduce_sum(log_phase_, axis=2),axis=1)  
            
            
            return self.log_probs, self.log_phases, tf.stop_gradient(cond_probs)
        
    def log_amplitude_nosym(self,samples,inputdim):
        numgpus = get_numavailable_gpus()
        
        if numgpus!=0:
            list_probs = [[] for i in range(numgpus)]
            list_phases = [[] for i in range(numgpus)]
            list_cond_prob = [[] for i in range (numgpus)]
            numsamplespergpu = tf.shape(samples)[0]//numgpus
            for i in range(numgpus):
                if (i!= numgpus-1):
                    with tf.device("/GPU:"+str(i)):
                        list_probs[i], list_phases[i], list_cond_prob[i] = self.log_amplitude_general(samples[i*numsamplespergpu:(i+1)*numsamplespergpu],inputdim)
                else:
                    with tf.device("/GPU:"+str(i)):
                        list_probs[i], list_phases[i], list_cond_prob[i] = self.log_amplitude_general(samples[i*numsamplespergpu:],inputdim)
            log_prob_temp = tf.concat(list_probs, 0)
            log_phase_temp = tf.concat(list_phases,0)
            cond_probs = tf.concat(list_cond_prob, 0)
        else:
            log_prob_temp, log_phase_temp, cond_probs = self.log_amplitude_general(samples, inputdim)
        return tf.complex(0.5*log_prob_temp,log_phase_temp), cond_probs
        
    def log_amplitudes_fromsymmetrygroup(self, list_samples, inputdim, group_character_signs):

        with tf.device('/CPU:0'):
            group_cardinal = len(list_samples)
            numsamples = tf.shape(list_samples[0])[0]
            numgpus = get_numavailable_gpus()

            list_probs = [[] for i in range(numgpus)]
            list_phases = [[] for i in range(numgpus)]
            list_cond_prob = [[] for i in range (numgpus)]
            list_samples = tf.reshape(tf.concat(list_samples, 0), [-1, self.Nx, self.Ny])
            numsamplespergpu = tf.shape(list_samples)[0]//numgpus #We assume that is divisible!


        for i in range(numgpus):
            with tf.device("/GPU:"+str(i)):
                log_prob_temp, log_phase_temp, cond_probs = self.log_amplitude_general(list_samples[i*numsamplespergpu:(i+1)*numsamplespergpu],inputdim)
                list_probs[i] = tf.exp(log_prob_temp)
                list_phases[i] = tf.complex(tf.cos(log_phase_temp), tf.sin(log_phase_temp))
                list_cond_prob[i] = cond_probs
        with tf.device('/CPU:0'):
            list_probs = tf.reshape(tf.concat(list_probs, 0), [group_cardinal,numsamples])
            list_phases = tf.reshape(tf.concat(list_phases, 0), [group_cardinal,numsamples])
            signed_phases = [list_phases[i]/group_character_signs[i] for i in range(len(group_character_signs))]
            list_cond_prob = tf.reshape(tf.concat(list_cond_prob, 0),[group_cardinal, numsamples, self.Nx, self.Ny])
            regularized_phase = tf.complex(regularized_identity(tf.math.real(sum(signed_phases)), epsilon = 1e-4),tf.math.imag(sum(signed_phases)))
            return tf.complex(0.5*tf.math.log(tf.reduce_sum(list_probs, axis = 0)/group_cardinal),tf.math.imag(tf.math.log(regularized_phase))), tf.reduce_sum(list_cond_prob, axis=0)/group_cardinal

################ Rot Symmetry ###################################################

    def log_amplitude_rotsym(self,samples,inputdim, group_character):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-1), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-3), [-1,self.Nx, self.Ny]))

            if group_character == "A":
                group_character_signs = [+1, +1, +1, +1]
            if group_character == "B":
                group_character_signs = [+1, -1, +1, -1]

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs)

    def log_amplitude_rotsym_spinparity(self,samples,inputdim, group_character, spinparity_value):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-1), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-3), [-1,self.Nx, self.Ny]))

            list_samples.append(tf.abs(1-samples))
            list_samples.append(tf.abs(1-list_samples[1]))
            list_samples.append(tf.abs(1-list_samples[2]))
            list_samples.append(tf.abs(1-list_samples[3]))

            if group_character == "A":
                group_character_signs = [+1, +1, +1, +1]
            if group_character == "B":
                group_character_signs = [+1, -1, +1, -1]

            group_character_signs_total = np.concatenate((group_character_signs, spinparity_value*group_character_signs), axis = 0)

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs_total)

################ Rotation reduced Symmetry ###################################################
    def log_amplitude_rotreducedsym_spinparity(self,samples,inputdim, group_character, spinparity_value):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-1), [-1,self.Nx, self.Ny]))

            list_samples.append(tf.abs(1-samples))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(tf.abs(1-samples), [-1,self.Nx, self.Ny, 1]),k=-1), [-1,self.Nx, self.Ny]))

            if group_character == "A":
                group_character_signs = [+1, +1]
            if group_character == "B":
                group_character_signs = [+1, -1]

            group_character_signs_total = np.concatenate((group_character_signs, spinparity_value*group_character_signs), axis = 0)

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs_total)

    def log_amplitude_rotreducedsym(self,samples,inputdim, group_character):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-1), [-1,self.Nx, self.Ny]))

            if group_character == "A":
                group_character_signs = [+1, +1]
            if group_character == "B":
                group_character_signs = [+1, -1]

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs)


################# c2v ##################
    def log_amplitude_c2vsym(self,samples,inputdim, group_character):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(samples[:,::-1])
            list_samples.append(samples[:,:,::-1])

            if group_character == "A1":
                group_character_signs = [+1, +1, +1, +1]
            if group_character == "A2":
                group_character_signs = [+1, +1, -1, -1]
            if group_character == "B1":
                group_character_signs = [+1, -1, +1, -1]
            if group_character == "B2":
                group_character_signs = [+1, -1, -1, +1]

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs)

    def log_amplitude_c2v_spinparity(self,samples,inputdim, group_character, spinparity_value):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(samples[:,::-1])
            list_samples.append(samples[:,:,::-1])

            list_samples.append(tf.abs(1-samples))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(tf.abs(1-samples), [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.abs(1-samples[:,::-1]))
            list_samples.append(tf.abs(1-samples[:,:,::-1]))

            if group_character == "A1":
                group_character_signs = [+1, +1, +1, +1]
            if group_character == "A2":
                group_character_signs = [+1, +1, -1, -1]
            if group_character == "B1":
                group_character_signs = [+1, -1, +1, -1]
            if group_character == "B2":
                group_character_signs = [+1, -1, -1, +1]

            group_character_signs_total = np.concatenate((group_character_signs, spinparity_value*group_character_signs), axis = 0)

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs_total)


################# c2 point group #############
    def log_amplitude_c2sym_spinparity(self,samples,inputdim, group_character, spinparity_value):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))

            list_samples.append(tf.abs(1-samples))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(tf.abs(1-samples), [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))

            if group_character == "A":
                group_character_signs = [+1, +1]
            if group_character == "B":
                group_character_signs = [+1, -1]

            group_character_signs_total = np.concatenate((group_character_signs, spinparity_value*group_character_signs), axis = 0)

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs_total)


    def log_amplitude_c2sym(self,samples,inputdim, group_character):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))

            if group_character == "A":
                group_character_signs = [+1, +1]
            if group_character == "B":
                group_character_signs = [+1, -1]

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs)

##################### c2d #########################
    def log_amplitude_c2dsym(self,samples,inputdim, group_character):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.transpose(a=samples, perm = [0,2,1]))
            list_samples.append(tf.transpose(a=list_samples[2], perm = [0,2,1]))

            if group_character == "A1":
                group_character_signs = [+1, +1, +1, +1]
            if group_character == "A2":
                group_character_signs = [+1, +1, -1, -1]
            if group_character == "B1":
                group_character_signs = [+1, -1, +1, -1]
            if group_character == "B2":
                group_character_signs = [+1, -1, -1, +1]

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs)


    def log_amplitude_c2dsym_spinparity(self,samples,inputdim, group_character, spinparity_value):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.transpose(a=samples, perm = [0,2,1]))
            list_samples.append(tf.transpose(a=list_samples[2], perm = [0,2,1]))

            list_samples.append(tf.abs(1-samples))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(tf.abs(1-samples), [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.abs(1-list_samples[2]))
            list_samples.append(tf.abs(1-list_samples[3]))

            if group_character == "A1":
                group_character_signs = [+1, +1, +1, +1]
            if group_character == "A2":
                group_character_signs = [+1, +1, -1, -1]
            if group_character == "B1":
                group_character_signs = [+1, -1, +1, -1]
            if group_character == "B2":
                group_character_signs = [+1, -1, -1, +1]

            group_character_signs_total = np.concatenate((group_character_signs, spinparity_value*group_character_signs), axis = 0)

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs_total)


#C4v##################
    
    def log_amplitude_c4vsym(self,samples,inputdim, group_character):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-1), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-3), [-1,self.Nx, self.Ny]))
            list_samples.append(samples[:,::-1])
            list_samples.append(samples[:,:,::-1])
            list_samples.append(tf.transpose(a=samples, perm = [0,2,1]))
            list_samples.append(tf.transpose(a=list_samples[2], perm = [0,2,1]))

            if group_character == "A1":
                group_character_signs = [+1, +1, +1, +1, +1, +1, +1, +1]
            if group_character == "A2":
                group_character_signs = [+1, +1, +1, +1, -1, -1, -1, -1]
            if group_character == "B1":
                group_character_signs = [+1, -1, +1, -1, +1, +1, -1, -1]
            if group_character == "B2":
                group_character_signs = [+1, -1, +1, -1, -1, -1, +1, +1]

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs)

#### with spin parity projection
    def log_amplitude_c4vsym_spinparity(self,samples,inputdim, group_character, spinparity_value):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-1), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-3), [-1,self.Nx, self.Ny]))
            list_samples.append(samples[:,::-1])
            list_samples.append(samples[:,:,::-1])
            list_samples.append(tf.transpose(a=samples, perm = [0,2,1]))
            list_samples.append(tf.transpose(a=list_samples[2], perm = [0,2,1]))

            list_samples.append(tf.abs(1-samples))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(tf.abs(1-samples), [-1,self.Nx, self.Ny, 1]),k=-1), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(tf.abs(1-samples), [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(tf.abs(1-samples), [-1,self.Nx, self.Ny, 1]),k=-3), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.abs(1-samples)[:,::-1])
            list_samples.append(tf.abs(1-samples)[:,:,::-1])
            list_samples.append(tf.transpose(a=tf.abs(1-samples), perm = [0,2,1]))
            list_samples.append(tf.transpose(a=list_samples[2+8], perm = [0,2,1]))


            if group_character == "A1":
                group_character_signs = np.array([+1, +1, +1, +1, +1, +1, +1, +1])
            if group_character == "A2":
                group_character_signs = np.array([+1, +1, +1, +1, -1, -1, -1, -1])
            if group_character == "B1":
                group_character_signs = np.array([+1, -1, +1, -1, +1, +1, -1, -1])
            if group_character == "B2":
                group_character_signs = np.array([+1, -1, +1, -1, -1, -1, +1, +1])

            group_character_signs_total = np.concatenate((group_character_signs, spinparity_value*group_character_signs), axis = 0)

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs_total)
    
