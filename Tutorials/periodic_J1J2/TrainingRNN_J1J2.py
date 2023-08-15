import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #stop displaying tensorflow warnings
import numpy as np
import os
import time
import random
import itertools

from ComplexRNNwavefunction import RNNwavefunction

# Loading Functions --------------------------
def J1J2MatrixElements(J1,J2,Bz,sigmap, sigmaH, matrixelements, periodic = True, Marshall_sign = False):
    """
    -Computes the matrix element of the J1J2 model for a given configuration sigmap
    -We hope to make this function parallel in future versions to return the matrix elements of a large number of configurations
    -----------------------------------------------------------------------------------
    Parameters:
    J1, J2, Bz: np.ndarray of shape (N), (N) and (N), respectively, and dtype=float:
                J1J2 parameters
    sigmap:     np.ndarrray of dtype=int and shape (N)
                spin-state, integer encoded (using 0 for down spin and 1 for up spin)
                A sample of spins can be fed here.
    sigmaH: an array to store the diagonal and the diagonal configurations after applying the Hamiltonian on sigmap.
    matrixelements: an array where to store the matrix elements after applying the Hamiltonian on sigmap.
    periodic: bool, indicate if the chain is periodic on not.
    Marshall_sign: bool, indicate if the Marshall sign is applied or not.
    -----------------------------------------------------------------------------------
    Returns: num, float which indicate the number of diagonal and non-diagonal configurations after applying the Hamiltonian on sigmap
    """
    N=len(Bz)
    #the diagonal part is simply the sum of all Sz-Sz interactions plus a B field
    diag=np.dot(sigmap-0.5,Bz)

    num = 0 #Number of basis elements

    if periodic:
        limit = N
    else:
        limit = N-1

    for site in range(limit):
        if sigmap[site]!=sigmap[(site+1)%N]: #if the two neighouring spins are opposite
            diag-=0.25*J1[site] #add a negative energy contribution
        else:
            diag+=0.25*J1[site]
        if site<(limit-1) and J2[site] != 0.0:
            if sigmap[site]!=sigmap[(site+2)%N]: #if the two second neighouring spins are opposite
                diag-=0.25*J2[site] #add a negative energy contribution
            else:
                diag+=0.25*J2[site]

    matrixelements[num] = diag #add the diagonal part to the matrix elements

    sig = np.copy(sigmap)

    sigmaH[num] = sig

    num += 1

    #off-diagonal part:
    for site in range(limit):
        if J1[site] != 0.0:
          if sigmap[site]!=sigmap[(site+1)%N]:
              sig=np.copy(sigmap)
              sig[site]=sig[(site+1)%N] #Make the two neighbouring spins equal.
              sig[(site+1)%N]=sigmap[site]

              sigmaH[num] = sig #The last fours lines are meant to flip the two neighbouring spins (that the effect of applying J+ and J-)

              if Marshall_sign:
                  matrixelements[num] = -J1[site]/2
              else:
                  matrixelements[num] = +J1[site]/2

              num += 1

    for site in range(limit-1):
      if J2[site] != 0.0:
        if sigmap[site]!=sigmap[(site+2)%N]:
            sig=np.copy(sigmap)
            sig[site]=sig[(site+2)%N] #Make the two neighbouring spins equal.
            sig[(site+2)%N]=sigmap[site]

            sigmaH[num] = sig #The last fours lines are meant to flip the two neighbouring spins (that the effect of applying J+ and J-)
            matrixelements[num] = +J2[site]/2

            num += 1
    return num

def J1J2Slices(J1, J2, Bz, sigmasp, sigmas, H, sigmaH, matrixelements, Marshall_sign):
    """
    Returns: A tuple -The list of slices (that will help to slice the array sigmas)
             -Total number of configurations after applying the Hamiltonian on the list of samples sigmasp (This will be useful later during training, note that it is not constant for J1J2 as opposed to TFIM)
    ----------------------------------------------------------------------------
    Parameters:
    J1, J2, Bz: np.ndarray of shape (N), (N) and (N), respectively, and dtype=float:
                J1J2 parameters
    sigmasp:    np.ndarrray of dtype=int and shape (numsamples,N)
                spin-states, integer encoded (using 0 for down spin and 1 for up spin)
    sigmas: an array to store the diagonal and the diagonal configurations after applying the Hamiltonian on all the samples sigmasp.
    H: an array to store the diagonal and the diagonal matrix elements after applying the Hamiltonian on all the samples sigmasp.
    sigmaH: an array to store the diagonal and the diagonal configurations after applying the Hamiltonian on a single sample.
    matrixelements: an array where to store the matrix elements after applying the Hamiltonian on sigmap on a single sample.
    ----------------------------------------------------------------------------
    """

    slices=[]
    sigmas_length = 0

    for n in range(sigmasp.shape[0]):
        sigmap=sigmasp[n,:]
        num = J1J2MatrixElements(J1,J2,Bz,sigmap, sigmaH, matrixelements, Marshall_sign)#note that sigmas[0,:]==sigmap, matrixelements and sigmaH are updated
        slices.append(slice(sigmas_length,sigmas_length + num))
        s = slices[n]

        H[s] = matrixelements[:num]
        sigmas[s] = sigmaH[:num]

        sigmas_length += num #Increasing the length of matrix elements sigmas

    return slices, sigmas_length
#--------------------------

# ---------------- Running VMC with RNNs for J1J2 Model -------------------------------------
def run_J1J2(numsteps = 10**5, systemsize = 20, J1_  = 1.0, J2_ = 0.0, Marshall_sign = False, num_units = 50, num_layers = 1, numsamples = 500, learningrate = 2.5*1e-4, seed = 111):

    N=systemsize #Number of spins
    lr = np.float64(learningrate)

    #Seeding
    tf.reset_default_graph()
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.set_random_seed(seed)  # tensorflow pseudo-random generator


    # Intitializing the RNN-----------
    units=[num_units]*num_layers #list containing the number of hidden units for each layer of the networks

    input_dim=2 #Dimension of the Hilbert space for each site (here = 2, up or down)
    numsamples_=20 #only for initialization; later I'll use a much larger value (see below)
    wf=RNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed) #contains the graph with the RNNs
     #contains the graph with the RNNs
    sampling=wf.sample(numsamples_,input_dim) #call this function once to create the dense layers


    with wf.graph.as_default(): #now initialize everything
        inputs=tf.placeholder(dtype=tf.int32,shape=[numsamples_,N]) #the inputs are the samples of all of the spins
        #defining adaptive learning rate
        global_step = tf.Variable(0, trainable=False)
        learningrate_placeholder=tf.placeholder(dtype=tf.float32,shape=[])
        learningrate_withexpdecay = tf.train.exponential_decay(learningrate_placeholder, global_step, decay_steps = 100, decay_rate = 1.0, staircase=True) #Adaptive Learning (decay_rate = 1 -> no decay)
        amplitudes=tf.reduce_sum(tf.log(wf.log_amplitude(inputs,input_dim)), axis=1) #The probs are obtained by feeding the sample of spins.
        optimizer=tf.train.AdamOptimizer(learning_rate=learningrate_withexpdecay, beta1=0.9, beta2 = 0.999, epsilon = 1e-8)
        init=tf.global_variables_initializer()
    # End Intitializing

    #Starting Session------------
    #Activating GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess=tf.Session(graph=wf.graph, config=config)
    sess.run(init)
    #---------------------------

    #Counting the number of parameters
    with wf.graph.as_default():
        variables_names =[v.name for v in tf.trainable_variables()]
    #     print(variables_names)
        sum = 0
        values = sess.run(variables_names)
        for k,v in zip(variables_names, values):
            v1 = tf.reshape(v,[-1])
            # print(k,v1.shape)
            sum +=v1.shape[0]
        print('The number of variational parameters of the cRNN wavefunction is {0}'.format(sum))
        print('\n')

    #Running the training -------------------

    path=os.getcwd()

    J1=+J1_*np.ones(N)
    J2=+J2_*np.ones(N)

    Bz=+0.0*np.ones(N)

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
            Eloc=tf.placeholder(dtype=tf.complex64,shape=[numsamples])
            samp=tf.placeholder(dtype=tf.int32,shape=[numsamples,N])
            log_amplitudes_=tf.reduce_sum(tf.log(wf.log_amplitude(samp,inputdim=2)), axis =1 )

            #now calculate the fake cost function: https://stackoverflow.com/questions/33727935/how-to-use-stop-gradient-in-tensorflow
            cost = 2*tf.real(tf.reduce_mean(tf.conj(log_amplitudes_)*tf.stop_gradient(Eloc)) - tf.conj(tf.reduce_mean(log_amplitudes_))*tf.reduce_mean(tf.stop_gradient(Eloc)))
            #Calculate Gradients---------------

            gradients, variables = zip(*optimizer.compute_gradients(cost))

            #End calculate Gradients---------------

            optstep=optimizer.apply_gradients(zip(gradients,variables),global_step=global_step)
            sess.run(tf.variables_initializer(optimizer.variables()),feed_dict={learningrate_placeholder: lr})

            saver=tf.train.Saver() #define tf saver

    meanEnergy=[]
    varEnergy=[]


    #Running The training

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
          # max_grad = tf.reduce_max(tf.abs(gradients[0]))

          samples_ = wf.sample(numsamples=numsamples,inputdim=2)
          samples = np.ones((numsamples, N), dtype=np.int32)

          inputs=tf.placeholder(dtype=tf.int32,shape=(None,N))
          log_amps=wf.log_amplitude(inputs,inputdim=2)

          local_energies = np.zeros(numsamples, dtype = np.complex64) #The type complex should be specified, otherwise the imaginary part will be discarded

          sigmas=np.zeros((2*N*numsamples,N), dtype=np.int32) #Array to store all the diagonal and non diagonal sigmas for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)
          H = np.zeros(2*N*numsamples, dtype=np.float32) #Array to store all the diagonal and non diagonal matrix elements for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)
          log_amplitudes = np.zeros(2*N*numsamples, dtype=np.complex64) #Array to store all the diagonal and non diagonal log_probabilities for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)

          sigmaH = np.zeros((2*N,N), dtype = np.int32) #Array to store all the diagonal and non diagonal sigmas for each sample sigma
          matrixelements=np.zeros(2*N, dtype = np.float32) #Array to store all the diagonal and non diagonal matrix elements for each sample sigma (the number of matrix elements is bounded by at most 2N)

          for it in range(len(meanEnergy),numsteps+1):

              samples=sess.run(samples_)

              #Getting the sigmas with the matrix elements
              slices, len_sigmas = J1J2Slices(J1,J2,Bz,samples, sigmas, H, sigmaH, matrixelements, Marshall_sign)

              #Process in steps to get log amplitudes
              # print("Generating log amplitudes Started")
              start = time.time()
              steps = len_sigmas//30000+1 #Process the sigmas in steps to avoid allocating too much memory

              # print("number of required steps :" + str(steps))

              for i in range(steps):
                if i < steps-1:
                    cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
                else:
                    cut = slice((i*len_sigmas)//steps,len_sigmas)

                log_amplitudes[cut] = np.sum(np.log(sess.run(log_amps,feed_dict={inputs:sigmas[cut]})), axis=1)
                # print(i+1)

              end = time.time()
              # print("Generating log amplitudes ended "+ str(end - start))

              #Generating the local energies
              for n in range(len(slices)):
                s=slices[n]
                local_energies[n] = H[s].dot(np.exp(log_amplitudes[s]-log_amplitudes[s][0]))

              meanE = np.mean(local_energies)
              varE = np.var(np.real(local_energies))

              #adding elements to be saved
              meanEnergy.append(meanE)
              varEnergy.append(varE)

              if it%100 == 0:
                print('mean(E): {0}, var(E): {1}, #samples {2}, #Step {3} \n\n'.format(meanE,varE,numsamples, it))

              lr_ = 1/((1/lr)+(it/10)) #learning rate decay
              sess.run(optstep,feed_dict={Eloc:local_energies,samp:samples,learningrate_placeholder: lr_})
          basis = np.array(list(itertools.product([0, 1], repeat=N)))
        
          amps_all_basis = sess.run(log_amps, feed_dict={inputs: basis})
          np.save('amps_all_basis.npy', amps_all_basis)
          print(amps_all_basis.shape)
    return meanEnergy, varEnergy

    #----------------------------------------
