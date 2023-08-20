import tensorflow as tf
print("Tensorflow version =",tf.__version__)
import numpy as np
import os
import time
import random
import argparse
import itertools
import RNNwavefunction
from RNNwavefunction import RNNwavefunction
import gc
from tensorflow.python.client import device_lib
import sys

def get_numavailable_gpus():
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])
parser = argparse.ArgumentParser()
parser.add_argument('--system_size', type = int, default=12)
parser.add_argument('--numsamples', type = float, default=512)
parser.add_argument('--rnn_unit', type = float, default=32)
parser.add_argument('--Bx', type = float, default=3)
parser.add_argument('--num_unit', type = int, default=1000)    #this should be equal to num_first_sample if no parallelization is conducted
parser.add_argument('--num_first_sample', type = int, default=1000)
parser.add_argument('--num_second_sample', type = int, default=200)
parser.add_argument('--position_', type = int, default=11)
parser.add_argument('--param', type=int, default = 1)
args = parser.parse_args()



systemsize = args.system_size
Bx = args.Bx
num_unit = args.num_unit
position_ = args.position_
numsamples = args.numsamples
Marshall_sign = True
rnn_unit = args.rnn_unit
num_layers = 1

learningrate = 5*1e-4

seed = args.param
J1_  = 1.0
N = systemsize #Number of spins
lr = np.float64(learningrate)

units = [rnn_unit]
B = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 3.5]
for Bx in (B):
    #Seeding
    tf.reset_default_graph()
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.set_random_seed(seed)  # tensorflow pseudo-random generator


    path=os.getcwd()

    ending='_units'
    for u in units:
        ending+='_{0}'.format(u)

    savename = '_Jz1Bx'+str(Bx)

    filename='Check_Points/RNNwavefunction_N'+str(N)+'_samp'+str(numsamples)+ savename +'_GRURNN_OBC' + ending +'.ckpt'

    '''

    ckpt_meta_path = filename+".ckpt"
    print(os.path.exists(ckpt_meta_path))
    print(os.path.exists("../Check_Points/XXZ/RNNwavefunction_N64_samp128_lradap0.0005_complexGRURNN_J1J20.24_units_64_zeromag.ckpt.meta"))
    ../Check_Points/1DTFIM/RNNwavefunction_N96_samp128_Jz1Bx0.5_GRURNN_OBC_units_256.ckpt
                           RNNwavefunction_N96_samp128_Jz1Bx2.0_GRURNN_OBC_units_128.ckpt.meta
    '''
    #                      RNNwavefunction_N64_samp128_lradap0.0005_complexGRURNN_J1J20.24_units_64_zeromag.ckpt.meta
    #/../Check_Points/J1J2/RNNwavefunction_N64_samp128_lradap0.0005_complexGRURNN_J1J20.24_units_64_zeromag.ckpt

    # Intitializing the RNN-----------

    input_dim=2 #Dimension of the Hilbert space for each site (here = 2, up or down)
    numsamples_=20 #only for initialization; later I'll use a much larger value (see below)


    wf=RNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed) #contains the graph with the RNNs


    '''
     #contains the graph with the RNNs
    sampling=wf.sample(numsamples_,input_dim) #call this function once to create the dense layers
    '''
    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
            if (get_numavailable_gpus()==0):
                device = '/CPU:0'
            else:
                device = '/GPU:0'

            global_step = tf.Variable(0, trainable=False)
            learningrate_placeholder=tf.placeholder(dtype=tf.float64,shape=[])
            learningrate_withexpdecay = tf.train.exponential_decay(learningrate_placeholder, global_step, decay_steps = 100, decay_rate = 1.0, staircase=True) #Adaptive Learning (decay_rate = 1 -> no decay)

            inputs=tf.placeholder(dtype=tf.int32,shape=[None,N]) #the inputs are the samples of all of the spins

            with tf.device(device):

                sample_first_input = tf.stop_gradient(wf.sample(num_unit, inputdim=2))
                log_prob=tf.stop_gradient(wf.log_probability(inputs,inputdim=2))
                cond_prob = tf.stop_gradient(wf.cond_prob(inputs, inputdim =2))

                #cost = 2*tf.real(tf.reduce_mean(tf.conj(log_amplitudes_)*tf.stop_gradient(Eloc)) - tf.conj(tf.reduce_mean(log_amplitudes_))*tf.reduce_mean(tf.stop_gradient(Eloc)))
            optimizer=tf.train.AdamOptimizer(learning_rate=learningrate_withexpdecay, beta1=0.9, beta2 = 0.999, epsilon = 1e-8)
            #gradients, variables = zip(*optimizer.compute_gradients(cost))
            #clipped_gradients = [tf.clip_by_value(g, -10, 10) for g in gradients]
            #optstep=optimizer.apply_gradients(zip(clipped_gradients,variables),global_step=global_step)
            init=tf.global_variables_initializer()
            saver=tf.train.Saver()

    # End Intitializing

    #Starting Session------------
    #Activating GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(graph=wf.graph, config=config)
    sess.run(init)
    #---------------------------

             #define tf saver
    with wf.graph.as_default():
        total_parameters = 0
        for variable in tf.compat.v1.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(shape)
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            print(variable_parameters)
            total_parameters += variable_parameters
        print(total_parameters)

    with tf.compat.v1.variable_scope(wf.scope,reuse=tf.compat.v1.AUTO_REUSE):
        with wf.graph.as_default():
            try:
                print("Loading the model from checkpoint")
                saver.restore(sess,filename)
                print("Loading was successful!")
            except:
                print("Loading was failed!")
                print(filename)
                sys.exit()
                
    # The code starts from here. Above is for loading the model
    def ith_step(wavefun, position, step, L, num_first_sample, num_second_sample, num_unit): 
        # position: sets of 2d coordinate indicating the ith step
        # k: how many steps in the cmi we want to conduct
        # L: the size of the lattice 
        #num_sample_first = sample[:int(sample.shape[0]/1000)]
        #num_sample_second = sample[int(sample.shape[0]/1000):int(sample.shape[0]/1000)*2]
        #num_sample_first_prob = sample_prob_tensor[:int(sample.shape[0]/1000)]
        #num_sample_second_prob = sample_prob_tensor[int(sample_prob_tensor.shape[0]/1000):int(sample.shape[0]/1000)*2] #divide the sample into two 
        if (get_numavailable_gpus()==0):
            device = '/CPU:0'
        else:
            device = '/GPU:0'

        print("device:", device)
        with tf.compat.v1.variable_scope(wavefun.scope,reuse=tf.compat.v1.AUTO_REUSE):
            with wavefun.graph.as_default():

                tv_diff =[]
                tv_first_var = []
                tv_second_var = []
                label = np.zeros((L))
                for dis in range (position+1):        # label B
                    label[dis] = np.abs(position-dis)
                label[position] = -1                  # make the label of C become -1

    # get the coordinate that we need to match
                for step_size in range (0, step+1):
                    output = 0
                    first_var = 0
                    second_var = 0
                    if step_size>=1:
                        pos_temp = np.where(label == step_size) # find the coordinate of B
                        label[pos_temp] = -1                    # make the label of B become -1
                    label_where = np.where(label==-1)           # find the coordinate of with -1 label 
                    print("label_where:", label_where) 
                    print("label_where0:", label_where[0])

                    for j in range (num_first_sample//num_unit):
                        sample_first =  sess.run(sample_first_input)
                        sample_first_prob = sess.run(cond_prob, feed_dict={inputs: sample_first})[:,position]  #first sampling
                        #print("sample_first_prob:", sample_first_prob)
                        print(np.array(label))
                        #print("label_where:", label_where)

                        label_value = sample_first[:, label_where[0]]  #the spin value of -1 label, if the coordinate hits B, replace the value by the original value in B.
                        cond_prob_second = tf.stop_gradient(wf.label_sample(num_first_sample = args.num_unit, num_second_sample = args.num_second_sample, inputdim = 2, label = label_where[0], label_value=label_value, position=position_))
                        print(cond_prob_second)
                        aci_first_prob = np.array(sess.run(cond_prob_second)).reshape(num_unit, num_second_sample)  
                        aci_first_prob_mean = np.mean(aci_first_prob, axis = 1)
                        aci_first_prob_var = np.var(aci_first_prob, axis = 1)
                        print("second_var:", np.mean(aci_first_prob_var))
                        print("first_var:", np.var(2*np.abs(sample_first_prob-aci_first_prob_mean)))
                        output += np.sum(2*np.abs(sample_first_prob - aci_first_prob_mean))/num_first_sample
                        first_var += np.var(2*np.abs(sample_first_prob-aci_first_prob_mean))
                        second_var += np.mean(aci_first_prob_var)

                    first_var/= num_first_sample//num_unit
                    second_var /= num_first_sample//num_unit

                    tv_diff.append(output)
                    tv_first_var.append(first_var)
                    tv_second_var.append(second_var)

            '''
            for i in range(sample_first.shape[0]):
                print("i:", i)
                label_value = sample_first[i][label_where]
                cond_prob_second = tf.stop_gradient(wavefun.label_sample(num_first_sam*100, inputdim = 2, label = label_position, label_value=label_value, position=position))
                aci_first_prob = np.array(sess.run(cond_prob_second))
                aci_first_prob_mean = np.mean(aci_first_prob)
                aci_first_prob_var = np.var(aci_first_prob)
                print("mean:", aci_first_prob_mean)
                print("var:", aci_first_prob_var)
                output += np.abs((sample_first_prob[i]-aci_first_prob_mean)/sample_first.shape[0])
                '''
            return tv_diff, tv_first_var, tv_second_var
        '''
        if choice =="Euclidean":
            return output/sample_first.shape[0], length, label, real_distance
        else:
            return output/sample_first.shape[0], length, label
        '''
    final_diff, final_first_var, final_second_var= ith_step(wf, position_, 11, N, args.num_first_sample, args.num_second_sample,  args.num_unit)
    print(final_diff)

    np.save("TV/"+str(Bx)+"/tv_"+str(Bx)+"position_"+str(position_)+"_param"+str(args.param), np.array(final_diff))
    np.save("TV/"+str(Bx)+"/tv_second_var"+str(Bx)+"position_"+str(position_)+"param"+str(args.param), np.array(final_second_var))
    np.save("TV/"+str(Bx)+"/tv_first_var"+str(Bx)+"position_"+str(position_)+"param"+str(args.param), np.array(final_first_var))



