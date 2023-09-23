import tensorflow as tf
print("Tensorflow version =",tf.__version__)
import numpy as np
import os
import time
import random
import argparse
import itertools
import ComplexRNNwavefunction
from ComplexRNNwavefunction import RNNwavefunction
import gc
from tensorflow.python.client import device_lib
import sys
sys.argv = [sys.argv[0]]
def get_numavailable_gpus():
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])
parser = argparse.ArgumentParser()
parser.add_argument('--system_size', type = int, default=96)
parser.add_argument('--numsamples', type = float, default=128)
parser.add_argument('--rnn_unit', type = float, default=128)
parser.add_argument('--J2_', type = float, default=0.0)
parser.add_argument('--num_unit', type = int, default=1000)
parser.add_argument('--num_first_sample', type = int, default=1000)
parser.add_argument('--num_second_sample', type = int, default=500)
parser.add_argument('--position_', type = int, default=60)
parser.add_argument('--param', type=int, default = 1)
args = parser.parse_args()



systemsize = args.system_size
J2_ = args.J2_
num_unit = args.num_unit
position_ = args.position_
numsamples = args.numsamples
Marshall_sign = True
rnn_unit = args.rnn_unit
num_layers = 1

learningrate = 5*1e-4

seed = 111
J1_  = 1.0
N = systemsize #Number of spins
lr = np.float64(learningrate)

units = [rnn_unit]

#Seeding
tf.reset_default_graph()
random.seed(seed)  # `python` built-in pseudo-random generator
np.random.seed(seed)  # numpy pseudo-random generator
tf.set_random_seed(seed)  # tensorflow pseudo-random generator


path=os.getcwd()

ending='_units'
for u in units:
    ending+='_{0}'.format(u)

savename = '_J1J2'+str(J2_)

filename='../Check_Points/J1J2/RNNwavefunction_N'+str(N)+'_samp'+str(numsamples)+'_lradap'+str(lr)+'_complexGRURNN'+ savename + ending +'_zeromag.ckpt'

'''
ckpt_meta_path = filename+".ckpt"
print(os.path.exists(ckpt_meta_path))
print(os.path.exists("../Check_Points/J1J2/RNNwavefunction_N64_samp128_lradap0.0005_complexGRURNN_J1J20.24_units_64_zeromag.ckpt.meta"))
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
        learningrate_placeholder=tf.placeholder(dtype=tf.float32,shape=[])
        learningrate_withexpdecay = tf.train.exponential_decay(learningrate_placeholder, global_step, decay_steps = 100, decay_rate = 1.0, staircase=True) #Adaptive Learning (decay_rate = 1 -> no decay)

        inputs=tf.placeholder(dtype=tf.int32,shape=[None,N]) #the inputs are the samples of all of the spins

        with tf.device(device):
                        
            sample_first_input = tf.stop_gradient(wf.sample(num_unit, inputdim=2))
            log_amplitudes_=tf.stop_gradient(wf.log_amplitude(inputs,inputdim=2))
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

with tf.compat.v1.variable_scope(wf.scope,reuse=tf.compat.v1.AUTO_REUSE):
    with wf.graph.as_default():
        config_sess = tf.stop_gradient(wf.sample(150000, inputdim=2))
        config_ = sess.run(config_sess)
L = systemsize
x = 80
spin_correlation = np.zeros(L)
for i in range(L):
    spin_correlation[i] += np.mean((2*config_[:,i]-1)*(2*config_[:,x]-1))
        
spin_correlation_matrix = np.zeros(L)
count = np.zeros(L)
for i in range(L):
    spin_correlation_matrix[np.abs(i-x)] += spin_correlation[i] 
    count[np.abs(i-x)] += 1

correlation = np.abs(spin_correlation_matrix[np.nonzero(spin_correlation_matrix)])


import matplotlib.pyplot as plt
plt.scatter(np.arange(correlation.shape[0]),correlation)
plt.ylabel("correlation")
plt.xlabel("distance")
plt.title("correlation_decay_with J2="+str(args.J2_), family = "serif")
plt.legend()
plt.savefig("figure/correlation_decay_with J2="+str(args.J2_)+".png", dpi=150)
plt.show()

plt.scatter(np.arange(correlation.shape[0]),np.log(correlation))
plt.ylabel("log_correlation")
plt.xlabel("distance")
plt.title("log_correlation_decay_with J2="+str(args.J2_), family = "serif")
plt.legend()
plt.savefig("figure/log_correlation_decay_with J2="+str(args.J2_)+".png", dpi=150)
plt.show()

np.save("figure/correlation_x80_J2"+str(args.J2_), correlation)

