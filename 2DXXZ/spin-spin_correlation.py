import tensorflow as tf
print("Tensorflow version =",tf.__version__)
import numpy as np
import os
import time
import random
import argparse
from Helper_Functions import *
import itertools
import MDComplexRNN
from MDComplexRNN import RNNwavefunction
from MDTensorizedGRUcell import MDRNNcell
from tensorflow.python.client import device_lib

def get_numavailable_gpus():
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])
import sys

sys.argv = [sys.argv[0]]

parser = argparse.ArgumentParser()
parser.add_argument('--J', type = float, default=0.0)
parser.add_argument('--numgpus', type = int, default=2)
parser.add_argument('--numsamples', type = int, default=128)
parser.add_argument('--rnn_unit', type = int, default=128)
parser.add_argument('--num_first_sample', type = int, default=100)
parser.add_argument('--num_second_sample', type = int, default=100)
parser.add_argument('--num_unit', type = int, default=100)
parser.add_argument('--position_y', type = int, default=8)
parser.add_argument('--position_x', type = int, default=8)
parser.add_argument('--param', type = int, default=1)
args = parser.parse_args()
print(3)
mag_fixed = True
magnetization = 0
units = [args.rnn_unit]
Nx = 10
Ny = 10
testing_sample = 10

numsamples = args.numsamples
J = args.J
numgpus=args.numgpus
RNN_symmetry = "nosym"
spinparity_fixed = False
ending='_units'
group_character ="A1"
for u in units:
    ending+='_{0}'.format(u)


savename = '_numsamples_'+str(numsamples)+'_magfixed'+str(mag_fixed)+'_mag'+str(magnetization)+'_J2'+str(J)+"_symmetry_"+RNN_symmetry+"_num_gpus"+str(numgpus)
backgroundpath = './Check_Points/Size_'+str(Nx)+'x'+ str(Ny)
filename_checkpoint = './Check_Points/Size_'+str(Nx)+'x'+ str(Ny)+'/RNNwavefunction_2DTCRNN_'+str(Nx)+'x'+ str(Ny)+ending+savename+'.ckpt'
print(filename_checkpoint)
'''
print(filename_checkpoint)
RNNwavefunction_2DTCRNN_10x10_units_128_numsamples_128_magfixedTrue_mag0_J20.0_symmetry_nosym_num_gpus2.ckpt.meta
RNNwavefunction_2DTCRNN_10x10_units_128_numsamples_128_magfixedTrue_mag0_J20.0_symmetry_nosym_num_gpus2.ckpt.meta
'''
tf.compat.v1.reset_default_graph()
seed = args.param
random.seed(seed)  # `python` built-in pseudo-random generator
np.random.seed(seed)  # numpy pseudo-random generator
tf.compat.v1.set_random_seed(seed)
wf=RNNwavefunction(Nx, Ny,units=units,cell=MDRNNcell,seed = seed, mag_fixed = mag_fixed, magnetization = magnetization)# tensorflow pseudo-random generator

       
with tf.compat.v1.variable_scope(wf.scope,reuse=tf.compat.v1.AUTO_REUSE):
    with wf.graph.as_default():
        #defining adaptive learning rate
        global_step = tf.Variable(0, trainable=False)
        learningrate_placeholder=tf.compat.v1.placeholder(dtype=tf.float32,shape=[])

        samples_tensor = wf.sample(numsamples=numsamples,inputdim=2)
        inputs=tf.compat.v1.placeholder(dtype=tf.int32,shape=(None,Nx, Ny))

        Eloc=tf.compat.v1.placeholder(dtype=tf.complex64,shape=[numsamples])
        samples_placeholder=tf.compat.v1.placeholder(dtype=tf.int32,shape=[numsamples,Nx,Ny])
        Temperature_placeholder = tf.compat.v1.placeholder(dtype=tf.float32,shape=())
        
        if RNN_symmetry == "c4vsym":
            if spinparity_fixed:
                log_amps, cond_prob=wf.log_amplitude_c4vsym_spinparity(inputs, inputdim=2, group_character = group_character, spinparity_value = spinparity_value)
                log_amps_tensor, cond_prob_tensor=wf.log_amplitude_c4vsym_spinparity(samples_placeholder,inputdim=2, group_character = group_character, spinparity_value = spinparity_value)
            else:
                log_amps, cond_prob=wf.log_amplitude_c4vsym(inputs,inputdim=2, group_character = group_character)
                log_amps_tensor, cond_prob_tensor=wf.log_amplitude_c4vsym(samples_placeholder,inputdim=2, group_character = group_character)
        else :
            log_amps, cond_prob=wf.log_amplitude_nosym(inputs,inputdim=2)
            log_amps_tensor, cond_prob_tensor=wf.log_amplitude_nosym(samples_placeholder,inputdim=2)

        cost = 2*tf.math.real(tf.reduce_mean(input_tensor=tf.multiply(tf.math.conj(log_amps_tensor),tf.stop_gradient(Eloc))) - tf.reduce_mean(input_tensor=tf.math.conj(log_amps_tensor))*tf.reduce_mean(input_tensor=tf.stop_gradient(Eloc))) + 4*Temperature_placeholder*( tf.reduce_mean(tf.math.real(log_amps_tensor)*tf.stop_gradient(tf.math.real(log_amps_tensor))) - tf.reduce_mean(tf.math.real(log_amps_tensor))*tf.reduce_mean(tf.stop_gradient(tf.math.real(log_amps_tensor))) )
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learningrate_placeholder)
        gradients, variables = zip(*optimizer.compute_gradients(cost))
        optstep=optimizer.apply_gradients(zip(gradients,variables),global_step=global_step)
        init=tf.compat.v1.global_variables_initializer()
        saver=tf.compat.v1.train.Saver()


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(graph=wf.graph, config=config)
sess.run(init)
'''
with wf.graph.as_default():
    variables_names =[v.name for v in tf.compat.v1.trainable_variables()]
    sum = 0
    values = sess.run(variables_names)
    for k,v in zip(variables_names, values):
        v1 = tf.reshape(v,[-1])
        print(k,v1.shape)
        sum +=v1.shape[0]
    print('The sum of params is {0}'.format(sum))

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
'''

with tf.compat.v1.variable_scope(wf.scope,reuse=tf.compat.v1.AUTO_REUSE):
    with wf.graph.as_default():

        try:
            print("Loading the model from checkpoint")
            saver.restore(sess,filename_checkpoint)
            print("Loading was successful!")
        except:
            try:
                print("Loading the model from old system size")
                saver.restore(sess,filename_oldsyssize)
                print("Loading was successful!")
            except:
                print("Loading was failed!")

with tf.compat.v1.variable_scope(wf.scope,reuse=tf.compat.v1.AUTO_REUSE):
    with wf.graph.as_default():
        config_sess = tf.stop_gradient(wf.sample(150000, inputdim=2))
        config = sess.run(config_sess)

L = 10
y = 9
x = 4
spin_correlation = np.zeros((L,L))
for i in range(L):
    for j in range(L):
        spin_correlation[i,j] += np.mean((2*config[:,i,j]-1)*(2*config[:,y,x]-1))
        

spin_correlation_matrix = np.zeros((L, L))
count = np.zeros((L, L))
for i in range(L):
    for j in range(L):
        spin_correlation_matrix[np.abs(i-y), np.abs(j-x)] += spin_correlation[i,j] 
        count[np.abs(i), np.abs(j)] += 1

count = np.zeros(15)
spin_correlation_dis = np.zeros(15)
for i in range (L):
    for j in range (L):
        spin_correlation_dis[np.abs(i-y)+np.abs(j-x)]+= spin_correlation[i,j]
        count[np.abs(i-y)+np.abs(j-x)]+=1

log_correlation = np.log(np.abs(spin_correlation_dis/count)) 
np.save("figure/log_correlation_y9x4_J="+str(args.J)+".npy", log_correlation)
import matplotlib.pyplot as plt
plt.scatter(np.arange(15), log_correlation)
plt.ylabel("log_correlation")
plt.xlabel("distance")
plt.title("log_correlation_decay_with J="+str(args.J), family = "serif")
plt.savefig("figure/log_correlation_decay_with J="+str(args.J)+".png", dpi=150)
plt.show()
plt.scatter(np.arange(15), np.exp(log_correlation))
plt.ylabel("correlation")
plt.xlabel("distance")
plt.title("correlation_decay_with J="+str(args.J), family = "serif")
plt.savefig("figure/correlation_decay_with J="+str(args.J)+".png", dpi=150)
plt.show()