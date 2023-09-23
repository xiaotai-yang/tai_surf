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
import sys
def get_numavailable_gpus():
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

parser = argparse.ArgumentParser()
parser.add_argument('--J', type = float, default=0.0)
parser.add_argument('--num_unit', type = int, default=50)
parser.add_argument('--numsamples', type = int, default=128)
parser.add_argument('--num_first_sample', type = int, default=100)
parser.add_argument('--num_second_sample', type = int, default=200)
parser.add_argument('--position_y', type = int, default=4)
parser.add_argument('--position_x', type = int, default=6)
parser.add_argument('--param', type=int, default = 1)
parser.add_argument('--numgpus', type=int, default = 2)

args = parser.parse_args()
J = args.J
position_ = [args.position_y, args.position_x]

mag_fixed = True
magnetization = 0
units = [128]
Nx = 10
Ny = 10
testing_sample = 10

numsamples = args.numsamples

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
seed = 111
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

        
        samples_placeholder=tf.compat.v1.placeholder(dtype=tf.int32,shape=[numsamples,Nx,Ny])
        
        
        if RNN_symmetry == "c4vsym":
            if spinparity_fixed:
                log_amps, cond_prob=wf.log_amplitude_c4vsym_spinparity(inputs, inputdim=2, group_character = group_character, spinparity_value = spinparity_value)
                log_amps_tensor, cond_prob_tensor=wf.log_amplitude_c4vsym_spinparity(samples_placeholder,inputdim=2, group_character = group_character, spinparity_value = spinparity_value)
            else:
                log_amps, cond_prob=wf.log_amplitude_c4vsym(inputs,inputdim=2, group_character = group_character)
                log_amps_tensor, cond_prob_tensor=wf.log_amplitude_c4vsym(samples_placeholder,inputdim=2, group_character = group_character)
        else :
            log_amps, cond_prob=wf.log_amplitude_nosym(inputs,inputdim=2)
            #log_amps_tensor, cond_prob_tensor=wf.log_amplitude_nosym(samples_placeholder,inputdim=2)

        #cost = 2*tf.math.real(tf.reduce_mean(input_tensor=tf.multiply(tf.math.conj(log_amps_tensor),tf.stop_gradient(Eloc))) - tf.reduce_mean(input_tensor=tf.math.conj(log_amps_tensor))*tf.reduce_mean(input_tensor=tf.stop_gradient(Eloc))) + 4*Temperature_placeholder*( tf.reduce_mean(tf.math.real(log_amps_tensor)*tf.stop_gradient(tf.math.real(log_amps_tensor))) - tf.reduce_mean(tf.math.real(log_amps_tensor))*tf.reduce_mean(tf.stop_gradient(tf.math.real(log_amps_tensor))) )
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learningrate_placeholder)
        #gradients, variables = zip(*optimizer.compute_gradients(cost))
        #optstep=optimizer.apply_gradients(zip(gradients,variables),global_step=global_step)
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
'''
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
            saver.restore(sess,filename_checkpoint)
            print("Loading was successful!")
        except:
            try:
                print("Loading the model from old system size")
                saver.restore(sess,filename_oldsyssize)
                print("Loading was successful!")
            except:
                print("Loading was failed!")
                sys.exit()
    '''
        try:
            print("Trying to load energies!")

            meanEnergy=np.loadtxt(backgroundpath + '/meanEnergy_2DTGCRNN_'+str(Nx)+'x'+ str(Ny) +ending + savename +'.txt', converters={0: lambda s: complex(s.decode().replace('+-', '-'))}, dtype = complex).tolist()
            print(meanEnergy)
            varEnergy=np.loadtxt(backgroundpath + '/varEnergy_2DTGCRNN_'+str(Nx)+'x'+ str(Ny) +ending + savename +'.txt').tolist()
        except:
            print("Failed! No need to load energies if running for the first time!")
    '''
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
    label[position[0]][position[1]] = -1
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
        
    dis_label[A[0]][A[1]] = -1
    return dis_label, A_distance   # reverse 

'''
test = cond_label([3,3], 10)
test_1 = distance_label(np.array([3,3]),10)
print(test)
print(test_1)
'''

def ith_step(wavefun, position, step, L, num_first_sample, num_second_sample, num_units ,choice,RNN_symmetry): 
    # position: set of 2d coordinate indicating the point we want to start
    # k: how many steps in the cmi we want to conduct
    # L: the size of the lattice
    #num_sample_first = sample[:int(sample.shape[0]/1000)]
    #num_sample_second = sample[int(sample.shape[0]/1000):int(sample.shape[0]/1000)*2]
    #num_sample_first_prob = sample_prob_tensor[:int(sample.shape[0]/1000)]
    #num_sample_second_prob = sample_prob_tensor[int(sample_prob_tensor.shape[0]/1000):int(sample.shape[0]/1000)*2] #divide the sample into two
    if (get_numavailable_gpus()!=0):
        device = '/CPU:0'
    else:
        device = '/GPU:0'
    with tf.device(device):
        print("device:", device)
        with tf.compat.v1.variable_scope(wavefun.scope,reuse=tf.compat.v1.AUTO_REUSE):
            with wavefun.graph.as_default():
                tv_diff =[[] for i in range (len(position))]
                tv_first_var = [[] for i in range (len(position))]
                tv_second_var = [[] for i in range (len(position))]
                sample_first_input = tf.stop_gradient(wavefun.sample(int(num_units), inputdim=2))
                if choice == "Euclidean":
                    label, real_distance = distance_label(np.array(position), L)
                else: 
                    label =  cond_label(position, L)
                #length = np.max(label)

                for step_size in range (0, step+1):
                    output = 0
                    first_var = 0
                    second_var = 0
                    if (step_size>=1):
                        pos_temp = np.where(label == step_size)
                        label[pos_temp] = -1
                    label_where = np.where(label==-1)
                    label_position = [[] for i in range (len(label_where[0]))]
                    for a in range (len(label_where)):
                        for b in range (len(label_where[a])):
                            label_position[b].append(label_where[a][b])

                    for j in range (num_first_sample//num_units):

                        sample_first =  sess.run(sample_first_input)
                        sample_first_prob = np.transpose(sess.run(cond_prob, feed_dict={inputs: sample_first}),(0,2,1))[:,position[0],position[1]] #transpose
                        sample_first = np.transpose(sample_first, (0,2,1))

                        print(np.array(label))
                        print("label_where:", label_where)
                        print("label_where0:", label_where[0])
                        print("label_position:", label_position)
                        label_value = sample_first[:, label_where[0], label_where[1]] #Here the slicing needs to be modified the shape first_sample, 
                        #np.save( "sample_first.npy",sample_first)
                        #np.save("label_where.npy", label_where)

                        print("label_value.shape:", label_value.shape)
                        cond_prob_second = tf.stop_gradient(wavefun.label_sample(num_units, num_second_sample, inputdim = 2, label = label_position, label_value=label_value, position=position))

                        aci_first_prob = np.array(sess.run(cond_prob_second)).reshape(num_units, num_second_sample)  #first_sample*
                        aci_first_prob_mean = np.mean(aci_first_prob, axis = 1)
                        aci_first_prob_var = np.var(aci_first_prob, axis = 1)
                        print("second_var:", np.mean(aci_first_prob_var))
                        print("first_var:", np.var(np.abs(1-aci_first_prob_mean/sample_first_prob)))
                        output+= np.sum(np.abs(1 - aci_first_prob_mean/sample_first_prob))/num_first_sample
                        first_var += np.var(np.abs(1-aci_first_prob_mean/sample_first_prob))
                        second_var += np.mean(aci_first_prob_var)

                    first_var/= num_first_sample//num_units
                    second_var /= num_first_sample//num_units

                    tv_diff[i].append(output)
                    tv_first_var[i].append(first_var)
                    tv_second_var[i].append(second_var)
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


final_diff, final_first_var, final_second_var= ith_step(wf, position_, 3, 10, 100, 50, 20, "else", RNN_symmetry)



np.save("tv_"+str(J)+"y"+str(position_[0])+"x"+str(position_[1])+"_param"+str(args.param), np.array(final_diff))
np.save("tv_second_var"+str(J)+"y"+str(position_[0])+"x"+str(position_[1])+"param"+str(args.param), np.array(final_second_var))
np.save("tv_first_var"+str(J)+"y"+str(position_[0])+"x"+str(position_[1])+"param"+str(args.param), np.array(final_first_var))



'''
    if (RNN_symmetry == "c4vsym"):     
        with tf.compat.v1.variable_scope(wavefun.scope,reuse=tf.compat.v1.AUTO_REUSE):
            with wavefun.graph.as_default():
                sample_first_input = tf.stop_gradient(wavefun.sample(int(num_first_sample), inputdim=2))
                sample_first =  sess.run(sample_first_input)
                
                list_samples = [sample_first_input]
                list_samples.append(tf.reshape(tf.image.rot90(tf.reshape( sample_first_input, [-1,L, L, 1]),k=-1), [-1,L, L]))
                list_samples.append(tf.reshape(tf.image.rot90(tf.reshape( sample_first_input, [-1,L, L, 1]),k=-2), [-1,L, L]))
                list_samples.append(tf.reshape(tf.image.rot90(tf.reshape( sample_first_input, [-1,L, L, 1]),k=-3), [-1,L, L]))
                list_samples.append( sample_first_input[:,::-1])
                list_samples.append( sample_first_input[:,:,::-1])
                list_samples.append(tf.transpose(a= sample_first_input, perm = [0,2,1]))
                list_samples.append(tf.transpose(a= sample_first_input, perm = [0,2,1]))

                
                sample_first_prob = np.transpose(sess.run(cond_prob, feed_dict={inputs: sample_first}),(0,2,1))[:,position[0],position[1]] #transpose
                for i in range (8):
                    sample_test = sess.run(list_samples[i])
                    sample_first_prob_test = np.prod(np.transpose(sess.run(cond_prob, feed_dict={inputs: sample_test}),(0,2,1)), axis=(1,2))
                    print("sample_first_prob_test:", sample_first_prob_test)
                sample_first = np.transpose(sample_first, (0,2,1))
                print(np.array(label))

                label_value = sample_first[:, label_where[0], label_where[1]] #Here the slicing needs to be modified the shape first_sample, 
                #np.save( "sample_first.npy",sample_first)
                #np.save("label_where.npy", label_where)

                print("label_value.shape:", label_value.shape)
                cond_prob_second = tf.stop_gradient(wavefun.label_sample(num_first_sample, num_second_sample, inputdim = 2, label = label_position, label_value=label_value, position=position))

                aci_first_prob = np.array(sess.run(cond_prob_second)).reshape(num_first_sample, num_second_sample)  #first_sample*
                aci_first_prob_mean = np.mean(aci_first_prob, axis = 1)
                aci_first_prob_var = np.var(aci_first_prob, axis = 1)
                print("second_var:", np.mean(aci_first_prob_var))
                print("first_var:", np.var(np.abs(1-aci_first_prob_mean/sample_first_prob)))
                output+= np.sum(np.abs(1 - aci_first_prob_mean/sample_first_prob))/num_first_sample


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


        return output
    else:
'''
'''
for i in range (2, Ny):
    for j in range(Nx):
        for num in range (5):
            output, second_var, first_var = ith_step(wf, [i, j], num, 10, 2500, 500, "else", RNN_symmetry)
            final_diff[(i-2)*Ny+j].append(output)
            final_second_var[(i-2)*Ny+j].append(second_var)
            final_first_var[(i-2)*Ny+j].append(first_var)
'''