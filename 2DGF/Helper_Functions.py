import numpy as np
import time
from math import ceil

#######################################################################
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
######################################################################


###################################################################################
def MatrixElements(Nx,Ny,sigmap,sigmas, matrixelements, XZ):

    #the diagonal part is simply the sum of all Sz-Sz interactions
    diag = 0
    num = 0 #Number of basis elements
    matrixelements[num] = diag #add the diagonal part to the matrix elements
    sig = np.copy(sigmap)
    sigmas[num] = sig
    num += 1
    if (XZ=='Z'):
        for ny in range(Ny):
            for nx in range(Nx):
                sig = np.copy(sigmap)
                sig[nx,ny] = 1-sigmap[nx,ny]
                sigmas[num] = sig
                if  nx == 0:    
                    if ny == 0:
                        if (sigmap[nx+1 ,ny]+sigmap[nx, ny+1])%2 == 0:
                            matrixelements[num] = -1
                        else:
                            matrixelements[num] = 1
                    elif ny == Ny - 1:
                        if (sigmap[nx+1, ny]+sigmap[nx, ny-1])%2 == 0:
                            matrixelements[num] = -1
                        else :
                            matrixelements[num] = 1
                    else :
                        if (sigmap[nx+1, ny]+sigmap[nx, ny+1]+sigmap[nx, ny-1])%2 == 0:
                            matrixelements[num] = -1
                        else :
                            matrixelements[num] = 1
                    num += 1

                elif  nx == Nx-1:   
                    if ny == 0:
                        if (sigmap[nx-1 ,ny]+sigmap[nx, ny+1])%2 == 0:
                            matrixelements[num] = -1
                        else:
                            matrixelements[num] = 1
                    elif ny == Ny - 1:
                        if (sigmap[nx-1, ny]+sigmap[nx, ny-1])%2 == 0:
                            matrixelements[num] = -1
                        else :
                            matrixelements[num] = 1
                    else :
                        if (sigmap[nx-1, ny]+sigmap[nx, ny+1]+sigmap[nx, ny-1])%2 == 0:
                            matrixelements[num] = -1
                        else :
                            matrixelements[num] = 1
                    num += 1

                elif  ny == 0 or ny == Ny - 1:   
                    if ny == 0:
                        if (sigmap[nx-1, ny]+sigmap[nx+1, ny]+sigmap[nx, ny+1])%2 == 0:
                            matrixelements[num] = -1
                        else :
                            matrixelements[num] = 1
                    elif ny == Ny-1:
                        if (sigmap[nx-1, ny]+sigmap[nx+1, ny]+sigmap[nx, ny-1])%2 == 0:
                            matrixelements[num] = -1
                        else :
                            matrixelements[num] = 1
                    num += 1

                else :
                    if (sigmap[nx,ny-1]+sigmap[nx,ny+1]+sigmap[nx+1,ny]+sigmap[nx-1,ny])%2==0:
                        matrixelements[num] = -1
                    else :
                        matrixelements[num] = 1
                    num += 1
    elif (XZ=='X'):
        for ny in range(Ny):
            for nx in range(Nx):
                sig = np.copy(sigmap)
                
                if  nx == 0:                       
                    if ny == 0:
                        sig[nx+1,ny] = 1-sigmap[nx+1,ny]
                        sig[nx,ny+1] = 1-sigmap[nx,ny+1]
                        sigmas[num] = sig
                    elif ny == Ny - 1:
                        sig[nx+1,ny] = 1-sigmap[nx+1,ny]
                        sig[nx,ny-1] = 1-sigmap[nx,ny-1]
                        sigmas[num] = sig
                    else :
                        sig[nx+1,ny] = 1-sigmap[nx+1,ny]
                        sig[nx,ny-1] = 1-sigmap[nx,ny-1]
                        sig[nx,ny+1] = 1-sigmap[nx,ny+1]
                        sigmas[num] = sig
                        
                elif  nx == Nx-1:   
                    if ny == 0:
                        sig[nx-1,ny] = 1-sigmap[nx-1,ny]
                        sig[nx,ny+1] = 1-sigmap[nx,ny+1]
                        sigmas[num] = sig
                    elif ny == Ny - 1:
                        sig[nx-1,ny] = 1-sigmap[nx-1,ny]
                        sig[nx,ny-1] = 1-sigmap[nx,ny-1]
                        sigmas[num] = sig
                    else :
                        sig[nx-1,ny] = 1-sigmap[nx-1,ny]
                        sig[nx,ny-1] = 1-sigmap[nx,ny-1]
                        sig[nx,ny+1] = 1-sigmap[nx,ny+1]
                        sigmas[num] = sig
                        

                elif  ny == 0 or ny == Ny - 1:
                    if ny == 0:
                        sig[nx+1,ny] = 1-sigmap[nx+1,ny]
                        sig[nx-1,ny] = 1-sigmap[nx-1,ny]
                        sig[nx,ny+1] = 1-sigmap[nx,ny+1]
                        sigmas[num] = sig
                    elif ny == Ny - 1:
                        sig[nx+1,ny] = 1-sigmap[nx+1,ny]
                        sig[nx-1,ny] = 1-sigmap[nx-1,ny]
                        sig[nx,ny-1] = 1-sigmap[nx,ny-1]
                        sigmas[num] = sig
                else :
                    sig[nx+1,ny] = 1-sigmap[nx+1,ny]
                    sig[nx-1,ny] = 1-sigmap[nx-1,ny]
                    sig[nx,ny+1] = 1-sigmap[nx,ny+1]
                    sig[nx,ny-1] = 1-sigmap[nx,ny-1]
                    sigmas[num] = sig
                    
                if sigmap[nx ,ny] == 0:
                    matrixelements[num] = -1
                elif sigmap[nx ,ny] == 1:
                    matrixelements[num] = 1
                num += 1            
    return num

def LocalEnergies(Nx,Ny,sigmasp,sigmas,H,sigmaH,matrixelements, XZ):
    """
    """
    slices=[]
    sigmas_length = 0

    numsamples =sigmasp.shape[0]

    for n in range(numsamples):
        sigmap=sigmasp[n,:]
        num = MatrixElements(Nx,Ny,sigmap, sigmaH, matrixelements, XZ)
        slices.append(slice(sigmas_length,sigmas_length + num))
        s = slices[n]

        if (len(H[s])!=num):
            print("error")
            print(H[s].shape,s, matrixelements[:num].shape)

        H[s] = matrixelements[:num]
        sigmas[s] = sigmaH[:num]

        sigmas_length += num #Increasing the length of matrix elements sigmas


    return slices,sigmas_length
########################################################################################


def Get_Samples_and_Elocs(Nx, Ny, samples, sigmas, H, log_amplitudes, sigmaH, matrixelements, samples_tensor, inputs, log_amps, sess, RNN_symmetry, XZ):

    local_energies = np.zeros(samples.shape[0], dtype = np.complex64) #The type complex should be specified, otherwise the imaginary part will be discarded

    samples=sess.run(samples_tensor)

    slices, len_sigmas = LocalEnergies(Nx,Ny,samples, sigmas, H, sigmaH, matrixelements, XZ)

    if RNN_symmetry == "c2dsym":
        numsampsteps = 4
    else:
        numsampsteps = 1

    steps = ceil(numsampsteps*len_sigmas/20000)

    for i in range(steps):
        if i < steps-1:
            cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
        else:
            cut = slice((i*len_sigmas)//steps,len_sigmas)

        log_amplitudes[cut] = sess.run(log_amps,feed_dict={inputs:sigmas[cut]})

    for n in range(len(slices)):
        s=slices[n]
        local_energies[n] = H[s].dot(np.exp(log_amplitudes[s]-log_amplitudes[s][0]))

    return samples, local_energies

def Get_Elocs(Nx, Ny, samples, sigmas, H, log_amplitudes, sigmaH, matrixelements, inputs, log_amps, sess, RNN_symmetry, XZ):

    local_energies = np.zeros(samples.shape[0], dtype = np.complex64) #The type complex should be specified, otherwise the imaginary part will be discarded

    slices, len_sigmas = LocalEnergies(Nx,Ny,samples,sigmas, H, sigmaH, matrixelements, XZ)

    if RNN_symmetry == "c2dsym":
        numsampsteps = 4
    else:
        numsampsteps = 1
    steps = ceil(numsampsteps*len_sigmas/20000)

    for i in range(steps):
        if i < steps-1:
            cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
        else:
            cut = slice((i*len_sigmas)//steps,len_sigmas)

        log_amplitudes[cut] = sess.run(log_amps,feed_dict={inputs:sigmas[cut]})
        print("Step:", i+1, "/", steps)

    for n in range(len(slices)):
        s=slices[n]
        local_energies[n] = H[s].dot(np.exp(log_amplitudes[s]-log_amplitudes[s][0]))

    return local_energies
