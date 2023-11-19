import netket as nk
import jax
from netket.operator.spin import sigmax,sigmaz
from scipy.sparse.linalg import eigsh
import numpy as np
import matplotlib.pyplot as plt
import time

L = 20
hi = nk.hilbert.Spin(s=1 / 2, N=L)
Gamma = -1
B = [-0.8, -1.5, -2.0, -2.5, -3.0, -3.5]

for V in B:
    H = sum([V*Gamma*sigmax(hi,i) for i in range(L)])

    H += sum([sigmaz(hi,i)*sigmaz(hi,(i+1)) for i in range(L-1)])
    sp_h = H.to_sparse()
    eig_vals, eig_vecs = eigsh(sp_h, k=1, which="SA")
    print("eigenvalues with scipy sparse:", eig_vals)

    probs_exact =  np.abs(eig_vecs.ravel()) ** 2
    spin_correlation_exact = []

    for i in range (L-1):
        p_middle = np.sum(np.array(np.split(probs_exact, 2**i)), axis=0)
        p_1, p_2 = np.split(p_middle, 2)
        p_1 = np.sum(np.array(np.split(p_1, 2**(L-i-2))), axis=0)
        p_2 = np.sum(np.array(np.split(p_2, 2**(L-i-2))), axis=0)
        p = np.concatenate((p_1, p_2),axis=None)
        spin_correlation_exact.append(p[0]+p[-1]-p[1]-p[2]-(p[0]+p[1]-p[2]-p[3])*(p[0]-p[1]+p[2]-p[3]))
    p_itself = np.sum(np.array(np.split(probs_exact, 2**(L-1))), axis = 0)
    spin_correlation_exact.append(np.sum(p_itself**2)-(p_itself[0]-p_itself[1])**2)
    np.save("spin_correlation_exact_L=20_B="+str(V)+".npy", np.array(spin_correlation_exact))
    plt.plot(np.flip(np.log(np.abs(np.array(spin_correlation_exact)))))
    plt.savefig("spin_correlation_exact_L=20_B="+str(V)+".png")
    plt.show()


    # Here we consider four cuts xx|xxxxxx|xxxx|xxxx|xx, the first and the last one determine the size of ABC, the middle two
    # determine how we decompose the system ABC.
    N = 2**20
    system_size = 20
    # initialization
    p_abc = np.array(probs_exact)+1e-30

    cmi = []
    for i in range (system_size-1):
        cmi.append(np.array([]))
    for x in range (system_size):
        print(x)
        for i in range (x+1,system_size):
            print(i)
            for j in range (i, system_size):     # i, j is the cut position
                for k in range (j+1 ,system_size+1):

                    p_abc_new = np.sum(np.array(np.split(np.sum(np.split(p_abc, 2**k), axis = 1), 2**x)) , axis = 0)

                    p_ab = np.sum(np.array(np.split(p_abc_new, 2**(j-x))), axis = 1)

                    p_bc = np.sum(np.array(np.split(np.sum(np.array(np.split(np.sum(np.split(p_abc, 2**k), axis = 1), 2**x)) , axis = 0), 2**(i-x))), axis = 0)
                    if (j == i):
                        cmi[j-i] = np.append(cmi[j-i],(np.sum(-p_ab*np.log(p_ab)) -np.sum(p_bc*np.log(p_bc)) + np.sum(p_abc_new*np.log(p_abc_new))))
                    else :
                        p_b = np.sum(np.array(np.split(p_ab, 2**(i-x))), axis = 0)
                        cmi[j-i] = np.append(cmi[j-i],(np.sum(-p_ab*np.log(p_ab)) -np.sum(p_bc*np.log(p_bc)) + np.sum(p_abc_new*np.log(p_abc_new)) + np.sum(p_b*np.log(p_b))))

    for i in range(len(cmi)):
        cmi[i] = np.mean(cmi[i])
    np.save("cmi_L=20_B="+str(V)+".npy", np.array(cmi))
    plt.plot(np.flip(np.log(np.abs(cmi))))
    plt.savefig("cmi_L=20_B="+str(V)+".png")
    plt.show()
