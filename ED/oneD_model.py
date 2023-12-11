import netket as nk
import jax
from netket.operator.spin import sigmax,sigmaz, sigmap, sigmam
from scipy.sparse.linalg import eigsh
import numpy as np
import matplotlib.pyplot as plt
import time
from oneD_tool import *

L = 16
hi = nk.hilbert.Spin(s=1 / 2, N=L)
model = "1DTFIM"
if model == "1DXXZ":
    int_ = "delta"
    params = [1.2, 1.05, 1.01, 1., 0.99, 0.8]  #sigmaz interaction
elif model == "1DJ1J2":
    int_ = "J2"
    params = [0.2, 0.4, 0.45, 0.55, 0.8, 1.0, 1.2] #J2
elif model == "1DTFIM":
    int_ = "B"
    params =  [-0.2, -0.5, -0.8, -0.95, -0.98, -1.0, -1.02, -1.05, -1.2, -1.5, -2.0, -3.5, -6.0] #magnetic field
periodic = False


for param in params:
    if model == "1DJ1J2":
        H = sum([2*(sigmap(hi,i)*sigmam(hi,(i+1))+sigmam(hi,i)*sigmap(hi,(i+1))) + sigmaz(hi, i)*sigmaz(hi, (i+1))   for i in range(L-1)])
        H += param*sum([2*(sigmap(hi,i)*sigmam(hi,(i+2))+sigmam(hi,i)*sigmap(hi,(i+2))) + sigmaz(hi, i)*sigmaz(hi, (i+2))   for i in range(L-2)])
        if periodic:
            H += 2*(sigmap(hi,0)*sigmam(hi,L-1)+sigmam(hi,0)*sigmap(hi,L-1))+sigmaz(hi,0)*sigmaz(hi,L-1)
            H += param*2*(sigmap(hi,0)*sigmam(hi,L-2)+sigmam(hi,0)*sigmap(hi,L-2))+param*sigmaz(hi,0)*sigmaz(hi,L-2)
            H += param*2*(sigmap(hi,L-1)*sigmam(hi, 1)+sigmam(hi,L-1)*sigmap(hi, 1))+param*sigmaz(hi,L-1)*sigmaz(hi,1)
        #H/=4
    elif model == "1DTFIM":
        H = sum([param * sigmax(hi, i) for i in range(L)])
        H += sum([-1 * sigmaz(hi, i) * sigmaz(hi, (i + 1)) for i in range(L - 1)])
        if periodic:
            H += -sigmaz(hi, 0) * sigmaz(hi, L - 1)
    elif model == "1DXXZ":
        H = sum([2 * (sigmap(hi, i) * sigmam(hi, i + 1) + sigmam(hi, i) * sigmap(hi, i + 1)) + param * sigmaz(hi,
                                                                                                              i) * sigmaz(
            hi, i + 1) for i in range(L - 1)])
        if periodic:
            H += 2 * (sigmap(hi, 0) * sigmam(hi, 1) + sigmam(hi, 0) * sigmap(hi, 1)) + param * sigmaz(hi, 0) * sigmaz(
                hi, L - 1)
        #H /= 4
    sp_h = H.to_sparse()
    eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
    print("eigenvalues with scipy sparse:", eig_vals)
    probs_exact =  np.abs(eig_vecs[:, 0]) ** 2
    mag = np.sum(probs_exact*np.array(count_diff_ones_zeros(L)))
    shape = (2,) * (L)
    probs_exact = probs_exact.reshape(*shape)

    if (periodic == False):
        cmi = cmi_(probs_exact, L)
        mean_corr, var_corr = spin_correlation_all(probs_exact, L)

    else:
        cmi = cmi_periodic(probs_exact, L)
        mean_corr = spin_correlation_periodic(probs_exact, L)
    cmi_all = cmi_traceout(probs_exact, L)

    np.save("result/"+model+"/cmi_"+model+"_L"+str(L)+"_"+int_+"_"+str(param)+"periodic_"+str(periodic)+".npy", cmi)
    np.save("result/"+model+"/mean_corr_"+model+"_L"+str(L)+"_"+int_+"_"+str(param)+"periodic_"+str(periodic)+".npy", mean_corr)
    if periodic == False:
        np.save("result/"+model+"/var_corr_"+model+"_L"+str(L)+"_"+int_+"_"+str(param)+"periodic_"+str(periodic)+".npy", var_corr)
    np.save("result/"+model+"/cmi_traceout_"+model+"_L"+str(L)+"_"+int_+"_"+str(param)+"periodic_"+str(periodic)+".npy", cmi_all)
    np.save("result/"+model+"/mag_"+model+"_L"+str(L)+"_"+int_+"_"+str(param)+"periodic_"+str(periodic)+".npy", mag)
    np.save("result/"+model+"/gap_"+model+"_L"+str(L)+"_"+int_+"_"+str(param)+"periodic_"+str(periodic)+".npy", np.array(eig_vals[1]-eig_vals[0]))