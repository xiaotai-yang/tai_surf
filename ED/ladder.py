import jax
import netket as nk
import numpy as np
from netket.operator.spin import sigmax,sigmaz, sigmap, sigmam
import time
from scipy.sparse.linalg import eigsh
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ladder_tool import *

L = 8
periodic = False
hi = nk.hilbert.Spin(s=1 / 2, N =  L*2)

Jv_ = [0.2, 0.4, 0.6]
Jx_ = [0.2, 0.4, 0.6]


for Jv in  Jv_ :
    for Jx in Jx_:
        H = sum([2*(sigmap(hi, 2*y)*sigmam(hi, 2*(y+1))+sigmam(hi, 2*y)*sigmap(hi, 2*(y+1))+sigmap(hi, 2*y+1)*sigmam(hi, 2*(y+1)+1)+sigmam(hi, 2*y+1)*sigmap(hi, 2*(y+1)+1))+sigmaz(hi, 2*y)*sigmaz(hi, 2*(y+1))+sigmaz(hi, 2*y+1)*sigmaz(hi, 2*(y+1)+1) for y in range(L-1) ])  #J-parallel

        H += Jv*sum([2*(sigmap(hi, 2*y)*sigmam(hi, 2*y+1)+sigmam(hi, 2*y)*sigmap(hi, 2*y+1))+sigmaz(hi, 2*y)*sigmaz(hi, 2*y+1) for y in range(L)]) # J-vertical

        H += Jx*sum([2*(sigmap(hi, 2*y)*sigmam(hi, 2*(y+1)+1)+sigmam(hi, 2*y)*sigmap(hi, 2*(y+1)+1)+sigmap(hi, 2*y+1)*sigmam(hi, 2*(y+1))+sigmam(hi, 2*y+1)*sigmap(hi, 2*(y+1)))+sigmaz(hi, 2*y)*sigmaz(hi, 2*(y+1)+1)+sigmaz(hi, 2*y+1)*sigmaz(hi, 2*(y+1)) for y in range(L-1)])  #J-cross

        if (periodic == True):
        #periodic boundary conditions
            H+= 2*sigmap(hi, 0)*sigmap(hi, 2*L-2)+2*sigmam(hi, 0)*sigmam(hi, 2*L-2)+2*sigmap(hi, 1)*sigmap(hi, 2*L-1)+2*sigmam(hi, 1)*sigmam(hi, 2*L-1)+sigmaz(hi, 0)*sigmaz(hi, 2*L-2)+sigmaz(hi, 1)*sigmaz(hi, 2*L-1) #J-parallel

            H+= Jx*(2*sigmap(hi, 0)*sigmap(hi, 2*L-1)+2*sigmam(hi, 0)*sigmam(hi, 2*L-1)+2*sigmap(hi, 1)*sigmap(hi, 2*L-2)+2*sigmam(hi, 1)*sigmam(hi, 2*L-2)+sigmaz(hi, 0)*sigmaz(hi, 2*L-1)+sigmaz(hi, 1)*sigmaz(hi, 2*L-2)) #J-cross

        sp_h = H.to_sparse()
        eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
        print("eigenvalues with scipy sparse Jv="+str(Jv)+ "_Jx="+ str(Jx) +":", eig_vals)

        prob_exact = eig_vecs[:,0]**2
        shape = (2,) * (2*L)
        prob_exact = prob_exact.reshape(*shape)

        if periodic == True:
            mean_corr_dimer = correlation_periodic_dimer(prob_exact, L)
            mean_corr_single = correlation_periodic_single(prob_exact, L)
            cmi_dimer = cmi_periodic(prob_exact, L)

        else:
            mean_corr, var_corr = correlation_all(prob_exact, L)
            cmi_dimer = cmi_(prob_exact, L, "dimer")

        cmi_single = cmi_(prob_exact, L, "single")
        cmi_traceout = cmi_traceout_dimer(prob_exact, L)

        np.save("result/ladder/gap_ladder_L"+str(L)+"_Jv="+str(Jv)+"_Jx="+str(Jx)+"periodic_"+str(periodic)+".npy", np.array(eig_vals[1]-eig_vals[0]))
        np.save("result/ladder/cmi_ladder_dimer_L"+str(L)+"_Jv="+str(Jv)+"_Jx="+str(Jx)+"periodic_"+str(periodic)+".npy", cmi_dimer)
        np.save("result/ladder/cmi_ladder_single_L"+str(L)+"_Jv="+str(Jv)+"_Jx="+str(Jx)+"periodic_"+str(periodic)+".npy", cmi_single)
        np.save("result/ladder/cmi_ladder_traceout_L"+str(L)+"_Jv="+str(Jv)+"_Jx="+str(Jx)+"periodic_"+str(periodic)+".npy", cmi_traceout)

        if periodic == False:
            np.save("result/ladder/mean_corr_ladder_dimer_L"+str(L)+"_Jv="+str(Jv)+"_Jx="+str(Jx)+"periodic_"+str(periodic)+".npy", mean_corr)
            np.save("result/ladder/var_corr_ladder_dimer_L"+str(L)+"_Jv="+str(Jv)+"_Jx="+str(Jx)+"periodic_"+str(periodic)+".npy", var_corr)
        elif periodic == True:
            np.save("result/ladder/mean_corr_ladder_dimer_L"+str(L)+"_Jv="+str(Jv)+"_Jx="+str(Jx)+"periodic_"+str(periodic)+".npy", mean_corr_dimer)
            np.save("result/ladder/mean_corr_ladder_single_L"+str(L)+"_Jv="+str(Jv)+"_Jx="+str(Jx)+"periodic_"+str(periodic)+".npy", mean_corr_single)

cmi_single = {}
cmi_dimer = {}
cmi_traceout = {}
mean_corr_dimer= {}
var_corr_dimer = {}
mean_corr_single = {}
var_corr_single = {}
gap = {}

# Looping through the values and loading each file
for Jv in Jv_:
    for Jx in Jx_:
        filename = f"result/ladder/cmi_ladder_single_L{L}_Jv={Jv}_Jx={Jx}periodic_{periodic}.npy"
        cmi_single[Jv, Jx] = np.load(filename)
        filename = f"result/ladder/cmi_ladder_dimer_L{L}_Jv={Jv}_Jx={Jx}periodic_{periodic}.npy"
        cmi_dimer[Jv, Jx] = np.load(filename)
        if periodic == True:
            cmi_dimer[Jv, Jx] = np.flip(cmi_dimer[Jv, Jx])
        filename = f"result/ladder/cmi_ladder_traceout_L{L}_Jv={Jv}_Jx={Jx}periodic_{periodic}.npy"
        cmi_traceout[Jv, Jx] = np.load(filename, allow_pickle=True)
        filename = f"result/ladder/mean_corr_ladder_dimer_L{L}_Jv={Jv}_Jx={Jx}periodic_{periodic}.npy"
        mean_corr_dimer[Jv, Jx] = np.load(filename)

        if periodic == False:
            filename = f"result/ladder/var_corr_ladder_dimer_L{L}_Jv={Jv}_Jx={Jx}periodic_{periodic}.npy"
            var_corr_dimer[Jv, Jx] = np.load(filename)

        if periodic == True:
            filename = f"result/ladder/mean_corr_ladder_single_L{L}_Jv={Jv}_Jx={Jx}periodic_{periodic}.npy"
            mean_corr_single[Jv, Jx] = np.load(filename)
        filename = f"result/ladder/gap_ladder_L{L}_Jv={Jv}_Jx={Jx}periodic_{periodic}.npy"
        gap[Jv, Jx] = np.load(filename)
cmi_length = []
err_cmi_length =[]
corre_length_dimer = []
err_corre_length_dimer = []
corre_length_single = []
err_corre_length_single = []
cmi_length_traceout = [[ [] for i in range(len(Jx_))] for j in range(len(Jv_))]
err_cmi_length_traceout = [ [[] for i in range(len(Jx_))] for j in range(len(Jv_))]
for a in Jv_:
    for b in Jx_:
        i = (a, b)
        df = len(cmi_dimer[i])-2
        (cmi_len, b1), residuals, _, _, _ = np.polyfit(np.arange(len(cmi_dimer[i])), -np.log(cmi_dimer[i]), 1, full=True)
        mean_x = np.mean(np.arange(len(cmi_dimer[i])))
        err = np.sqrt(residuals / df) / np.sqrt(np.sum((np.arange(len(cmi_dimer[i])) - mean_x)**2))
        cmi_length.append(cmi_len)
        err_cmi_length.append(err)

        df = len(mean_corr_dimer[i])-2
        (corr_len, b2), residuals, _, _, _ = np.polyfit(np.arange(len(mean_corr_dimer[i])), -np.log(np.abs(mean_corr_dimer[i])), 1, full=True)
        mean_x = np.mean(np.arange(len(mean_corr_dimer[i])))
        err = np.sqrt(residuals / df) / np.sqrt(np.sum((np.arange(len(mean_corr_dimer[i])) - mean_x)**2))
        corre_length_dimer.append(corr_len)
        err_corre_length_dimer.append(err)

        if periodic == True:
            df = len(mean_corr_single[i])-2
            (corr_len, b2), residuals, _, _, _ = np.polyfit(np.arange(len(mean_corr_single[i])), -np.log(np.abs(mean_corr_single[i])), 1, full=True)
            mean_x = np.mean(np.arange(len(mean_corr_single[i])))
            err = np.sqrt(residuals / df) / np.sqrt(np.sum((np.arange(len(mean_corr_single[i])) - mean_x)**2))
            corre_length_single.append(corr_len)
            err_corre_length_single.append(err)

        for i in range(len(cmi_traceout[a, b])):
            df = len(cmi_traceout[a, b][i])-2
            (cmi_len, b1), residuals, _, _, _ = np.polyfit(np.arange(len(cmi_traceout[a, b][i])), -np.log(cmi_traceout[a, b][i]), 1, full=True)
            mean_x = np.mean(np.arange(len(cmi_traceout[a, b][i])))
            err = np.sqrt(residuals / df) / np.sqrt(np.sum((np.arange(len(cmi_traceout[a, b][i])) - mean_x)**2))
            cmi_length_traceout[Jv_.index(a)][Jx_.index(b)].append(cmi_len)
            err_cmi_length_traceout[Jv_.index(a)][Jx_.index(b)].append(err)

for a in Jv_:
    for b in Jx_:
        i = (a, b)
        plt.plot(np.log(cmi_single[i]), label="Jv="+str(a)+"_Jx="+str(b))
    plt.xlabel("distance")
    plt.ylabel("log_CMI_single")
    plt.legend()
    plt.savefig("figure/ladder/cmi_single_ladder_L"+str(L)+"_Jv="+str(a)+"_periodic_"+str(periodic)+".png")
    plt.show()
    plt.clf()

for a in Jv_:
    for b in Jx_:
        i = (a, b)
        plt.plot(np.log(cmi_dimer[i]), label="Jv="+str(a)+"_Jx="+str(b))
    plt.xlabel("distance")
    plt.ylabel("log_CMI_dimer")
    plt.legend()
    plt.savefig("figure/ladder/cmi_dimer_ladder_L"+str(L)+"_Jv="+str(a)+"_periodic_"+str(periodic)+".png")
    plt.show()
    plt.clf()

for a in Jv_:
    for b in Jx_:
        i = (a, b)
        plt.plot(np.log(np.abs(mean_corr_dimer[i])), label="Jv="+str(a)+"_Jx="+str(b))
        if periodic == False:
            plt.fill_between(np.arange(len(mean_corr_dimer[i])) ,np.log(np.abs(mean_corr_dimer[i])) - 1/mean_corr_dimer[i]*np.sqrt(var_corr_dimer[i]) , np.log(np.abs(mean_corr_dimer[i])) + 1/np.abs(mean_corr_dimer[i])*np.sqrt(var_corr_dimer[i]), alpha=0.2)
    plt.xlabel("distance")
    plt.ylabel("log_correlation_dimer")
    plt.legend()
    plt.savefig("figure/ladder/correlation_ladder_dimer_L"+str(L)+"_Jv="+str(a)+"_periodic_"+str(periodic)+".png")
    plt.show()
    plt.clf()

if periodic == True:
    for a in Jv_:
        for b in Jx_:
            i = (a, b)
            plt.plot(np.log(np.abs(mean_corr_single[i])), label="Jv="+str(a)+"_Jx="+str(b))
            if periodic == False:
                plt.fill_between(np.arange(len(mean_corr_single[i])) ,np.log(np.abs(mean_corr_single[i])) - 1/mean_corr_single[i]*np.sqrt(var_corr[i]) , np.log(np.abs(mean_corr_single[i])) + 1/np.abs(mean_corr_single[i])*np.sqrt(var_corr[i]), alpha=0.2)
        plt.xlabel("distance")
        plt.ylabel("log_correlation_single")
        plt.legend()
        plt.savefig("figure/ladder/correlation_ladder_single_L"+str(L)+"_Jv="+str(a)+"_periodic_"+str(periodic)+".png")
        plt.show()
        plt.clf()

for a in Jv_:
    for b in Jx_:
        i = (a, b)
        if periodic == False:
            plt.plot(np.log(np.abs(mean_corr_dimer[i][1:])), np.log(cmi_dimer[i]), label="Jv="+str(a)+"_Jx="+str(b))
            plt.fill_betweenx(np.log(cmi_dimer[i]), np.log(np.abs(mean_corr_dimer[i][1:])) - 1/np.abs(mean_corr_dimer[i][1:])*np.sqrt(var_corr_dimer[i][1:]) , np.log(np.abs(mean_corr_dimer[i][1:])) + 1/np.abs(mean_corr_dimer[i][1:])*np.sqrt(var_corr_dimer[i][1:]), alpha=0.2)
        else:
            plt.plot(np.log(np.abs(mean_corr_dimer[i])), np.log(cmi_dimer[i]), label="Jv="+str(a)+"_Jx="+str(b))
    plt.xlabel("log_correlation")
    plt.ylabel("log_CMI_dimer")
    plt.legend()
    plt.savefig("figure/ladder/correlation_dimer_cmi_ladder_L"+str(L)+"_Jv="+str(a)+"_periodic_"+str(periodic)+".png")
    plt.show()
    plt.clf()
if periodic == True:
    for a in Jv_:
        for b in Jx_:
            i = (a, b)
            plt.plot(np.log(np.abs(mean_corr_single[i])), np.log(cmi_dimer[i]), label="Jv="+str(a)+"_Jx="+str(b))
        plt.xlabel("log_correlation")
        plt.ylabel("log_CMI_dimer")
        plt.legend()
        plt.savefig("figure/ladder/correlation_single_cmi_ladder_L"+str(L)+"_Jv="+str(a)+"_periodic_"+str(periodic)+".png")
        plt.show()
        plt.clf()

gap_list = []
for a in Jv_:
    for b in Jx_:
        i = (a, b)
        gap_list.append(gap[i])

plt.scatter(gap_list, cmi_length, label="cmi", s=10)
plt.errorbar(gap_list, cmi_length, yerr=err_cmi_length[:][0], fmt='o', ecolor='g', capthick=1, capsize=10)
plt.xlabel("gap")
plt.ylabel("cmi_length")
plt.legend()
plt.savefig("figure/ladder/cmi_length_gap_ladder_L"+str(L)+"_periodic_"+str(periodic)+".png")
plt.show()
plt.clf()

plt.scatter(gap_list, corre_length_dimer, label="correlation", s=10)
plt.errorbar(gap_list, corre_length_dimer, yerr=err_corre_length_dimer[:][0], fmt='o', ecolor='g', capthick=1, capsize=10)
plt.xlabel("gap")
plt.ylabel("correlation_length")
plt.legend()
plt.savefig("figure/ladder/correlation_length_dimer_gap_ladder_L"+str(L)+"_periodic_"+str(periodic)+".png")
plt.show()
plt.clf()

if periodic == True:
    plt.scatter(gap_list, corre_length_single, label="correlation", s=10)
    plt.errorbar(gap_list, corre_length_single, yerr=err_corre_length_single[:][0], fmt='o', ecolor='g', capthick=1, capsize=10)
    plt.xlabel("gap")
    plt.ylabel("correlation_length")
    plt.legend()
    plt.savefig("figure/ladder/correlation_length_single_gap_ladder_L"+str(L)+"_periodic_"+str(periodic)+".png")
    plt.show()
    plt.clf()

for a in range(len(Jv_)):
    for b in range(len(Jx_)):
        plt.scatter(np.arange(len(cmi_length_traceout[a][b])), (cmi_length_traceout[a][b]), label="Jv="+str(Jv_[a])+"_Jx="+str(Jx_[b]), s=10)
        plt.errorbar(np.arange(len(cmi_length_traceout[a][b])), (cmi_length_traceout[a][b]), yerr=np.array(err_cmi_length_traceout[a][b]).ravel(), fmt='o', ecolor='g', capthick=1, capsize=10)
        plt.xlabel("trace_size")
        plt.ylabel("cmi_length")
        plt.legend()
        plt.savefig("figure/ladder/traceout/cmi_length_traceout_ladder_L"+str(L)+"Jv="+str(Jv_[a])+"_Jx="+str(Jx_[b])+"periodic_"+str(periodic)+".png")
        plt.show()
        plt.clf()