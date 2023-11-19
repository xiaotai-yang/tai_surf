import netket as nk
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import jax.random as random
from jax.random import PRNGKey, split, categorical
import jax.lax as lax
from jax.lax import scan
import jax.nn as nn
import time
from tqdm import tqdm
from functools import partial
import time
from math import ceil
from RNNfunction import *
import numpy as np

def local_element_indices_2d(num_body, pauli_array, loc_array):
    if pauli_array.shape[-1] != num_body:
        raise ValueError(f"Array has incorrect body of interactions {pauli_array.shape[-1]}. Expected body of interactions is {num_body}.")

    count_3s = jnp.sum(pauli_array == 3, axis = 1)
    count_1s = jnp.sum(pauli_array == 1, axis = 1)

    pauli_array_xz = {}
    xloc_arrays = {}
    zloc_arrays = {}
    yloc_arrays = {}
    xy_loc_arrays = {}

    for i in range(num_body+1):    #z_number
        for j in range (num_body+1-i):  #x_number
            mask = ((count_3s == i)&(count_1s == j))
            pauli_array_xz[i, j] = pauli_array[mask]

            mask_x = (pauli_array_xz[i, j] == 1)
            mask_y = (pauli_array_xz[i, j] == 2)
            mask_z = (pauli_array_xz[i, j] == 3)

            if mask_x.sum() != 0:
                xloc_arrays[i, j] = loc_array[mask][mask_x].reshape(-1, j, 2)
            elif mask_y.sum() != 0 or mask_z.sum() !=0:
                if mask_y.sum()!=0:
                    xloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i ,j].shape[0], 0, 2).astype(int)
                else:
                    xloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i, j].shape[0], 0, 2).astype(int)

            if mask_y.sum() !=0:
                yloc_arrays[i, j] = loc_array[mask][mask_y].reshape(-1, num_body-i-j, 2).astype(int)
            elif mask_x.sum() != 0 or mask_z.sum() !=0:
                if mask_x.sum()!=0:
                    yloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i, j].shape[0], 0, 2).astype(int)
                else:
                    yloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i, j].shape[0], 0, 2).astype(int)

            if mask_z.sum()!=0:
                zloc_arrays[i, j] = loc_array[mask][mask_z].reshape(-1, i, 2).astype(int)
            elif mask_x.sum() != 0 or mask_y.sum() !=0:
                if mask_y.sum()!=0:
                    zloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i, j].shape[0], 0, 2).astype(int)
                else:
                    zloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i, j].shape[0], 0, 2).astype(int)

    for ind in (xloc_arrays):
        xy_loc_arrays[ind] = jnp.concatenate((xloc_arrays[ind], yloc_arrays[ind]), axis=1).astype(int)
    return  xy_loc_arrays, yloc_arrays, zloc_arrays
@jax.jit
def total_samples_2d(samples, xyloc):
    def scan_array_element(sample_element, xyloc_arrays_element):
        scan_samples = sample_element.at[xyloc_arrays_element[:,0], xyloc_arrays_element[:,1]].set((sample_element[xyloc_arrays_element[:,0], xyloc_arrays_element[:,1]]+1)%2)
        return sample_element, scan_samples
    tmp = []
    for i in xyloc:
        tmp.append(lax.scan(scan_array_element, samples, xyloc[i])[1])

    return jnp.concatenate(tmp, axis = 0).reshape(-1, samples.shape[0], samples.shape[1])

@jax.jit
def new_coe_2d(samples, coe_array_off_diag, yloc, zloc):
    def ycoe(sample_element, yloc_arrays_element):
        scan_coe_tmp_y = ((0-1j)**sample_element[yloc_arrays_element[:,0],yloc_arrays_element[:,1]]).prod()
        return sample_element, scan_coe_tmp_y
    def zcoe(sample_element, zloc_arrays_element):
        scan_coe_tmp_z = ((-1+0j)**sample_element[zloc_arrays_element[:,0], zloc_arrays_element[:,1]]).prod()
        return sample_element, scan_coe_tmp_z

    def scan_ycoe(carry, yloc_ind):
        samples_, yloc_ = carry
        result = lax.cond(yloc_[yloc_ind].shape[0] != 0, lambda: lax.scan(ycoe, samples_, yloc_[yloc_ind])[1], lambda: 1, None)
        return carry, result
    def scan_zcoe(carry, zloc_ind):
        samples_, zloc_ = carry
        result = lax.cond(zloc_[zloc_ind].shape[0] != 0, lambda: lax.scan(zcoe, samples_, zloc_[zloc_ind])[1], lambda: 1, None)
        return carry, result
    tmp_y = []
    tmp_z = []
    for i in yloc:
        tmp_y.append(lax.cond(yloc[i].shape[1] != 0, lambda: lax.scan(ycoe, samples, yloc[i])[1], lambda: jnp.repeat(jnp.array([1. + 0j]), yloc[i].shape[0])))
    for i in zloc:
        tmp_z.append(lax.cond(zloc[i].shape[1] != 0, lambda: lax.scan(zcoe, samples, zloc[i])[1], lambda: jnp.repeat(jnp.array([1. + 0j]), zloc[i].shape[0])))
    coe_tmp_y = jnp.concatenate(tmp_y, axis = 0)
    coe_tmp_z = jnp.concatenate(tmp_z, axis = 0)

    return coe_tmp_y * coe_tmp_z *  coe_array_off_diag
@jax.jit
def diag_coe(samples, zloc_bulk_diag, zloc_edge_diag, zloc_corner_diag, coe_bulk_diag, coe_edge_diag, coe_corner_diag):
    def scan_z(samples, z_array):
        return ((-1) ** samples[z_array[:, 0], z_array[:, 1]]).prod()
    vmap_scan_z = vmap(scan_z, (None, 0))
    coe_bulk = (vmap_scan_z(samples, zloc_bulk_diag)*coe_bulk_diag).sum() if zloc_bulk_diag.size!=0 else 0
    coe_edge = (vmap_scan_z(samples, zloc_edge_diag)*coe_edge_diag).sum() if zloc_edge_diag.size!=0 else 0
    coe_corner = (vmap_scan_z(samples, zloc_corner_diag)*coe_corner_diag).sum() if zloc_corner_diag.size!=0 else 0
    return coe_bulk+coe_edge+coe_corner
def location_pauli_label(loc_array_bulk, loc_array_edge, loc_array_corner, Ny, Nx):
    label_bulk = {}
    label_edge = {}
    label_corner = {}
    # yi, xi : the location to label how many operation is acted on
    # y , x : The location of the loc_array that we can use to access the location to label
    for yi in range (Ny):
        for xi in range(Nx):
            y = 0
            tmp = jnp.array([])
            for i in loc_array_bulk:
                x = 0
                for j in i:
                    if (j == jnp.array([yi, xi])).all():
                        tmp = jnp.append(tmp, jnp.array([y, x]))
                    x += 1
                y += 1
            label_bulk[yi, xi] = tmp.reshape(-1, 2)
    for yi in range (Ny):
        for xi in range(Nx):
            y = 0
            tmp = jnp.array([])
            for i in loc_array_edge:
                x = 0
                for j in i:
                    if (j == jnp.array([yi, xi])).all():
                        tmp = jnp.append(tmp, jnp.array([y, x]))
                    x += 1
                y += 1
            label_edge[yi, xi] = tmp.reshape(-1, 2)
    for yi in range (Ny):
        for xi in range(Nx):
            y = 0
            tmp = jnp.array([])
            for i in loc_array_corner:
                x = 0
                for j in i:
                    if (j == jnp.array([yi, xi])).all():
                        tmp = jnp.append(tmp, jnp.array([y, x]))
                    x += 1
                y += 1
            label_corner[yi, xi] = tmp.reshape(-1, 2)
    return label_bulk, label_edge, label_corner

def loc_array(Ny, Nx):
    I, J = jnp.meshgrid(jnp.arange(1, Ny - 1), jnp.arange(1, Nx - 1), indexing='ij')
    bulk_coordinates = jnp.stack([I, J, I, J + 1, I + 1, J, I, J - 1, I - 1, J], axis=-1)
    loc_array_bulk = bulk_coordinates.reshape(-1, 5, 2)

    # Add edge coordinates excluding corners for the left and right sides of the grid
    edge_coordinates = []
    for i in range(1, Ny - 1):
        edge_coordinates.extend([[i, 0], [i, 1], [i + 1, 0], [i - 1, 0]])
        edge_coordinates.extend([[i, Nx - 1], [i, Nx - 2], [i + 1, Nx - 1], [i - 1, Nx - 1]])

    # Add edge coordinates excluding corners for the top and bottom of the grid
    for j in range(1, Nx - 1):
        edge_coordinates.extend([[0, j], [1, j], [0, j - 1], [0, j + 1]])
        edge_coordinates.extend([[Ny - 1, j], [Ny - 2, j], [Ny - 1, j - 1], [Ny - 1, j + 1]])

    # Convert list to a JAX array
    loc_array_edge = jnp.array(edge_coordinates).reshape(-1, 4, 2)

    loc_array_corner = jnp.array([[[0, 0], [0, 1], [1, 0]],
                                  [[0, Nx - 1], [0, Nx - 2], [1, Nx - 1]],
                                  [[Ny - 1, 0], [Ny - 1, 1], [Ny - 2, 0]],
                                  [[Ny - 1, Nx - 1], [Ny - 1, Nx - 2], [Ny - 2, Nx - 1]]])
    return loc_array_bulk, loc_array_edge, loc_array_corner
def pauli_cmi_pattern(pauli_array_bulk, pauli_array_edge, pauli_array_corner, label_bulk, label_edge, label_corner, cmi_pattern, key, sparsity, L):
    if (cmi_pattern == "no_decay"):
        print("Pattern: no_decay")
        for i in label_bulk:
            pauli_array_bulk = pauli_array_bulk.at[label_bulk[i][:,0].astype(int), label_bulk[i][:,1].astype(int)].set(-pauli_array_bulk[label_bulk[i][:,0].astype(int), label_bulk[i][:,1].astype(int)]+4)
        for i in label_edge:
            pauli_array_edge = pauli_array_edge.at[label_edge[i][:,0].astype(int), label_edge[i][:,1].astype(int)].set(-pauli_array_edge[label_edge[i][:,0].astype(int), label_edge[i][:,1].astype(int)]+4)
        for i in label_corner:
            pauli_array_corner = pauli_array_corner.at[label_corner[i][:,0].astype(int), label_corner[i][:,1].astype(int)].set(-pauli_array_corner[label_corner[i][:,0].astype(int), label_corner[i][:,1].astype(int)]+4)
    elif(cmi_pattern == "random"):
        print("Pattern:random")
        for i in label_bulk:
            key, subkey = split(key, 2)
            p = jax.random.uniform(subkey, jnp.array([1]), float, 0 , 1)
            if p>0.5:
                pauli_array_bulk = pauli_array_bulk.at[label_bulk[i][:,0].astype(int), label_bulk[i][:,1].astype(int)].set(-pauli_array_bulk[label_bulk[i][:,0].astype(int), label_bulk[i][:,1].astype(int)]+4)
                pauli_array_edge = pauli_array_edge.at[label_edge[i][:,0].astype(int), label_edge[i][:,1].astype(int)].set(-pauli_array_edge[label_edge[i][:,0].astype(int), label_edge[i][:,1].astype(int)]+4)
                pauli_array_corner = pauli_array_corner.at[label_corner[i][:,0].astype(int), label_corner[i][:,1].astype(int)].set(-pauli_array_corner[label_corner[i][:,0].astype(int), label_corner[i][:,1].astype(int)]+4)
    elif (cmi_pattern == "ordered_random"):
        print("Pattern:ordered_random")
        for i in label_bulk:
            if (i[0]%2 == 0):
                if ((i[0]*L+i[1]+1)%sparsity==0):
                    pauli_array_bulk = pauli_array_bulk.at[
                        label_bulk[i][:, 0].astype(int), label_bulk[i][:, 1].astype(int)].set(
                        -pauli_array_bulk[label_bulk[i][:, 0].astype(int), label_bulk[i][:, 1].astype(int)] + 4)
                    pauli_array_edge = pauli_array_edge.at[
                        label_edge[i][:, 0].astype(int), label_edge[i][:, 1].astype(int)].set(
                        -pauli_array_edge[label_edge[i][:, 0].astype(int), label_edge[i][:, 1].astype(int)] + 4)
                    pauli_array_corner = pauli_array_corner.at[
                        label_corner[i][:, 0].astype(int), label_corner[i][:, 1].astype(int)].set(
                        -pauli_array_corner[label_corner[i][:, 0].astype(int), label_corner[i][:, 1].astype(int)] + 4)
            elif (i[0]%2 == 1):
                if (((i[0]+1)*L-i[1])%sparsity==0):
                    pauli_array_bulk = pauli_array_bulk.at[
                        label_bulk[i][:, 0].astype(int), label_bulk[i][:, 1].astype(int)].set(
                        -pauli_array_bulk[label_bulk[i][:, 0].astype(int), label_bulk[i][:, 1].astype(int)] + 4)
                    pauli_array_edge = pauli_array_edge.at[
                        label_edge[i][:, 0].astype(int), label_edge[i][:, 1].astype(int)].set(
                        -pauli_array_edge[label_edge[i][:, 0].astype(int), label_edge[i][:, 1].astype(int)] + 4)
                    pauli_array_corner = pauli_array_corner.at[
                        label_corner[i][:, 0].astype(int), label_corner[i][:, 1].astype(int)].set(
                        -pauli_array_corner[label_corner[i][:, 0].astype(int), label_corner[i][:, 1].astype(int)] + 4)
    else:
        print("Pattern:decay")
    return pauli_array_bulk, pauli_array_edge, pauli_array_corner

def off_diag_count(xy_loc_bulk, xy_loc_edge, xy_loc_corner):
    off_diag_bulk_count, off_diag_edge_count, off_diag_corner_count = 0, 0, 0
    for i in xy_loc_bulk:
        if i[1] != 0:
            off_diag_bulk_count += xy_loc_bulk[i].shape[0]
        else:
            diag_bulk_count = 1
    for i in xy_loc_edge:
        if i[1] != 0:
            off_diag_edge_count += xy_loc_edge[i].shape[0]
        else:
            diag_edge_count = 1
    for i in xy_loc_corner:
        if i[1] != 0:
            off_diag_corner_count += xy_loc_corner[i].shape[0]
        else:
            diag_corner_count = 1
    return off_diag_bulk_count, off_diag_edge_count, off_diag_corner_count