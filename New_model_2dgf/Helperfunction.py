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
    sample_tmp = samples
    for xyloc_ind in xyloc:
        if xyloc[xyloc_ind].size != 0:
            sample_tmp = jnp.append(sample_tmp, jax.lax.scan(scan_array_element, samples, xyloc[xyloc_ind])[1])
    return sample_tmp.reshape(-1, samples.shape[0], samples.shape[1])

@jax.jit
def new_coe_2d(samples, coe_array, yloc, zloc):
    def ycoe(sample_element, yloc_arrays_element):
        scan_coe_tmp_y = ((-1)**sample_element[yloc_arrays_element[:,0],yloc_arrays_element[:,1]]*1j).prod()
        return sample_element, scan_coe_tmp_y
    def zcoe(sample_element, zloc_arrays_element):
        scan_coe_tmp_z = ((-1)**sample_element[zloc_arrays_element[:,0], zloc_arrays_element[:,1]]).prod()
        return sample_element, scan_coe_tmp_z
    coe_tmp_y = jnp.array([0])
    coe_tmp_z = jnp.array([0])
    for yloc_ind in yloc:
        if yloc[yloc_ind].shape[0] != 0:
            coe_tmp_y = jnp.append(coe_tmp_y, jax.lax.scan(ycoe, samples, yloc[yloc_ind])[1])
        else:
            coe_tmp_y = jnp.append(coe_tmp_y, 1)
    for zloc_ind in zloc:
        if zloc_ind[1] == 0:
            coe_tmp_z = coe_tmp_z.at([0]).set(jax.lax.scan(zcoe, samples, zloc[zloc_ind])[1].sum())
        elif zloc[zloc_ind].shape[0] != 0:
            coe_tmp_z = jnp.append(coe_tmp_z, jax.lax.scan(zcoe, samples, zloc[zloc_ind])[1])
        else:
            coe_tmp_z = jnp.append(coe_tmp_z, 1)
    return coe_tmp_y*coe_tmp_z*jnp.concatenate((jnp.array([1]), coe_array), axis=0)

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
    bulk_coordinates = jnp.stack([I, J, I, J + 1, I + 1, J, I - 1, J, I, J - 1], axis=-1)
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