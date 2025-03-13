"""
@author: frbourassa
2023
"""

import os, json
from scipy import sparse
import numpy as np


# Data wrangling functions
def strip_names(df, char, axis=0):
    stripper = lambda x: x.strip(char)
    return df.rename(mapper=stripper, axis=axis)


def pad_names(df, char, axis=0):
    padder = lambda x: char + x + char
    return df.rename(mapper=padder, axis=axis)


# Data import and export functions
def dict_to_hdf5(gp, di):
    """ Store dictionary di values as datasets in hdf5 group gp"""
    for k in di.keys():
        gp[k] = di[k]
    return gp


def save_params_individually(gp, di):
    """ More robust version of dict_to_hdf5, recursively
    applies to array slices of uneven lengths """
    for k in di.keys():
        try:
            v = np.asarray(di[k])
        # Recursively try to split the uneven array into a dict of arrays
        # label them as k_i
        except ValueError:
            vdict = {k + "_" + str(i):di[k][i] for i in range(len(di[k]))}
            save_params_individually(gp, vdict)
        else:
            gp[k] = v
    return gp


def hdf5_to_dict(gp):
    """ Retrieve datasets of hdf5 group gp into a dictionary"""
    di = {}
    for k in gp:
        di[k] = gp.get(k)[()]
    return di

def csr_matrix_to_hdf5(gp, mat):
    """ Store scipy.sparse.csr_matrix to a HDF5 group, by storing
    its data, indices, indptr, shape attributes.
    Inspired from https://stackoverflow.com/questions/11129429/
    storing-numpy-sparse-matrix-in-hdf5-pytables
    """
    # If there is only one non-null value stored in this matrix
    if mat.data.size == 0:  # Empty matrix, avoid checking data[0] below
        gp.attrs["data_bits_packed"] = False
        gp.create_dataset("data", data=mat.data)
    elif np.all(mat.data == mat.data[0]):  # Only one data value
        gp.attrs["data_bits_packed"] = False
        gp.create_dataset("data", data=mat.data[0])
    # If there are Trues and Falses, pack bits of boolean arrays 
    # to save a bit of space.
    elif mat.dtype in [bool, np.bool_]:
        gp.attrs["data_bits_packed"] = True
        gp.create_dataset("data", data=np.packbits(mat.data))
    else:
        gp.attrs["data_bits_packed"] = False
        gp.create_dataset("data", data=mat.data)

    # Use smallest possible int type for indices and indptr
    gp.create_dataset("indices",  # index within row
                    data=mat.indices.astype(np.min_scalar_type(mat.shape[1])))
    gp.create_dataset("indptr",  # index of last element
                    data=mat.indptr.astype(np.min_scalar_type(mat.size)))
    gp.create_dataset("shape", data=mat.shape)
    return gp


def hdf5_to_csr_matrix(gp):
    is_packed = gp.attrs.get("data_bits_packed")
    if is_packed:
        dat = np.unpackbits(gp.get("data")).astype(bool)
        mat = sparse.csr_matrix(
                    (dat[:gp.get("indices").size],  # remove padding
                    gp.get("indices"), gp.get("indptr")),
                    shape=gp.get("shape"), dtype=bool
                )
    else:
        dat0 = gp.get("data")
        ind = gp.get("indices")
        if dat0.shape == ():
            dat = np.full(ind.shape, dat0)
        else:
            dat = dat0
        mat = sparse.csr_matrix(
                    (dat, ind, gp.get("indptr")),
                    shape=gp.get("shape"), dtype=gp.get("data").dtype
                )
    return mat


def save_defaults_json(di, fpath, overwrite=False):
    """ Save a dictionary to a JSON file, but if the file already exists
    update it with values for keys not already in it, without replacing
    existing values.
    """
    # File does not exist or can be overwritten
    if not os.path.isfile(fpath) or overwrite:
        with open(fpath, "w") as file:
            json.dump(di, file, indent=2)
        di_file = di
    else:
        with open(fpath, "r") as file:
            di_file = json.load(file)
        with open(fpath, "w") as file:
            for k in di.keys():
                di_file.setdefault(k, di[k])
            json.dump(di_file, file, indent=2)

        # If performance was important, we should use builtins only.
        # But side effect: the input dict is modified in-place, so avoid.
        #di.update(di_file)
        #json.dump(di, file, indent=2)

    return di_file


# Printing functions
def nice_dict_print(di):
    for k in di.keys():
        print(str(k) + ":", di[k])
    return di


# Appending to .npz file
def add_to_npz(filename, arrays_to_add):
    """
    Appending to npz file is not recommended,
    because we need to load and save again. 
    But sometimes it is convenient anyways. 

    Args:
        filename (str): name of the npz file to append to; 
            will be created if it doesn't exist
        arrays_to_add (dict): names and arrays to append
    
    Returns: 0 if appended, 1 if created new file
    """
    try:
        f = np.load(filename)
    except FileNotFoundError:
        existing_arrays = {}
        code_exit = 1
    else:
        existing_arrays = {k:f[k] for k in f.keys()}
        f.close()
        code_exit = 0
    # Re-save existing arrays, add the mixture_yvecs
    existing_arrays.update(arrays_to_add)
    np.savez_compressed(filename, **existing_arrays)
    return code_exit



# Reconstruct objects with .slope, .intercept, etc. attributes from saved dict.
class LinRegRes():
    def __init__(self, di):
        for k in di:
            setattr(self, k, di[k])
