"""
Attempt at coding a faster sparse matrix dot product using numba. 

Unfortunately, we get close, but still slower than scipy for our use-case.  
Hence, we can't easily speed these operations up (unless using GPUs). 
The neural tag calculations will have to take a long time in high dimensions. 

@author: frbourassa
November 2024
"""

import numpy as np
from scipy import sparse
from numba import jit, njit, uint8, int32, float64, int32


# Wrapper function taking the sparse matrices, then calling the 
# right numba-compiled product function
def fastdot(p, x):
    if isinstance(p, sparse.csr_matrix) and isinstance(x, np.ndarray):
        if x.ndim == 1:
            v = dot_csr_dense_vec(p.data, p.indices, p.indptr, p.shape, x)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return v

@njit
def dot_csr_dense_vec(data, indices, indptr, shape, x):
    v = np.zeros(shape[0])
    for i in range(shape[0]):
        # From Scipy's doc: the column indices for row i are stored in 
        # indices[indptr[i]:indptr[i+1]] and their corresponding
        # values are stored in data[indptr[i]:indptr[i+1]]
        v[i] = np.sum(x[indices[indptr[i]:indptr[i+1]]] 
                      * data[indptr[i]:indptr[i+1]])
    return v


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../")
    from modelfcts.tagging import create_sparse_proj_mat
    from utils.profiling import DeltaTimer

    n_s, n_k, n_tries = 1000, 40000, 10
    
    rng = np.random.default_rng(seed=1342124000810298494591)
    p = create_sparse_proj_mat(n_k, n_s, rng, fraction_filled=6/50)
    x = rng.random(size=n_s)

    # Reference result
    v1 = p.dot(x)
    # First attempt to compile and make sure the calculation is correct
    v2 = fastdot(p, x)
    assert np.allclose(v1, v2, rtol=1e-14)

    # Now, time it
    timer = DeltaTimer()
    timer.start()
    for _ in range(n_tries):
        v1 = p.dot(x)
    t1 = timer.delta()
    print("Time for scipy method:", t1/n_tries)
    for _ in range(n_tries):
        v2 = fastdot(p, x)
    t2 = timer.delta()
    print("Time for numba-compiled method:", t2/n_tries)
    
    





