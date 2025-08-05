""" Functions to compute neural tags of odors in the Kenyon cells layer,
based on the olfactory network model in Shen et al., PNAS, 2020. This sparse
projection to a high dimensional layer implements locality-sensitive hashing.

@author: frbourassa
September 2023
"""
import numpy as np
import scipy as sp
from scipy import sparse
from utils.export import csr_matrix_to_hdf5, hdf5_to_csr_matrix

### SPARSE STORAGE CLASS ###

def get_2dshape(ndshape):
    return (np.prod(ndshape[:-1]), ndshape[-1])

class SparseNDArray(sparse.lil_array):
    """ Sparse ndarray internally stored in memory as a 2D lil_array.
    Can be sliced externally as a ndarray; all indices except the last
    are raveled to access the internal 2d array.
    Made for fast access/storage operations, not fast arithmetic operations.
    When saved to disk, it is converted to a csr_matrix and saved with an
    extra ndshape attribute, so the SparseNDArray can be reconstructed.
    Either initialized empty with a shape as the sole non-keyword argument,
    or from an existing sparse matrix (useful for retrieving array
    saved to disk).

    Using lil array because usually this array is filled in a sorted order
    after creation, but consider using dok if the order in which elements
    are added is unpredictable.

    In all dimensions except the last, can only slice with indices or
    list of indices that can be passed to np.ravel_multi_index.
    Implement a slice-to-integer indices conversion method if fancier
    slicing is ever needed.
    """
    def __init__(self, arg1, ndshape=None, dtype=None, copy=False):
        # Catch the intended ndshape, convert to underlying lil_array shape
        if isinstance(arg1, (tuple, list, set)):
            if ndshape is not None:
                assert tuple(ndshape) == tuple(arg1), "Inconsistent shapes"
            self.ndshape = tuple(arg1)
            super().__init__(get_2dshape(self.ndshape),
                                dtype=dtype, copy=copy)
        elif isinstance(arg1,(np.ndarray, np.matrix)) or sparse.issparse(arg1):
            if ndshape is None:
                raise ValueError("Cannot create from existing 2D array "
                                    + "without specifying ndshape")
            elif np.prod(ndshape) != np.prod(arg1.shape):
                raise ValueError("Inconsistent ndshape {}".format(ndshape)
                            + "and underyling 2d shape {}".format(arg1.shape))
            self.ndshape = tuple(ndshape)
            super().__init__(arg1, shape=get_2dshape(self.ndshape),
                                        dtype=dtype, copy=copy)
        else:
            raise ValueError("arg1 must be either a shape list or tuple, "
                                "or an existing 2d array")

    def __getitem__(self, item):
        # Create the underlying 2d index
        item2d = (np.ravel_multi_index(item[:-1], self.ndshape[:-1]), item[-1])
        return super().__getitem__(item2d)

    def __setitem__(self, item, values):
        item2d = (np.ravel_multi_index(item[:-1], self.ndshape[:-1]), item[-1])
        return super().__setitem__(item2d, values)

    def to_hdf(self, hdf_gp):
        """ hdf_gp: h5Py File or group in that file for this matrix,
            or anything else with create_dataset properties.
        """
        csr_matrix_to_hdf5(hdf_gp, self.tocsr())
        hdf_gp.create_dataset("ndshape", data=self.ndshape)

    def todense(self):
        return super().todense().reshape(self.ndshape)

    # Class function, call SparseNDArray.read_from_hdf(hdf_group)
    def read_from_hdf(self, hdf_gp):
        csrmat = hdf5_to_csr_matrix(hdf_gp)
        ndarr = SparseNDArray(csrmat, ndshape=hdf_gp.get("ndshape"))
        return ndarr

### PROJECTION FUNCTIONS ###

def create_sparse_proj_mat(n_kc, n_rec, rgen, fraction_filled=6/50):
    n_per_row = int(fraction_filled*n_rec)
    data = np.ones(n_per_row*n_kc, dtype="uint8")
    row_ind = np.arange(n_per_row*n_kc, dtype=int) // n_per_row
    col_ind = np.ones((n_kc, n_per_row), dtype="int")
    for i in range(n_kc):
        col_ind[i] = rgen.choice(n_rec, size=n_per_row, replace=False)
    mat = sp.sparse.csr_matrix((data, (row_ind, col_ind.ravel())), shape=(n_kc, n_rec), dtype="uint8")
    return mat


def relu_copy(x):
    return x * (x > 0)

def project_neural_tag(y_vec, x_vec, projmat, **proj_kwargs):
    """ Project the input layer x_vec
    with the inhibitory feedback weights w_vec to the sparse kenyon cell (KC) output,
    thresholding KCs below kc_thresh and then keeping only a fraction kc_sparsity of active KCs.

    Args:
        y_vec (np.ndarray): PN layer vector
        x_vec (np.ndarray): input layer vector, used to adjust threshold
        projmat (np.ndarray or sp.sparse.csr_matrix): projection matrix
            from PNs to KCs, shape n_kc x n_receptors.
            Will use the .dot method of the matrix.
    Keyword args: proj_kwargs
        kc_sparsity (float): fraction of most activated KCs kept in the tag
        adapt_kc (bool): if no fixed threshold is passed, the threshold on KCs
            is determined from the average ORN activity. If adapt_kc is True,
            this average is then multiplied by the number of PN per KC,
            divided by 3 (3 PN per KC leads to threshold = average x element).
        n_pn_per_kc (int): number of PN connected to each KC, used
            to adjust threshold
        fix_thresh (float): fixed threshold for KCs
        project_thresh_fact (float): factor by which to multiply the
            determined projection factor

    Returns:
        z_set (set): sparse neural tag for the given activation odor. List of active KCs.
    """
    # Collect keyword arguments and their default values
    kc_sparsity = proj_kwargs.get("kc_sparsity", 0.05)
    adapt_kc = proj_kwargs.get("adapt_kc", True)
    n_pn_per_kc = proj_kwargs.get("n_pn_per_kc", 3)
    fix_thresh = proj_kwargs.get("fix_thresh", None)
    ptf = proj_kwargs.get("project_thresh_fact", 1.0)

    # Extract useful information
    n_kc = projmat.shape[0]
    n_rec = projmat.shape[1]
    # Number of connections from PNs to each KC. If three, then threshold equals mean of inputs
    # Otherwise we need to correct for the fact that the more KCs project to each PN,
    # the higher the signal of each PN will be compared to the case described in the paper (3 KCs to each PN)
    if fix_thresh is not None:
        kc_thresh = fix_thresh
    # We need to reduce the threshold to still catch new odors partially inhibited too
    elif adapt_kc:
        kc_thresh = np.mean(x_vec) * n_pn_per_kc / 3
    else:
        kc_thresh = np.mean(x_vec)
    kc_thresh *= ptf

    # 2. Project y_vec on KCs: this is the slowest part
    # But we can't really speed up scipy's C++ implementation...
    kc_vec = projmat.dot(relu_copy(y_vec))

    # 3. Threshold noise: will consider only positions in mask.
    # Keep only values strictly above threshold, so if thresh = 0,
    # then all values are masked and the tag is empty.
    mask = (kc_vec > kc_thresh).astype(np.bool_)
    kc_vec[np.logical_not(mask)] = 0.0

    # 4. Binarize: keep the np.ceil(0.05*n_kc) most active KCs non-zero KCs,
    # or all the nonzero ones. Return a set of the indices of those cells.
    # No arbitrary tie breaks: do not just sort and take the first 0.05n_kc
    thresh_keep = np.quantile(kc_vec, 1.0 - kc_sparsity)
    z_set = set(np.nonzero(np.logical_and(mask, kc_vec >= thresh_keep))[0])
    return z_set
