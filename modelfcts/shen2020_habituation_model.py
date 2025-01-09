import numpy as np
import scipy as sp
from scipy import sparse

from utils.metrics import jaccard


def generate_odorant(n_rec, rgen, mean_in=10.0):
    """ Generate input vector.

    Args:
        n_rec (int): number of receptor types, length of vectors
        mean_in (float): average value of the input vector.
        rgen (np.random.Generator): random generate (numpy >= 1.17)
    Returns:
        kappa1_vec (np.ndarray): 1d vector of receptor activities
    """
    return rgen.exponential(scale=mean_in, size=n_rec)


def generate_background(n_rec, rgen, lambda_in=10.0):
    """ Generate vectors eta and kappa^-1 for an odorant, with antagonism parameter rho.

    Args:
        n_rec (int): number of receptor types, length of vectors
        mean_in (float): average value of the input vector.
        rgen (np.random.Generator): random generate (numpy >= 1.17)
    Returns:
        kappa1_vec (np.ndarray): 1d vector of receptor activities
    """
    raise NotImplementedError("You should use generate_odorant instead")


def project_neural_tag_shen2020(x_vec, w_vec, projmat, kc_sparsity=0.05, adapt_kc=True, n_pn_per_kc=3, fix_thresh=None):
    """ Given the parameters of the Shen 2020 neural network, project the input layer x_vec
    with the inhibitory feedback weights w_vec to the sparse kenyon cell (KC) output,
    thresholding KCs below kc_thresh and then keeping only a fraction kc_sparsity of active KCs.

    Args:
        x_vec (np.ndarray): input vector, activation of each receptor type
        w_vec (np.ndarray): inhibition weights from LN1 to PN neurons
        projmat (np.ndarray or sp.sparse.csr_matrix): projection matrix from PNs to KCs,
            shape n_kc x n_receptors. Will use the .dot method of the matrix.

    Returns:
        z_set (set): sparse neural tag for the given activation odor. List of active KCs.
    """
    # Extract useful information
    n_kc = projmat.shape[0]
    n_rec = projmat.shape[1]
    # Number of connections from PNs to each KC. If three, then threshold equals mean of inputs
    # Otherwise we need to correct for the fact that the more KCs project to each PN,
    # the higher the signal of each PN will be compared to the case described in the paper (3 KCs to each PN)
    if fix_thresh is not None:
        kc_thresh = fix_thresh
    elif adapt_kc:
        kc_thresh = np.mean(x_vec) * n_pn_per_kc / 3
    else:
        kc_thresh = np.mean(x_vec)

    # 1. Project on PNs, including inhibition from LN1
    x_vec = np.maximum(x_vec - w_vec, 0)
    #x_vec = x_vec - w_vec
    #x_vec[x_vec<0] = 0

    # 2. Project on KCs
    y_vec = projmat.dot(x_vec)

    # 3. Threshold noise:: will consider only positions in mask.
    mask = (y_vec >= kc_thresh).astype(bool)
    y_vec[np.logical_not(mask)] = 0.0
    #mask2 = (projmat.dot(x_vec/6) >= kc_thresh)
    #print("Non-zero elements after thresholding by mean {}: ".format(kc_thresh), np.count_nonzero(mask))
    #print("Compare to expected:", np.count_nonzero(mask2))

    # 4. Binarize: keep the np.ceil(0.05*n_kc) most active KCs non-zero KCs,
    # or all the nonzero ones. Return a set of the indices of those cells.
    # No arbitrary tie breaks, so can't just sort and take the first 0.05n_kc args
    # TODO: make sure this is a fine format and we don't need the full vector
    # Otherwise, use an array of booleans for the binary vector, or a sp.sparse?
    thresh_keep = np.quantile(y_vec, 1.0 - kc_sparsity)
    z_set = set(np.nonzero(np.logical_and(mask, y_vec >= thresh_keep))[0])
    #y_vec[y_vec < thresh_keep] = 0.0
    #z_set = set(np.nonzero(y_vec)[0])
    return z_set


def jaccard(s1, s2):
    if len(s1) > 0 or len(s2) > 0:  # exclude the special case of s1 and s2 empty sets
        return len(s1 & s2) / len(s1 | s2)
    else:
        return 0.0


def create_sparse_proj_mat(n_kc, n_rec, rgen, fraction_filled=6/50):
    n_per_row = int(fraction_filled*n_rec)
    data = np.ones(n_per_row*n_kc, dtype="uint8")
    row_ind = np.arange(n_per_row*n_kc, dtype=int) // n_per_row
    col_ind = np.ones((n_kc, n_per_row), dtype="int")
    for i in range(n_kc):
        col_ind[i] = rgen.choice(n_rec, size=n_per_row, replace=False)
    mat = sp.sparse.csr_matrix((data, (row_ind, col_ind.ravel())), shape=(n_kc, n_rec), dtype="uint8")
    return mat


def time_evolve_habituation_fixed(w_vec0, backgnd, nstep, learnrates):
    """ Take initial conditions for w_i and C_tot, evolve them in time against a fixed odor backgnd.
    Nothing random here as we don't fluctuate the odor.

    Args:
        w_vec0 (np.ndarray): initial weights 1D vector, should be of same length
            as vectors in backgnd (number of receptors)
        backgnd (np.ndarray): input vector of the background against which we habituate
        nstep (int): number of time steps to take.

    Returns:
        w_vect (np.ndarray): final weights vector
    """
    # Extract parameters
    if nstep > 1e6:
        raise ValueError("Consider asking for less than 1e6 steps at a time")
    alpha, beta = learnrates

    # Initialize variables
    w_vec = w_vec0.copy()
    t = 0  # number of steps taken

    # Iterate until satisfing the number of steps asked for
    while t < nstep:
        # Update w, remove the max(s-w, 0) thing for the update rule, see if it still works (it should)
        #x_vec = np.maximum(backgnd - w_vec, 0)
        #w_vec = w_vec + alpha * x_vec - beta* w_vec
        w_vec = w_vec + alpha * backgnd - (alpha + beta)* w_vec
        t += 1

    return w_vec


def combine_odorants(od1, od2, frac1):
    """ Compute the new input vector after linearly combining two odorant vectors,
    as frac1*od1 + (1-frac1)*od2

    Args:
        od1, od2 (np.ndarrays): input vectors of odorants 1 and 2
        frac1 (float): number between 0 and 1, proportion of od1 in new mixture.
    Returns:
        od_mix (np.ndarrays): input vector of the mixture.
    """
    if frac1 < 0.0 or frac1 > 1.0:
        raise ValueError("frac1 should be a float in [0.0, 1.0], not {}".format(frac1))
    return frac1*od1 + (1-frac1)*od2
