""" Various distances and other metrics.

@author: frbourassa
June 2022
"""
import numpy as np
import itertools

def l2_norm(vecs, axis=-1):
    r""" Computes l2 norm of vectors stored along the last axis of vecs.
    Args:
        vecs can be either a single vector (1d) or an  arbitrary array of vectors,
            where the last dimension indexes elements of vectors.
        axis (int): which axis to sum along.

    Returns: array of distances of same shape as vecs
        except for the summation axis, removed.
    """
    return np.sqrt(np.sum(vecs**2, axis=axis))

def l1_norm(vecs, axis=-1):
    r""" |x| = \sum_i |x_i|"""
    return np.sum(np.abs(vecs), axis=axis)

def lp_norm(vecs, p, axis=-1):
    r""" |x| = (\sum_i |x_i|^p)**(1.0/p) """
    # Use numpy implementation
    return np.linalg.norm(vecs, ord=p, axis=axis)

def linf_norm(vecs, axis=-1):
    """ |x| = max_i(|x_i|) """
    return np.max(np.abs(vecs), axis=axis)

def cosine_dist(x, y, axis=-1):
    r""" d(x, y) = 1 - (x \cdot y)/(|x| |y|)"""
    xnorm, ynorm = l2_norm(x), l2_norm(y)
    return 1.0 - x.dot(np.moveaxis(y, axis, 0)) / xnorm / ynorm

def index_closest(x, refs, metric="r2"):
    """ Return the index of the element in refs that is closest to x"""
    if metric == "r2":
        return np.argmin((refs - x)**2)
    else:
        raise NotImplementedError("Did not bother coding that metric yet")
    return None


def frobnorm(mat):
    """ Compute Frobenius norm of matrix A,
    ||A||^2 = Tr(A^T A). """
    return np.trace(mat.T.dot(mat))


def subspace_align_error(mat, target):
    """ Compute min_Q ||Q.dot(mat) - target||^2 / ||target||^2.
    The solution to that orthogonal Procrustes problem is
        Q = U V^T, where USV^T is the SVD of target.dot(mat.T)
    according to Wikipedia, citing the solution of Schönemann, 1966.
    (https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)
    """
    # Solve Procrustes problem
    u, s, vh = np.linalg.svd(target.dot(mat.T))
    q = u.dot(vh)
    # Compute alignment error
    numer = frobnorm(q.dot(mat) - target)
    denom = frobnorm(target)
    dist = 0.0
    if not np.allclose(denom, 0.0):
        dist = numer / denom
    elif numer == 0.0:
        dist = 0.0
    else: # division of a positive number by 0
        dist = np.inf
    return dist


def jaccard(s1, s2):
    # exclude the special case of s1 and s2 empty sets
    if len(s1) > 0 or len(s2) > 0:
        return len(s1 & s2) / len(s1 | s2)
    else:
        return 0.0


# From the itertools documentation
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r)
            for r in range(len(s)+1))
