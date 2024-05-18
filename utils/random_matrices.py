"""
A few utility functions to generate random matrices to prepare simulations
or tests: random orthogonal ensemble, random covariance matrices.

@author: frbourassa
May 17, 2024
"""
import numpy as np


# Generate a random orthogonal matrix
def random_orthogonal_mat(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # Scipy or my version both give similar test results
    #return sp.stats.ortho_group.rvs(dim=n, size=1, random_state=rng)
    q, r = np.linalg.qr(rng.standard_normal(size=[n, n]), mode="complete")
    return q.dot(np.diagflat(np.sign(np.diagonal(r))))


def random_covariance_mat(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # Standard deviation matrix, diagonalized
    gtilde_std = np.sqrt(np.diagflat(rng.random(n)))
    # Generate a random orthogonal matrix R to rotate this diagonal
    # set of variances and create correlated gaussians
    r_mat = random_orthogonal_mat(n, rng=rng)
    # Square root decomposition of the actual covariance matrix, U sqrt(D)
    psi_mat = r_mat.dot(gtilde_std)
    # Return Sigma = U D U^T = (R sqrt(D)) (R sqrt(D))^T = psi psi^T
    cov_mat = psi_mat.dot(psi_mat.T)
    return cov_mat
