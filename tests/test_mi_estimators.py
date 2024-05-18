"""
Test cases of the continuous, binless MI estimator implemented in
utils.continuous_mi_estimator based on Kraskov et al., PRE, 2004.

@author: frbourassa
May 17, 2024
"""

import numpy as np
import sys
if not ".." in sys.path:  # For import of local functions
    sys.path.insert(1, "..")

from utils.continuous_mi_estimator import (
    kraskov_continuous_info,
    kraskov_differential_entropy
)
from utils.random_matrices import random_covariance_mat


def compare_mi_test(estim, exact, tol=0.1):
    msg = "Estimate: {:.6E}, true: {:.6E}".format(estim, exact)
    if abs(estim - exact) > tol:
        raise AssertionError("Difference too large!", msg)
    else:
        print("Error within tolerance.", msg)
    return 0


def mi_mvn(cov_mat, x_idx, y_idx, base=np.e):
    """ Compute the MI between variables indexed in x_idx and variables
    indexed in y_idx. They each have Gaussian marginal distributions as well.

    MI = \log (det(cov_mat) / (det(cov_mat[x_idx, x_idx]) * det(cov_mat[y_idx, y_idx])))
    """
    n_tot, n_x, n_y = cov_mat.shape[0], x_idx.shape[0], y_idx.shape[0]
    assert n_tot == n_x + n_y
    assert np.all(np.sort(np.concatenate([x_idx, y_idx])).astype(int)
                    == np.arange(n_tot, dtype=int))
    cov_mat_x = cov_mat[x_idx][:, x_idx]
    cov_mat_y = cov_mat[y_idx][:, y_idx]
    det_z = np.linalg.det(cov_mat)
    det_x = np.linalg.det(cov_mat_x)
    det_y = np.linalg.det(cov_mat_y)
    if det_x > 0.0 and det_y > 0.0 and det_z > 0.0:
        mi = 0.5 * np.log(det_x * det_y / det_z)
    else:
        mi = np.nan

    return mi / np.log(base)


def diff_ent_mvn(cov_mat, base=np.e):
    det_x = np.linalg.det(cov_mat)
    n = cov_mat.shape[0]
    ent = 0.5 * n * (np.log(2.0*np.pi) + 1.0) + 0.5 * np.log(det_x)
    return ent / np.log(base)


def main_test_independent():
    nsamp = int(1e4)
    meanx = np.zeros(2)
    meany = np.ones(2)
    cov = np.eye(2)
    rndgen = np.random.default_rng(seed=12323452)
    xvecs = rndgen.multivariate_normal(mean=meanx, cov=cov, size=[nsamp, 2])
    yvecs = rndgen.multivariate_normal(mean=meany, cov=cov, size=[nsamp, 2])

    # Add identical points to check what happens then
    xvecs = np.concatenate([xvecs, xvecs[-2:-1]], axis=0)
    yvecs = np.concatenate([yvecs, yvecs[-2:-1]], axis=0)

    # Fast, approximate algorithm using KDTrees and vectorized queries
    print("Testing version 1 on independent samples...")
    mi_fast = kraskov_continuous_info(xvecs, yvecs, version=1, k=6, base=2)
    compare_mi_test(mi_fast, 0.0, tol=1e-1)

    print("Testing version 2 on independent samples...")
    mi_fast = kraskov_continuous_info(xvecs, yvecs, version=2, k=6, base=2)
    compare_mi_test(mi_fast, 0.0, tol=1e-1)

    return 0


def main_test_random_mvn():
    nsamp = int(1e5)
    rndgen = np.random.default_rng(0xedbf0accecc93b0ea42d3c788232d232)
    n_x, n_y = 2, 2
    n_tot = n_x + n_y
    x_ind = np.arange(n_x)
    y_ind = np.arange(n_x, n_tot)
    cov_mat_tot = random_covariance_mat(n_tot, rng=rndgen)
    mean_tot = np.zeros(n_tot)
    analytical_mi = mi_mvn(cov_mat_tot, x_ind, y_ind, base=2)
    # Samples
    xy_samples = rndgen.multivariate_normal(mean_tot, cov_mat_tot, size=nsamp)

    # MI estimators
    print("Testing version 1 on MVN samples...")
    mi_fast_1 = kraskov_continuous_info(
        xy_samples[:, x_ind], xy_samples[:, y_ind], version=1, k=6, base=2
    )
    compare_mi_test(mi_fast_1, analytical_mi, tol=1e-1)

    print("Testing version 2 on MVN samples...")
    mi_fast_2 = kraskov_continuous_info(
        xy_samples[:, x_ind], xy_samples[:, y_ind], version=2, k=6, base=2
    )
    compare_mi_test(mi_fast_2, analytical_mi, tol=1e-1)

    return 0


def main_test_entropy():
    nsamp = int(1e5)
    n_dim = 3
    rndgen = np.random.default_rng(0xddbf0ac0efc93b0ea42d3dacb232d111)
    cov_mat = random_covariance_mat(n_dim, rng=rndgen)
    mean_tot = np.zeros(n_dim)
    samples = rndgen.multivariate_normal(mean_tot, cov_mat, size=nsamp)

    # Exact entropy
    analytical_ent = diff_ent_mvn(cov_mat, base=2)

    # Estimated entropy
    ent_estim = kraskov_differential_entropy(samples, k=6, version=1, base=2)
    compare_mi_test(ent_estim, analytical_ent, tol=1e-1)

    return 0

if __name__ == "__main__":
    # Test case 1 : independent X and Y, each 2D
    #main_test_independent()

    # Test with multivariate gaussians, for which we know the analytical MI
    #main_test_random_mvn()

    # Test differential entropy estimator.
    main_test_entropy()
