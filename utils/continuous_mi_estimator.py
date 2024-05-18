#! /usr/bin/env python3
""" Python implementation of the mutual information estimator proposed by
Kraskov et al., 2004, "Estimating mutual information", PRE.
Computes mutual information between two continuous vector-valued random
variables, from an array of samples (X, Y).

There are two similar versions of the estimator. The formulas are:

$$ MI(X, Y) = \psi(k) - <\psi(n_x+1) + \psi(n_y+1)> + \psi(N)   (v1) $$
$$ MI(X, Y) = \psi(k) - 1/k - <\psi(n_x) + \psi(n_y)> + \psi(N)   (v2) $$

where:
    - \psi is the digamma function;
    - N is the number of points;
    - k is the number of nearest-neighbors of the same category used to
        estimate the probability density;
    - n_x(i), n_y(i): in the first estimator, numbers of points within distance
        epsilon_i of point x_i in the X subspace or Y subspace, where epsilon_i i
        s the distance to the kth neighbor in the joint space.
        In the second estimator, n_x(i) is the number of points in the X
        subspace within distance epsilon_x(i), which is the X norm of the kth
        neighborhood in X subspace (max X distance to one of the k neighbors).
        Likewise for n_y(i): points within epsilon_y(i) in the Y subspace.
    - <\psi(n_x)> and <\psi(n_y)> are averaged over all samples.

where:
    - psi is the digamma function;
    - N is the number of points;
    - N_x is the number of points of category X=x;
    - k is the number of nearest-neighbors of the same category used to
        estimate the probability density;
    - m is the number of nearest-neighbors of any category within the ball
        extending up to the kth nearest-neighbor of the same category x;
    - <psi(N_x)> and <psi(m)> are averaged over samples of X (each category
        X=x is weighted by the number of occurences of x).

The first estimator is supposed to have smaller statistical error, but more
systematic error than the second one. The systematic errors are problematic
only for high dimensions.

Speed improved by vectorizing some operations, using Scipy's cKDTree
for nearest-neighbor search, and using its multi-core capabilities.

WARNING: limited testing of the code was carried out. It worked fine on the
test case provided below and for the authors' use cases, but results are not
guaranteed in other applications (especially with very high-dimensional data).

@author: François Bourassa (frbourassa)
May 16, 2024
"""

import numpy as np
from scipy import special
from scipy.spatial import cKDTree
import psutil


def to_2d_array(a):
    """ Convert a list of points or a 1D array to a 2D arrays where
    the first axis indexes samples, the second indexes dimensions
    """
    aa = np.asarray(a, dtype=type(a[0]))
    aa = aa.reshape(aa.shape[0], -1)
    return aa


def kraskov_continuous_info(x, y, k=3, version=1, base=np.e, eps=0):
    """
    Estimates mutual information between two continuous, vector-valued random
    variables x and y using nearest-neighbor statistics.
    Relatively fast Python implementation, using Scipy's cKDTree, of the I^(1)
    estimator in Kraskov et al., 2004, "Estimating mutual information", PRE.

    Author of this implementation: Francois Bourassa (Github: frbourassa)

    We only use the maximum norm, because this is required for distance in the
    joint space, |z-z'| = max(|x-x'|, |y-y'|). While in principle the subspace
    norms |x-x'|, |y-y'| can be different, there is no fast way to compute
    the |z-z'| distance with a cKDTree without computing all pairwise distances
    |x-x'|, |y-y'| first, except if these distances are themselves max norms,
    because then max(|x-x'|, |y-y'|) = max((x-x')_1, (x-x')_2, ..., (y-y')_N),
    i.e. we can compute the max over all individual x and y coordinates.

    For the first estimator (versions==1), x and y distances must be normalized
    in advance, to have similar scales. Otherwise, the first estimator is very
    poor since distances in the joint space are very different from distances
    in one of the subspaces, and n_x and n_y tend to k or N, respectively.

    The second estimator (version==2) is more robust in that respect, but it is
    more computationally costly, because we need to recompute distances in the
    X, Y subspaces to the kth neighbor in the joint space (i.e. compute
    epsilon_x and epsilon_y). The first estimator does not need this step
    because the radii used are the joint space epsilon.

    Args:
        x (np.ndarray): X values of samples, shaped [n_samples, n_dim_x]
        y (np.ndarray): Y values of samples, shaped [n_samples, n_dim_y]
        k (int): Number of nearest neighbors for density estimation
        version (int): 1 or 2, which version of the Kraskov estimator.
            Default: 1 because faster to compute, but sensitive to high
            dimensions and to large scale differences between X and Y.
        base (float): Logarithm base in which the MI is computed (default: e)
        eps (float): Relative tolerance on the radius up to which neighbors
            are included (default: 0, exact computation).

    Returns:
        float: Mutual information estimate
    """
    # Make sure x and y are 2D arrays
    x = to_2d_array(x)
    y = to_2d_array(y)
    xy = np.concatenate([x, y], axis=1)

    # Number of workers for parallel processing, use half of them.
    n_workers = min(1, psutil.cpu_count() // 2)

    # Build KDTrees in the joint and marginal spaces.
    n_points = x.shape[0]
    assert n_points == y.shape[0], "Mismatch in the number of samples"
    num_dims = x.shape[1] + y.shape[1]
    joint_tree = cKDTree(xy, leafsize=max(16, int(k*num_dims/4)))

    # Check that there are no exactly identical points
    identical_pairs = joint_tree.query_pairs(
        r=0.0, eps=0.0, output_type="ndarray"
    )
    # If any, perturb them slightly to avoid numerical instabilities
    dup_pts = np.unique(identical_pairs[:, 0])
    if identical_pairs.shape[0] > 0:
        print("Found identical points; perturbing them")
        # Average nn distance as a perturbation
        perturb = 1e-6*np.mean(joint_tree.query(xy, p=np.inf, k=[2])[0])
        # Perturb the first point of each pair
        increments = np.random.random(size=(dup_pts.shape[0], xy.shape[1]))
        xy[dup_pts, :] = xy[dup_pts, :] + perturb*(increments - 0.5)
        # Update the tree, x and y arrays too
        joint_tree = cKDTree(xy, leafsize=max(16, int(k*num_dims/4)))
        x = xy[:, :x.shape[1]]
        y = xy[:, x.shape[1]:]
    # Build x and y trees
    x_tree = cKDTree(x, leafsize=max(16, int(k*num_dims/4)))
    y_tree = cKDTree(y, leafsize=max(16, int(k*num_dims/4)))



    if version == 1:
        # The epsilon(i) are distances to kth neighbors in the joint space.
        epsilon_i = np.squeeze(joint_tree.query(
            xy, k=[k+1], p=np.inf, eps=eps, workers=n_workers
        )[0])

        # For each point, count the n_x(i) and n_y(i): number of points within
        # epsilon(i) neighborhood of point i in the X and Y subspaces.
        # Should be strictly less than epsilon_i
        n_x = x_tree.query_ball_point(
            x=x, r=epsilon_i, p=np.inf, eps=eps, workers=n_workers,
            return_length=True
        )
        n_y = y_tree.query_ball_point(
            x=y, r=epsilon_i, p=np.inf, eps=eps, workers=n_workers,
            return_length=True
        )
        # Query includes the point itself so n_x, n_y are in fact n_x+1, n_y+1
        # already

    elif version == 2:
        # The epsilon_x(i) and epsilon_y(i) are the X and Y dimensions
        # of the k-neighbor region found in the joint space.
        # So, first find all points in the joint k-neighborhood of each point
        neighbors = np.asarray(joint_tree.query(
            xy, k=k+1, p=np.inf, eps=eps, workers=n_workers
        )[1])
        # Find the subspace distances epsilon_x, epsilon_y
        # of these k-neighborhoods
        epsilon_xi = np.max(np.abs(x[:, np.newaxis, :] - x[neighbors]), axis=(1, 2))
        epsilon_yi = np.max(np.abs(y[:, np.newaxis, :] - y[neighbors]), axis=(1, 2))

        # For each point, count the n_x(i) and n_y(i): number of points within
        # epsilon_x(i) or epsilon_x(i) of point i in the X and Y subspaces.
        n_x = x_tree.query_ball_point(
            x=x, r=epsilon_xi, p=np.inf, eps=eps, workers=n_workers,
            return_length=True
        )
        n_y = y_tree.query_ball_point(
            x=y, r=epsilon_yi, p=np.inf, eps=eps, workers=n_workers,
            return_length=True
        )
        # Query includes the point itself so need to subtract 1
        n_x -= 1
        n_y -= 1

    else:
        raise ValueError("version kwarg can only be 1 or 2")

    # Compute the average psi(n_x) and psi(n_i)
    av_psi_nx = np.mean(special.psi(n_x))
    av_psi_ny = np.mean(special.psi(n_y))

    # Computing the estimator
    f = special.psi(k) - av_psi_nx - av_psi_ny + special.psi(n_points)
    if version == 2:
        f -= 1/k

    return f / np.log(base)


def kraskov_differential_entropy(x, k=3, version=1, base=np.e, eps=0):
    """ Estimate entropy of a random variable (can be vector-valued)
    given samples, using the estimator derived in
    eq. 20 (version 1) of Kraskov et al., 2004, PRE.
    Too lazy to derive the formula and implement version 2.

    Using the maximum norm, so c_d density is 1 in the estimator.

    Args:
        as in kraskov_continuous_info, for x only.

    Returns:
        entropy (float): differential entropy, in the base units specified.
    """
    if version != 1:
        raise NotImplementedError("Only implemented version 1 for entropy")

    # Make sure x and y are 2D arrays
    x = to_2d_array(x)

    # Number of workers for parallel processing, use half of them.
    n_workers = min(1, psutil.cpu_count() // 2)

    # Build KDTrees in the joint and marginal spaces.
    n_points = x.shape[0]
    num_dims = x.shape[1]
    x_tree = cKDTree(x, leafsize=max(16, int(k*num_dims/4)))

    # Check that there are no exactly identical points
    identical_pairs = x_tree.query_pairs(
        r=0.0, eps=0.0, output_type="ndarray"
    )
    # If any, perturb them slightly to avoid numerical instabilities
    dup_pts = np.unique(identical_pairs[:, 0])
    if identical_pairs.shape[0] > 0:
        print("Found identical points; perturbing them")
        # Average nn distance as a perturbation
        perturb = 1e-6*np.mean(x_tree.query(x, p=np.inf, k=[2])[0])
        # Perturb the first point of each pair
        increments = np.random.random(size=(dup_pts.shape[0], x.shape[1]))
        x[dup_pts, :] = x[dup_pts, :] + perturb*(increments - 0.5)
        # Update the tree, x and y arrays too
        x_tree = cKDTree(x, leafsize=max(16, int(k*num_dims/4)))

    # Version 1 of the estimator (eq. 20)
    # The epsilon(i) are distances to kth neighbors in the joint space.
    epsilon_i = np.squeeze(x_tree.query(
        x, k=[k+1], p=np.inf, eps=eps, workers=n_workers
    )[0])  # These are really epsilon(i) / 2, the distances to k-neighbors

    dens_term = num_dims / n_points * np.sum(np.log(epsilon_i * 2.0))

    # Computing the estimator
    f = special.psi(n_points) + dens_term - special.psi(k)

    return f / np.log(base)
