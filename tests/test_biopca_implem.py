import numpy as np
import sys
import matplotlib.pyplot as plt

if not ".." in sys.path:
    sys.path.insert(1, "..")

from modelfcts.biopca import integrate_inhib_ifpsp_network_skip
from modelfcts.backgrounds import update_ou_kinputs
from utils.statistics import seed_from_gen
from utils.random_matrices import random_orthogonal_mat


# Call signature of integrate_inhib_ifpsp_network_skip:
#   ml_init, l_init, update_bk, bk_init,
#   biopca_params, inhib_params, bk_params, tmax, dt,
#   seed=None, noisetype="normal", skp=1

def update_mvnormal(bk_vari, bk_params, stdnorm_vec, dt):
    # bk_vari and dt are arguments for general compatibility: not used.
    psi = bk_params[0]  # Only parameter is the Cholesky of the covariance!
    return psi.dot(stdnorm_vec), stdnorm_vec


# Version that uses the approximate inverse L
def integrate_biopca_ifpsp(m_init, l_init, update_bk, bk_init, biopca_params,
                bk_params, tmax, dt, seed=None, noisetype="normal"):
    # Note: keep lambda matrix as 1d diagonal only, replace dot products by:
    # Lambda.dot(A): Lambda_ii applied to row i, replace by Lambda_diag[:, None]*A element-wise
    # A.dot(Lambda): Lambda_ii applied to column i, just A*Lambda broadcasts right
    n_neu = m_init.shape[0]  # Number of neurons N_I
    n_dim = m_init.shape[1]  # Number of input neurons N_D
    bk_vari_init, bk_vec_init = bk_init
    assert n_dim == bk_vec_init.shape[0], "Mismatch between dimension of m and background"
    mrate, lrate, lambda_diag = biopca_params

    rng = np.random.default_rng(seed=seed)
    tseries = np.arange(0, tmax, dt)

    # Containers for the solution over time
    m_series = np.zeros([tseries.shape[0], n_neu, n_dim])  # series of M^T (N_IxN_D)
    l_series = np.zeros([tseries.shape[0], n_neu, n_neu])  # series of L (N_IxN_I)
    cbar_series = np.zeros([tseries.shape[0], n_neu])  # series of projections
    bkvec_series = np.zeros([tseries.shape[0], n_dim])  # Input vecs, convenient to compute inhibited output

    ## Initialize running variables, separate from the containers above to avoid side effects.
    c = np.zeros(n_neu)  # un-inhibited neuron activities (before applying L)
    cbar = np.zeros(n_neu)  # inhibited neuron activities (after applying L)
    bk_vari = bk_vari_init.copy()
    bkvec = bk_vec_init.copy()
    mmat = m_init.copy()
    lmat = l_init.copy()

    # Inverse of diagonal of L is used a few times per iteration
    # Indices to access diagonal and off-diagonal elements of L
    # Will be used often, so prepare in advance. Replace dot product
    # with diagonal matrix by element-wise products.
    diag_idx = np.diag_indices(l_init.shape[0])
    inv_l_diag = 1.0 / l_init[diag_idx]  # 1d flattened diagonal
    # Use this difference the only time M_d is needed per iteration
    # Faster to re-invert inv_l_diag than to slice lmat again
    # l_offd = lmat - dflt(1.0 / inv_l_diag)
    newax = np.newaxis
    dflt = np.diagflat

    # Initialize neuron activity with m and background at time zero
    c = inv_l_diag*(mmat.dot(bkvec))  # L_d^(-1) M^T x_0
    # Lateral inhibition between neurons
    cbar = c - inv_l_diag*np.dot(lmat-dflt(1.0 / inv_l_diag), c)

    # Store back some initial values in containers
    cbar_series[0] = cbar
    m_series[0] = m_init
    l_series[0] = l_init
    bkvec_series[0] = bkvec

    # Generate N(0, 1) noise samples in advance
    if (tseries.shape[0]-1)*bk_vari.size > 1e7:
        raise ValueError("Too much memory needed; consider calling multiple times for shorter times")
    if noisetype == "normal":
        noises = rng.normal(0, 1, size=(tseries.shape[0]-1,*bk_vari.shape))
    elif noisetype == "uniform":
        noises = rng.random(size=(tseries.shape[0]-1, *bk_vari.shape))
    else:
        raise NotImplementedError("Noise option {} not implemented".format(noisetype))

    t = 0
    for k in range(0, len(tseries)-1):
        ### Online PCA weights
        # Synaptic plasticity: update mmat, lmat to k+1 based on cbar at k
        # Adding new length-1 axes is faster than np.outer for c.x^T, c.c^T
        mmat = mmat + dt * mrate * (cbar[:, newax].dot(bkvec[newax, :]) - mmat)
        lmat = lmat + dt * mrate * lrate * (cbar[:, newax].dot(cbar[newax, :])
                        - lambda_diag[:, newax] * lmat * lambda_diag)
        # Update too the variable saving the inverse of the diagonal of L
        inv_l_diag = 1.0 / lmat[diag_idx]

        t += dt

        # Update background to time k+1, to be used in next time step (k+1)
        bkvec, bk_vari = update_bk(bk_vari, bk_params, noises[k], dt)

        # Neural dynamics (two-step) at time k+1, to be used in next step
        c = inv_l_diag*(mmat.dot(bkvec))  # L_d^(-1) M^T x_0
        # Lateral inhibition between neurons
        cbar = c - inv_l_diag*np.dot(lmat - dflt(1.0/inv_l_diag), c)

        # Save current state
        knext = (k+1)
        m_series[knext] = mmat
        l_series[knext] = lmat
        bkvec_series[knext] = bkvec
        cbar_series[knext] = cbar  # Save activity of neurons at time k+1

    return tseries, bkvec_series, m_series, l_series, cbar_series


def test_olfactory_back():
    ###  Prepare background parameters
    ### General simulation parameters
    n_dimensions = 4
    n_components = 3
    n_neurons = 3

    # Simulation time scales
    duration = 50000.0
    deltat = 1.0
    tau_nu = 2.0  # Correlation time scale of the background nu_gammas (same for all)
    learnrate = 0.003  # Learning rate of M
    rel_lrate = 2.0  # Learning rate of L, relative to learnrate
    lambda_range = 0.5
    # Choose Lambda diagonal matrix as advised in Minden et al., 2018
    lambda_mat_diag = np.asarray([1.0 - lambda_range*k / (n_neurons - 1) for k in range(n_neurons)])
    biopca_rates = [learnrate, rel_lrate, lambda_range]

    inhib_rates = [25e-5, 5e-5]  # alpha, beta

    # Initial synaptic weights: as advised in Minden et al., 2018
    rgen_meta = np.random.default_rng(seed=0x8496f883e85163519eb26fb84733ebad)
    init_mmat = rgen_meta.standard_normal(size=[n_neurons, n_dimensions]) / np.sqrt(n_dimensions)
    init_lmat = np.eye(n_neurons, n_neurons)  # Supposed to be near-identity, start as identity

    # Choose three LI vectors in (+, +, +) octant: [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], etc.
    back_components = 0.1*np.ones([n_components, n_dimensions])
    for i in range(n_components):
        if i < n_dimensions:
            back_components[i, i] = 0.8
        else:  # If there are more components than there are dimensions (ORNs)
            back_components[i, i % n_dimensions] = 0.8 - i
        # Normalize
        back_components[i] = back_components[i] / np.sqrt(np.sum(back_components[i]**2))

    # Initial background vector and initial nu values
    averages_nu = np.ones(n_components) / np.sqrt(n_components)
    init_nu = np.zeros(n_components)
    init_bkvec = averages_nu.dot(back_components)
    # Initial background params, ordered with nu first for the update_ou_kinputs function
    init_back_list = [init_nu, init_bkvec]

    ## Compute the matrices in the Ornstein-Uhlenbeck update equation
    # Update matrix for the mean term:
    # Exponential decay with time scale tau_nu over time deltat
    update_mat_A = np.identity(n_components)*np.exp(-deltat/tau_nu)

    # Steady-state covariance matrix
    sigma2 = 0.09
    correl_rho = 0.0  # Set to zero for comparison with analytical prediction
    steady_covmat = correl_rho * sigma2 * np.ones([n_components, n_components])  # Off-diagonals: rho
    steady_covmat[np.eye(n_components, dtype=bool)] = sigma2  # diagonal: ones

    # Cholesky decomposition of steady_covmat gives sqrt(tau/2) B
    # Update matrix for the noise term: \sqrt(tau/2(1 - exp(-2*deltat/tau))) B
    psi_mat = np.linalg.cholesky(steady_covmat)
    update_mat_B = np.sqrt(1.0 - np.exp(-2.0*deltat/tau_nu)) * psi_mat

    back_params = [update_mat_A, update_mat_B, back_components, averages_nu]



    ### Run test implementation
    common_seed = seed_from_gen(rgen_meta)
    print(common_seed)
    res1 = integrate_biopca_ifpsp(init_mmat, init_lmat, update_ou_kinputs,
            init_back_list, biopca_rates, back_params, duration, deltat,
            seed=common_seed, noisetype="normal")
    tser1, bkvecser1, mser1, lser1, cbarser1 = res1

    # Run inhibition implementation
    res2 = integrate_inhib_ifpsp_network_skip(init_mmat, init_lmat, update_ou_kinputs,
            init_back_list, biopca_rates, inhib_rates, back_params, duration,
            deltat, seed=common_seed, noisetype="normal")
    tser2, bkser2, bkvecser2, mser2, lser2, cbarser2, wser2, sser2 = res2

    assert np.allclose(mser1, mser2)
    assert np.allclose(lser1, lser2)
    assert np.allclose(cbarser1, cbarser2)
    print("Both implementations give same PCA learning on olfactory back.")
    print("Now checking what the learning looks like")

    fig, ax = plt.subplots()
    for i in range(n_neurons):
        li, = ax.plot(tser1, lser1[:, i, i], lw=2.5)
        ax.plot(tser2, lser2[:, i, i], ls="--", lw=0.5)
    plt.show()
    plt.close()


def test_mvnormal_back():
    ### Prepare simulation parameters
    rgen = np.random.default_rng(seed=0x88977399c6600332f6b98129df9126c2)

    # Dimensionalities
    n_n = 10  # input
    n_k = 3  # output

    # Duration
    duration = 1e5

    # Standard deviation matrix diagonalized
    gtilde_std = np.sqrt(np.diagflat([1.0, 0.75, 0.5] + [0.2]*(n_n - 3)))

    # Lambda matrix
    lambd_mat = np.asarray([1, 0.85, 0.7])

    tau_const = 0.5  # Use value reported in the paper

    # Initial matrices. M: normally-distributed, mean 0 and variance 1/N
    init_m = rgen.standard_normal(size=[n_k, n_n]) / np.sqrt(n_n)
    # L: identity matrix
    init_l = np.eye(n_k)
    deltat = 1.0

    # Algorithm parameters
    biopca_params = [1.1e-3, 1.0/tau_const, lambd_mat]

    # Generate a random orthogonal matrix R for this test run
    r_mat = random_orthogonal_mat(n_n, rgen)

    # Square root decomposition of the actual covariance matrix
    psi_mat = r_mat.dot(gtilde_std)

    # Background parameters with current psi_mat
    init_bk = [np.zeros(n_n, dtype=bool), psi_mat.dot(rgen.standard_normal(size=n_n))]
    back_params = [psi_mat]

    ### Run test implementation
    common_seed = seed_from_gen(rgen)
    print(common_seed)
    res1 = integrate_biopca_ifpsp(init_m, init_l, update_mvnormal,
            init_bk, biopca_params, back_params, duration, deltat,
            seed=common_seed, noisetype="normal")
    tser1, bkvecser1, mser1, lser1, cbarser1 = res1

    # Run inhibition implementation
    inhib_rates = [0.00025, 0.00005]
    res2 = integrate_inhib_ifpsp_network_skip(init_m, init_l, update_mvnormal,
            init_bk, biopca_params, inhib_rates, back_params, duration,
            deltat, seed=common_seed, noisetype="normal")
    tser2, bkser2, bkvecser2, mser2, lser2, cbarser2, wser2, sser2 = res2

    assert np.allclose(mser1, mser2)
    assert np.allclose(lser1, lser2)
    assert np.allclose(cbarser1, cbarser2)
    print("Both implementations give same PCA learning on m.v. normal back.")
    print("Now checking what the learning looks like")

    fig, ax = plt.subplots()
    for i in range(n_k):
        li, = ax.plot(tser1, lser1[:, i, i], lw=2.5)
        ax.plot(tser2, lser2[:, i, i], ls="--", lw=0.5)
    plt.show()
    plt.close()

if __name__ == "__main__":
    #test_olfactory_back()
    test_mvnormal_back()
