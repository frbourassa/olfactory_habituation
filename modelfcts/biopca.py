"""
This module implements the Biologically Plausible Online PCA (biopca)
neural network model proposed by Minden, Pehlevan, and Chklovskii, 2018.

Changing a bit their notation, inhibitory neurons are connected to the input
layer x by a weight matrix M, where each column contains the weight vector
entering one inhibitory neuron. Moreover, inhibitory neurons are connected
to one another by the weight matrix L, which has the property of
being quasi-diagonal built in the learning algorithm. These neurons
learn a principal subspace projection (PSP) of the input data

Minden notation -> our notation
-------------------------------
their W -> my M
their M -> my L
their \tilde{y} -> my c
their y -> my \bar{c}
At least, their x and my are the same: input vector.

In my habituation network model, neurons in the inhibitory layer learn
projections of the input odors according to some model that can find
independent components, either IBCM, PCA, or ICA.
The inhibitory weight matrix connecting those neurons to projection neurons
(PNs) with inhibition is learnt in the same way for all models, that is,
with the gradient rule to minimize s = R(x - Wc) that I have
derived elsewhere.The goal is to compare the success of different projection
pursuit algorithms (IBCM, PCA, ICA) at inhibiting an olfactory backgrounds.

So, in this module, the PCA case is implemented, in the online version of
Minden et al., 2018. The functions provided here simulate the learning over
time in response to a fluctuating mixture of background odors.

@author: frbourassa
June 2022
"""
import numpy as np
from modelfcts.ideal import relu_inplace
from utils.metrics import l2_norm, l1_norm, lp_norm


def build_lambda_matrix(l_max, l_range, n_neu):
    # Choose Lambda diagonal matrix as advised in Minden et al., 2018
    if l_range >= 1.0:
        raise ValueError("lambda_range should be < 1, "
                        + "currently it is = {}".format(l_range))
    if n_neu > 1:
        lambda_mat = np.asarray([1.0 - l_range*k / (n_neu - 1)
                        for k in range(n_neu)])
    else:
        lambda_mat = np.ones(1)
    return l_max * lambda_mat


def integrate_inhib_ifpsp_network_skip(ml_inits, update_bk, bk_init,
                biopca_params, inhib_params, bk_params, tmax, dt,
                seed=None, noisetype="normal", skp=1, **model_options):
    """ Integrate the feedforward inhibition network with online PCA learning
    rule for the inhibitory neurons, from the Minden et al., 2018 model
    The input process is updated by the function update_bk, which
    takes pre-generated noise and bk_params.
    The intended usage here is for an input that is a general linear
    combination of background vectors, i.e. x = A\nu, with \nu a concentration
    vector and A a matrix where each row is an odor's ORN stimulation vector.

    Using a ReLU activation function on projection neurons.

    Note: what we call L in this function is really L' = L^{-1}, 
    the inverse of the L matrix in our general olfactory network circuit. 
    L' gets updated and is returned in l_series by this function. 

    Args:
        ml_inits (list of 2 np.ndarray):
            m_init: 2d array, shape (number neurons, number dimensions)
            The transpose of the M matrix where each column is an m vector, ie
            synaptic weights connecting one inhibitory neuron to input neurons.
            l_init: 2d array, shape (number neurons)^2, of
            synaptic weights coupling inhibitory neurons to each other.
        update_bk (callable): function that updates the background variables and
            the background vector
        bk_init (list of two 1d np.ndarrays): [bk_vari_init, bk_vec_init]
            bk_vari_init (np.ndarray): array of background random variables,
                shaped [odor, n_var_per_odor], or 1d if one var. per odor.
            bk_vec_init (np.ndarray): initial background vector, must have size n_orn
        biopca_params (list):
            mrate (float): learning rate of M (alpha_t in Minden 2018)
            lrate (float): learning rate of L, relative to mrate
                (inverse of tau in Minden 2018)
            lambda_range (float): lambda matrix range of diagonal elements
                (between 0 and 1). Largest lambda element is 1, smallest is 0
            xrate (float): if remove_mean is True, learning
                rate for the mean background.
        inhib_params (list): alpha, beta: list of parameters for the inhibitory
            neurons update. Should have alpha > beta here.
            For the early times averaging of synaptic weights, will use beta
            and alpha as alpha and beta (to keep m vector close to origin).
        bk_params (list): list of parameters passed to update_bk (3rd argument)
        tmax (float): max time
        dt (float): time step (alpha_t rate in Minden 2018)
        seed (int): seed for the random number generator
        noisetype (str): either "normal" or "uniform"
        skp (int): save only every skp time step
    Keyword arguments: model_options
        remove_mean (bool): whether to learn the average input and subtract
            it before PCA projection
        remove_lambda (bool): if True, multiply projection by Lambda^-1,
            i.e., get an orthonormal basis.
        activ_fct (str): either "ReLU" or "identity"
        w_norms (tuple of 2 ints): (p, q) either 1 or 2. Default: (2, 2)
            Lp norm for minimization, Lq norm for regularization.

    Returns:
        tseries, bk_series, bkvec_series, m_series, l_series,
        xmean_series, cbar_series, w_series, y_series
    """
    remove_mean = model_options.get("remove_mean", False)
    remove_lambda = model_options.get("remove_lambda", False)
    activ_fct = str(model_options.get("activ_fct", "ReLU")).lower()
    w_norms = model_options.get("w_norms", (2, 2))
    m_init, l_init = ml_inits
    # Note: keep lambda matrix as 1d diagonal only, replace dot products by:
    # Lambda.dot(A): Lambda_ii applied to row i, replace by Lambda_diag[:, None]*A element-wise
    # A.dot(Lambda): Lambda_ii applied to column i, just A*Lambda broadcasts right
    n_neu = m_init.shape[0]  # Number of neurons N_I
    n_orn = m_init.shape[1]  # Number of input neurons N_ORN
    bk_vari_init, bk_vec_init = bk_init
    assert n_orn == bk_vec_init.shape[0], "Mismatch between dimension of m and background"
    alpha, beta = inhib_params
    # xrate will be a dummy value if remove_mean == False
    mrate, lrate, lambda_max, lambda_range, xrate = biopca_params
    lrate_l = lrate / lambda_max**2

    # Choose Lambda diagonal matrix as advised in Minden et al., 2018
    lambda_diag = build_lambda_matrix(lambda_max, lambda_range, n_neu)
    rng = np.random.default_rng(seed=seed)
    tseries = np.arange(0, tmax, dt*skp)

    # Check that the biggest matrices, W or M, will not use too much memory
    if tseries.shape[0] * n_orn * n_neu > 5e8 / 8:  # 500 MB per series max
        raise ValueError("Excessive memory use by saved series; increase skp")

    # Containers for the solution over time
    bk_series = np.zeros([tseries.shape[0]] + list(bk_vari_init.shape))
    m_series = np.zeros([tseries.shape[0], n_neu, n_orn])  # series of M^T (N_IxN_D)
    l_series = np.zeros([tseries.shape[0], n_neu, n_neu])  # series of L (N_IxN_I)
    cbar_series = np.zeros([tseries.shape[0], n_neu])  # series of projections
    w_series = np.zeros([tseries.shape[0], n_orn, n_neu])  # Inhibitory weights W (N_DxN_I)
    bkvec_series = np.zeros([tseries.shape[0], n_orn])  # Input vecs, convenient to compute inhibited output
    y_series = np.zeros([tseries.shape[0], n_orn])  # series of proj. neurons
    if remove_mean:
        xmean_series = np.zeros([tseries.shape[0], n_orn])
    else:
        xmean_series = None

    ## Initialize running variables, separate from the containers above to avoid side effects.
    c = np.zeros(n_neu)  # un-inhibited neuron activities (before applying L)
    cbar = np.zeros(n_neu)  # inhibited neuron activities (after applying L)
    wmat = w_series[0].copy()  # Initialize with null inhibition
    bk_vari = bk_vari_init.copy()
    bkvec = bk_vec_init.copy()
    mmat = m_init.copy()
    lmat = l_init.copy()
    yvec = bk_vec_init.copy()
    if activ_fct == "relu":
        relu_inplace(yvec)
    elif activ_fct == "identity":
        pass
    else:
        raise ValueError("Unknown activation function: {}".format(activ_fct))
    if remove_mean:
        xmean = np.zeros(bkvec.shape)
    else:
        xmean = 0.0

    # Inverse of diagonal of L is used a few times per iteration
    # Indices to access diagonal and off-diagonal elements of L
    # Will be used often, so prepare in advance. Replace dot product
    # with diagonal matrix by element-wise products.
    diag_idx = np.diag_indices(l_init.shape[0])
    inv_l_diag = 1.0 / l_init[diag_idx]  # 1d flattened diagonal
    # Use this difference the only time M_d is needed per iteration
    # Faster to re-invert inv_l_diag than to slice lmat again
    # l_offd = lmat - dflt(1.0 / inv_l_diag)  # is faster than
    # l_offd = lmat - dflt(lmat[diag_idx])
    newax = np.newaxis
    dflt = np.diagflat

    # Initialize neuron activity with m and background at time zero
    c = inv_l_diag * (mmat.dot(bkvec - xmean))
    cbar = c - inv_l_diag*np.dot(lmat-dflt(1.0 / inv_l_diag), c)
    if remove_lambda:
        cbar = cbar / lambda_diag

    # Store back some initial values in containers
    cbar_series[0] = cbar
    bk_series[0] = bk_vari
    m_series[0] = m_init
    l_series[0] = l_init
    bkvec_series[0] = bkvec
    y_series[0] = yvec

    # Generate N(0, 1) noise samples in advance
    if (tseries.shape[0]*skp-1)*bk_vari.size > 2e7:
        raise ValueError("Too much memory needed; consider calling multiple times for shorter times")
    if noisetype == "normal":
        noises = rng.normal(0, 1, size=(tseries.shape[0]*skp-1,*bk_vari.shape))
    elif noisetype == "uniform":
        noises = rng.random(size=(tseries.shape[0]*skp-1, *bk_vari.shape))
    else:
        raise NotImplementedError("Noise option {} not implemented".format(noisetype))

    t = 0
    for k in range(0, len(tseries)*skp-1):
        # Learning the mean: independent of everything else.
        if remove_mean:
            xmean = xmean + dt * xrate * (bkvec - xmean)
        # Else, xmean stays 0

        ### Inhibitory  weights
        # They depend on cbar and yvec at time step k, which are still in cbar, yvec
        # cbar, shape [n_neu], should broadcast against columns of wmat,
        # while yvec, shape [n_orn], should broadcast across rows (copied on each column)
        if w_norms[0] == 2:  # default L2 norm, nice and smooth
            alpha_term = alpha * cbar[newax, :] * yvec[:, newax]
        elif w_norms[0] == 1:  # L1 norm
            aynorm = alpha * l1_norm(yvec)
            alpha_term = aynorm * cbar[newax, :] * np.sign(yvec[:, newax])
        elif w_norms[0] > 2:  # Assuming some Lp norm with p > 2
            # Avoid division by zero for p > 2 by clipping ynorm
            ynorm = max(1e-9, lp_norm(yvec, p=w_norms[0]))
            yterm = np.sign(yvec) * np.abs(yvec/ynorm)**(w_norms[0]-1) * ynorm
            alpha_term = alpha * cbar[newax, :] * yterm[:, newax]
        else:
            raise ValueError("Cannot deal with Lp norms with p < 0 or non-int")

        if w_norms[1] == 2:
            beta_term = beta * wmat
        elif w_norms[1] == 1:
            beta_term = beta * l1_norm(wmat.ravel()) * np.sign(wmat)
        elif w_norms[1] > 2:
            wnorm = max(1e-9, lp_norm(wmat.ravel(), p=w_norms[1]))
            wterm = np.sign(wmat) * np.abs(wmat/wnorm)**(w_norms[1]-1)
            beta_term = beta * wterm * wnorm
        else:
            raise ValueError("Cannot deal with Lp norms with p < 0 or non-int")

        wmat += dt * (alpha_term - beta_term)

        ### Online PCA weights
        # Synaptic plasticity: update mmat, lmat to k+1 based on cbar at k
        mmat += dt * mrate * (cbar[:, newax].dot(bkvec[newax, :]) - mmat)
        lmat += dt * mrate * lrate_l * (cbar[:, newax].dot(cbar[newax, :])
                        - lambda_diag[:, newax] * lmat * lambda_diag)
        # Update too the variable saving the inverse of the diagonal of L
        inv_l_diag = 1.0 / lmat[diag_idx]

        t += dt

        # Update background to time k+1, to be used in next time step (k+1)
        bkvec, bk_vari = update_bk(bk_vari, bk_params, noises[k], dt)

        # Neural dynamics (two-step) at time k+1, to be used in next step
        c = inv_l_diag * (mmat.dot(bkvec - xmean))  # L_d^(-1) M^T x
        # Lateral inhibition between neurons
        cbar = c - inv_l_diag*np.dot(lmat - dflt(1.0/inv_l_diag), c)
        if remove_lambda:
            # Remove the Lambda scale of eigenvectors, so the W matrix does
            # not need to compensate too much.
            # So we use Lambda^{-1}L^{-1}M as a projector as prescribed in Minden 2018
            cbar = cbar / lambda_diag

        # Lastly, projection neurons at time step k+1.
        # xmean is 0 if we don't remove the mean
        yvec = bkvec - xmean - wmat.dot(cbar)
        if activ_fct == "relu":
            relu_inplace(yvec)

        # Save current state only if at a multiple of skp
        if (k % skp) == (skp - 1):
            knext = (k+1) // skp
            w_series[knext] = wmat
            m_series[knext] = mmat
            l_series[knext] = lmat
            bk_series[knext] = bk_vari
            bkvec_series[knext] = bkvec
            cbar_series[knext] = cbar  # Save activity of neurons at time k+1
            y_series[knext] = yvec
            if remove_mean:
                xmean_series[knext] = xmean
    return (tseries, bk_series, bkvec_series, m_series, l_series,
                xmean_series, cbar_series, w_series, y_series)


def biopca_respond_new_odors(odors, mlx, wmat, biopca_rates, options={}):
    """
    Args:
        odors (np.ndarray): indexed [..., n_orn]
            so can take dot product properly with m and store many
            odors along arbitrary other axes.
        current_mlx (list of 2 np.ndarrays):
            current_m (np.ndarray): shape [n_neurons, n_orn]
            current_l (np.ndarray): shape [n_neurons, n_neurons]
            current_x (np.ndarray): shape [n_orn]
        current_w (np.ndarray): shape [n_orn, n_neurons]
        biopca_rates (np.ndarray): [lrate, mrate, lambda_range, xmean_rate]
    Keyword args:
        remove_mean (bool): whether to learn the average input and subtract
            it before PCA projection
        remove_lambda (bool): if True, multiply projection by Lambda^-1,
            i.e., get an orthonormal basis.
        activ_fct (str): either "ReLU" or "identity"
    """
    mmat, lmat, xmean = mlx
    activ_fct = options.get("activ_fct", "ReLU")
    remove_mean = options.get("remove_mean", False)
    remove_lambda = options.get("remove_lambda", True)
    inv_l_diag = 1.0 / np.diagonal(lmat)
    # Choose Lambda diagonal matrix as advised in Minden et al., 2018
    lambda_max = biopca_rates[2]
    lambda_range = biopca_rates[3]
    n_neu = mmat.shape[0]
    lambda_diag = build_lambda_matrix(lambda_max, lambda_range, n_neu)
    if remove_mean:
        # Subtracting the mean background from c and from s too
        # c is just the projection of the variation around the mean
        # inv_l_diag broadcast across odors in first axes, OK in this order
        c = inv_l_diag * ((odors - xmean).dot(mmat.T))  # L_d^(-1) M^T x_0
    else:
        c = inv_l_diag*(odors.dot(mmat.T))  # L_d^(-1) M^T x_0
    # Lateral inhibition between neurons
    cbar = c - inv_l_diag*np.dot(c, (lmat - np.diagflat(1.0/inv_l_diag)).T)
    # Remove lambda scale from eigenvectors
    if remove_lambda:
        cbar = cbar / lambda_diag
    # Lastly, projection neurons after inhibition
    yvec = odors - xmean if remove_mean else odors
    yvec = yvec - cbar.dot(wmat.T)
    if str(activ_fct).lower() == "identity":
        pass
    elif str(activ_fct).lower() == "relu":
        relu_inplace(yvec)
    else:
        raise ValueError("Unknown activation function: {}".format(activ_fct))
    return yvec
