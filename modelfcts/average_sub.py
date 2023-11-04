""" Habituation model inspired from Shen, Dasgupta, Navlakha, 2020,
where the background is averaged and subtracted from the projection neurons.
Written for final comparison to other models.

In terms of our notation with n_I inhibitory neurons, a matrix M and inhibitory
weights W, there should be one inhibitory neuron, connected to all PNs.
Moreover, we should have c = 1 fixed instead of being projections;
thus, we are not learning M and just defining c = 1 instead of c = Mx.
Then, our gradient-based rule for the diagonal weights in W gives
average background inhibition, with an averaging time rate beta.
    $$ dw_i/dt = \alpha ReLU(x_i - w_i) - \beta w_i $$

@author: frbourassa
September 2023
"""
import numpy as np
from modelfcts.ideal import relu_inplace

def integrate_inhib_average_sub_skip(m_init, update_bk,
    bk_init, avg_params, inhib_params, bk_params, tmax, dt,
    seed=None, noisetype="normal", skp=1, **model_options):
    """
    Args:
        m_init (np.ndarray): dummy, M is useless and will stay
            equal to m_init throughout.
            Used to infer n_orn, so it should have shape [1, n_orn].
        update_bk (callable): function that updates the background variables and
            the background vector
        bk_init (list of two 1d np.ndarrays): [bk_vari_init, bk_vec_init]
            bk_vari_init (np.ndarray): array of background random variables,
                shaped [odor, n_var_per_odor], or 1d if one var. per odor.
            bk_vec_init (np.ndarray): initial background vector, must have size n_orn
        avg_params (None): no rates because m_{ij}=0 and c_i=1
        inhib_params (list): alpha, beta: list of parameters for the inhibitory
            neurons update. Should have alpha > beta here.
            For the early times averaging of synaptic weights, will use beta
            and alpha as alpha and beta (to keep m vector close to origin).
        bk_params (list): list of parameters passed to update_bk (3rd argument)
        tmax (float): max time
        dt (float): time step
        seed (int): seed for the random number generator
        noisetype (str): either "normal" or "uniform"
        skp (int): save only every skp time step
    Keyword args: model options
        activ_fct (str): either "ReLU" or "identity"

    Returns:
        tseries, bk_series, bkvec_series, w_series, s_series
        The two None returns are there to mimic the signature of other
        integration functions.
    """
    activ_fct = str(model_options.get("activ_fct", "ReLU")).lower()
    n_orn = m_init.shape[1]  # Number of ORNs
    bk_vari_init, bk_vec_init = bk_init
    assert n_orn == bk_vec_init.shape[0], "Mismatch between dimension of m and background"
    alpha, beta = inhib_params

    rng = np.random.default_rng(seed=seed)
    tseries = np.arange(0, tmax, dt*skp)

    # Containers for the solution over time
    bk_series = np.zeros([tseries.shape[0]] + list(bk_vari_init.shape))
    w_series = np.zeros([tseries.shape[0], n_orn, 1])  # Inhibitory weights
    bkvec_series = np.zeros([tseries.shape[0], n_orn])  # Input vecs, convenient to compute inhibited output
    s_series = np.zeros([tseries.shape[0], n_orn])

    ## Initialize running variables, separate from the containers above to avoid side effects.
    wmat = w_series[0].copy()  # Initialize with null inhibition
    bk_vari = bk_vari_init.copy()
    bkvec = bk_vec_init.copy()
    svec = bkvec.copy()
    if activ_fct == "relu":
        relu_inplace(svec)
    elif activ_fct == "identity":
        pass
    else:
        raise ValueError("Unknown activation function: {}".format(activ_fct))

    # Store back some initial values in containers
    bk_series[0] = bk_vari
    bkvec_series[0] = bkvec

    # Generate N(0, 1) noise samples in advance
    if (tseries.shape[0]*skp-1)*bk_vari.size > 1e7:
        raise ValueError("Too much memory needed; consider calling multiple times for shorter times")
    if noisetype == "normal":
        noises = rng.normal(0, 1, size=(tseries.shape[0]*skp-1,*bk_vari.shape))
    elif noisetype == "uniform":
        noises = rng.random(size=(tseries.shape[0]*skp-1, *bk_vari.shape))
    else:
        raise NotImplementedError("Noise option {} not implemented".format(noisetype))

    t = 0
    for k in range(0, len(tseries)*skp-1):
        t += dt
        ### Inhibitory weights
        # They depend on cbar and svec at time step k, which are still in cbar, svec
        # cbar, shape [n_neu], should broadcast against columns of wmat,
        # while svec, shape [n_orn], should broadcast across rows (copied on each column)
        wmat = wmat + dt * (alpha*svec[:, np.newaxis] - beta*wmat)

        # Update background to time k+1, to be used in next time step
        bkvec, bk_vari = update_bk(bk_vari, bk_params, noises[k], dt)

        # Lastly, projection neurons at time step k+1
        svec = bkvec - wmat[:, 0]
        if activ_fct == "relu":
            relu_inplace(svec)

        # Save current state only if at a multiple of skp
        if (k % skp) == (skp - 1):
            knext = (k+1) // skp
            w_series[knext] = wmat
            bk_series[knext] = bk_vari
            bkvec_series[knext] = bkvec
            s_series[knext] = svec

    return tseries, bk_series, bkvec_series, w_series, s_series


def average_sub_respond_new_odors(odors, wmat, options={}):
    activ_fct = str(options.get("activ_fct", "ReLU")).lower()
    resp = odors - wmat[:, 0]
    if activ_fct == "relu":
        relu_inplace(resp)  # Broadcasting to all odors
    elif activ_fct == "identity":
        pass
    else:
        raise ValueError("Unknown activation function: {}".format(activ_fct))
    return resp
