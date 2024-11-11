""" Module containing function for the integration of single IBCM neurons
and networks of those neurons, and their usage for inhibition of an olfactory
background.

@author: frbourassa
September 2021
"""
import numpy as np
from modelfcts.ideal import relu_inplace
from utils.metrics import l2_norm, l1_norm, lp_norm

### IBCM NEURON alone, no inhibition
def integrate_ibcm(m_init, update_bk, bk_init, bk_params, tmax, dt,
                   learnrate, seed=14345124, noisetype="normal", tavg=10):
    r""" Integrate the IBCM equation when the input updated by the derivative
    function update_bk, which takes pre-generated noise and bk_params.
    The intended usage here is for an input that is a linear
    combination of two LI vectors, and the proportion of each
    component in the input fluctuates around 1/2 following a
    Ornstein-Uhlenbeck process with rate specified in bk_params.

    Args:
        m_init (np.ndarray): 1 array, shape (number dimensions,)
        update_bk (callable): function that updates the background variables and
            the background vector
        bk_init (list of two 1d np.ndarrays): [bk_vari_init, bk_vec_init]
            bk_vari_init (np.ndarray): array of background random variables,
                shaped [odor, variables_per_odor], or 1d if one var. per odor.
            bk_vec_init (np.ndarray): initial background vector, must have size n_orn
        bk_params (list): list of parameters passed to update_bk (3rd argument)
        tmax (float): max time
        dt (float): time step
        learnrate (float): kinetic learning rate \mu in the IBCM equation
        seed (int): seed for the random number generator
        noisetype (str): either "normal" or "uniform"
        tavg (float): default: 10

    Returns:
        tseries, m_series, bk_series, c_series, cbar_series, w_series, bkvec_series
    """
    n_orn = m_init.shape[0]
    bk_vari_init, bk_vec_init = bk_init
    assert n_orn == bk_vec_init.shape[0], "Mismatch between dimension of m and background"

    rng = np.random.default_rng(seed=seed)
    tseries = np.arange(0, tmax, dt)

    # Containers for the solution over time
    bk_series = np.zeros([tseries.shape[0]] + list(bk_vari_init.shape))
    m_series = np.zeros([tseries.shape[0],n_orn])
    c_series = np.zeros([tseries.shape[0]])
    bkvec_series = np.zeros([tseries.shape[0], n_orn])  # Input vecs, convenient to compute inhibited output
    theta_series = np.zeros(tseries.shape[0])

    ## Initialize running variables, separate from the containers above to avoid side effects.
    bk_vari = bk_vari_init.copy()
    bkvec = bk_vec_init.copy()
    m = m_init.copy()
    c = m.dot(bkvec)  # neuron activity
    c2_avg = c*c  # Average neuron activity squared

    # Store back some initial values in containers
    c_series[0] = c
    bk_series[0] = bk_vari
    m_series[0] = m_init
    bkvec_series[0] = bkvec
    theta_series[0] = c2_avg

    # Generate N(0, 1) noise samples in advance
    if (tseries.shape[0]-1)*bk_vari.size > 2e7:
        raise ValueError("Too much memory needed; consider calling multiple times for shorter durations")
    if noisetype == "normal":
        noises = rng.normal(0, 1, size=(tseries.shape[0]-1, *bk_vari.shape))
    elif noisetype == "uniform":
        noises = rng.random(size=(tseries.shape[0]-1, *bk_vari.shape))
    else:
        raise NotImplementedError("Noise option {} not implemented".format(noisetype))

    t = 0
    for k in range(0, len(tseries)-1):
        t += dt
        ### IBCM neurons
        # Update first the synaptic weights m to time k+1 before updating c, c2,
        # because dm/dt depends on c, c2_avg at time k
        m = m + learnrate * dt * c * (c - c2_avg) * bkvec

        # Store the updated synaptic weights
        m_series[k+1] = m

        # Now, update to time k+1 the threshold (c2_avg) using cbar at time k
        # to be used to update m in the next time step
        c2_avg = c2_avg + (c*c - c2_avg)/tavg * dt
        # This Euler scheme could cause numerical stability problems if dt is too large,
        # or tavg too small, or the average vector too large.

        # Update background to time k+1, to be used in next time step
        bkvec, bk_vari = update_bk(bk_vari, bk_params, noises[k], dt)
        bk_series[k+1] = bk_vari
        bkvec_series[k+1] = bkvec

        # Lastly, compute activity of IBCM neurons at next time step, k+1,
        # with the updated background and synaptic weight vector m
        # Compute un-inhibited activity of each neuron with current input (at time k)
        c = m.dot(bkvec)
        c_series[k+1] = c
        theta_series[k+1] = c2_avg

    return tseries, m_series, bk_series, c_series, theta_series, bkvec_series


### IBCM NETWORK alone, no inhibition model for now
def integrate_ibcm_network(m_init, update_bk, bk_init, bk_params, tmax, dt,
        learnrate, seed=14345124, noisetype="normal", tavg=10, coupling=0.1):
    r""" Integrate the IBCM equation when the input updated by the derivative
    function update_bk, which takes pre-generated noise and bk_params.
    The intended usage here is for an input that is a linear
    combination of two LI vectors, and the proportion of each
    component in the input fluctuates around 1/2 following a
    Ornstein-Uhlenbeck process with rate specified in bk_params.

    Args:
        m_init (np.ndarray): 2d array, shape (number neurons, number dimensions)
        update_bk (callable): function that updates the background variables and
            the background vector
        bk_init (list of two 1d np.ndarrays): [bk_vari_init, bk_vec_init]
            bk_vari_init (np.ndarray): array of background random variables,
                shaped [odor, variables_per_odor], or 1d if one var. per odor.
            bk_vec_init (np.ndarray): initial background vector, must have size n_orn
        bk_params (list): list of parameters passed to update_bk (3rd argument)
        tmax (float): max time
        dt (float): time step
        learnrate (float): kinetic learning rate \mu in the IBCM equation
        seed (int): seed for the random number generator
        noisetype (str): either "normal" or "uniform"
        tavg (float): default: 10
        coupling (float): eta, between 0 and 1

    Returns:
        tseries, m_series, bk_series, c_series, cbar_series, theta_series, bkvec_series
    """
    n_neu = m_init.shape[0]  # Number of neurons
    n_orn = m_init.shape[1]
    bk_vari_init, bk_vec_init = bk_init
    assert n_orn == bk_vec_init.shape[0], "Mismatch between dimension of m and background"

    rng = np.random.default_rng(seed=seed)
    tseries = np.arange(0, tmax, dt)

    # Containers for the solution over time
    bk_series = np.zeros([tseries.shape[0]] + list(bk_vari_init.shape))
    m_series = np.zeros([tseries.shape[0], n_neu, n_orn])
    cbar_series = np.zeros([tseries.shape[0], n_neu])
    c_series = np.zeros([tseries.shape[0], n_neu])
    bkvec_series = np.zeros([tseries.shape[0], n_orn])  # Input vecs, convenient to compute inhibited output
    theta_series = np.zeros([tseries.shape[0], n_neu])

    ## Initialize running variables, separate from the containers above to avoid side effects.
    c = np.zeros(n_neu)  # un-inhibited neuron activities
    cbar = np.zeros(n_neu)  # inhibited neuron activities
    cbar2_avg = np.zeros(n_neu)  # Average inhibited neuron activities squared, initialized at zero
    bk_vari = bk_vari_init.copy()
    bkvec = bk_vec_init.copy()
    m = m_init.copy()

    # Initialize neuron activity with m and background at time zero
    c = m.dot(bkvec)
    cbar = c - coupling*(np.sum(c) - c)  # -c to cancel the subtraction of c[i] itself
    cbar2_avg = cbar**2

    # Store back some initial values in containers
    c_series[0] = c
    cbar_series[0] = cbar
    bk_series[0] = bk_vari
    m_series[0] = m_init
    bkvec_series[0] = bkvec
    theta_series[0] = cbar2_avg

    # Generate N(0, 1) noise samples in advance
    if (tseries.shape[0]-1)*bk_vari.size > 2e7:
        raise ValueError("Too much memory needed; consider calling multiple times for shorter durations")
    if noisetype == "normal":
        noises = rng.normal(0, 1, size=(tseries.shape[0]-1, *bk_vari.shape))
    elif noisetype == "uniform":
        noises = rng.random(size=(tseries.shape[0]-1, *bk_vari.shape))
    else:
        raise NotImplementedError("Noise option {} not implemented".format(noisetype))

    t = 0
    for k in range(0, len(tseries)-1):
        t += dt
        ### IBCM neurons
        # Update first their synaptic weights m to time k+1 before updating c, cbar,
        # because dm/dt depends on c, cbar, cbar2_avg at time k
        # Phi function for each neuron
        phiterms_vec = cbar * (cbar - cbar2_avg)
        # Now, careful with broadcast: for each neuron (dimension 0 of m and cbar), we need a scalar element
        # of phiterms_vec times the whole bkvec, for dimension 1 of m.
        # This can be done vectorially with a dot product (n_neu, 1)x(1, n_components)
        rhs_scalar = phiterms_vec - coupling*(np.sum(phiterms_vec) - phiterms_vec)
        # Euler integrator and learning rate
        m = m + learnrate*dt*rhs_scalar[:, np.newaxis].dot(bkvec[np.newaxis, :])

        # Store the updated synaptic weights
        m_series[k+1] = m

        # Now, update to time k+1 the threshold (cbar2_avg) using cbar at time k
        # to be used to update m in the next time step
        cbar2_avg = cbar2_avg + (cbar*cbar - cbar2_avg)/tavg * dt
        # This Euler scheme could cause numerical stability problems if dt is too large,
        # or tavg too small, or the average vector too large.

        # Update background to time k+1, to be used in next time step
        bkvec, bk_vari = update_bk(bk_vari, bk_params, noises[k], dt)
        bk_series[k+1] = bk_vari
        bkvec_series[k+1] = bkvec

        # Lastly, compute activity of IBCM neurons at next time step, k+1,
        # with the updated background and synaptic weight vector m
        # Compute un-inhibited activity of each neuron with current input (at time k)
        c = m.dot(bkvec)
        c_series[k+1] = c
        cbar = c - coupling*(np.sum(c) - c)  # -c to cancel the subtraction of c[i] itself
        # np.sum(c) is a scalar and c a vector, so it broadcasts properly.
        cbar_series[k+1] = cbar  # Save activity of neurons at time k+1
        theta_series[k+1] = cbar2_avg

    return tseries, m_series, bk_series, c_series, cbar_series, theta_series, bkvec_series


### NETWORK OF IBCM NEURONS WITH OPTIMAL INHIBITORY WEIGHTS
# One main function instead of multiple variants
# Only marginal performance sacrifices by checking the various options.
def integrate_inhib_ibcm_network_options(vari_inits, update_bk, bk_init,
    ibcm_params, inhib_params, bk_params, tmax, dt,
    seed=None, noisetype="normal", skp=1, **options):
    r""" Integrate the IBCM equation when the input updated by the derivative
    function update_bk, which takes pre-generated noise and bk_params.

    The intended usage here is for an input that is a general linear
    combination of background vectors, i.e. x = A\nu, with \nu a concentration
    vector and A a matrix where each row is an odor's ORN stimulation vector.

    Option "skp": 1 by default (all steps recorded), else one every skp
    step is recorded. Useful when going to large number of neurons
        or dimension, to avoid filling computer memory. But this makes code
        a bit slower (if statement at every iteration).

    Option "variant" == "law":
    Law and Cooper, 1994 modification, modified again: divide the learning rate
    by theta + k_{theta}. Speeds up initial convergence, and only
    freeze learning once large theta reached.

    Option "saturation" == "tanh": apply a mild tanh saturation to c:
    c = s*lambd * tanh(m.x / s / lambd). To avoid undesired saturation,
    s has to be scaled up proportionally to \Lambda.

    Option "activ_fct": either "ReLU" or "identity"

    Args:
        vari_inits (list of np.ndarrays):
            m_init (np.ndarray): 2d array, shape (number neurons, number dim.)
                The transpose of the M matrix where each column is an m vector,
                synaptic weights connecting one inhibitory neuron to inputs.
            theta_init (np.ndarray): initial thresholds Theta.
            w_init (np.ndarray): initial w


        update_bk (callable): function that updates the background variables and
            the background vector
        bk_init (list of two 1d np.ndarrays): [bk_vari_init, bk_vec_init]
            bk_vari_init (np.ndarray): array of background random variables,
                shaped [odor, n_var_per_odor], or 1d if one var. per odor.
            bk_vec_init (np.ndarray): initial background vector, must have size n_orn
        ibcm_params (list): learnrate (mu), tavg (tau_theta), coupling (eta),
            lambd (float): scale of synaptic weights, set via Theta equation;
                default value is 1.0
            sat (normalization factor in tanh activation function),
            ktheta (threshold cbar at which the learning rate decreases),
            decay_relative (float): decay rate, relative to learning rate
            The seven should always be provided, just put dummy values for
            unused ones depending on options.
            learnrate and sat are relative to lambd=1, they are scaled
            appropriately if lambda is different.
        inhib_params (list): alpha, beta: list of parameters for the inhibitory
            neurons update. Should have alpha > beta here.
            For the early times averaging of synaptic weights, will use beta
            and alpha as alpha and beta (to keep m vector close to origin).
        bk_params (list): list of parameters passed to update_bk (3rd argument)
        tmax (float): max time
        dt (float): time step

    Keyword args:
        seed (int): seed for the random number generator
        noisetype (str): either "normal" or "uniform"
        skp (int): save only every skp time step

    Options:
        activ_fct (str): activation fct of PN, either "ReLU" or "identity"
        saturation (str): "linear" or "tanh"
        variant (str): either "intrator" or "law"
        decay (bool): if True, add small decay
        w_norms (tuple of 2 ints): (p, q) either 1 or 2. Default: (2, 2)
            Lp norm for minimization, Lq norm for regularization.

    Returns:
        [tseries, bk_series, bkvec_series, m_series,
        cbar_series, theta_series, w_series, s_series]
    """
    # Get some of the keyword arguments
    saturation = options.get("saturation", "linear")
    variant = options.get("variant", "intrator")
    activ_fct = str(options.get("activ_fct", "ReLU")).lower()
    decay = options.get("decay", False)
    w_norms = options.get("w_norms", (2, 2))

    # Legacy option to just pass initial M
    if isinstance(vari_inits, np.ndarray):
        m_init = vari_inits
        n_neu = m_init.shape[0]  # Number of neurons
        n_orn = m_init.shape[1]
        w_init = np.zeros([n_orn, n_neu])
        theta_init = None
    elif isinstance(vari_inits, list) and len(vari_inits) == 1:
        m_init = np.asarray(vari_inits[0])
        n_neu = m_init.shape[0]  # Number of neurons
        n_orn = m_init.shape[1]
        w_init = np.zeros([n_orn, n_neu])
        theta_init = None
    else:
        m_init, theta_init, w_init = vari_inits
        n_neu = m_init.shape[0]  # Number of neurons
        n_orn = m_init.shape[1]

    bk_vari_init, bk_vec_init = bk_init
    assert n_orn == bk_vec_init.shape[0], "Mismatch between dimension of m and background"
    alpha, beta = inhib_params
    learnrate, tavg, coupling, lambd, sat, ktheta, decay_relative = ibcm_params
    # Compensate for lambda different from 1, if applicable
    mu_abs = learnrate / lambd

    rng = np.random.default_rng(seed=seed)
    tseries = np.arange(0, tmax, dt*skp)

    # Check that the biggest matrices, W or M, will not use too much memory
    if tseries.shape[0] * n_orn * n_neu > 5e8 / 8:  # 500 MB per series max
        raise ValueError("Excessive memory use by saved series; increase skp")

    # Containers for the solution over time
    bk_series = np.zeros([tseries.shape[0]] + list(bk_vari_init.shape))
    m_series = np.zeros([tseries.shape[0], n_neu, n_orn])
    cbar_series = np.zeros([tseries.shape[0], n_neu])
    w_series = np.zeros([tseries.shape[0], n_orn, n_neu])  # Inhibitory weights
    bkvec_series = np.zeros([tseries.shape[0], n_orn])  # Input vecs, convenient to compute inhibited output
    s_series = np.zeros([tseries.shape[0], n_orn])
    theta_series = np.zeros([tseries.shape[0], n_neu])

    ## Initialize running variables, separate from the containers above to avoid side effects.
    m = m_init.copy()
    bk_vari = bk_vari_init.copy()
    bkvec = bk_vec_init.copy()
    c = m.dot(bkvec)  # un-inhibited neuron activities
    # Initialize neuron activity with m and background at time zero
    cbar = c - coupling*(np.sum(c) - c)  # -c to cancel the subtraction of c[i] itself
    if saturation == "tanh":
        sat_abs = sat * lambd
        cbar = sat_abs * np.tanh(cbar / sat_abs)
    else: sat_abs = None
    if theta_init is None:
        # Important to initialize cbar2_avg to non-zero values, because we divide by this!
        cbar2_avg = np.maximum(cbar*cbar / lambd, learnrate*lambd)
    else:
        cbar2_avg = theta_init.copy()
    wmat = w_init.copy()
    svec = bk_vec_init - wmat.dot(cbar)
    if activ_fct == "relu":
        relu_inplace(svec)
    elif activ_fct == "identity":
        pass
    else:
        raise ValueError("Unknown activation fct: {}".format(activ_fct))

    # Store back some initial values in containers
    cbar_series[0] = cbar
    bk_series[0] = bk_vari
    m_series[0] = m_init
    bkvec_series[0] = bkvec
    s_series[0] = svec
    theta_series[0] = cbar2_avg
    w_series[0] = wmat

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
    newax = np.newaxis
    for k in range(0, len(tseries)*skp-1):
        t += dt
        ### Inhibitory  weights
        # They depend on cbar and svec at time step k, which are still in cbar, svec
        # cbar, shape [n_neu], should broadcast against columns of wmat,
        # while svec, shape [n_orn], should broadcast across rows (copied on each column)
        if w_norms[0] == 2:  # default L2 norm, nice and smooth
            alpha_term = alpha * cbar[newax, :] * svec[:, newax]
        elif w_norms[0] == 1:  # L1 norm
            asnorm = alpha * l1_norm(svec)
            alpha_term = asnorm * cbar[newax, :] * np.sign(svec[:, newax])
        elif w_norms[0] > 2:  # Assuming some Lp norm with p > 2
            # Avoid division by zero for p > 2 by clipping snorm
            snorm = max(1e-9, lp_norm(svec, p=w_norms[0]))
            sterm = np.sign(svec) * np.abs(svec/snorm)**(w_norms[0]-1) * snorm
            alpha_term = alpha * cbar[newax, :] * sterm[:, newax]
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

        ### IBCM neurons
        # Phi function for each neuron.
        if variant == "intrator":
            phiterms_vec = cbar * (cbar - cbar2_avg)
        #  Law and Cooper modification for faster convergence.
        elif variant == "law":
            phiterms_vec = cbar * (cbar - cbar2_avg) / (ktheta + cbar2_avg/lambd)
        else:
            raise ValueError("Unknown variant: {}".format(variant))

        if saturation == "tanh":
            phiterms_vec *=  1.0 - (cbar/sat_abs)**2
        # Now, careful with broadcast: for each neuron (dimension 0 of m and cbar), we need a scalar element
        # of phiterms_vec times the whole bkvec, for dimension 1 of m.
        # This can be done vectorially with a dot product (n_neu, 1)x(1, n_components)
        rhs_scalar = phiterms_vec - coupling*(np.sum(phiterms_vec) - phiterms_vec)
        # Euler integrator and learning rate
        # learnrate_t = learnrate if t < 150000 else learnrate / 5
        # Reducing learning rate after a while may help.
        # Consider feedback on mu through some metric of how well neurons
        # are inhibiting the background, e.g. s average activity.
        m += mu_abs*dt*rhs_scalar[:, np.newaxis].dot(bkvec[np.newaxis, :])
        # In principle, should add low decay to background subspace
        # To make sure 1) only learn the background space, 2) de-habituate after
        # The decay term is proportional to m, not m^2 like the IBCM term
        # so we needed to divide learnrate by Lambda for the IBCM term
        # but not for this linear decay term, which should use just learnrate
        if decay and variant == "law":
            m -= dt * decay_relative * learnrate / (ktheta + cbar2_avg[:, np.newaxis]/lambd) * m
        elif decay and variant == "intrator":
            m -= dt * decay_relative * learnrate * m
        # Now, update to time k+1 the threshold (cbar2_avg) using cbar at time k
        # to be used to update m in the next time step
        cbar2_avg += dt * (cbar*cbar / lambd - cbar2_avg)/tavg

        # Update background to time k+1, to be used in next time step
        bkvec, bk_vari = update_bk(bk_vari, bk_params, noises[k], dt)

        # Then, compute activity of IBCM neurons at next time step, k+1,
        # with the updated background and synaptic weight vector m
        # Compute un-inhibited activity of each neuron with current input (at time k)
        c = m.dot(bkvec)
        cbar = c - coupling*(np.sum(c) - c)  # -c to cancel the subtraction of c[i] itself
        if saturation == "tanh":
            cbar = sat_abs * np.tanh(cbar / sat_abs)
        # np.sum(c) is a scalar and c a vector, so it broadcasts properly.

        # Lastly, projection neurons at time step k+1
        svec = bkvec - wmat.dot(cbar)
        if activ_fct == "relu":
            relu_inplace(svec)

        # Save current state only if at a multiple of skp
        if (k % skp) == (skp - 1):
            knext = (k+1) // skp
            w_series[knext] = wmat
            m_series[knext] = m
            bk_series[knext] = bk_vari
            bkvec_series[knext] = bkvec
            cbar_series[knext] = cbar  # Save activity of neurons at time k+1
            s_series[knext] = svec
            theta_series[knext] = cbar2_avg

    return [tseries, bk_series, bkvec_series, m_series,
            cbar_series, theta_series, w_series, s_series]


# For legacy reasons, keep other versions with hard-coded options
# but they are now wrappers around the _options version.
def integrate_inhib_ibcm_network(m_init, update_bk, bk_init,
    ibcm_params, inhib_params, bk_params, tmax, dt,
    seed=None, noisetype="normal", activ_fct="ReLU"):
    """ See integrate_inhib_ibcm_network_options documentation """
    model_options = {
        "activ_fct": activ_fct,
        "saturation": "linear",
        "variant": "intrator",
        "decay": False
    }
    # Add missing rates
    ibcm_params2 = list(ibcm_params) + [0.0]*3
    res = integrate_inhib_ibcm_network_options(m_init, update_bk, bk_init,
        ibcm_params2, inhib_params, bk_params, tmax, dt,
        seed=seed, noisetype=noisetype, skp=1, **model_options)
    return res


def integrate_inhib_ibcm_network_tanh(m_init, update_bk, bk_init,
    ibcm_params, inhib_params, bk_params, tmax, dt,
    seed=None, noisetype="normal", activ_fct="ReLU"):
    """ See  integrate_inhib_ibcm_network_options documentation """
    model_options = {
        "activ_fct": activ_fct,
        "saturation": "tanh",
        "variant": "intrator",
        "decay": False
    }
    ibcm_params2 = list(ibcm_params) + [0.0]*2
    res = integrate_inhib_ibcm_network_options(m_init, update_bk, bk_init,
        ibcm_params2, inhib_params, bk_params, tmax, dt,
        seed=seed, noisetype=noisetype, skp=1, **model_options)
    return res


def integrate_inhib_ibcm_network_skip(m_init, update_bk, bk_init,
    ibcm_params, inhib_params, bk_params, tmax, dt,
    seed=None, noisetype="normal", skp=1, activ_fct="ReLU"):
    """ See  integrate_inhib_ibcm_network_options documentation """
    model_options = {
        "activ_fct": activ_fct,
        "saturation": "linear",
        "variant": "intrator",
        "decay": False
    }
    ibcm_params2 = list(ibcm_params) + [0.0]*3
    res = integrate_inhib_ibcm_network_options(m_init, update_bk, bk_init,
        ibcm_params2, inhib_params, bk_params, tmax, dt,
        seed=seed, noisetype=noisetype, skp=skp, **model_options)
    return res


def integrate_inhib_ibcm_network_skip_tanh(m_init, update_bk, bk_init,
    ibcm_params, inhib_params, bk_params, tmax, dt,
    seed=None, noisetype="normal", skp=1, activ_fct="ReLU"):
    """ See  integrate_inhib_ibcm_network_options documentation """
    model_options = {
        "activ_fct": activ_fct,
        "saturation": "tanh",
        "variant": "intrator",
        "decay": False
    }
    ibcm_params2 = list(ibcm_params) + [0.0]*2
    res = integrate_inhib_ibcm_network(m_init, update_bk, bk_init,
        ibcm_params2, inhib_params, bk_params, tmax, dt,
        seed=seed, noisetype=noisetype, skp=skp, **model_options)
    return res


# Law and Cooper modification to accelerate convergence in large simulations
# I only coded the tanh version since I assume this is only needed for
# large, less stable networks.
def integrate_inhib_lawibcm_network_skip_tanh(m_init, update_bk, bk_init,
    ibcm_params, inhib_params, bk_params, tmax, dt,
    seed=None, noisetype="normal", skp=1, activ_fct="ReLU"):
    """ See  integrate_inhib_ibcm_network_options documentation """
    model_options = {
        "activ_fct": activ_fct,
        "saturation": "tanh",
        "variant": "law",
        "decay": False
    }
    ibcm_params2 = list(ibcm_params) + [0.0]  # Adding dummy decay rate
    res = integrate_inhib_ibcm_network_options(m_init, update_bk, bk_init,
        ibcm_params2, inhib_params, bk_params, tmax, dt,
        seed=seed, noisetype=noisetype, skp=skp, **model_options)
    return res


def ibcm_respond_new_odors(odors, mmat, wmat, ibcm_rates, options={}):
    """
    Args:
        odors (np.ndarray): indexed [..., n_orn]
            so can take dot product properly with m and store many
            odors along arbitrary other axes.
        mmat (np.ndarray): indexed [n_neurons, n_orn]
        wmat (np.ndarray): indexed [n_orn, n_neurons]
        ibcm_rates (np.ndarray): contains coupling eta and tanh amplitude
        options (dict):
            activ_fct (str): either "ReLU" or "identity"
            saturation (str): "tanh" or "linear"
    """
    saturation = options.get("saturation", "linear")
    activ_fct = str(options.get("activ_fct", "ReLU")).lower()
    eta = ibcm_rates[2]
    lambd = ibcm_rates[3]
    if saturation == "tanh":
        sat = ibcm_rates[4] * lambd
    else: sat = None
    # Compute activation of neurons to the mixtures (new+background)
    # Given the IBCM and inhibitory neurons' current state
    # (either latest or some average state of the neurons)
    c = odors.dot(mmat.T)
    cbar = c - eta*(np.sum(c, axis=-1, keepdims=True) - c)
    # Saturation
    if saturation == "tanh":
        cbar = sat * np.tanh(cbar / sat)

    # cbar shape: odors.shape[:-1], n_neurons
    # New odor after inhibition by the network, ReLU activation on s
    # Inhibit with the mean cbar*wser, to see how on average the new odor will show
    svec = odors - cbar.dot(wmat.T)
    if str(activ_fct).lower() == "identity":
        pass
    elif str(activ_fct).lower() == "relu":
        relu_inplace(svec)
    else:
        raise ValueError("Unknown activation function: {}".format(activ_fct))
    return svec


def compute_mbars_cgammas_cbargammas(ms, eta, backvecs):
    r"""
    Compute the time series of \bar{m} = m_i - \eta \sum_{j\neq i} m_j,
    of c_{\gamma} = \vec{m} \cdot \vec{x}_{\gamma},
    and of \bar{c}_{\gamma} = \vec{\bar{m}} \cdot \vec{x}_{\gamma}
    Returns:
        mbs (np.ndarray): \bar{m} series
        cgs (np.ndarray): c_{\gamma} series
        cbgs (np.ndarray): \bar{c}_{\gamma} series
    """
    mbs = ms * (1 + eta) - eta*np.sum(ms, axis=1, keepdims=True)
    cgs = ms.dot(backvecs.T)
    cbgs = mbs.dot(backvecs.T)
    return mbs, cgs, cbgs
