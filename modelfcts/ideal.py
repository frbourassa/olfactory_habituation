""" Ideal linear background inhibition model, which subtracts the new odor
component parallel to the background as well.

The ReLU function is defined here too.

@author: frbourassa
September 30, 2023
"""
import numpy as np
from utils.metrics import l2_norm


### IDEAL INHIBITION ###
def relu_inplace(x):
    """ WARNING: modifies x in-place. Ideal for new arrays, views,
    or combinations/operations on arrays"""
    return np.maximum(x, 0.0, x)
#def relu_inplace(x):
#    return (np.abs(x) + x) / 2


def find_projector(a):
    """ Calculate projector a a^+, which projects
    a column vector on the vector space spanned by columns of a.
    """
    a_inv = np.linalg.pinv(a)
    return a.dot(a_inv)

def find_parallel_component(x, basis, projector=None):
    """
    Args:
        x (np.ndarray): 1d array of length D containing the vector to decompose.
        basis (np.ndarray): 2d matrix of size DxK where each column is one
            of the linearly independent background vectors.
        projector (np.ndarray): 2d matrix A A^+, the projector on the vector
            space spanned by columns of basis.
    Return:
        x_par (np.ndarray): component of x found in the vector space of basis
            The perpendicular component can be obtained as x - x_par.
    """
    # If the projector is not provided yet
    if projector is None:
        # Compute Moore-Penrose pseudo-inverse and AA^+ projector
        projector = find_projector(basis)
    x_par = projector.dot(x)
    return x_par

def ideal_linear_inhibitor(x_n_par, x_n_ort, x_back, f, factor, **opt):
    """ Calculate the ideal projection neuron layer, which assumes
    perfect inhibition (down to beta/(2*alpha+beta)) of the component of the mixture
    parallel to the background odors' vector space, while leaving the orthogonal
    component of the new odor untouched.

    Args:
        x_n_par (np.1darray): new odor, component parallel to background vector space
        x_n_ort (np.1darray): new odor, component orthogonal to background vector space
        x_back (np.2darray): background samples, one per row
        f (float): mixture fraction (hard case is f=0.2)
        factor (float): reduction factor of the parallel component.
    Keyword args: opt
        activ_fct (str): either "ReLU" or "identity"

    Returns:
        s (np.1darray): projection neurons after perfect linear inhibition
    """
    # Allow broadcasting for multiple x_back vectors
    #factor = beta / (2*alpha + beta)
    s = factor * f*x_n_par + f*x_n_ort
    # I thought the following would have been even better, but turns out it is worse for small f
    #s = f*x_n_par + f*x_n_ort
    s = s.reshape(1, -1) + factor * x_back
    if str(opt.get("activ_fct", "ReLU")).lower() == "relu":
        relu_inplace(s)
    else:
        pass
    return s


def compute_ideal_factor(
        nu_new, moments, dims, odor_gen_fct, gen_args, reps=1000
    ):
    """Optimal reduction factor of the parallel component to minimize
    the L2 distance with new odors. Assuming odor vectors,
    new and background alike, are generated independently by odor_gen_fct
    and then normalized individually.
    Typically, this works by sampling i.i.d. exponential random variables
    for elements, and then normalizing the vector.
    Also assuming background concentrations are i.i.d. with average avgnu
    and variance sigma2_nu.
    This factor depends on the new concentration, nu_new.
    dims = [n_B, n_S].
    moments = [avgnu, sigma2_nu]
    reps (int): number of samples scales as reps**2: backgrounds x new odors
    """
    avgnu, sigma2_nu = moments
    n_b, n_s = dims
    # Compute the average odor vector. Use 10^5 samples
    n_samples = reps**2
    vec_samples = odor_gen_fct([n_s, n_samples], *gen_args)
    vec_samples = vec_samples / l2_norm(vec_samples, axis=0)
    mean_vec_element = np.mean(vec_samples)  # All elements equal by symmetry
    mean_vec_norm2 = n_s * mean_vec_element**2

    # Compute the average norm^2 of the parallel component of new odors
    # in a background of n_b odors. Generate new background vectors,
    # use a subsample of vec_samples above as new odors
    back_samples = odor_gen_fct([n_s, reps*n_b], *gen_args)
    back_samples = back_samples / l2_norm(back_samples, axis=0)
    x_par_norms2 = np.zeros(reps**2)  # Container for all x_{n, par}^2 samples
    for n in range(reps):
        back_vecs = back_samples[:, n*n_b:(n+1)*n_b]
        proj = find_projector(back_vecs)
        # Take some of the previously generated samples as new odors
        for m in range(n*reps, (n+1)*reps):
            x_par = find_parallel_component(
                        vec_samples[:, m], back_vecs, projector=proj
                    )
            x_par_norms2[m] = np.sum(x_par**2)
    x_par_norm2 = np.mean(x_par_norms2)

    # Compute terms that appear in the optimum solution
    cross_product = nu_new * avgnu * n_b * mean_vec_norm2
    mean_back2 = avgnu**2 * (n_b + n_b*(n_b-1)*mean_vec_norm2) + n_b*sigma2_nu
    mean_par2 = x_par_norm2 * nu_new**2

    optimal_factor = cross_product + mean_par2
    optimal_factor /= (mean_back2 + 2*cross_product + mean_par2)
    return optimal_factor


def compute_ideal_factor_toy(nu_new, sigma2, n_s, odor_gen_fct, gen_args):
    """ Like compute_ideal_factor but for the toy background model,
    $x_B = x_d + \nu x_s$.
    """
    # Compute the average vector. Use 10^5 samples
    # TODO: fix with an estimate of x_n^2 norm
    vec_samples = odor_gen_fct([n_s, int(1e5)], *gen_args)
    vec_samples = vec_samples / l2_norm(vec_samples, axis=0)
    mean_vec_element = np.mean(vec_samples)  # All elements equal
    mean_vec_norm2 = n_s * mean_vec_element**2
    cross_product = nu_new*mean_vec_norm2
    mean_back2 = 0.5 + (0.5 - 2*sigma2)*mean_vec_norm2 + 2*sigma2

    optimal_factor = cross_product + nu_new**2
    optimal_factor /= (mean_back2 + 2*cross_product + nu_new**2)
    return optimal_factor


### Optimal matrices for manifold learning, iid odor concentrations
def compute_optimal_matrices(back_odors, new_odors, moments_conc, new_concs):
    """ Optimal matrix for manifold learning in a background with
    iid odor concentrations with moments_conc, back_odor vectors, 
    geared towards response to new_odors at concentrations new_concs. 
    For background odor mixture s and new odor x, the optimum is
    W = LM^+ with L = <s(x + s)^T>, M = <(s+x)(s+x)^T> """
    # Compute optimal matrix for each new concentration. Need to compute
    # the average new odor <s_n>, and average <s_n s_n^T> first. 
    avg_new_odors = [c * np.mean(new_odors, axis=0) for c in new_concs]
    # Outer product of each new odor with itself: s_n s_n^T
    # Memory use: chunksize * dimension**2 * 8bytes, limit to 1.2 GB per process
    chunksize = int(1.2e9 / (new_odors.shape[1]**2 * 8))
    if chunksize >= new_odors.shape[0]:  # One chunk is enough
        avg_new_odor_mat = np.mean(new_odors[:, :, None] * new_odors[:, None, :], axis=0)
        # Multiply by <c^2> = <c>^2 + sigma^2 for each new c
    else:  # Too many odors at once, need to loop over chunks
        nchunks = new_odors.shape[0] // chunksize
        last_chunksize = new_odors.shape[0] % chunksize
        avg_new_odor_mat = np.zeros([new_odors.shape[1], new_odors.shape[1]])
        for i in range(nchunks):
            istart = i*chunksize
            iend = (i+1)*chunksize
            avg_new_odor_mat += np.sum(
                new_odors[istart:iend, :, None] * new_odors[istart:iend, None, :], 
                axis=0) / new_odors.shape[0]
        if last_chunksize > 0:
            istart = nchunks*chunksize
            avg_new_odor_mat += np.sum(
                new_odors[istart:, :, None] * new_odors[istart:, None, :], 
                axis=0) / new_odors.shape[0]
    covmats_new_odors = [avg_new_odor_mat * (c**2 + moments_conc[1]) for c in new_concs]
            
    del avg_new_odor_mat
    # moments of the background vectors too
    avg_back = moments_conc[0] * np.sum(back_odors, axis=0)
    # \sum_\rho s_\rho s_\rho^T
    covmat_back = np.sum(back_odors[:, :, None]*back_odors[:, None, :], axis=0)
    covmat_back = moments_conc[1] * covmat_back + np.outer(avg_back, avg_back)
    # With these components, we are ready to compute W optimal for each background
    optimal_ws = []
    for i in range(len(new_concs)):
        m = (covmats_new_odors[i] + covmat_back 
            + np.outer(avg_new_odors[i], avg_back)
            + np.outer(avg_back, avg_new_odors[i])
        )
        left_mat = np.outer(avg_back, avg_new_odors[i]) + covmat_back
        w_opt = left_mat.dot(np.linalg.pinv(m))
        optimal_ws.append(w_opt)
    optimal_ws = np.asarray(optimal_ws)
    return optimal_ws


### Re-run W for any model
def rerun_w_dynamics(w_init, xc_series, inhib_params, dt, skp=1, scale=1.0, **options):
    """ Run W dynamics given the background and cbar series of some model
    simulation.

    Args:
        w_init (np.ndarray): initial W weigths, shape [n_R, n_I]
        xc_series (list of 2 np.ndarray): bkvecser, cbarser, xavgser (optional)
            of some model. They cannot have skipped steps.
        inhib_params (list of 2 floats): alpha, beta
        dt (float): time step of the simulations.
        skp (int): save every skp time step of w
        scale (float): cbar is multiplied by scale.

    Options:
        activ_fct (str): "ReLU" or "identity"

    Returns:
        wser (np.ndarray): shape [time, n_R, n_I]
        sser (np.ndarray): shape [time, n_R]
    """
    # Extract parameters
    activ_fct = str(options.get("activ_fct", "ReLU")).lower()
    remove_mean = options.get("remove_mean", False)
    if remove_mean:
        bkvecser, cbarser, xavgser = xc_series
    else:
        bkvecser, cbarser = xc_series
        xavgser = [0.0]
    alpha, beta = inhib_params
    n_times = bkvecser.shape[0]
    n_r = bkvecser.shape[1]
    n_i = cbarser.shape[1]
    # Prepare container series
    w_series = np.zeros([n_times // skp, n_r, n_i])
    s_series = np.zeros([n_times // skp, n_r])
    # Current state variables
    wmat = w_init.copy()
    cbar = cbarser[0] * scale
    bkvec = bkvecser[0]
    xavg = xavgser[0]
    svec = bkvecser[0] - wmat.dot(cbar)
    if activ_fct == "relu":
        relu_inplace(svec)
    elif activ_fct == "identity":
        pass
    else:
        raise ValueError("Unknown activation fct: {}".format(activ_fct))
    # Store initial values
    w_series[0] = w_init
    s_series[0] = svec

    # Start looping
    t = 0
    for k in range(0, n_times-1):
        t += dt
        # Access current values
        cbar = cbarser[k] * scale
        bkvec = bkvecser[k]
        if remove_mean:  # Only change xavg from 0.0 if remove_mean
            xavg = xavgser[k]
        wmat += dt * (alpha*cbar[np.newaxis, :]*svec[:, np.newaxis] - beta*wmat)

        # Lastly, projection neurons at time step k+1
        svec = bkvec - xavg - wmat.dot(cbar)
        if activ_fct == "relu":
            relu_inplace(svec)

        # Save current state
        if (k % skp) == (skp - 1):
            knext = (k+1) // skp
            w_series[knext] = wmat
            s_series[knext] = svec

    return w_series, s_series
