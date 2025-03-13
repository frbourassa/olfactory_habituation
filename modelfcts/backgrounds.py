""" Module containing functions for the time evolution of background odor
mixtures according to different stochastic processes.

@author: frbourassa
September 2021
Updated June 2022
"""
import numpy as np
from modelfcts.distribs import (truncexp1_inverse_transform,  # whiff concs
            powerlaw_cutoff_inverse_transform)   # whiff or blank durations


# Realistic model of olfactory receptor activation patterns:
# each component is i.i.d. exponential
def generate_odorant(n_rec, rgen, lambda_in=0.1):
    """ Generate ORN affinity vector for an odorant, with mean 1/lambda_in

    Args:
        n_rec (int): number of receptor types, length of vectors
        rgen (np.random.Generator): random generate (numpy >= 1.17)
        lambda_in (float): lambda parameter of the exp distribution
            Equals the inverse of the average of each vector component
    Returns:
        kappa1_vec (np.ndarray): 1d vector of receptor activities
    """
    return rgen.exponential(scale=1.0/lambda_in, size=n_rec)


### ORNSTEIN-UHLENBECK PROCESSES ###
# Function to update the fluctuating background variable
def update_ou_kinputs(nu_bk, params_bk, noises, dt):
    r"""
    Update a background made of L odorants with concentrations combined linearly:
        x(t) = \sum_{\alpha} \nu_{\alpha} x_{\alpha}
    The concentrations nu_alpha fluctuate according to a multivariate
    Ornstein-Uhlenbeck process with some correlation between them. The general update rule is of the form
        \vec{\nu}(t + dt) = M_A \vec{\nu}(t) + M_B \vec{n}
    where M_A and M_B are related to the matrix coefficients A and B in the Langevin equation
    as indicated above (exp factors and Cholesky decomposition), and \vec{n} is a vector of
    independent normal(0, 1) samples.
    This function is agnostic to A and B and just requires the overall matrix coefficients
    in front of of x(t) and \vec{n} in the update rule.

    Args:
        nu_bk (np.ndarray): array of length K, containing concentrations nu_i
            of odorants in the background
        params_bk (list of arrays): mateA, matJB, vecs_nu, means_nu: matrices of shape (K, K)
            involved in the update rule and matrix of background component vectors x_{\alpha}, shape (K, d)
            where d is the number of dimensions. Each row is a different component.
            Also a 1d array of the mean values of the nu variables. They are simulated with zero mean
            and their average is added before the background is computed
        noises: 1d array of pre-generated normal(0, 1) samples, one per component in nu_bk
    """
    mateA, matJB, vecs_nu, means_nu = params_bk
    nu_bk_new = np.dot(mateA, nu_bk) + np.dot(matJB, noises)
    # Update background vector, putting back the mean value of the nu's
    bkvec = np.squeeze(np.dot((nu_bk_new+means_nu)[np.newaxis, :], vecs_nu))

    return bkvec, nu_bk_new


# Special case of two input components combined with nu and 1-nu.
def update_ou_2inputs(nu_bk, params_bk, noise, dt):
    """
    Update a background made of k odorants with randomly fluctuating proportions
    Args:
        nu_bk (np.ndarray): array of length 1, containing proportion nu of odorant a
        params_bk (list of arrays): mean_nu, upcoef_mean, upcoef_noise, vecs_nu
            Average nu value, coefficients in the exact update rule for the O-U process,
            and linear components of the background odor, in a 2D array with shape[0]==k.
        noises: pre-generated normal(0, 1) sample, one per component in nu_bk
    """
    # Extract parameters
    mean_nu, upcoef_mean, upcoef_noise, vecs_nu = params_bk

    # Update the proportions nu: exact update rule from Gillespie 1996 (c of Gillespie = 2/tau*sigma^2)
    # Coef for mean: exp(-dt/tau)
    # Coef for noise: sqrt(sigma^2*(1 - exp(-2dt/tau)))
    # Works for mean=0, so remove the mean first, then put it back after update
    nu_bk = nu_bk - mean_nu  # At time t
    nu_bk = upcoef_mean*nu_bk + upcoef_noise*noise  # At time t+dt now
    nu_bk = nu_bk + mean_nu  # Put mean back at t+dt

    # Compute new background vector:  bk_j = \sum_i nu_i vecs_{ij}
    bkvec = (0.5 + nu_bk) * vecs_nu[0] + (0.5 - nu_bk) * vecs_nu[1]

    return bkvec, nu_bk


def update_ou_2inputs_clip(nu_bk, params_bk, noise, dt):
    """ Same update as update_ou_2inputs, but the concentration of each odor
    is clipped to be non-negative. This ensures that ORN activities
    are non-negative as well if the background vectors x_a, x_b
    are non-negative.
    """
    # Extract parameters
    mean_nu, upcoef_mean, upcoef_noise, vecs_nu = params_bk

    # Update the proportions nu: exact update rule from Gillespie 1996 (c of Gillespie = 2/tau*sigma^2)
    # Coef for mean: exp(-dt/tau)
    # Coef for noise: sqrt(sigma^2*(1 - exp(-2dt/tau)))
    # Works for mean=0, so remove the mean first, then put it back after update
    nu_bk = nu_bk - mean_nu  # At time t
    nu_bk = upcoef_mean*nu_bk + upcoef_noise*noise  # At time t+dt now
     # Put mean back at t+dt and clip
    nu_bk = np.clip(nu_bk + mean_nu, a_min=-0.5, a_max=0.5)

    # Compute new background vector:  bk_j = \sum_i nu_i vecs_{ij}
    bkvec = (0.5 + nu_bk) * vecs_nu[0] + (0.5 - nu_bk) * vecs_nu[1]

    return bkvec, nu_bk


### NON-NORMAL PROCESSES BASED ON TRANSFORMATIONS OF ORNSTEIN-UHLENBECK ###
def update_thirdmoment_kinputs(x_bk, params_bk, noises, dt):
    r"""
    Update a background made of L odorants with concentrations combined linearly:
        x(t) = \sum_{\alpha} \nu_{\alpha} x_{\alpha}
    The concentrations nu_alpha are given by
        \nu_\alpha = s_\alpha + x_\alpha + \epsilon x_\alpha
    where the $x_\alpha$ fluctuate according to a multivariate
    Ornstein-Uhlenbeck process with some correlation between them. The general update rule is of the form
        \vec{x}(t + dt) = M_A \vec{x}(t) + M_B \vec{n}
    where M_A and M_B are related to the matrix coefficients A and B in the Langevin equation
    as indicated above (exp factors and Cholesky decomposition), and \vec{n} is a vector of
    independent normal(0, 1) samples.
    This function is agnostic to A and B and just requires the overall matrix coefficients
    in front of of x(t) and \vec{n} in the update rule.

    Args:
        x_bk (np.ndarray): array of length K, containing normal processes x_\alpha underlying
            the concentrations \nu_\alpha of odorants in the background
        params_bk (list of arrays): mateA, matJB, vecs_nu, means_nu, epsil: matrices of shape (K, K)
            involved in the update rule and matrix of background component vectors x_{\alpha}, shape (K, d)
            where d is the number of dimensions. Each row is a different component.
            Also a 1d array of the mean values of the nu variables. They are simulated with zero mean
            and their average is added before the background is computed.
            epsil is the magnitude of the quadratic contribution to nu.
        noises: 1d array of pre-generated normal(0, 1) samples, one per component in nu_bk
    """
    # First, update gaussian x
    mateA, matJB, vecs_nu, means_nu, epsil = params_bk
    x_bk_new = np.dot(mateA, x_bk) + np.dot(matJB, noises)
    # Then compute nu
    nu_bk_new = means_nu + x_bk_new + epsil*x_bk_new*x_bk_new
    # Update background vector, putting back the mean value of the nu's
    bkvec = np.squeeze(np.dot(nu_bk_new[np.newaxis, :], vecs_nu))

    return bkvec, x_bk_new

# Steady-state concentration distribution sampling, small third moment
def sample_ss_distrib_thirdmoment(means_nu, covmat_nu, epsil, size=1, rgen=None):
    """ Steady-state probability distribution: let x be multivariate  normal
    with mean means_nu and covariance matrix covmat_nu, then this function
    returns samples of x + epsil*x**2. """
    if rgen is None:
        rgen = np.random.default_rng()
    x_samp = rgen.multivariate_normal(means_nu, covmat_nu, size=size)
    return x_samp + epsil * x_samp**2

# Steady-state background vector sampling, small third moment distribution
def sample_background_thirdmoment(means_nu, covmat_nu, epsil, vecs_nu, size=1, rgen=None):
    # Get samples of nu variables first
    nu_samp = sample_ss_distrib_thirdmoment(means_nu, covmat_nu, epsil, size=size, rgen=rgen)

    # Then, combine background vectors with those nu coefficients
    # Each row in nu_samp is a different sample, so taking the dot product
    # the first dimension still indexes samples, the second gives components.
    vec_samp = np.dot(nu_samp, vecs_nu)
    return vec_samp


# LOG-NORMAL PROCESS (simulate the log with Ornstein-Uhlenbeck)
logof10 = np.log(10.0)
def update_logou_kinputs(nu_bk, params_bk, noises, dt):
    r"""
    Update a background made of L odorants with concentrations combined linearly:
        x(t) = \sum_{\alpha} 10**{\nu_{\alpha}} x_{\alpha}
    The logarithm of concentrations, nu_alpha, fluctuate according to a
    multivariate Ornstein-Uhlenbeck process, optionally with some correlation.
     The general update rule is of the form
        \vec{\nu}(t + dt) = M_A \vec{\nu}(t) + M_B \vec{n}
    where M_A and M_B are related to the matrix coefficients A and B in the Langevin equation
    as indicated above (exp factors and Cholesky decomposition), and \vec{n} is a vector of
    independent normal(0, 1) samples.
    This function is agnostic to A and B and just requires the overall matrix coefficients
    in front of of x(t) and \vec{n} in the update rule.

    Args:
        nu_bk (np.ndarray): array of length K, containing log10 of concentrations nu_i
            of odorants in the background
        params_bk (list of arrays): mateA, matJB, vecs_nu, means_nu: matrices of shape (K, K)
            involved in the update rule and matrix of background component vectors x_{\alpha}, shape (K, d)
            where d is the number of dimensions. Each row is a different component.
            Also a 1d array of the mean values of the nu variables. They are simulated with zero mean
            and their average is added before the background is computed
        noises: 1d array of pre-generated normal(0, 1) samples, one per component in nu_bk
    """
    mateA, matJB, vecs_nu, means_nu = params_bk
    nu_bk_new = np.dot(mateA, nu_bk) + np.dot(matJB, noises)
    # Update background vector, putting back the mean value of the nu's and
    # exponentiating because the nu's are the logarithms of
    # log-normal concentrations
    concentrations = np.exp((nu_bk_new+means_nu)*logof10)  # 10**nu_\alpha
    bkvec = np.squeeze(np.dot(concentrations[np.newaxis, :], vecs_nu))

    return bkvec, nu_bk_new


### ALTERNATING PROCESS ###
def update_alternating_inputs(idx_bk, params_bk, noises, dt):
    """ Select randomly the next background input.
    Args:
        nu_bk (np.ndarray): array of length k-1, containing proportions nu_i of odorants
        params_bk (list):  Contains the following parameters
            cumul_probs (np.ndarray): cumulative probabilities up to the kth input vector.
            vecs (np.ndarray): 2d array where each row is one of the possible input vectors
        noises (np.1darray): pre-generated uniform(0, 1) samples, in an array of length 1,
            to choose next input vector.
        """
    # Index of the next input
    cumul_probs, vecs = params_bk
    idx = np.argmax(cumul_probs > noises[0])
    return vecs[idx], np.asarray([idx])


### UTILITY FUNCTIONS ###
## Decompose on some non-orthogonal basis, for the purpose here of
# the basis x_d = 1/2(x_a + x_b), x_s = (x_a - x_b)
def decompose_nonorthogonal_basis(vec, basis):
    """ Each column of basis contains one of the basis vectors"""
    coefs = np.linalg.solve(basis, vec)
    return coefs


### POWER LAW WAIT TIMES BETWEEN WHIFFS ###
def update_tc_odor(tc, dt, unif, *args):
    """ Update t and c of an odor, if necessary after time step dt,
    using uniform(0, 1) noise samples and parameters in args.
    For further info, see update_powerlaw_times_concs documentation.
    """
    twlo, twhi, tblo, tbhi, c0, alpha = args
    # Update needed if dt or less left to the wait time.
    if (tc[0] - dt) <= 0:
        # Determine whether we were in a whiff or a blank
        if tc[1] > 0 :  # we were in a whiff
            # Pull t_b, duration of new blank
            tc[0:1] = powerlaw_cutoff_inverse_transform(unif[0:1], tblo, tbhi)
            # Set c to zero
            tc[1] = 0.0
        else:  # we were in a blank
            # Pull t_w, duration of new whiff
            tc[0:1] = powerlaw_cutoff_inverse_transform(unif[0:1], twlo, twhi)
            # Set conc c of the whiff
            tc[1:2] = truncexp1_inverse_transform(unif[1:], c0, alpha)
    else:
        tc[0] = tc[0] - dt
    return tc


def update_powerlaw_times_concs(tc_bk, params_bk, noises, dt):
    """
    Simulate turbulent odors by pulling wait times until the end of a whiff
        or until the next blank, and a concentration of the whiff.
        For each odor, check whether the time left until switch is <= zero;
        if so, pull either
            - another wait time t_w if current c=0, and pull the new c > 0
              (we were in a blank and are starting a whiff)
            - another wait time t_b if current c > 0, and set c = 0
              (we were in a whiff and are starting a blank)
        Otherwise, decrement t by dt and don't change c.

    Args:
        tc_bk (np.ndarray): array of t, c for each odor in the background,
            where t = time left until next change, c = current concentration
            of the odor. Shaped [n_odors, 2]
        params_bk (list): contains the following elements (a lot needed!):
            whiff_tmins (np.ndarray): lower cutoff in the power law
                of whiff durations, for each odor
            whiff_tmaxs (np.ndarray): upper cutoff in the power law
                of whiff durations, for each odor
            blank_tmins (np.ndarray): same as whiff_tmins but for blanks
            blank_tmaxs (np.ndarray): same as whiff_tmaxs but for blanks
            c0s (np.ndarray): c0 concentration scale for each odor
            alphas (np.ndarray): alpha*c0 is the lower cutoff of p_c
            vecs (np.ndarray): 2d array where each row is one of the
                possible input vectors
        noises (np.ndarray): fresh U(0, 1) samples, shaped [n_odors, 2],
            in case we need to pull a new t and/or c.
            TODO: most noises are wasted; for now memory isn't an issue
            but this is a place where the code can be optimized a lot.
            But generating 10^7 noise samples takes ~ 100 ms, this is not
            costing much time, so really, only memory is wasted.
        dt (float): time step duration, in simulation units
    """
    # Update one odor's t and c at a time, if necessary
    tc_bk_new = np.zeros(tc_bk.shape)
    for i in range(tc_bk.shape[0]):
        tc_bk_new[i] = update_tc_odor(tc_bk[i], dt, noises[i],
                                *[p[i] for p in params_bk[:-1]])

    # Compute backgound vector (even if it didn't change)
    # TODO: this could be optimized too by giving the current back vec
    # as an input, but this requires editing the ibcm simulation functions
    vecs_nu = params_bk[-1]
    new_bk_vec = np.squeeze(np.dot(tc_bk_new[:, 1:2].T, vecs_nu))
    return new_bk_vec, tc_bk_new


def update_powerlaw_times_concs_saturate(tc_bk, params_bk, noises, dt):
    # Update one odor's t and c at a time, if necessary
    tc_bk_new = np.zeros(tc_bk.shape)
    for i in range(tc_bk.shape[0]):
        tc_bk_new[i] = update_tc_odor(tc_bk[i], dt, noises[i],
                                *[p[i] for p in params_bk[:-1]])

    # Compute backgound vector (even if it didn't change)
    vecs_nu = params_bk[-1]
    new_bk_vec = np.squeeze(np.dot(tc_bk_new[:, 1:2].T, vecs_nu))
    # Apply tanh saturation of the mixture. Typical vectors have magnitude
    # well below 1.
    new_bk_vec = 3.0*np.tanh(new_bk_vec/3.0)
    return new_bk_vec, tc_bk_new


# Steady-state sampling from the powerlaw and exp1 background
def sample_ss_conc_powerlaw(*args, size=1, rgen=None):
    """ Steady-state probability distribution:
    either zero concentration or non-zero, with
    probability 1-chi or chi respectively,
    where chi = E(t_w)/(E(t_b) + E(t_w)). Then,
    if non-zero, sample concentration from the truncated exp1 law.
    """
    twlo, twhi, tblo, tbhi, c0, alpha = args
    n_odors = len(twlo)
    if rgen is None:
        rgen = np.random.default_rng()
    # 1. Generate uniform samples to determine which concentrations will be above zero
    r_sampl = rgen.random(size=[size, n_odors])
    chi_prob = 1.0 / (1.0 + np.sqrt(tblo*tbhi/twlo/twhi))
    # 1. Determine which odors will have non-zero concentration
    wh = (r_sampl < chi_prob)
    nu_samp = np.zeros(r_sampl.shape)
    # 2. Generate uniform samples and transform to conc. distribution
    # Need to treat one odor (column) at a time to use the right params
    for i in range(n_odors):
        r_sampl = rgen.random(size=np.sum(wh[:, i]))
        nu_samp[wh[:, i], i] = truncexp1_inverse_transform(r_sampl, c0[i], alpha[i])

    return nu_samp

def sample_background_powerlaw(vecs_nu, *args, size=1, rgen=None):
    """
    Args:
        vecs_nu (np.ndarray): array of background odor vectors,
            indexed [odor, dimension]
        args:
            whiff_tmins (np.ndarray): lower cutoff in the power law
                of whiff durations, for each odor
            whiff_tmaxs (np.ndarray): upper cutoff in the power law
                of whiff durations, for each odor
            blank_tmins (np.ndarray): same as whiff_tmins but for blanks
            blank_tmaxs (np.ndarray): same as whiff_tmaxs but for blanks
            c0s (np.ndarray): c0 concentration scale for each odor
            alphas (np.ndarray): alpha*c0 is the lower cutoff of p_c
        size (int): number of background samples to generate (default: 1)
        rgen (np.random.Generator): random generator (optional).
    """
    # Get samples of concentration variables nu first
    nu_samp = sample_ss_conc_powerlaw(*args, size=size, rgen=rgen)

    # Then, combine background vectors with those nu coefficients
    # Each row in nu_samp is a different sample, so taking the dot product
    # the first dimension still indexes samples, the second gives components.
    vec_samp = np.dot(nu_samp, vecs_nu)
    return vec_samp


### Turbulent background with Gaussian noise on OSNs ###
def box_muller_2d(samples, pair_axis=0):
    """ Transform pairs of uniform(0, 1) random samples to 
    pairs of normal(0, 1) using the Box-Muller transform. 
    
    Args:
        samples (np.ndarray): 1d or 2d array of samples
        pair_axis (int): axis of length 2 containing
            pairs of samples. Default: 0
    """
    # Since we don't know the axis of pairs yet, use the .take method: 
    # https://numpy.org/doc/stable/reference/generated/numpy.take.html
    # my_array.take(indices=range(2, 7), axis=3)
    ampli = np.sqrt(-2.0*np.log(samples.take(indices=0, axis=pair_axis)))
    twopisamples = 2.0*np.pi*samples.take(indices=1, axis=pair_axis)
    z1 = ampli * np.cos(twopisamples)
    z2 = ampli * np.sin(twopisamples)
    return np.stack([z1, z2], axis=pair_axis)

def update_powerlaw_gauss_noise(tc_bk, params_bk, noises, dt):
    """
    Simulate turbulent odors by pulling wait times until the end of a whiff
        or until the next blank, and a concentration of the whiff.
        For each odor, check whether the time left until switch is <= zero;
        if so, pull either
            - another wait time t_w if current c=0, and pull the new c > 0
              (we were in a blank and are starting a whiff)
            - another wait time t_b if current c > 0, and set c = 0
              (we were in a whiff and are starting a blank)
        Otherwise, decrement t by dt and don't change c.
    Then add uncorrelated Gaussian white noise to each ORN in 
    the background vector. 

    Args:
        tc_bk (np.ndarray): array of t, c for each odor in the background,
            where t = time left until next change, c = current concentration
            of the odor. 
            And remaining n_R//2 rows are current Gaussian noise. 
            Shaped [n_odors + n_R//2, 2]
        params_bk (list): contains the following elements (a lot needed!):
            whiff_tmins (np.ndarray): lower cutoff in the power law
                of whiff durations, for each odor
            whiff_tmaxs (np.ndarray): upper cutoff in the power law
                of whiff durations, for each odor
            blank_tmins (np.ndarray): same as whiff_tmins but for blanks
            blank_tmaxs (np.ndarray): same as whiff_tmaxs but for blanks
            c0s (np.ndarray): c0 concentration scale for each odor
            alphas (np.ndarray): alpha*c0 is the lower cutoff of p_c
            noise_ampli (1-element array): amplitude (standard dev.) 
                of Gaussian white noise added to each ORN on top of the background. 
            vecs (np.ndarray): 2d array where each row is one of the
                possible input vectors
        noises (np.ndarray): fresh U(0, 1) samples, shaped [n_odors + [n_R//2], 2],
            in case we need to pull a new t and/or c.
        dt (float): time step duration, in simulation units
    """
    # Update one odor's t and c at a time, if necessary
    tc_bk_new = np.zeros(tc_bk.shape)
    # infer number of odors based on number of dimensions
    vecs_nu = params_bk[-1]
    n_odors, n_orn = vecs_nu.shape
    for i in range(n_odors):
        tc_bk_new[i] = update_tc_odor(tc_bk[i], dt, noises[i, :2],
                                *[p[i] for p in params_bk[:6]])
    # Transform the remaining n_R samples to normal samples, 
    # to be added as normal noise. Save them as RVs too
    tc_bk_new[n_odors:] = box_muller_2d(noises[n_odors:], pair_axis=1)
    norm_noises = tc_bk_new[n_odors:].flatten()

    # Compute backgound vector (even if it didn't change)
    # TODO: this could be optimized too by giving the current back vec
    # as an input, but this requires editing the ibcm simulation functions
    new_bk_vec = np.dot(tc_bk_new[:n_odors, 1], vecs_nu)
    #noise_ampli = params_bk[-2]
    new_bk_vec += params_bk[-2] * norm_noises[:new_bk_vec.shape[0]]  # If odd n_R, remove last noise
    return new_bk_vec, tc_bk_new


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Test the power law background
    params_background = [
        [1.0, 2.0, 1.0],        # whiff_tmins
        [100., 200.0, 200.0],   # whiff_tmaxs
        [2.0, 2.0, 1.0],        # blank_tmins
        [1000.0, 200.0, 100.0],  # blank_tmaxs
        [1.0, 2.0, 1.0],        # c0s
        [0.5, 0.2, 0.5],        # alphas
        np.asarray([[2, 1, 1], [1, 2, 1], [1, 1, 2]])/np.sqrt(6)  # vecs
    ]
    params_background = [p[0:1] for p in params_background]
    # Initial values
    tc_init = np.asarray([[2.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    tc_init = tc_init[0:1]
    # Time course parameters
    deltat = 1.0
    timesteps = np.arange(0.0, 1e4, deltat)
    rgen = np.random.default_rng(seed=0xfcea6b41f4e5005baf07dd59baf7c040)
    unif_samples = rgen.random(size=[timesteps.size] + list(tc_init.shape))
    tc_series = np.zeros([timesteps.size, *tc_init.shape])
    tc_series[0] = tc_init
    vec_series = np.zeros([timesteps.size, params_background[-1].shape[1]])
    for i in range(timesteps.size - 1):
        vec_series[i], tc_series[i+1] = update_powerlaw_times_concs(
                            tc_series[i], params_background,
                            unif_samples[i], deltat)

    # Plot the time series of concentrations
    fig, ax = plt.subplots()
    for j in range(len(params_background[0])):
        ax.plot(timesteps, tc_series[:, j, 1], label="Odor {}".format(j))
    ax.set(xlabel="Time (dt)", ylabel="Concentration")
    ax.legend()
    plt.show()
    plt.close()
