"""
Functions to generate odor affinity vectors and combine odors
according to the nonlinear OSN activation function from 
Kadakia and Emonet, eLife, 2019. 

See more explanations in the notebook 
supplementary_tests/nonlinear_osn_turbulent.ipynb

@author: frbourassa
August 2025
"""

import numpy as np
from modelfcts.distribs import inverse_transform_tanhcdf
from modelfcts.backgrounds import (
    update_tc_odor, 
    sample_ss_conc_powerlaw
)


def generate_odor_tanhcdf(n_rec, rgen, k1val=0.01, alpha=0.38751946, 
        logb=-1.29460373, unit_scale=5e-4, k2range=1e3):
    """ Generate vectors of K and K^* for an odor, following 
    a fit on the data of Si et al., 2019 for K^* affinities and
    Kadakia and Emonet, eLife, 2019 to set all K = k1val = 0.01 default. 
    
    K = small affinity, 0.01 in Kadakia and Emonet
    K^* = larger affinities, can be up to 10^9 in Si et al.'s units, here 
    we clip them to k2range times the lower cutoff sensitivity in Si et al., 
    which is at roughly (1/b)^{1/alpha} * unit_scale, to prevent extreme values. 
    
    We use the distribution 1-CDF = tanh(1/bx^a) fitted on Si et al. data
    in the notebook si2019_hill_tanh_distribution_fits.ipynb, 
    which has a power-law tail cut a cutoff at lower concentrations. 
    Best parameter fits: log10(b) = -1.29460373, 
                        alpha = 0.38751946 (power-law exponent)
    
    The inverse EC50s from Si et al. correspond to affinities like $K$ or $K^*$.
    But they are in relative units of inverse dilution, which do not match 
    the typical scale of our dimensionless turbulent concentrations. 
    Here, we scale the default dilution units to match the two scales and 
    maintain OSN activation around their half-maximum, producing non-negligible 
    nonlinearity in the OSNs without making them always saturated. 
    
    Args:
        n_rec (int or tuple): number of receptor types, length of vectors
            n_rec can be an int (n_son, to make a single odor) or a tuple
            (n_components, n_osn) to make two matrices of odor vectors.
        rgen (np.random.Generator): random generator (numpy >= 1.17)
        k1val (float): value of all K affinities
        k2inv_bounds (tuple of 2 floats): lowest and highest 1/K* possible
        alpha (float): power law has CDF with exponent alpha, 
            PDF with exponent alpha-1
    
    Returns:
        kmats (np.ndarray): matrix of inactive and active complex binding 
            affinities, shaped [n_rec, 2]
    """
    k1vec = np.full(n_rec, k1val * unit_scale)
    
    r = rgen.random(size=n_rec)
    k2vec = inverse_transform_tanhcdf(r, logb, alpha) * unit_scale
    # In the sampled distribution, the lower cutoff is at x^alpha=1/b, roughly
    k2vec = np.clip(k2vec, 0.0, k2range * unit_scale * 10.0**(-logb/alpha))
    
    return np.stack([k1vec, k2vec], axis=-1)


def combine_odors_affinities(concs, kmats, epsils, fmax=1.0):
    """ Combine odors coming in with concentrations conc and defined
    by active and inactive binding affinities kappa1, kappa2. 
    OSN types have free energy differences epsils. 

    Args:
        concs (np.ndarray): 1d array of odor concentrations, indexed [n_odors]
        kmats (np.ndarray): shape [n_odors, n_osns, 2], 
            affinities of inactive and active complexes, respectively
        epsils (np.ndarray): shape [n_osns], free energy difference 
            of each OSN type
        fmax (float): maximum amplitude, default 1, but we usually scale
            to 1/sqrt(n_dimensions) to maintain OSN activities in 
            the magnitude of s for which IBCM, BioPCA rates have been 
            chosen in other simulations. 

    Returns:
        activ (np.ndarray): 1d array of ORN activation, indexed [n_receptors]
    """
    k1mat, k2mat = kmats[:, :, 0], kmats[:, :, 1]
    # Dot products over odors
    kc1 = concs.dot(k1mat)
    kc2 = concs.dot(k2mat)
    logterm = (1.0 + kc1) / (1.0 + kc2)
    activs = fmax / (1.0 + np.exp(epsils) * logterm)
    return activs


# Function to update the background by combining odors at the current 
# concentrations with the OSN model
def update_powerlaw_times_concs_affinities(tc_bk, params_bk, noises, dt):
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
            fmax (float): maximum OSN activation amplitude (normalization)
            epsils (np.ndarray): free energy difference of each OSN type, 
                shaped [n_osn]. 
            vecs (np.ndarray): 3d array where axis 2 has length 2, 
                the first sub-array giving the K vectors of each odor, 
                and the second, giving K* vectors. 
        noises (np.ndarray): fresh U(0, 1) samples, shaped [n_odors, 2],
            in case we need to pull a new t and/or c.
        dt (float): time step duration, in simulation units
    """
    # Update one odor's t and c at a time, if necessary
    tc_bk_new = np.zeros(tc_bk.shape)
    for i in range(tc_bk.shape[0]):
        tc_bk_new[i] = update_tc_odor(tc_bk[i], dt, noises[i],
                                *[p[i] for p in params_bk[:-3]])

    # Compute backgound vector (even if it didn't change)
    kmats = params_bk[-1]  # k, k*
    epsils = params_bk[-2]
    fmax = params_bk[-3]
    new_bk_vec = combine_odors_affinities(tc_bk_new[:, 1], 
                                    kmats, epsils, fmax=fmax)
    return new_bk_vec, tc_bk_new


# Sampling concentrations is the same, but sampling background vectors
# requires applying the nonlinear response function
def sample_background_powerlaw_nl_osn(kmats, *args, size=1, rgen=None):
    """
    Args:
        k_mats (np.ndarray): array of background odor affinities K, K^*,
            indexed [2, odor, dimension]
        args:
            whiff_tmins (np.ndarray): lower cutoff in the power law
                of whiff durations, for each odor
            whiff_tmaxs (np.ndarray): upper cutoff in the power law
                of whiff durations, for each odor
            blank_tmins (np.ndarray): same as whiff_tmins but for blanks
            blank_tmaxs (np.ndarray): same as whiff_tmaxs but for blanks
            c0s (np.ndarray): c0 concentration scale for each odor
            alphas (np.ndarray): alpha*c0 is the lower cutoff of p_c
            fmax (float): maximum OSN activation amplitude (normalization)
            epsils (np.ndarray): free energy difference of each OSN type, 
                shaped [n_osn]. 
            vecs (np.ndarray): 3d array where axis 0 has length 2, 
                the first sub-array giving the K vectors of each odor, 
                and the second, giving K* vectors. 
        size (int): number of background samples to generate (default: 1)
        rgen (np.random.Generator): random generator (optional).
    """
    # Get samples of concentration variables nu first
    nu_samp = sample_ss_conc_powerlaw(*args[:6], size=size, rgen=rgen)

    # Then, combine background vectors with those concentrations
    epsils, fmax_osn = args[6], args[7]
    vec_samp = combine_odors_affinities(nu_samp, kmats, epsils, fmax=fmax_osn)
    return vec_samp, nu_samp





### Simplified version with one affinity per odor ###
def combine_odors_compet(concs, kmat, epsils, fmax=1.0):
    """ Combine odors coming in with concentrations conc and defined
    by active and inactive binding affinities kappa1, kappa2. 
    OSN types have free energy differences epsils. 

    Args:
        concs (np.ndarray): 1d array of odor concentrations, indexed [n_odors]
        kmat (np.ndarray): shape [n_odors, n_osns], OR affinities
        epsils (np.ndarray): shape [n_osns], free energy difference 
            of each OSN type
        fmax (float): maximum amplitude, default 1, but we usually scale
            to 1/sqrt(n_dimensions) to maintain OSN activities in 
            the magnitude of s for which IBCM, BioPCA rates have been 
            chosen in other simulations. 

    Returns:
        activ (np.ndarray): 1d array of ORN activation, indexed [n_receptors]
    """
    # Dot products over odors
    kc = concs.dot(kmat)
    activs = fmax * kc / (np.exp(epsils) + kc)
    return activs


def update_powerlaw_times_concs_compet(tc_bk, params_bk, noises, dt):
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
            vecs (np.ndarray): 3d array where axis 0 has length 2, 
                the first sub-array giving the K vectors of each odor, 
                and the second, giving K* vectors. 
            epsils (np.ndarray): free energy difference of each OSN type, 
                shaped [n_osn]. 
            fmax (float): maximum OSN activation amplitude (normalization)
        noises (np.ndarray): fresh U(0, 1) samples, shaped [n_odors, 2],
            in case we need to pull a new t and/or c.
        dt (float): time step duration, in simulation units
    """
    # Update one odor's t and c at a time, if necessary
    tc_bk_new = np.zeros(tc_bk.shape)
    for i in range(tc_bk.shape[0]):
        tc_bk_new[i] = update_tc_odor(tc_bk[i], dt, noises[i],
                                *[p[i] for p in params_bk[:-3]])

    # Compute backgound vector (even if it didn't change)
    kvecs = params_bk[-3]  # k, k*
    epsils = params_bk[-2]
    fmax = params_bk[-1]
    new_bk_vec = combine_odors_compet(tc_bk_new[:, 1], kvecs, 
                                      epsils, fmax=fmax)
    return new_bk_vec, tc_bk_new
