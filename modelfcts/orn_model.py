"""
Functions to generate odor eta and kappa vectors and combine odors
according to the nonlinear ORN activation function defined in
Reddy et al., eLife, 2018.

@author: frbourassa
June 2022
"""

import numpy as np


def generate_odorant_kappaeta(rho, n_rec, std_kappa1, rgen):
    """ Generate vectors eta and kappa^-1 for an odorant, with antagonism parameter rho.

    Args:
        rho (float): correlation coef between -1 and 1
        n_rec (int): number of receptor types, length of vectors
        std_kappa1 (float): standard deviation of the inverse binding affinities (4, typically)
        rgen (np.random.Generator): random generate (numpy >= 1.17)
    Returns:
        kappa1_vec (np.ndarray): 1d vector of binding affinities inverses
        eta_vec (np.ndarray): 1d vector of activation efficacies
    """
    logeta_vec = rgen.normal(size=n_rec)
    omega_vec = rgen.normal(size=n_rec)
    logkappa1_vec = std_kappa1 * (rho*logeta_vec + np.sqrt(1-rho**2)*omega_vec)
    return np.exp(logkappa1_vec), np.exp(logeta_vec)

def combine_odors_ornmodel(concs, invkappavecs, etavecs, n_cng=4, fmax=1.0):
    """ Combine odors coming in with concentrations conc and defined
    by binding affinities kappavecs and activation efficacies etavecs.
    Activation function has parameter fmax (amplitude) and n_cng (exponent).

    Args:
        concs (np.ndarray): 1d array of odor concentrations, indexed [n_odors]
        invkappavecs (np.ndarray): 2d array of inverse kappa vector for each
            odorant, indexed [n_odors, n_receptors]
        etavecs (np.ndarray): 2d array of eta vector for each odorant,
            indexed [n_odors, n_receptors]
        n_cng (int): power in the denominator of the response function
        fmax (float): max amplitude of an ORN activity.

    Returns:
        activ (np.ndarray): 1d array of ORN activation, indexed [n_receptors]
    """
    # Total concentration, scalar
    ctot = np.sum(concs)
    if ctot <= 0.0:
        activ = np.zeros(invkappavecs.shape[1])
    else:
        # Concentration fraction, indexed [n_odors, 1] for broadcasting
        betas = concs[:, np.newaxis] / ctot
        # \kappa_{mix}^{-1}, indexed [n_receptors]
        invkappa_mix = np.sum(betas * invkappavecs, axis=0)
        # \eta_{mix}, indexed [n_receptors].  Do not multiply by kappa_mix
        # because it is canceled in the activation function anyways.
        eta_mix = np.sum(etavecs * betas * invkappavecs, axis=0)# / invkappa_mix
        # Ratio at the denominator of F(\nu), indexed [n_receptors]
        ratio = (1.0 + ctot*invkappa_mix) / (eta_mix * ctot)  # *invkappa_mix
        # Total activation
        activ = fmax / (1.0 + ratio**n_cng)
    return activ
