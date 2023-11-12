""" Functions to evaluate the analytical solutions for habituation
with the BioPCA model.

@author: frbourassa
November 2023
"""
import numpy as np
from utils.metrics import l2_norm

### Special solution for toy background with 2 odors, 1 varying proportion ###
def fixedpoints_pca_2vectors(components, sigma2, ab_rates):
    """
    Special PCA solution for toy background x = (1/2+nu)*x_a + (1/2-nu)*x_b,
    with one PCA neuron and average background subtraction assumed.
    x_s = (x_a - x_b), x_d = (x_a + x_b)/2

    Args:
        components (np.ndarray): x_a and x_b, odors components
        sigma2 (float): variance of nu (the composition of the mixture)
        ab_rates (list of 2 floats): alpha and beta

    Returns:
        m_pca (np.ndarray): M matrix, a 1xn_R vector in fact, parallel to x_s
        l_pca (np.ndarray): L' matrix, a scalar in fact, equal to eigenvalue
        w_pca (np.ndarray): W matrix, a n_R vector in fact, parallel to x_s
    """
    x_d = np.sum(components, axis=0)/2.0
    x_s = np.diff(components, axis=0)
    n_r = components.shape[1]
    ba_ratio = ab_rates[1] / ab_rates[0]

    l_pca = sigma2 * l2_norm(x_s)
    m_pca = sigma2 * np.sqrt(l2_norm(x_s)) * x_s
    w_pca = m_pca / (l_pca + ba_ratio)
    return m_pca, l_pca, w_pca


def pca_fixedpoint_s_2vectors_instant(components, sigma2, ab_rates, xser):
    ba_ratio = ab_rates[1] / ab_rates[0]
    x_d = np.sum(components, axis=0)/2.0
    x_s = np.diff(components, axis=0)

    return ba_ratio / (ba_ratio + sigma2 * l2_norm(x_s)) * (xser - x_d)

def pca_fixedpoint_s_2vectors_variance(components, sigma2, ab_rates):
    ba_ratio = ab_rates[1] / ab_rates[0]
    x_d = np.sum(components, axis=0)/2.0
    x_s = np.diff(components, axis=0)
    pc = sigma2 * l2_norm(x_s)
    return ba_ratio / (ba_ratio + pc) * pc
