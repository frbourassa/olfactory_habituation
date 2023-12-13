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


def compute_optimal_factor(nu_new, moments, dims, odor_gen_fct, gen_args):
    """Optimal reduction factor of the parallel component to minimize
    the L2 distance with new odors. Assuming odor vectors,
    new and background alike, are generated independently by odor_gen_fct
    and then normalized individually.
    Typically, this works by sampling i.i.d. exponential random variables
    for elements, and then normalizing the vector.
    Also assuming background concentrations are i.i.d. with average avgnu
    and variance sigma2_nu.
    This factor depends on the new concentration, nu_new.
    dims = [n_B, n_R].
    moments = [avgnu, sigma2_nu]
    """
    avgnu, sigma2_nu = moments
    n_b, n_r = dims
    # Compute the average vector. Use 10^5 samples
    vec_samples = odor_gen_fct([n_r, int(1e5)], *gen_args)
    vec_samples = vec_samples / np.sqrt(l2_norm(vec_samples, axis=0))
    mean_vec_element = np.mean(vec_samples)  # All elements equal
    mean_vec_norm2 = n_r * mean_vec_element**2

    # Compute terms that appear in the optimum solution
    cross_product = nu_new * avgnu * n_b * mean_vec_norm2
    mean_back2 = avgnu**2 * (n_b + n_b*(n_b-1)*mean_vec_norm2) + n_b*sigma2_nu

    optimal_factor = cross_product + nu_new**2
    optimal_factor /= (mean_back2 + 2*cross_product + nu_new**2)
    return optimal_factor


def compute_optimal_factor_toy(nu_new, sigma2, n_r, odor_gen_fct, gen_args):
    """ Like compute_optimal_factor but for the toy background model,
    $x_B = x_d + \nu x_s$.
    """
    # Compute the average vector. Use 10^5 samples
    vec_samples = odor_gen_fct([n_r, int(1e5)], *gen_args)
    vec_samples = vec_samples / np.sqrt(l2_norm(vec_samples, axis=0))
    mean_vec_element = np.mean(vec_samples)  # All elements equal
    mean_vec_norm2 = n_r * mean_vec_element**2
    cross_product = nu_new*mean_vec_norm2
    mean_back2 = 0.5 + (0.5 - 2*sigma2)*mean_vec_norm2 + 2*sigma2

    optimal_factor = cross_product + nu_new**2
    optimal_factor /= (mean_back2 + 2*cross_product + nu_new**2)
    return optimal_factor
