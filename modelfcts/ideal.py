""" Ideal linear background inhibition model, which subtracts the new odor
component parallel to the background as well.

The ReLU function is defined here too.

@author: frbourassa
September 30, 2023
"""
import numpy as np


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

def ideal_linear_inhibitor(x_n_par, x_n_ort, x_back, f, alpha, beta, **opt):
    """ Calculate the ideal projection neuron layer, which assumes
    perfect inhibition (down to beta/(2*alpha+beta)) of the component of the mixture
    parallel to the background odors' vector space, while leaving the orthogonal
    component of the new odor untouched.

    Args:
        x_n_par (np.1darray): new odor, component parallel to background vector space
        x_n_ort (np.1darray): new odor, component orthogonal to background vector space
        x_back (np.2darray): background samples, one per row
        f (float): mixture fraction (hard case is f=0.2)
        alpha (float): inhibitory weights learning rate alpha
        beta (float): inhibitory weights decaying rate beta
    Keyword args: opt
        activ_fct (str): either "ReLU" or "identity"

    Returns:
        s (np.1darray): projection neurons after perfect linear inhibition
    """
    # Allow broadcasting for multiple x_back vectors
    factor = beta / (2*alpha + beta)
    s = factor * f*x_n_par + f*x_n_ort
    # I thought the following would have been even better, but turns out it is worse for small f
    #s = f*x_n_par + f*x_n_ort
    s = s.reshape(1, -1) + factor * x_back
    if str(opt.get("activ_fct", "ReLU")).lower() == "relu":
        relu_inplace(s)
    else:
        pass
    return s
