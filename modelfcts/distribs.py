""" Functions implementing probability distributions related
to turbulent olfactory backgrounds. Up to now, I have implemented:

- Distribution of inter-whiff or inter-blanks times (same form, change cutoff):
    p_t(t) = 1/A (t/tau)^(-3/2)     if t > tau, zero otherwise

- Distribution of concentration in a whiff:
    p_c(c) = uniform                if c <= alpha*c0
           = 1/A exp(-c/c_0) / c    if c > alpha*c0


With these probability distributions, it is possible to simulate an olfactory
background with approximately the statistics described in
    Celani, Villermaux, Vergassola, PRX, 2014

The strategy is to simulate waiting times between whiffs and durations of
whiffs using p_t, and choose a constant concentration from p_c for each whiff.

@author: frbourassa
June 2022
"""
import numpy as np
import scipy as sp
import scipy.special
from scipy.optimize import root_scalar, RootResults

###
### E_1 INTEGRAND DENSITY WITH LOW-END SATURATION, FOR CONCENTRATIONS ###
###

## Inverse exponential integral
# The first two derivatives of E_1(y) for Halley's method.
# Assumes y > 0 for now. Need same call form as implicit_inverse_exp1_equation
# so x needs to be a second argument
def deriv_exp1(y, x):
    return -np.exp(y) / y


def deriv2_exp1(y, x):
    return np.exp(-y) / y * (1.0 + 1.0 / y)


# Implicit equation for y, the inverse of exp1
def implicit_inverse_exp1_equation(y, x):
    return sp.special.exp1(y) - x


def implicit_inverse_exp1_equation_log(y, logx):
    return np.log(sp.special.exp1(y)) - logx


# Inverse exponential integral using brentq method.
def inverse_exp1(x):
    if x <= 0:
        raise ValueError("x = E_1(y) does not take negative values for y > 0")
    # If x is small, better solve in log scale
    elif x < 5:
        # Define decent limits based on x
        logx = np.log(x)
        ylo = max(1e-16, (-5-logx)/1.1)
        yhi = max(10, 5-logx)
        # Solve E_1(y) = x for y, which is the result to return
        res = root_scalar(implicit_inverse_exp1_equation_log, x0=1.0,
            args=(np.log(x),), method="brentq", fprime=False, fprime2=False,
            bracket=[ylo, yhi])
    # Otherwise, solve in linear scale, bracket should be
    # between some very small positive number and y=0.5, since E_1(0.5) < 5 already
    elif x < 25:
        ylo = 1e-14  # E1(1e-12) = 25, approximately.
        yhi = 0.5
        res = root_scalar(implicit_inverse_exp1_equation, x0=0.1,
            args=(x,), method="brentq", fprime=deriv_exp1, fprime2=deriv2_exp1,
            bracket=[ylo, yhi])
    # Beyond x=25, very hard to solve at all, cancellation errors occur.
    # Use the Taylor series E1(y) = -gamma - log(y)
    else:
        #raise ValueError("Can't invert x=E1(y) beyond x = 25, cancellations")
        res = RootResults(root=np.exp(-x - np.euler_gamma), iterations=1,
            function_calls=1, flag=0)
        res.converged = True
    if res.converged:
        return res.root
    else:
        raise RuntimeError("Failed to invert exp1, flag: ", res.flag)

# Vectorized function to actually use
vec_inverse_exp1 = np.vectorize(inverse_exp1, otypes=[float])

## Actual E_1 distribution
# Transform U(0, 1) samples into the truncated exp1 distribution.
def truncexp1_inverse_transform(unif, c0, alpha_c0):
    """ Assumes unif is a np.ndarray """
    norm_constant_a = np.exp(-alpha_c0) + sp.special.exp1(alpha_c0)
    # Where unif < e^(-alpha)/A
    wh = (unif <= np.exp(-alpha_c0) / norm_constant_a)
    c = np.zeros(unif.shape)
    c[wh] = alpha_c0 * norm_constant_a * c0 * np.exp(alpha_c0) * unif[wh]
    wh = np.logical_not(wh)
    c[wh] = c0 * vec_inverse_exp1(norm_constant_a * (1.0 - unif[wh]))
    return c


# Probability density for the E_1 integrand with low-end saturation.
def truncexp1_density(c, c0, alpha_c0):
    norm_constant_a = np.exp(-alpha_c0) + sp.special.exp1(alpha_c0)
    wh = (c <= alpha_c0 * c0)
    dens = np.zeros(c.shape)
    dens[wh] = np.exp(-alpha_c0) / (alpha_c0 * norm_constant_a * c0)
    wh = np.logical_not(wh)
    c_slice = c[wh]
    dens[wh] = np.exp(-c_slice/c0) / (c_slice * norm_constant_a)
    return dens


# Average concentration of whiffs
def truncexp1_average(c0, alpha_c0):
    norm_constant_a = np.exp(-alpha_c0) + sp.special.exp1(alpha_c0)
    avg_c = (1 + 0.5*alpha_c0) * c0 * np.exp(-alpha_c0) / norm_constant_a
    return avg_c


###
### TRUNCATED POWER LAW, FOR WAITING TIMES ###
###
# Transform U(0, 1) samples into power law samples
def powerlaw_cutoff_inverse_transform(unif, tmin, tmax):
    factor = (1.0 - np.sqrt(tmin/tmax))
    return tmin / ((1.0 - unif * factor)**2)
