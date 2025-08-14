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
import math
import scipy.special
from scipy.optimize import root_scalar, RootResults

# New Scipy >= 1.12: RootResults takes "method" argument
# we want the new behavior for older versions
def create_root_results(root, iterations, function_calls, flag, method=None):
    try:
        rr = RootResults(root, iterations, function_calls, flag, method=method)
    except TypeError:  # old version compatibility
        rr = RootResults(root, iterations, function_calls, flag)
    return rr


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


def implicit_inverse_exp1_equation_logy(logy, x):
    return sp.special.exp1(math.exp(logy)) - x


# Inverse exponential integral using brentq method.
def inverse_exp1(x):
    if x <= 0:
        raise ValueError("x = E_1(y) does not take negative values for y > 0")
    # If x is small, better solve in log scale
    # In our simulations of turbulent backgrounds, we are always in this regime
    elif x < 1.0:  # This is where the error starts to grow if we solve in log scale
        # Define decent limits based on x
        logx = np.log(x)
        ylo = max(1e-16, (-5-logx)/1.1)
        yhi = max(10, 5-logx)
        # Solve E_1(y) = x for y, which is the result to return
        res = root_scalar(implicit_inverse_exp1_equation_log, x0=1.0,
            args=(np.log(x),), method="brentq", fprime=False, fprime2=False,
            bracket=[ylo, yhi])
    # Otherwise, solve for log(y)=z, but x and E_1(y) in linear scale, bracket should be
    # between some very small positive number and y=0.5, since E_1(0.5) < 5 already
    # E1(1e-8) \approx 18, error on the Taylor series of E1 at order y^2 is then 1e-16
    # then we don't need to solve using root_scalar
    elif x < 30:
        logylo = math.log(1e-15)
        logyhi = math.log(0.5)
        res = root_scalar(implicit_inverse_exp1_equation_logy, x0=math.log(0.1),
            args=(x,), method="brentq", fprime=None, fprime2=None,
            bracket=[logylo, logyhi])
        if res.converged:
            res.root = np.exp(res.root)
    # Beyond x=25, very hard to solve at all, cancellation errors occur.
    # Use the Taylor series E1(y) = -gamma - log(y)
    else:
        #raise ValueError("Can't invert x=E1(y) beyond x = 25, cancellations")
        res = create_root_results(root=np.exp(-x - np.euler_gamma), iterations=1,
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


# More general truncated power-law: Y between y_0, y_1, 
# pdf with exponent a-1. Transform U(0, 1) samples into such Y. 
def power_range_inverse_transform(unif, y0, y1, alpha):
    y0a = y0**alpha
    y1a = y1**alpha
    y = (unif*(y1a-y0a) + y0a)**(1.0 / alpha)
    return y

### POWER-LAW TAIL WITH tanh PLATEAU
# For affinities sampling in a nonlinear OSN adaptation model
def inverse_transform_tanhcdf(r, logb, alpha):
    """ Inverse transform method to sample from a distribution with
    complementary CDF tanh(1/(b*x^a)) """
    return (10.0**logb * np.arctanh(r))**(-1.0 / alpha)

