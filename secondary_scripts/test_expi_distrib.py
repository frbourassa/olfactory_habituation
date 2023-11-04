"""
Test sampling from a power law.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import root_scalar, RootResults
import scipy.special

from time import perf_counter

cmyk_blue = "#3E529F"
cmyk_red = "#DA3833"
cmyk_green = "#307F54"

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

# Implicit equation for y, the inverse of exp1, solved in log.
# Also return first and second derivatives, since results can be
# reused in this way.
def implicit_inverse_exp1_equation_log_full(y, logx):
    expi = sp.special.exp1(y)
    logexpi = np.log(expi)
    deriv = -np.exp(-y) / (y * expi)
    deriv2 = deriv*deriv - deriv * (1.0 + 1.0 / y)
    return logexpi - logx, deriv, deriv2

def implicit_inverse_exp1_equation_log(y, logx):
    return np.log(sp.special.exp1(y)) - logx

# Inverse exponential integral using Halley's method.
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

vec_inverse_exp1 = np.vectorize(inverse_exp1)

# Test the inverse E_1 function just coded
def test_inverse_exp1():
    xrange = np.arange(0.01, 50, 0.1)
    inverse_soln = vec_inverse_exp1(xrange)
    exp1_vals = sp.special.exp1(inverse_soln)  # Should give xrange back
    fig, ax = plt.subplots()
    ax.plot(xrange, exp1_vals, lw=4)
    ax.plot(xrange, xrange, color="r", lw=1.5, ls="--")
    ax.set(xlabel="x", ylabel=r"$E_1(E_1^{-1}(x))$")
    plt.show()
    plt.close()


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


def truncexp1_density(c, c0, alpha_c0):
    norm_constant_a = np.exp(-alpha_c0) + sp.special.exp1(alpha_c0)
    wh = (c <= alpha_c0 * c0)
    dens = np.zeros(c.shape)
    dens[wh] = np.exp(-alpha_c0) / (alpha_c0 * norm_constant_a * c0)
    wh = np.logical_not(wh)
    c_slice = c[wh]
    dens[wh] = np.exp(-c_slice/c0) / (c_slice * norm_constant_a)
    return dens


if __name__ == "__main__":
    # 1. Test the inverse exp1
    #test_inverse_exp1()

    # 2. Test the power law with unlimited upper value
    rgen = np.random.default_rng(seed=0xe746043073634d83a2bc6fb83ba4b2fd)
    unif_samples = rgen.random(size=int(1e6))

    conc0 = 1.0
    alpha = 0.5
    start_time = perf_counter()
    print("Starting to convert {} samples".format(unif_samples.size))
    concs_samples = truncexp1_inverse_transform(unif_samples, conc0, alpha)
    end_time = perf_counter()
    print("Finished converting {} samples".format(unif_samples.size))
    time_per_sample = (end_time - start_time) / unif_samples.size
    print("This took on average {} s per sample".format(time_per_sample))

    fig, ax = plt.subplots()
    # Note: this pdf is better seen with linear x scale, y log scale.
    counts, binseps = np.histogram(concs_samples, bins="doane")
    binwidths = np.diff(binseps)
    # Center of bins on a log scale, but given in linear coordinates
    bin_centers = (binseps[1:] + binseps[:-1])/2
    pdf = counts / binwidths / concs_samples.size
    cdf = np.cumsum(pdf)
    ax.bar(x=binseps[:-1], align="edge", height=pdf, width=binwidths,
            color=cmyk_blue, label="Samples")
    ax.set(xlabel=r"$c$", ylabel=r"$p_c(c)$", yscale="log")

    bin_axis = np.linspace(binseps[0], binseps[-1], 201)
    dens = truncexp1_density(bin_axis, conc0, alpha)
    ax.plot(bin_axis, dens, color=cmyk_red, lw=3.,
        label=r"$p_c(c) \sim \frac{e^{-c/c_0}}{Ac}$")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig("../figures/tests/concentration_pdf_test.pdf",
        transparent=True, bbox_inches="tight")
    plt.show()
    plt.close()
