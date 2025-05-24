""" Module with functions to check steady-state distributions,
detection of new odors, etc. for various background processes.
Collection of functions previously scattered across notebooks.

@author: frbourassa
June 2022
"""
import numpy as np
import matplotlib.pyplot as plt

from modelfcts.distribs import truncexp1_density
from utils.metrics import subspace_align_error, l2_norm


# Function to check the s.s. distribution of the power-law and exp1 process
def check_conc_samples_powerlaw_exp1(conc_samples, twlo, twhi, tblo, tbhi, c0s, alphas):
    """ Check that the turbulent background simulated over time
    obeys the right concentration statistics, with a fraction chi of samples
    where c > 0 (whiffs) and a fraction 1 - chi where c = 0 (blanks),
    and samples in whiffs (c > 0) obeying the exp1 integrand distribution.

    Args:
        conc_samples (np.ndarray): concentration samples (zero or non-zero),
            indexed [odor, sample]
        twlo, twhi, tblo, tbhi, c0s, alphas: see the params_bk (list) argument
            of the function modelfcts.backgrounds.update_powerlaw_times_concs
    Returns:
        fig, axes: histograms of the concentration samples overlaid with
            the analytical distribution based on t_w, t_b, and c parameters.
    """
    chis = 1.0 / (1.0 + np.sqrt(tblo*tbhi/twlo/twhi))
    cmyk_blue = "#3E529F"
    cmyk_red = "#DA3833"
    cmyk_green = "#307F54"
    fig, axes = plt.subplots(1, conc_samples.shape[0])
    axes = axes.flatten()
    fig.set_size_inches(2.75*len(axes), 2.75)
    for i in range(len(axes)):
        ax = axes[i]
        # First, histogram of nonzero samples
        nonzero_samples = conc_samples[i]
        nonzero_samples = nonzero_samples[nonzero_samples > 0.0]
        # Note: this pdf is better seen with linear x scale, y log scale.
        counts, binseps = np.histogram(nonzero_samples, bins="doane")
        binwidths = np.diff(binseps)
        # Center of bins on a log scale, but given in linear coordinates
        bin_centers = (binseps[1:] + binseps[:-1])/2
        pdf = counts / binwidths / conc_samples[i].size
        cdf = np.cumsum(pdf)
        ax.bar(x=binseps[:-1], align="edge", height=pdf, width=binwidths,
                color=cmyk_blue, label="Samples")
        ax.set(xlabel=r"$c$", ylabel=r"$p_c(c)$", yscale="log")

        # Add a point for zero samples, compare to analytical
        ax.plot(0.0, np.sum((conc_samples[i] == 0.0)) / conc_samples[i].size,
                color=cmyk_blue, marker="o", ls="none", mec="k", ms=8)

        bin_axis = np.linspace(binseps[0], binseps[-1], 201)
        conc0, alpha = c0s[i], alphas[i]
        dens = chis[i]*truncexp1_density(bin_axis, conc0, alpha)
        ax.plot(bin_axis, dens, color=cmyk_red, lw=3.,
            label=r"$p_c(c) \sim \frac{e^{-c/c_0}}{Ac}$")
        ax.plot(0.0, 1.0 - chis[i], marker="*", color=cmyk_red, ls="none",
                mec=cmyk_red, ms=4.5)
        ax.legend()
    fig.tight_layout()
    return fig, axes


# Background process: save all elements of histograms
# of concentration during whiffs, t_blank, t_whiff. Use all i.i.d. odors
def check_turbulent_background_stats(tcser, back_rates):
    # Extract parameters for analytical predictions
    for i in range(len(back_rates)):
        if not np.all(back_rates[i] == back_rates[i][0]):
            raise ValueError("Odors not i.i.d., this functions assumes so")
    twlo, twhi, tblo, tbhi, c0, alpha = [a[0] for a in back_rates]
    chi = 1.0 / (1.0 + np.sqrt(tblo*tbhi/twlo/twhi))

    stats = {}

    # First, check concentrations
    conc_ser = tcser[:, :, 1]
    nonzero_samples = conc_ser[conc_ser > 0.0].flatten()
    conc_counts, conc_bins = np.histogram(nonzero_samples, bins="doane")
    stats["conc_bins"] = conc_bins
    conc_binwidths = np.diff(conc_bins)  # This can be recomputed easily
    # Center of bins on a log scale, but given in linear coordinates
    conc_bin_centers = (conc_bins[1:] + conc_bins[:-1])/2  # easily recomputed
    stats["conc_pdf"] = conc_counts / conc_binwidths / conc_ser.size
    stats["conc_prob_zero"] = 1.0 - nonzero_samples.size / conc_ser.size

    # Analytical distribution  of concentrations
    conc_axis = np.linspace(stats["conc_bins"][0], stats["conc_bins"][-1], 201)  # easily recomputed
    stats["conc_analytical_pdf"] = chi*truncexp1_density(conc_axis, c0, alpha)
    stats["conc_analytical_prob_zero"] = 1.0 - chi

    # Now, analyze whiff and blank durations
    # Whiffs start when concentration goes up; blanks, when concentration goes down
    conc_diffs = np.diff(conc_ser, prepend=conc_ser[0:1], axis=0)  # Prepend so the first element is 0
    whiff_starts = np.nonzero(conc_diffs > 0.0)
    blank_starts = np.nonzero(conc_diffs < 0.0)
    tstarts = {"w":whiff_starts, "b":blank_starts}

    for i in tstarts:  # loop over whiffs and blanks
        if i == "w":
            thresh_min, thresh_max = twlo, twhi
        else:
            thresh_min, thresh_max = tblo, tbhi
        t_samples = tcser[tstarts[i][0], tstarts[i][1], 0].flatten()  # durations
        # Choose log-spaced bins
        t_counts, t_bins = np.histogram(np.log10(t_samples), bins="doane")
        # Center of bins on a log scale, but given in linear coordinates
        bin_centers_forlog = 10**((t_bins[1:] + t_bins[:-1])/2)
        # For plotting, bin limits in linear scale; set the log scale on the plot
        t_bins = 10**t_bins
        stats["t_{}_bins".format(i)] = t_bins
        t_binwidths = np.diff(t_bins)
        stats["t_{}_pdf".format(i)] = t_counts / t_binwidths / t_samples.size
        # Analytical density with an upper cutoff.
        # Should be a straight line in log scale, no need for a more finely sampled axis.

        stats["t_{}_axis".format(i)] = bin_centers_forlog
        norm_factor = 2 * thresh_min * (1.0 - np.sqrt(thresh_min / thresh_max))
        stats["t_{}_analytical_pdf".format(i)] = (bin_centers_forlog/thresh_min)**(-3/2) / norm_factor

    return stats


def jacobian_row_wmat_l2_avgstats(cgammas, back_moments, w_rates):
    """ Jacobian of dW/dt, given M and statistics of the background.
    All rows of W have the same jacobian, the full ijxij jac is block-diagonal
    So only compute the jacobian for one row derived wrt the same row.
    This is to check stability of the W fixed point (irrespective of numerics)

    Args:
        cgammas (np.ndarray): matrix of dot products of L.M with each
            background component x_gamma, indexed [i, gamma]
        back_moments (list of floats of np.ndarrays): average nu, sigma^2
            of all components (if floats) or of each component gamma (arrays)
        w_rates (list of floats): alpha, beta
    Returns:
        jac (np.ndarray): jacobian matrix of one row of W by itself,
            which is one block of the Jacobian for the full W matrix
    """
    avgnu, sigma2s = back_moments
    alpha, beta = w_rates
    n_i = cgammas.shape[0]
    # Compute vector of c_ds: c^I_d
    c_d = avgnu * np.sum(cgammas, axis=1)
    c_dmat = c_d[:, None] * c_d[None, :]
    # Compute matrix of c_gamma products, summed over gamma
    c_gamma_mat = np.sum(sigma2s * cgammas[:, None, :]*cgammas[None, :, :], axis=2)
    #  Compute <c^j c^l> correlations.
    ccmat = c_dmat + c_gamma_mat
    # Compute jacobian
    jac = -beta * np.identity(n_i) - alpha * ccmat
    return jac


def stability_row_wmat_l2_instant(cvec, w_rates, dt):
    """ Compute the eigenvalues of the Euler integrator's iterative map for W,
    given an instantaneous \bar{c} value, the inhibitory neurons' activities.
    Imagining that the c's remain constant thereafter, is the Euler integrator
    on a stable tangent for this step? Check by computing the eigenvalues
    of the matrix multiplying W in this iterative map,
    W^{t+1} = W^t + dt * alpha*x.c^T - dt*(alpha*c.c^T + beta*identity).W
    so the matrix of which to check eigenvalues and ensure they are < 1 is
        A = 1 - dt*(alpha*c.c^T + beta*identity)
    since the term dt * alpha*x.c^T doesn't contain W and thus doesn't enter
    the linear stability analysis. 

    Args:
        cvec (np.ndarray): vector of inhibitory neurons' activities,
            i.e. LMx
        w_rates (list of floats): alpha, beta
        dt (float): time step

    Returns:
        eigvals (np.ndarray): eigenvalues of the linear update operator,
            that is, 1 + dt*jacobian.
    """
    # Symmetric matrix, can use eigvalsh
    alpha, beta = w_rates
    n_i = cvec.shape[0]
    #up_mat = np.identity(n_i)*(1.0 - dt*beta) - dt*alpha*np.outer(cvec, cvec)
    #eigvals = np.linalg.eigvalsh(up_mat)
    # Obviously, with the outer product here, one eigenvalue is
    # 1-dt*(beta+alpha*cvec**2), the others are 1-beta*dt
    # So no need to compute the whole matrix and diagonalize.
    eigvals = np.full(n_i, fill_value=1.0 - dt*beta)
    eigvals[0] = 1.0 - dt*beta - dt*alpha*l2_norm(cvec)**2
    return eigvals


def compute_max_lambda_w_stability(hvec_norm_lambda0, w_rates, dt):
    """ Compute the maximum scaling factor Lambda multiplying the
    average (or RMS, etc.) default cvec norm when Lambda is equal to its
    default value, Lambda_0, at which the numerical integrator
    becomes unstable. 

    Args:
        hvec_norm_lambda0 (np.ndarray): typical norm of the LN activity vector, 
            (either average or RMS or max) when Lambda is equal to its default. 
        w_rates (list of floats): alpha, beta
        dt (float): time step
    
    Returns:
        lambda_limit (float): Lambda factor multiplying hvec_norm_lambda0
        at which the integrator starts being unstable. 
        (so this is really the max. Lambda/Lambda_0 factor). 
    """
    alpha, beta = w_rates
    # Largest eigenvalue is 1.0 - dt*beta - dt*alpha*l2_norm(cvec)**2
    # Set it to magnitude 1 (limit of instability), invert for cvec_norm2
    hvec_norm2_limit = (2.0 - dt*beta) / (dt * alpha)
    # Compare this limit cvec2 norm to cvec_lambda0, extract the scale
    lambda_limit = np.sqrt(hvec_norm2_limit) / hvec_norm_lambda0
    return lambda_limit


### Functions to analyze biologically plausible online PCA results
# Compute the real PCA (without subtracting the mean)
# and compare it to the algorithm results.
def compute_pca_meankept(samp, do_proj=False, vari_thresh=1.0, force_svd=False, demean=False):
    """ Given an array of samples, compute the empirical covariance and
    diagonalize it to obtain the principal components and principal values,
    which are what is returned.

    If less than 10*d samples, take SVD of the sample matrix directly
    divided by 1/sqrt(N-1), because this amounts to eigendecomposition of
    the covariance matrix, but with better numerical stability and accuracy
    (but it's a lot slower).

    Args:
        samp (np.array): nxp matrix for n samples of p dimensions each.
            Pass the values of a dataframe for proper slicing.
        do_proj (bool): if True, also project the sample points
        vari_thresh (float in [0., 1.]): include principal components until
            a fraction vari_thresh of the total variance is explained.
        force_svd (bool): if True, use SVD of the data matrix directly.
        demean (bool): if True, subtract the mean before computing SVD,
            i.e., compute true covariance PCA.
    Returns:
        p_values (np.ndarray): 1d array of principal values, descending order.
        p_components (np.ndarray): 2d array of principal components.
            p_components[:, i] is the vector for p_values[i]
        samp_proj (np.ndarray): of shape (samp.shape[0], n_comp) where n_comp
            is the number of principal components needed to explain
            vari_thresh of the total variance.
    """
    if demean:
        samp_loc = samp - np.mean(samp, axis=0)
    else:
        samp_loc = samp
    # Few samples: use SVD on the de-meaned data directly.
    if force_svd or samp_loc.shape[0] <= 10*samp_loc.shape[1]:
        svd_res = np.linalg.svd(samp_loc.T / np.sqrt(samp_loc.shape[0] - 1))
        # U, Sigma, V. Better use transpose so small first dimension,
        # because higher accuracy in eigenvectors in U
        # Each column of U is an eigenvector of samp^T*samp/(N-1)
        p_components = svd_res[0]
        p_values = svd_res[1]**2  # Singular values are sqrt of eigenvalues

    # Many samples are available; use covariance then eigen decomposition
    else:
        covmat = np.dot(samp_loc.T, samp_loc) / (samp_loc.shape[0] - 1)
        p_values, p_components = np.linalg.eigh(covmat)
        # Sort in decreasing order; eigh returns increasing order
        p_components = p_components[:, ::-1]
        p_values = p_values[::-1]

    if do_proj:
        vari_explained = 0.
        total_variance = np.sum(p_values)
        n_comp = 0
        while vari_explained < total_variance*vari_thresh:
            vari_explained += p_values[n_comp]
            n_comp += 1
            if n_comp > p_values.shape[0]: break
        samp_proj = samp_loc.dot(p_components[:, :n_comp])

    else:
        samp_proj = None

    return p_values, p_components, samp_proj


def compute_projector_series(mser, lser):
    nt = mser.shape[0]
    # Note that lser here is really L' = L^{-1}
    nk = lser.shape[1]
    linvdiag = 1.0 / np.diagonal(lser, axis1=1, axis2=2)
    loffd = lser.copy()
    loffd[:, np.arange(nk), np.arange(nk)] = 0.0

    linvser = linvdiag[:, :, None] * (np.tile(np.eye(nk), (nt, 1, 1)) - loffd * linvdiag[:, None, :])
    # So we are simply computing LM at each time point here. 
    fser = np.einsum("...ij,...jk", linvser, mser)  # ... allows broadcasting
    return fser


def analyze_pca_learning(xser, mser, lser, lambda_diag, demean=False):
    # Exact PCA: eigenvalue decomposition of xx^T / (n_samples-1)
    nk = lser.shape[1]
    nt = xser.shape[0]
    eigvals, eigvecs, _ = compute_pca_meankept(xser, do_proj=False, demean=demean)

    # Determine basis learnt by algorithm and return
    fser = compute_projector_series(mser, lser)
    learntvecs = ((1.0/lambda_diag[None, :, None]) * fser)
    # Each row of learntvecs[t] is an eigenvector learnt at time t

    # Values on the diagonal of L' are supposed to be eigenvalues
    # Recall lser returned by PCA integration is really lprime_ser, L'=L^{-1} 
    # so we can directly take the diagonal of this lser to get eigenvalues
    learntvals = np.diagonal(lser, axis1=1, axis2=2)
    # Sort them in decreasing order
    sort_arg = np.argsort(np.mean(learntvals[nt//2:], axis=0))[::-1]
    learntvals = learntvals[:, sort_arg]

    # Off-diagonal values are supposed to tend to zero
    loffd = lser.copy()
    loffd[:, np.arange(nk), np.arange(nk)] = 0.0
    offd_avg_abs = np.mean(np.abs(loffd), axis=(1, 2))

    # Subspace alignment
    # Target is eigvects and should be the second arg of subspace_align_error
    error_series = np.asarray([subspace_align_error(learntvecs[i].T, eigvecs[:, :nk]) 
                               for i in range(nt)])

    return [eigvals, eigvecs], [learntvals, learntvecs], fser, offd_avg_abs, error_series
