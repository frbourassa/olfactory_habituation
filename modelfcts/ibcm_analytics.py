""" Module with various analytical solution functions of the IBCM model
that were previously in Jupyter notebooks.

@author: frbourassa
"""
import numpy as np
from utils.metrics import l2_norm, powerset
from modelfcts.ideal import relu_inplace

### Special solution for toy background with 2 odors, 1 varying proportion ###
def fixedpoints_barm_2vectors(components, sigma, eta, lambd=1.0, n_r=2):
    """ Analytical fixed points for IBCM neurons in terms of the reduced m variables.
    These are simpler because they have the same possible values for all neurons individually,
    no matter whether they are at the same or different fixed points. So really, there are only
    two possible vectors, but we repeat them in different combinations here. """
    ss_mat = np.zeros([2*2, 2, n_r])
    norm_a = np.sum(components[0]**2)
    norm_b = np.sum(components[1]**2)
    overlap = components[0].dot(components[1])

    # Fixed points where both neurons' synaptic weight vectors are equal
    cplus = (1 + 0.5/sigma) * lambd
    cminus = (1 - 0.5/sigma) * lambd

    # (+, +)
    ss_mat[0, 0] = (cplus*norm_b - overlap*cminus)/(norm_a*norm_b - overlap**2) * components[0]
    ss_mat[0, 0] += (cminus*norm_a - overlap*cplus)/(norm_a*norm_b - overlap**2) * components[1]
    ss_mat[0, 1] = ss_mat[0, 0]

    # (-, -)
    ss_mat[1, 0] = (cminus*norm_b - overlap*cplus)/(norm_a*norm_b - overlap**2) * components[0]
    ss_mat[1, 0] += (cplus*norm_a - overlap*cminus)/(norm_a*norm_b - overlap**2) * components[1]
    ss_mat[1, 1] = ss_mat[1, 0]

    # Fixed points where the two neurons are at opposite fixed points
    cplus = (1 + 1 / (2*sigma)) * lambd
    cminus = (1 - 1 / (2*sigma)) * lambd

    # (+, -)
    ss_mat[2, 0] = (cplus*norm_b - overlap*cminus)/(norm_a*norm_b - overlap**2) * components[0]
    ss_mat[2, 0] += (cminus*norm_a - overlap*cplus)/(norm_a*norm_b - overlap**2) * components[1]
    ss_mat[2, 1] = (cminus*norm_b - overlap*cplus)/(norm_a*norm_b - overlap**2) * components[0]
    ss_mat[2, 1] += (cplus*norm_a - overlap*cminus)/(norm_a*norm_b - overlap**2) * components[1]

    # (-, +)
    ss_mat[3, 0] = ss_mat[2, 1]
    ss_mat[3, 1] = ss_mat[2, 0]

    fixed_pts_labels = ["(+, +)", "(-, -)", "(+, -)", "(-, +)"]

    return ss_mat, fixed_pts_labels


# Fixed point for w vector
def fixedpoints_w_2vectors(rates, fixed_mbar, bk_components, sigma2, lambd=1.0):
    """
    Compute the steady-state inhibitory weights of the two IBCM neurons with
    input synaptic weights specified in fixed_mbar.
    Args:
        inhib_rates: [alpha, beta]
        fixed_mbar: array of average synaptic weight vectors of each individual neuron
            at the fixed point, shape (n_neurons, n_dimensions)
        bk_components: array of background components x_a, x_b.
            shape (n_components=2, n_dimensions)
        sigma2 (float):  variance of nu
        lambd (float): Lambda scale parameter.
    """
    # First, check at which fixed point each neuron is.
    # The sign of m.dot(x_s) tells that
    alph, bet = rates
    x_d = 0.5 * (bk_components[0] + bk_components[1])
    x_s = bk_components[0] - bk_components[1]
    signs = np.sign(fixed_mbar.dot(x_s)).astype(int)  # Gives \pm 1/sigma
    if not np.sum(signs) == 0:
        raise ValueError("Analytical calculation not applicable because "
                         +"both neurons are at the same fixed point")
    factor = alph*lambd / (2*alph*lambd**2 + bet)
    fixed_wvecs = factor * (x_d + signs.reshape(-1, 1) * np.sqrt(sigma2) * x_s)
    return fixed_wvecs


def fixedpoint_s_2vectors_instant(rates, x_instant, lambd=1.0, options={}):
    activ_fct = str(options.get("activ_fct", "ReLU")).lower()
    alph, bet = rates
    if activ_fct == "relu":
        return relu_inplace(bet / (2*alph*lambd**2 + bet) * x_instant)
    else:
        return bet / (2*alph*lambd**2 + bet) * x_instant


def analytical_convergence_times_2d(init_c_ds, norms2_x_ds, mu, sigm2, alph=0.9, lambd=1.0):
    """ Predict times for c_d and c_s to converge to fixed points.
    Valid for small sigma^2, when we expect c_d to converge before c_s
    If sigma^2 or the initial value of c_s are too large, the prediction
    for td is still good probably, but for ts it will be bad,
    because we assumed that c_d reaches its steady-state value of 1
    to compute the time ts.
    Args:
        init_c_ds (list of 2 floats): initial value of m.x_d and m.x_s
        norms2_x_ds (list of 2 floats): squared norm of x_d and x_s
        mu (float): learning rate
        sigm2 (float): variance of nu
        lambd (float): Lambda scale factor. The convergence time should be the
            same for any Lambda (mere change of variable rescaling m;
            same dynamics). We assume that init_c_ds contains the Lambda
            scale, we just need to cancel it to reduce to the Lambda=1 case.

    Returns:
        td (float): time for c_d to reach steady-state
        ts (float): time for c_s to reach steady-state,
            assuming c_d reached steady-state much faster.
    """
    td = (lambd/init_c_ds[0] - 1.0)
    td += np.log(alph*(lambd - init_c_ds[0]) / (1.0 - alph) / init_c_ds[0])
    td /= mu * norms2_x_ds[0]
    # Keeping c_s = epsilon_s when solving for t_d
    #k = sigm2 * init_c_ds[1]**2
    #td = np.log((1.0 - init_c_ds[0])/(1-alph))
    #td += (np.arctan(alph/k) - np.arctan(init_c_ds[0]/k)) / k
    #td += 0.5*np.log((alph**2 + k**2)/(init_c_ds[0]**2 + k**2))
    #td /= (1 + k**2) * mu * norms2_x_ds[0]

    # Time to converge to 90 %
    #sig = np.sqrt(sigm2)
    #td = np.log(alph*np.sqrt(1.0 - sigm2*init_m_sd[1]**2)/(sig*init_m_sd[1]*np.sqrt(1-alph**2)))
    ts = np.log(alph * lambd / np.sqrt(sigm2) / init_c_ds[1])
    ts = ts / (mu * norms2_x_ds[1]*sigm2) + td
    return td, ts


# To analyze a simulation of 1 IBCM neuron and find convergence time
def find_convergence_time(tpts, mdd, mds, sigm2, alph=0.9, lambd=1.0):
    r"""
    Args:
        tpts (np.ndarray): time points
        mdd (np.ndarray): time series of m \cdot \vec{x}_d
        mds (np.ndarray): time series of m \cdot \vec{x}_s
        sigm2 (float): variance of nu
    """
    # Check when mds reaches close to lambda (analytical ss value)
    # and when mdd reaches close to \pm 1 / sigma
    td = tpts[np.argmax(mdd > alph*lambd)]
    ts = tpts[np.argmax(np.abs(mds) > alph*lambd / np.sqrt(sigm2))]
    return td, ts


### Fixed points of the IBCM model for a linear combination ###
### of odors with non-zero third moment                     ###
def fixedpoint_thirdmoment_onecval(avgnu, variance, epsilon, nb, m3=1.0, lambd=1.0):
    # Reject the case c_gamma = 0, that is the unstable origin.
    # Linear equation, easily solved
    numerator = nb**2 * avgnu**3 + 3*variance*nb*avgnu + epsilon*m3
    denominator = nb**3 * avgnu**4 + 2*variance*nb**2*avgnu**2 + variance**2*nb
    # Add the Lambda scale at the end
    y = numerator / denominator * lambd
    cd = y*nb*avgnu
    u2 = nb*y**2
    return y, y, cd, u2


# Exact fixed points
def fixedpoint_thirdmoment_exact(moments_nu, k1, k2, verbose=False, lambd=1.0):
    """ Exact equation for the fixed point with k1 dot products c_gammas
    equal to the larger value y1, and k2 equal to the remaining value
    Args:
        moments_nu (list of 3 floats): <nu>, sigma^2, m_3
        k1 (int): number of dot products equal to the larger value
        k2 (int): number of dot products equal to the lesser value
        verbose (bool): if True, print extra information
        lambd (float): Lambda scale parameter.
    Returns:
        y1, y2
        cd, u2
    """
    avgnu, sigma2, m3 = moments_nu
    # All c_gammas equal. Probably unstable, can compute anyways.
    if k1 == 0 or k2 == 0:
        return fixedpoint_thirdmoment_onecval(avgnu, sigma2, m3, max(k1, k2), m3=1.0, lambd=lambd)

    # Compute the quadratic factor alpha that relates y1 and y2
    a1 = (sigma2 - avgnu**2 * k1 - m3*avgnu/sigma2) * k1
    a2 = (sigma2 - avgnu**2 * k2 - m3*avgnu/sigma2) * k2
    b = 2*avgnu**2*k1*k2 + m3*avgnu/sigma2*(k1 + k2) + m3/avgnu
    alpha_minus = (b - np.sqrt(b**2 - 4.0*a1*a2)) / (2.0 * a2)
    # This is either the - root if y1 > 0 or the + root if y1 < 0
    alpha_plus = (b + np.sqrt(b**2 - 4.0*a1*a2)) / (2.0 * a2)
    # This is either the + root if y1 > 0 or the - root if y1 < 0
    if verbose:
        print("a1 =", a1)
        print("a2 =", a2)
        print("b =", b)
        print("alpha minus =", alpha_minus)
        print("alpha plus =", alpha_plus)

    # Compute both roots to be sure, take the one with y1 > 0
    both_roots = []
    ok_roots = []
    for alpha in [alpha_minus, alpha_plus]:
        # Compute y1
        y1_numer = 2*avgnu * (k1 + alpha*k2) + m3/sigma2 * (1.0 + alpha)
        y1_denom = avgnu**2 * (k1 + alpha*k2)**2 + sigma2 * (k1 + alpha**2 * k2)
        y1 = y1_numer / y1_denom
        # In case we find y1 < 0, we are in fact calculating the + root,
        # because alpha has in reality b - sgn(y1)*sqrt(b^2 - 4a1a2).
        # We should replace for the alpha plus root in this case to keep y1 > y2.
        y2 = alpha*y1
        both_roots.append((y1, y2))
        ok_roots.append(int(y1 >= y2))

    # Select the right root
    if verbose:
        print(both_roots)
    assert sum(ok_roots) == 1, "Both roots have y1 > y2 or the opposite"
    idx_root = ok_roots.index(True)
    y1, y2 = both_roots[idx_root]

    # Compute cd and u2, for reference
    cd = avgnu * (k1*y1 + k2*y2)
    u2 = k1 * y1**2 + k2 * y2**2

    # Check this is really a solution of the average fixed-point equation
    for val in [y1, y2]:
        rhs = (avgnu * (cd**2 + sigma2*u2)*(1 - cd)
               - sigma2*(cd**2 - 2*cd + sigma2*u2)*val
               + m3*val**2)
        assert abs(rhs/avgnu) < 1e-12, "Wrong root"

    # The other quadratic root (with a plus in alpha) gives y2 > y1, and is in
    # fact the fixed point one would find with k1 and k2 swapped.
    # Add Lambda back at the very end
    return y1*lambd, y2*lambd, cd*lambd, u2*lambd**2


# Exact W fixed points for general distribution of neurons across odors
def ibcm_fixedpoint_w_thirdmoment(inhib_rates, moments_nu, back_vecs, cs_cn, specif_gammas):
    r""" Exact analytical steady-state solution for the matrix W,
    given the specificity of each neuron, and the values of c_n and c_s.
    The latter can be either from the analytical M solution, or from
    numerical solution -- allowing to compare W simulations to prediction
    even when the M predition fails due to large correlations.

    This assumes each IBCM neuron is specific to one odor, which are the only
    fixed point that are stable, based on all simulations.

    Args:
        inhib_rates (list of 2 floats): alpha, beta
        moments_nu (list of 3 floats): <nu>, sigma^2, m_3, only the first 2
            are used.
        back_vecs (np.ndarray): background odor vectors, one per row,
            thus shaped n_B x n_R, indexed [odor, orn dimension].
        cs_cn (list of 2 floats): values of y_1 and y_2, the specific
            and non-specific dot product values, respectively.
            They should contain the scale Lambda.
        specif_gammas (np.ndarray of ints): vector giving \gamma_j of
            each neuron j, that is, the background odor to which each neuron
            is specific. 1d array of ints with length n_I.

    The Lambda scale parameter is not needed: assumed to be contained in cs_cn,
        it does not directly appear in the W equation otherwise.
    It does give that W elements scale as Lambda / (cst + Lambda^2),
        as in the toy model; MW tends to have scale 1 as Lambda increases.

    Returns:
        W (np.2darray): the W matrix, indexed [orn dimension, IBCM neuron],
            i.e. each column contains weights stemming from IBCM neuron j.
    """
    # Extract parameters
    alpha, beta = inhib_rates
    avgnu, sigma2, _ = moments_nu
    c_s, c_n = cs_cn
    cdiff = c_s - c_n
    n_B = back_vecs.shape[0]
    n_I = specif_gammas.shape[0]

    # 1. Compute the number of neurons specific to each odor, n_gammas
    specif_gammas = specif_gammas.astype(int)
    n_gammas = np.bincount(specif_gammas, minlength=n_B)

    # 2. Compute a few terms that appear often.
    x_d = avgnu * np.sum(back_vecs, axis=0)
    c_d = avgnu * (c_s + (n_B - 1)*c_n)
    cst_A = c_d**2 + sigma2*c_d*c_n / avgnu + sigma2*c_n*cdiff
    b_gammas = beta/alpha + n_gammas * sigma2 * cdiff**2
    cst_K = np.sum(n_gammas / b_gammas)
    denoms_bak = b_gammas * (1.0 + cst_A * cst_K)
    # The 1d vectors n_gammas/b_gammas is aligned with columns,
    # so transpose back_vecs to multiply one coef per x_gamma, then sum.
    weighted_x = np.sum(n_gammas / b_gammas * back_vecs.T, axis=1)

    # 3. Compute the w_{\gamma} vectors
    # that is, the column in W for a neuron specific to odor \gamma
    # Shape w_gammas: [n_orn, n_B], each column contains one w_{\gamma} vector
    w_gammas = sigma2*cdiff / b_gammas * back_vecs.T
    w_gammas += (c_d + sigma2/avgnu*c_n) / denoms_bak * x_d[:, None]
    w_gammas -= cst_A * sigma2 * cdiff / denoms_bak * weighted_x[:, None]

    # 4. Assemble the w_{gamma} columns according to the neurons' specificity
    wmat = np.asarray(w_gammas[:, specif_gammas])
    return wmat


# Fixed point of a single neuron
def jacobian_fixedpoint_thirdmoment(
    moments, ibcm_params, which_specif, back_comps, m3=1.0, options={}):
    """ which_specif: boolean array equal to True for specific gammas. """
    ## 1. Evaluate x, y, cd, u^2 at the fixed point
    # From the list of c_gammas which are specific, count k1 and k2
    assert which_specif.size == back_comps.shape[0]
    which_specif = which_specif.astype(bool)
    k1 = np.sum(which_specif.astype(bool))
    k2 = which_specif.size - k1

    avgnu, variance, epsilon = moments
    # Neglect non-linear saturation of neurons and decay, but not ktheta
    mu, tau_theta, eta, lambd, sat, ktheta, decay_relative = ibcm_params
    mu_abs = mu / lambd
    c_sp, c_nsp, cd, u2 = fixedpoint_thirdmoment_exact([avgnu, variance,
                            epsilon*m3], k1, k2, verbose=False, lambd=lambd)
    x_d = avgnu * np.sum(back_comps, axis=0)
    cgammas_vec = np.where(which_specif, c_sp, c_nsp)
    # 2. Evaluate the jacobian blocks
    n_dims = x_d.shape[0]
    n_comp = k1 + k2
    jac = np.zeros([n_dims + 1, n_dims + 1])
    # Scalar element
    jac[-1, -1] = -1.0 / tau_theta
    # Vector blocks
    avg_cx = cd * x_d + variance * cgammas_vec.dot(back_comps)
    # Last column: derivative of theta with respect to m
    jac[:n_dims, -1] = 2.0 / tau_theta * avg_cx / lambd
    # Matrix block
    theta_ss = (cd**2 + variance * u2) / lambd
    x_gammas_outer = back_comps[:, :, None] * back_comps[:, None, :]
    xd_outer = np.outer(x_d, x_d)
    xd_xgammas_outer = x_d[None, :, None] * back_comps[:, None, :]
    avg_xx = xd_outer + variance*np.sum(x_gammas_outer, axis=0)
    avg_cxx = (cd * avg_xx
                + variance*np.sum(cgammas_vec[:, None, None]*xd_xgammas_outer, axis=0)
                + variance*np.sum(cgammas_vec[:, None, None]*xd_xgammas_outer, axis=0).T
                + epsilon*m3*np.sum(cgammas_vec[:, None, None]*x_gammas_outer, axis=0)
            )
    variant = str(options.get("variant", "intrator"))
    if variant == "intrator":
        jac[:-1, :-1] = mu_abs * (2*avg_cxx - theta_ss*avg_xx)
        # Last row: derivative of m with respect to theta
        jac[-1, :n_dims] = -mu_abs * avg_cx
    elif variant == "law":
        # Changing the learning rate of mu
        jac[:-1, :-1] = mu_abs/(ktheta+theta_ss/lambd) * (2*avg_cxx - theta_ss*avg_xx)
        # Last row: derivative of m with respect to theta
        jac[-1, :n_dims] = -mu_abs/(ktheta+theta_ss/lambd) * avg_cx
        # Extra term in the derivative of mu equations w.r.t. theta
        # is zero because <c(c-theta)x> = 0 at the fixed point!
    else:
        raise ValueError("Variant option" + variant + "unknown")

    # Add effect of the Law IBCM modification? Because this changes stability
    # when learning rate would be pushed too far for the regular IBCM model

    # Build the complete matrix
    return jac

def ibcm_all_largest_eigenvalues(
        moments, ibcm_params, back_comps, m3=1.0, cut=1e-16, options={}
    ):
    """ For one IBCM neuron, compute the non-zero eigenvalue
    with the largest real part for each possible fixed point,
    as defined by possible specificity to each odor.

    Args:
        moments (list): average, variance, third central moment of
            background odor concentrations, assumed i.i.d.
        ibcm_params (list): the pure IBCM rates, mu, tau_theta, eta
    """
    n_components = back_comps.shape[0]
    all_largest_eigenvalues = {}
    for specif in powerset(range(n_components)):
        specif_vec = np.zeros(n_components, dtype=bool)
        if len(specif) > 0:
            specif_vec[list(specif)] = True
        jacob = jacobian_fixedpoint_thirdmoment(
                    moments, ibcm_params, specif_vec, back_comps, m3=m3,
                    options=options
                )
        eigvals = np.linalg.eigvals(jacob)
        # Keep non-zero eigvals only; the zero ones reflect the fact
        # that dynamics lie in the background subspace.
        eigvals = eigvals[np.absolute(eigvals) > cut]
        max_idx = np.argmax(np.real(eigvals))
        all_largest_eigenvalues[specif] = eigvals[max_idx]
    return all_largest_eigenvalues


def ibcm_saddle_eigenvalues(
        moments, ibcm_params, back_comps, m3=1.0, cut=1e-16, options={}
    ):
    """ For one IBCM neuron, compute the non-zero eigenvalue
    with the largest real part for each possible fixed point,
    as defined by possible specificity to each odor.

    Args:
        moments (list): average, variance, third central moment of
            background odor concentrations, assumed i.i.d.
        ibcm_params (list): the pure IBCM rates, mu, tau_theta, eta
    """
    n_components = back_comps.shape[0]
    specif_vec = np.zeros(n_components, dtype=bool)
    jacob = jacobian_fixedpoint_thirdmoment(
                moments, ibcm_params, specif_vec, back_comps, m3=m3,
                options=options
            )
    eigvals = np.linalg.eigvals(jacob)
    # Keep non-zero eigvals only; the zero ones reflect the fact
    # that dynamics lie in the background subspace.
    eigvals = eigvals[np.absolute(eigvals) > cut]
    return eigvals


def lambda_pca_equivalent(h_dots, moments_conc, n_b, w_alpha_beta, verbose=False):
    """ Compute the Lambda scaling factor for the BioPCA model
    so its magnitude of background reduction is equivalent to
    that of the IBCM models. We impose equal prefactors rather
    than perfectly equal PN squared activities, <y^2>, since
    they are hard to compare anyways since PCA is helped
    by a separate average subtraction circuit. 
    
    Args:
        h_dots: [h_s, h_]: dot products of IBCM m with specific and
            non-specific components, respectively
        moments_conc (list): [mean, variance] concentration moments
        n_b (int): number of background components
        w_alpha_beta (list): [alpha, beta] W rates. 
        verbose (bool): if True, print details of the calculation
    """
    # Compute IBCM reduction factor first
    b_ov_a = w_alpha_beta[1] / w_alpha_beta[0]
    hs, hn = h_dots
    hd = moments_conc[0] * (hs + (n_b-1)*hn)
    b_ov_a = w_alpha_beta[1] / w_alpha_beta[0]
    ibcm_a = hd**2 + moments_conc[1]/moments_conc[0]*hd*hn + moments_conc[1]*hn*(hs - hn)
    ibcm_b = b_ov_a + moments_conc[1]*(hs - hn)**2
    ibcm_b_over_a = ibcm_b / (ibcm_b + n_b*ibcm_a)
    ibcm_fact = ibcm_b_over_a * b_ov_a / (b_ov_a + moments_conc[1]*(hs - hn)**2)
    lambda_pca = np.sqrt(b_ov_a * (1 - ibcm_fact) / ibcm_fact / moments_conc[1])
    if verbose:
        print("***Calculation details for the PCA Lambda scaling factor***")
        print("IBCM: hs and hn = ", hs, hn)
        print("IBCM: hd = <c> * (hs - (n_b-1) h_n) =", hd)
        print("IBCM: A = ", ibcm_a)
        print("IBCM: B = ", ibcm_b)
        print("IBCM beta / (beta + alpha*sigma^2*(hs-hn)^2) factor:", 
              b_ov_a / (b_ov_a + moments_conc[1]*(hs - hn)**2))
        print("IBCM extra factor: B / (B + N_B*A) =", ibcm_b_over_a)
        print("IBCM combined factor: f=", ibcm_fact)
        print("Lambda_PCA = sqrt(beta/alpha / sigma^2 * (1-f)/f) =", lambda_pca)
        print("*** ***")
    return lambda_pca
    

# Test the function computing the W analytical prediction
if __name__ == "__main__":
    back_vecs = np.zeros([6, 25])
    back_vecs[:6, :6] = np.eye(6)
    inhib_rates = [1.0, 0.2]
    moments_nu = (0.3, 0.09, 0.1)
    cs_cn = (5.0, -1.0)
    specif_gammas = np.asarray((0, 1, 2, 3, 4, 5, 0, 1))
    print(ibcm_fixedpoint_w_thirdmoment(
            inhib_rates, moments_nu, back_vecs, cs_cn, specif_gammas)
    )
