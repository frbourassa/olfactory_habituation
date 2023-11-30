""" Module with various analytical solution functions of the IBCM model
that were previously in Jupyter notebooks.

@author: frbourassa
"""
import numpy as np
from utils.metrics import l2_norm
from modelfcts.ideal import relu_inplace

### Special solution for toy background with 2 odors, 1 varying proportion ###
def fixedpoints_m_2vectors(components, sigma, eta, n_r=2):
    """ components: x_a and x_b, components of the fluctuating background mixture
    sigma: standard deviation of nu (the composition of the mixture)
    eta: coupling coefficient < 1

    Returns a 3D array where each element on axis 0 is a fixed point,
        the axis 1 indexes the neurons, and axis 2, the components.
        So the return array is a stack of matrices where each row is a neuron.

    Also returns a list of labels giving the order of fixed points:
        (+, +), (-, -), (+, -), (-, +)
    """
    ss_mat = np.zeros([2*2, 2, n_r])
    norm_a = np.sum(components[0]**2)
    norm_b = np.sum(components[1]**2)
    overlap = components[0].dot(components[1])

    # Fixed points where both neurons' synaptic weight vectors are equal
    cplus = (1 + 0.5/sigma) / (1 - eta)
    cminus = (1 - 0.5/sigma) / (1 - eta)

    # (+, +)
    ss_mat[0, 0] = (cplus*norm_b - overlap*cminus)/(norm_a*norm_b - overlap**2) * components[0]
    ss_mat[0, 0] += (cminus*norm_a - overlap*cplus)/(norm_a*norm_b - overlap**2) * components[1]
    ss_mat[0, 1] = ss_mat[0, 0]

    # (-, -)
    ss_mat[1, 0] = (cminus*norm_b - overlap*cplus)/(norm_a*norm_b - overlap**2) * components[0]
    ss_mat[1, 0] += (cplus*norm_a - overlap*cminus)/(norm_a*norm_b - overlap**2) * components[1]
    ss_mat[1, 1] = ss_mat[1, 0]

    # Fixed points where the two neurons are at opposite fixed points
    cplus = 1 / (1 - eta) + 1 / (2*sigma*(1 + eta))
    cminus = 1 / (1 - eta) - 1 / (2*sigma*(1 + eta))

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


def fixedpoints_barm_2vectors(components, sigma, eta, n_r=2):
    """ Analytical fixed points for IBCM neurons in terms of the reduced m variables.
    These are simpler because they have the same possible values for all neurons individually,
    no matter whether they are at the same or different fixed points. So really, there are only
    two possible vectors, but we repeat them in different combinations here. """
    ss_mat = np.zeros([2*2, 2, n_r])
    norm_a = np.sum(components[0]**2)
    norm_b = np.sum(components[1]**2)
    overlap = components[0].dot(components[1])

    # Fixed points where both neurons' synaptic weight vectors are equal
    cplus = (1 + 0.5/sigma)
    cminus = (1 - 0.5/sigma)

    # (+, +)
    ss_mat[0, 0] = (cplus*norm_b - overlap*cminus)/(norm_a*norm_b - overlap**2) * components[0]
    ss_mat[0, 0] += (cminus*norm_a - overlap*cplus)/(norm_a*norm_b - overlap**2) * components[1]
    ss_mat[0, 1] = ss_mat[0, 0]

    # (-, -)
    ss_mat[1, 0] = (cminus*norm_b - overlap*cplus)/(norm_a*norm_b - overlap**2) * components[0]
    ss_mat[1, 0] += (cplus*norm_a - overlap*cminus)/(norm_a*norm_b - overlap**2) * components[1]
    ss_mat[1, 1] = ss_mat[1, 0]

    # Fixed points where the two neurons are at opposite fixed points
    cplus = 1 + 1 / (2*sigma)
    cminus = 1 - 1 / (2*sigma)

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
def fixedpoints_w_2vectors(rates, fixed_mbar, bk_components, sigma2):
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

    fixed_wvecs = alph / (2*alph + bet) * (x_d + signs.reshape(-1, 1) * np.sqrt(sigma2) * x_s)
    return fixed_wvecs


def fixedpoint_s_2vectors_instant(rates, x_instant, options={}):
    activ_fct = str(options.get("activ_fct", "ReLU")).lower()
    alph, bet = rates
    if activ_fct == "relu":
        return relu_inplace(bet / (2*alph + bet) * x_instant)
    else:
        return bet / (2*alph + bet) * x_instant


def fixedpoint_s_2vectors_mean(rates, bk_components, options={}):
    activ_fct = str(options.get("activ_fct", "ReLU")).lower()
    x_d = 0.5 * (bk_components[0] + bk_components[1])
    alph, bet = rates
    if activ_fct == "relu":
        return relu_inplace(bet / (2*alph + bet) * x_d)
    else:
        return bet / (2*alph + bet) * x_d


def fixedpoint_s_2vectors_norm2(rates, bk_components, sigm2):
    x_d = 0.5 * (bk_components[0] + bk_components[1])
    x_s = bk_components[0] - bk_components[1]
    alph, bet = rates
    return (bet / (2*alph + bet))**2 * (l2_norm(x_d) + sigm2*l2_norm(x_s))


### Fixed points of the IBCM model for a linear combination ###
### of odors with non-zero third moment                     ###
def fixedpoint_thirdmoment_onecval(avgnu, variance, epsilon, nb, m3=1.0):
    # Reject the case c_gamma = 0, that is the unstable origin.
    # Linear equation, easily solved
    numerator = nb**2 * avgnu**3 + 3*variance*nb*avgnu + epsilon*m3
    denominator = nb**3 * avgnu**4 + 2*variance*nb**2*avgnu**2 + variance**2*nb
    y = numerator / denominator
    cd = y*nb*avgnu
    u2 = nb*y**2
    return y, y, cd, u2


def fixedpoint_thirdmoment_perturbtheory(avgnu, variance, epsilon, k1, k2, m3=1.0, order=1):
    """ Calculate the two possible values taken by the dot product of a neuron's \vec{\bar{m}}
    with each component, from a perturbation solution at first order in the magnitude
    of the third moment m_3 of the \nu_{\alpha}. It depends on $k_1$ and $k_2$, the number
    of components that have either of the two possible dot product values with \vec{\bar{m}}.

    We always take the plus sign in x (value appearing k_1 times); to have the minus sign,
    switch k_1 and k_2, and then y will be the minus sign solution.

    Valid for the special case where all nus have the same average and variance,
    and zero correlation.

    The solution proceeds in two times: first, we find first-order corrections
    to $u^2 = \sum_i c_i^2 = \frac{1}{variance} + \epsilon v$   and
    $c_d = avgnu*\sum_i c_i = 1 + \epsilon w$. Second, we recover x and y, the two
    possible values taken by each of the dot products c_i, from inverting
    $u^2 = k_1 x^2 + k_2 y^2$ and $c_d = avgnu*(k_1 x + k_2 y)$. There are two
    possibilities for x and y for a given u^2 and c_d:
        $x = c_d/(N*avgnu) \pm \frac{1}{N} \sqrt{k_2/k_1}\sqrt{Nu^2 - avgnu^2 c_d^2} $
        $y = c_d/(N*avgnu) \mp \frac{1}{N} \sqrt{k_1/k_2}\sqrt{Nu^2 - avgnu^2 c_d^2} $
    x and y take opposite signs (choosing sign in x expression forces opposite sign in y).

    I'm pretty sure the only stable fixed point is when one k_i = 1, and only
    one of the signs for x (don't know which yet), but I don't have analytical
    evidence for it, so I return both possible pairs of (x, y) for a given k_1, k_2.

    For N_K > 3, I will have to compute such pairs for all choices of k_1 and pick
    the solution(s) that match simulations.
    Eventually, I will be able to empirically decide on which k_i value and which pair
    of (x, y) sign gives stable fixed points (e.g. k_1=1 and plus sign
    - always taking x to be the unique value).


    Args:
        avgnu (float): average value of the nus.
        variance (float): variance of the nus.
        epsilon (float): small amplitude of the third centered moment.
        k1 (int): number of components with one dot product value.
        k2 (int): number of components with the other dot product.
            k1+k2 = N, the number of components.
        m3 (float): third moment amplitude; multiplies epsilon.
            Default: 1.0, epsilon alone is sufficient.
    Returns:
        x, y: the two possible values of dot products
        cd, u2: the sum of c_gammas and c_gamma^2s, respectively
    """
    # All c_gammas equal. Probably unstable, can compute anyways.
    if k1 == 0 or k2 == 0:
        return fixedpoint_thirdmoment_onecval(avgnu, variance, epsilon, max(k1, k2), m3=m3)

    nn = k1 + k2  # number of components

    # First, compute corrections to u^2 and c_d, choosing the plus sign in x
    radical = np.sqrt(nn/variance - 1/(avgnu*avgnu))
    if order == 1:
        sigma4v = m3*(
            2/avgnu/nn
            + ((k2/k1 - k1/k2)/variance + (1/k2 - 1/k1)/(avgnu*avgnu))
                / ((np.sqrt(k2/k1) + np.sqrt(k1/k2))*radical)
        )
        # Uncorrected x, appears in expression for w in terms of sigma4v
        x0 = 1/nn/avgnu + np.sqrt(k2/k1)*radical/nn
        w = (
            m3/2/nn/avgnu/k1 * (k2/variance - 1/(avgnu*avgnu))
            + (2*m3/nn/avgnu - sigma4v)*x0/2/avgnu
        )
        u2 = 1/variance + epsilon*sigma4v/variance/variance  # We had v * sigma^4
        cd = 1 + epsilon*w

        # Second, compute x and y, remembering we chose the plus sign in x
        # for the derivation. If you want the negative signfor a given k_1,
        # call the function with k_1 and k_2 switched; then y will be the x you want.
        radical2 = np.sqrt(nn*u2 - (cd/avgnu)**2)
        x = cd/nn/avgnu + np.sqrt(k2/k1)*radical2/nn
        y = cd/nn/avgnu - np.sqrt(k1/k2)*radical2/nn

    # Zeroth order x and y, uncorrected, if cd=1 and u2 = 1/variance still
    # Does not really make sense to stop at zeroth order, because then degeneracy is not lifted
    # and the neurons can take any value on the N-2 dimensional subspace of fixed points.
    elif order == 0:
        x = 1/nn/avgnu + np.sqrt(k2/k1)*radical/nn
        y = 1/nn/avgnu - np.sqrt(k1/k2)*radical/nn
        u2 = 1/variance
        cd = 1
        # They don't match simulations as well as first-order, fortunately

    # Second order or more?
    else:
        raise NotImplementedError("I never tried to calculate above first-order perturbation theory!")

    # I think the stable fixed points are for k1<k2, plus sign in x. That makes the
    # less frequent value x the largest, and the most frequent, y, small and < 0.
    # That looks like the numerical fixed points we get.
    return x, y, cd, u2


# Exact fixed points
def fixedpoint_thirdmoment_exact(moments_nu, k1, k2, verbose=False):
    """ Exact equation for the fixed point with k1 dot products c_gammas
    equal to the larger value y1, and k2 equal to the remaining value
    Args:
        moments_nu (list of 3 floats): <nu>, sigma^2, m_3
        k1 (int): number of dot products equal to the larger value
        k2 (int): number of dot products equal to the lesser value
        verbose (bool): if True, print extra information
    Returns:
        y1, y2
        cd, u2
    """
    # Compute the quadratic factor alpha that relates y1 and y2
    avgnu, sigma2, m3 = moments_nu
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
    return y1, y2, cd, u2


# Exact W fixed points for general distribution of neurons across odors
def ibcm_fixedpoint_w_thirdmoment(inhib_rates, moments_nu, back_vecs, cs_cn, specif_gammas):
    """ Exact analytical steady-state solution for the matrix W,
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
        specif_gammas (np.ndarray of ints): vector giving \gamma_j of
            each neuron j, that is, the background odor to which each neuron
            is specific. 1d array of ints with length n_I.

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
    n_gammas = np.bincount(specif_gammas)

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
def jacobian_fixedpoint_thirdmoment(moments, ibcm_params, which_specif, back_comps, m3=1.0, order=1):
    """ which_specif: boolean array equal to True for specific gammas. """
    ## 1. Evaluate x, y, cd, u^2 at the fixed point
    # From the list of c_gammas which are specific, count k1 and k2
    assert which_specif.size == back_comps.shape[0]
    which_specif = which_specif.astype(bool)
    k1 = np.sum(which_specif.astype(bool))
    k2 = which_specif.size - k1

    avgnu, variance, epsilon = moments
    mu, tau_theta, eta = ibcm_params
    c_sp, c_nsp, cd, u2 = fixedpoint_thirdmoment_exact([avgnu,
                        variance, epsilon*m3], k1, k2, verbose=False)
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
    # Last column
    jac[:n_dims, -1] = 2.0 / tau_theta * avg_cx
    # Last row
    jac[-1, :n_dims] = -mu * avg_cx
    # Matrix block
    theta_ss = cd**2 + variance * u2
    x_gammas_outer = back_comps[:, :, None] * back_comps[:, None, :]
    xd_outer = np.outer(x_d, x_d)
    xd_xgammas_outer = x_d[None, :, None] * back_comps[:, None, :]
    avg_xx = xd_outer + variance*np.sum(x_gammas_outer, axis=0)
    avg_cxx = (cd * avg_xx
                + variance*np.sum(cgammas_vec[:, None]*xd_xgammas_outer, axis=0)
                + variance*np.sum(cgammas_vec[:, None]*xd_xgammas_outer, axis=0).T
                + epsilon*m3*np.sum(cgammas_vec[:, None, None]*x_gammas_outer, axis=0)
            )
    jac[:-1, :-1] = mu * (2*avg_cxx - theta_ss*avg_xx)

    # Build the complete matrix
    return jac

# Test the function computing the W analytical prediction
if __name__ == "__main__":
    back_vecs = np.zeros([6, 25])
    back_vecs[:6, :6] = np.eye(6)
    inhib_rates = [1.0, 0.2]
    moments_nu = (0.3, 0.09, 0.1)
    cs_cn = (5.0, -1.0)
    specif_gammas = np.asarray((0, 1, 2, 3, 4, 5, 0, 1))
    print(ibcm_fixedpoint_w_thirdmoment(inhib_rates, moments_nu, back_vecs, cs_cn, specif_gammas))
