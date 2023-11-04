from sympy import (
    symbols,
    MatrixSymbol,
    Matrix,
    MatMul,
    zeros,
    det,
    Identity,
    solve,
    solveset,
    simplify,
    factor
)
from time import perf_counter

def main_compute_jacobian_eigenvalues(n_d, n_b):
    # Define parameters
    mu, sigma, avgnu, tau_theta = symbols("mu,sigma,avgnu,tau_theta")
    # Take as an example 3 odor components in 4d.
    xvecs = []
    cgammas = []
    for i in range(n_b):
        xvecs.append(MatrixSymbol("x_{}".format(i), n_d, 1))
        cgammas.append(symbols("c_{}".format(i)))
    xmat = Matrix(xvecs)
    cgammasvec = Matrix([[c] for c in cgammas])

    mmat = cgammasvec[0] * (MatMul(xvecs[0], xvecs[0].T)
                            + MatMul(xvecs[0], xvecs[0].T))
    for g in range(n_b):
        for p in range(n_b):
            if g == 0 and p == 0:
                continue  # Already included
            mmat = mmat + cgammasvec[g] * (MatMul(xvecs[g], xvecs[p].T)
                                            + MatMul(xvecs[p], xvecs[g].T))
    mmat = 2*mu*sigma*avgnu * mmat
    print(mmat)
    print(type(mmat))
    print("Shape:", mmat.shape)

    # eigvals does not work on abstract matrices
    #print(mmat.eigenvals())

    # Try to compute determinant or characteristic polynomial...
    lamda = symbols("lamda")
    determ = det(mmat - Identity(n_d)*lamda)
    print(determ)
    solution = solveset(determ, lamda)
    print(solution)

    return mmat, solution

def main_compute_jacobian_eigenvalues_elements(n_d, n_b):
    # Define individual elements of the matrix, as suggested in
    # https://stackoverflow.com/questions/37026935/speeding-up-computation-of-symbolic-determinant-in-sympy
    # Define parameters
    mu, sigma, avgnu, tau_theta = symbols("mu,sigma,avgnu,tau_theta")

    # Template matrix for determinant expression in n_d dimensions
    template_mat = Matrix(n_d, n_d, symbols('A:{0}:{0}'.format(n_d)))
    # TODO: make full jacobian matrix with n_d+1 dimensions (for theta)
    template_det = template_mat.det()

    # Take as an example 3 odor components in 4d.
    xvecs = Matrix(n_b, n_d, symbols("x:{}:{}".format(n_b, n_d)))  # Each row is a vector
    print(xvecs)
    cgammas = Matrix(n_b, 1, symbols("c:{}".format(n_b)))
    print(cgammas)
    mmat = cgammas[0, 0] * (MatMul(xvecs[0, :].T, xvecs[0, :])
                            + MatMul(xvecs[0, :].T, xvecs[0, :]))

    for g in range(n_b):
        for p in range(n_b):
            if g == 0 and p == 0:
                continue  # Already included
            mmat = mmat + cgammas[g] * (MatMul(xvecs[g, :].T, xvecs[p, :])
                                        + MatMul(xvecs[p, :].T, xvecs[g, :]))
    mmat = 2*mu*sigma*avgnu * mmat
    print(mmat)

    lamda = symbols("lamda")
    mmat_eq = mmat - Identity(n_d)*lamda
    determ = template_det.subs(zip(list(template_mat), list(mmat_eq)))
    #determ = factor(determ)
    #determ = simplify(determ)
    time1 = perf_counter()
    determ = determ.as_poly(lamda)
    print("\nDeterminant:\n", determ)
    print()
    # One root should be 0, check that
    time2 = perf_counter()
    print("Time to get determinant as polynomial:", time2 - time1, "s")
    print("One solution is zero:", simplify(determ.subs(lamda, 0)))
    time3 = perf_counter()
    print("Time to evaluate at 0:", time3 - time2, "s")

    # Solve for eigenvalues! Takes a minute, equation too large honestly.
    solution = solve(determ, lamda)
    time4 = perf_counter()
    print("Solution: eigenvalues = ", solution)
    print("Time to get eigenvalues:", time4 - time3)

    # Evaluate for sample values of parameters TODO
    #param_values = [(mu, 1e-3), (sigma, 0.09), (avgnu, 0.3), (tau_theta, 200.0)]
    #for sol in solutions:
    #    sol.subs()

    return mmat, solution


if __name__ == "__main__":
    #main_compute_jacobian_eigenvalues(4, 3)
    main_compute_jacobian_eigenvalues_elements(4, 3)
    result = """
    Solution: eigenvalues =  [
    0,

    -2*avgnu*mu*sigma*sqrt((x00**2 + 2*x00*x10 + 2*x00*x20 + x01**2 + 2*x01*x11 + 2*x01*x21 + x02**2 + 2*x02*x12 + 2*x02*x22 + x03**2 + 2*x03*x13 + 2*x03*x23 + x10**2 + 2*x10*x20 + x11**2 + 2*x11*x21 + x12**2 + 2*x12*x22 + x13**2 + 2*x13*x23 + x20**2 + x21**2 + x22**2 + x23**2)*(c0**2*x00**2 + c0**2*x01**2 + c0**2*x02**2 + c0**2*x03**2 + 2*c0*c1*x00*x10 + 2*c0*c1*x01*x11 + 2*c0*c1*x02*x12 + 2*c0*c1*x03*x13 + 2*c0*c2*x00*x20 + 2*c0*c2*x01*x21 + 2*c0*c2*x02*x22 + 2*c0*c2*x03*x23 + c1**2*x10**2 + c1**2*x11**2 + c1**2*x12**2 + c1**2*x13**2 + 2*c1*c2*x10*x20 + 2*c1*c2*x11*x21 + 2*c1*c2*x12*x22 + 2*c1*c2*x13*x23 + c2**2*x20**2 + c2**2*x21**2 + c2**2*x22**2 + c2**2*x23**2)) + 2*avgnu*mu*sigma*(c0*x00**2 + c0*x00*x10 + c0*x00*x20 + c0*x01**2 + c0*x01*x11 + c0*x01*x21 + c0*x02**2 + c0*x02*x12 + c0*x02*x22 + c0*x03**2 + c0*x03*x13 + c0*x03*x23 + c1*x00*x10 + c1*x01*x11 + c1*x02*x12 + c1*x03*x13 + c1*x10**2 + c1*x10*x20 + c1*x11**2 + c1*x11*x21 + c1*x12**2 + c1*x12*x22 + c1*x13**2 + c1*x13*x23 + c2*x00*x20 + c2*x01*x21 + c2*x02*x22 + c2*x03*x23 + c2*x10*x20 + c2*x11*x21 + c2*x12*x22 + c2*x13*x23 + c2*x20**2 + c2*x21**2 + c2*x22**2 + c2*x23**2),

    2*avgnu*mu*sigma*sqrt((x00**2 + 2*x00*x10 + 2*x00*x20 + x01**2 + 2*x01*x11 + 2*x01*x21 + x02**2 + 2*x02*x12 + 2*x02*x22 + x03**2 + 2*x03*x13 + 2*x03*x23 + x10**2 + 2*x10*x20 + x11**2 + 2*x11*x21 + x12**2 + 2*x12*x22 + x13**2 + 2*x13*x23 + x20**2 + x21**2 + x22**2 + x23**2)*(c0**2*x00**2 + c0**2*x01**2 + c0**2*x02**2 + c0**2*x03**2 + 2*c0*c1*x00*x10 + 2*c0*c1*x01*x11 + 2*c0*c1*x02*x12 + 2*c0*c1*x03*x13 + 2*c0*c2*x00*x20 + 2*c0*c2*x01*x21 + 2*c0*c2*x02*x22 + 2*c0*c2*x03*x23 + c1**2*x10**2 + c1**2*x11**2 + c1**2*x12**2 + c1**2*x13**2 + 2*c1*c2*x10*x20 + 2*c1*c2*x11*x21 + 2*c1*c2*x12*x22 + 2*c1*c2*x13*x23 + c2**2*x20**2 + c2**2*x21**2 + c2**2*x22**2 + c2**2*x23**2)) + 2*avgnu*mu*sigma*(c0*x00**2 + c0*x00*x10 + c0*x00*x20 + c0*x01**2 + c0*x01*x11 + c0*x01*x21 + c0*x02**2 + c0*x02*x12 + c0*x02*x22 + c0*x03**2 + c0*x03*x13 + c0*x03*x23 + c1*x00*x10 + c1*x01*x11 + c1*x02*x12 + c1*x03*x13 + c1*x10**2 + c1*x10*x20 + c1*x11**2 + c1*x11*x21 + c1*x12**2 + c1*x12*x22 + c1*x13**2 + c1*x13*x23 + c2*x00*x20 + c2*x01*x21 + c2*x02*x22 + c2*x03*x23 + c2*x10*x20 + c2*x11*x21 + c2*x12*x22 + c2*x13*x23 + c2*x20**2 + c2*x21**2 + c2*x22**2 + c2*x23**2)
    ]
    """
