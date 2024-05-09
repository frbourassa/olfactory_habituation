"""
Scripts to check how the performance of each model (IBCM, BioPCA) changes
as a function of the Lp norms chosen for the minimization and regularization
terms in the W cost function, optimizing the $\alpha$ and $\beta$ rates
along the way.

This requires running multiple simulations, hence we use multiprocessing
to grid-search (Lp, Lq), the L-norms of the minimization and regularization
terms, respectively, and coarsely optimizing alpha, beta as well for each
(Lp, Lq) choice.

For IBCM and PCA, the dynamics are independent of W norm choices,
 so we can just run the full dynamics once, save the x and c series,
 and re-run W dynamics for various choices of W norms.
There has to be no skip in these series though.

Then, W simulations only return a few aggregate statistics.
We use Lambda=1 for IBCM and Lambda=11 for PCA, since these values lead
to roughly similar performance for both models with (L2, L2) norms.

@author: frbourassa
May 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import json

from modelfcts.ideal import relu_inplace, rerun_w_dynamics
from modelfcts.ibcm import integrate_inhib_ibcm_network_options
from modelfcts.biopca import integrate_inhib_ifpsp_network_skip
from modelfcts.ibcm_analytics import fixedpoint_thirdmoment_exact

from modelfcts.distribs import truncexp1_average
from modelfcts.backgrounds import sample_ss_conc_powerlaw
from simulfcts.habituation_recognition import (
    initialize_integration,
    main_habituation_runs,
    main_recognition_runs
)

# Save full x, c time series to re-integrate, also save m snapshots already
# Update: doesn't work due to memory usage of full simulations,
# M and W series too large during the simulation.
def save_simul_results_w(id, res, attrs, gp, snap_i):
    result_items = {
        "IBCM": ["tdump", "back_conc_snaps", "bkvecser", "mser",
                 "cbarser", "thetadump", "wdump", "sdump"],
        "PCA": ["tdump", "back_conc_snaps", "bkvecser", "mser",
                 "lser", "xmeanser", "cbarser", "sdump", "sdump"],
    }
    try:
        item_names = result_items.get(attrs["model"])
    except KeyError:
        raise NotImplementedError("Treat output of other models")
    for i, lbl in enumerate(result_items[attrs["model"]]):
        if lbl.endswith("dump"): continue  # don't save this one
        elif lbl == "back_conc_snaps" and res[i].ndim == 3:
            dset = res[i][snap_i, :, -1]  # Keep only concentrations
        elif lbl.endswith("ser"):
            dset = res[i]
        elif lbl == "s_snaps":
            dset = res[i][snap_i]
            transient = 8 * res[i].shape[0] // 10
            snorm = l2_norm(res[i][transient:], axis=1)
            s_stats = np.asarray([
                np.mean(snorm), np.var(snorm),
                np.mean((snorm - np.mean(snorm))**3)
            ])
            gp.create_dataset("s_stats", data=s_stats)
        elif lbl.endswith("snaps"):
            dset = res[i][snap_i]
        gp.create_dataset(lbl, data=dset.copy())
    return gp


# Update: not useful anymore.
def integrate_w_given_xc(xser, cser, w_init, inhib_params, dt, **options):
    # Get some of the keyword arguments
    activ_fct = str(options.get("activ_fct", "ReLU")).lower()
    w_norms = options.get("w_norms", (2, 2))
    skp = options.get("skp", 1)
    n_neu = c_init.shape[1]  # Number of neurons
    n_orn = x_init.shape[1]
    nsteps = xser.shape[0]
    assert nsteps == cser.shape[0]

    if w_init is None:
        w_init = np.zeros([n_orn, n_neu])

    alpha, beta = inhib_params

    # Containers for the solution over time
    w_series = np.zeros([tseries.shape[0], n_orn, n_neu])  # Inhibitory weights
    s_series = np.zeros([tseries.shape[0], n_orn])

    ## Initialize running variables, separate from the containers above to avoid side effects.
    bkvec = xser[0]
    cbar = cser[0]
    wmat = w_init.copy()
    svec = bkvec - wmat.dot(cbar)
    if activ_fct == "relu":
        relu_inplace(svec)
    elif activ_fct == "identity":
        pass
    else:
        raise ValueError("Unknown activation fct: {}".format(activ_fct))

    # Store back some initial values in containers
    s_series[0] = svec
    w_series[0] = wmat

    t = 0
    for k in range(0, nsteps-1):
        # Get current x, c
        bkvec = xser[k]
        cbar = cser[k]
        t += dt
        ### Inhibitory  weights
        # They depend on cbar and svec at time step k, which are still in cbar, svec
        # cbar, shape [n_neu], should broadcast against columns of wmat,
        # while svec, shape [n_orn], should broadcast across rows (copied on each column)
        alpha_term = alpha*cbar[np.newaxis, :]*svec[:, np.newaxis]
        if w_norms[0] == 1:
            alpha_term /= max(l2_norm(svec), 1e-9)  # avoid zero division
        elif w_norms[0] > 2:  # Assuming even Lp norm
            alpha_term *= l2_norm(svec)**(w_norms[0]-2)

        if w_norms[1] == 2:
            beta_term = beta*wmat
        elif w_norms[1] == 1:
            beta_term = beta * np.sign(wmat)
        else:
            beta_term = 0.0

        wmat += dt * (alpha_term - beta_term)

        # Update background and c to time k+1, to be used in next time step
        bkvec = xser[k+1]
        cbar = cser[k+1]

        # Compute projection neurons at time step k+1
        svec = bkvec - wmat.dot(cbar)
        if activ_fct == "relu":
            relu_inplace(svec)

        # Save current state only if at a multiple of skp
        if (k % skp) == (skp - 1):
            knext = (k+1) // skp
            w_series[knext] = wmat
            s_series[knext] = svec

    return w_series, s_series


if __name__ == "__main__":
    # Results folder
    folder = os.path.join("results", "performance_w")

    # Global seed, used to spawn more seeds for different background instances
    common_seed = 0xe4a1f15c70ecc52736db51e441a451df

    # Dimensionalities
    n_r = 25  # n_R: choose 25 (half of full Drosophila dimensionality)
    n_b = 6   # n_B: check against 6 background odors.
    n_i = 24  # n_I: depends on model choice. Use 24 for IBCM (avg. 4 / odor)
    n_k = 1000  # n_K: number of Kenyon cells for neural tag generation
    dimensions_array = np.asarray([n_r, n_b, n_i, n_k])

    # Grid search over p, q, alpha, beta
    p_choices = (1, 2, 16)
    q_choices = (1, 2)
    alpha_grids_p = {  # Depends on p, decrease alpha for larger p, vice-versa
        1: 10**np.arange(-5, -3.4, 0.5),
        2: 10**np.arange(-5.5, -3.9, 0.5),
        16: 10**np.arange(-5.5, -3.9, 0.5)
    }
    beta_grids_q = {  # Depends on q, decrease beta for small q, vice-versa
        1: 10**np.arange(-7.0, -4.9, 1.0),
        2: [5e-6, 2e-5, 8e-5]
    }

    # Global test parameters
    skip_steps = 20
    new_test_concs = np.asarray([0.5, 1.0])  # to multiply by average whiff c.
    n_runs = 8  # nb of backgrounds and habituation runs to test
    n_test_times = 10  # nb of late time points at which habituation is tested
    n_back_samples = 10  # nb background samples tested at every time
    n_new_odors = 100  # nb new odors at each test time

    repeats_array = np.asarray([
                        n_runs, n_test_times, n_back_samples,
                        n_new_odors, len(new_test_concs), skip_steps
                    ])

    # Other parameters common to all models
    duration_dt = np.asarray([360000.0, 1.0])
    start_test_t = duration_dt[0] - n_test_times * 2000.0
    snapshot_times = np.linspace(start_test_t, duration_dt[0], n_test_times)
    # Avoid going to exactly the total time, it is not available
    snapshot_times -= duration_dt[1]*skip_steps
    w_alpha_beta = np.asarray([1e-4, 2e-5])  # default values for full simuls
    projection_arguments = {
        "kc_sparsity": 0.05,
        "adapt_kc": True,
        "n_pn_per_kc": 3,
        "project_thresh_fact": 0.05
    }
    activ_fct_choice = "identity"
    turbulent_back_params = [
        np.asarray([1.0] * n_b),        # whiff_tmins
        np.asarray([500.] * n_b),       # whiff_tmaxs
        np.asarray([1.0] * n_b),        # blank_tmins
        np.asarray([800.0] * n_b),      # blank_tmaxs
        np.asarray([0.6] * n_b),        # c0s
        np.asarray([0.5] * n_b),        # alphas
    ]
    # Adjust new odor concentrations to average whiff concentration
    avg_whiff_conc = np.mean(truncexp1_average(*turbulent_back_params[4:]))
    print("Average whiff concentration: {:.4f}".format(avg_whiff_conc))
    new_test_concs *= avg_whiff_conc

    # Compute moments of the background concentration process
    dummy_rgen = np.random.default_rng(0x66d5b6b67eb49bf6b4d33610c055c19e)
    conc_samples = sample_ss_conc_powerlaw(
                        *turbulent_back_params, size=int(1e5), rgen=dummy_rgen
                    )
    mean_conc = np.mean(conc_samples)
    moments_conc = np.asarray([
        mean_conc,
        np.var(conc_samples),
        np.mean((conc_samples - mean_conc)**3)
    ])
    print("Computed numerically the concentration moments:", moments_conc)

    # Compute prediction of fixed points for IBCM,
    # to estimate the baseline Lambda for BioPCA
    ibcm_preds = fixedpoint_thirdmoment_exact(moments_conc, 1, n_b-1)
    cs_minus_cn = abs(ibcm_preds[0] - ibcm_preds[1])

    ### Run IBCM simulations for each Lambda choice
    ibcm_attrs = {
        "model": "IBCM",
        "background": "turbulent",
        # need to save 128-bit to str, too large for HDF5
        "main_seed": str(common_seed)
    }
    ibcm_params = {
        "dimensions": dimensions_array,
        "repeats": repeats_array,
        # learnrate, tau_avg, eta, lambda, sat, ktheta, decay_relative
        "m_rates": np.asarray([0.00125, 1600.0, 0.6/n_i, 1.0, 50.0, 0.1, 0.005]),
        "w_rates": w_alpha_beta,
        "time_params": duration_dt,
        "back_params": turbulent_back_params,
        "snap_times": snapshot_times,
        "new_concs": new_test_concs,
        "moments_conc": moments_conc
    }
    ibcm_options = {
        "activ_fct": activ_fct_choice,
        "saturation": "tanh",
        "variant": "law",   # for turbulent background
        "decay": True
    }
    # To keep track of the i, j index of each (p, q) and (alpha, beta) choice
    table_i_pq = {}
    table_ij_ab = {}
    print("Running IBCM simulations of habituation and new odor recognition"
        + " for various p, q, alpha, beta and saving")
    for i, pq in enumerate(itertools.product(p_choices, q_choices)):
        table_i_pq[str(i)] = pq
        ibcm_options["w_norms"] = pq
        agrid, bgrid = alpha_grids_p[pq[0]], beta_grids_q[pq[1]]
        for j, ab in enumerate(itertools.product(agrid, bgrid)):
            table_ij_ab["{}_{}".format(i, j)] = ab
            ibcm_params["w_rates"] = ab
            ibcm_file_name = os.path.join(
                folder, "ibcm_simuls_for_w_{}_{}.h5".format(i, j)
            )
            print("Habituation for p={}, q={}, a={}, b={}".format(*pq, *ab))
            main_habituation_runs(
                ibcm_file_name, ibcm_attrs, ibcm_params, ibcm_options)

            print("Recognition for p={}, q={}, a={}, b={}".format(*pq, *ab))
            main_recognition_runs(
                ibcm_file_name, ibcm_attrs, ibcm_params,
                ibcm_options, projection_arguments)

    ### Run n_runs BioPCA simulation for each p, q, alpha, beta choice
    # Change number of inhibitory neurons, need less with PCA
    n_i = n_b
    dimensions_array = np.asarray([n_r, n_b, n_i, n_k])
    pca_file_name = os.path.join(folder, "biopca_performance_lambda.h5")
    biopca_attrs = {
        "model": "PCA",
        "background": "turbulent",
        # Intentionally the same seed to test all models against same backs
        "main_seed": str(common_seed)
    }
    # learnrate, rel_lrate, lambda_max, lambda_range, xavg_rate
    # After a first try, it seems that PCA with Lambda = 10 is pretty
    # much like IBCM with Lambda=1,
    lambda_pca = cs_minus_cn
    print(lambda_pca)
    biopca_rates = np.asarray([1e-4, 2.0, lambda_pca, 0.5, 1e-4])
    biopca_params = {
        "dimensions": dimensions_array,
        "repeats": repeats_array,
        "m_rates": biopca_rates,
        "w_rates": w_alpha_beta,
        "time_params": duration_dt,
        "back_params": turbulent_back_params,
        "snap_times": snapshot_times,
        "new_concs": new_test_concs,
        "moments_conc": moments_conc
    }
    biopca_options = {
        "activ_fct": activ_fct_choice,
        "remove_mean": False,  # helps PCA too much
        "remove_lambda": False
    }

    print("Running BioPCA simulations of habituation and new odor recognition"
            + " for various p, q, alpha, beta and saving")
    for i, pq in enumerate(itertools.product(p_choices, q_choices)):
        biopca_options["w_norms"] = pq
        agrid, bgrid = alpha_grids_p[pq[0]], beta_grids_q[pq[1]]
        for j, ab in enumerate(itertools.product(agrid, bgrid)):
            biopca_params["w_rates"] = ab
            biopca_file_name = os.path.join(
                folder, "biopca_simuls_for_w_{}_{}.h5".format(i, j)
            )
            print("Habituation for p={}, q={}, a={}, b={}".format(*pq, *ab))
            main_habituation_runs(
                biopca_file_name, biopca_attrs, biopca_params, biopca_options)

            print("Recognition for p={}, q={}, a={}, b={}".format(*pq, *ab))
            main_recognition_runs(
                biopca_file_name, biopca_attrs, biopca_params,
                biopca_options, projection_arguments)

    # Save the (p, q, alpha, beta) tables grids to a JSON file
    with open(os.path.join(folder, "table_i_pnorm-qnorm.json"), "w") as f:
        json.dump(table_i_pq, f)
    with open(os.path.join(folder, "table_ij_alpha-beta.json"), "w") as f:
        json.dump(table_ij_ab, f)
