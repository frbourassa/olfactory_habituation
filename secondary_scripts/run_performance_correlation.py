"""
Scripts to check how the performance of each model (IBCM, BioPCA) changes
as we increase the magnitude of correlation (Pearson, rho) between odors. 
To further test robustness against asymmetry between odors, 
we put unequal variance on the different odors?

This requires running multiple simulations for each correlation value, 
hence we use multiprocessing. Launch one  "run_performance_recognition.py"-like
simulation for each tested correlation, to be saved to a separate file. 

For example runs at different correlation values, I will use a separate
Jupyter notebook running and saving relevant details for plotting. 
The present script is only for assessing performance across
multiple runs as a function of correlation strength. 

@author: frbourassa
July 2025
"""
import numpy as np
import os, sys
if ".." not in sys.path:
    sys.path.insert(1, "..")

from simulfcts.habituation_recognition import (
    main_habituation_runs,
    main_recognition_runs
)
from simulfcts.idealized_recognition import (
    idealized_recognition_from_runs, 
    jaccard_between_random_odors
)
from modelfcts.distribs import truncexp1_average
from modelfcts.backgrounds import (
    sample_ss_mixed_concs_powerlaw, logof10, mean_turbulent_concs
)
from modelfcts.ibcm_analytics import (
    fixedpoint_thirdmoment_exact, lambda_pca_equivalent
)
from utils.statistics import seed_from_gen
import multiprocessing


def get_target_cholesky(correl, n_comp):
    target_covmat_scaled = np.zeros((n_comp, n_comp))
    target_covmat_scaled[[-1, -2], [-2, -1]] = correl
    target_covmat_scaled[np.diag_indices(n_comp)] = 1.0
    if abs(correl) < 1.0:
        target_cholesky = np.linalg.cholesky(target_covmat_scaled)
    else:
        target_cholesky = np.zeros((n_comp, n_comp))
        target_cholesky[np.diag_indices(n_comp)] = 1.0
        target_cholesky[-1, -1] = 0.0
        target_cholesky[-1, -2] = correl  # Replace odor 1 by odor 0 for rho = +-1
    return target_cholesky


if __name__ == "__main__":
    # Results folder
    # Need to initialize subprocesses as spawn to avoid collision between
    # processes sharing the same memory. spawn is default on MacOS, so
    # I had no issues locally, but I had lock delays on the Linux server
    # where fork is the default and causes problems in multithreading. 
    # So make sure spawn is default everywhere, I multiprocess at a high
    # level so child processes do not get started often. 
    multiprocessing.set_start_method('spawn')
    folder = os.path.join("..", "results", "performance_correlation")
    do_main_runs = True

    # Dimensionalities -- fixed
    n_s = 50  # n_S: stay small, fly size
    n_b = 4   # n_B: 4 odors, 2 of which are correlated up to being one odor
    n_i = 24  # n_I: depends on model choice. Use 24 for IBCM
    n_k = 2000  # n_K: number of Kenyon cells for neural tag generation
    dimensions_array = np.asarray([n_s, n_b, n_i, n_k])
    correl_range = np.around(np.arange(-0.9, 1.01, 0.1), 12)  # rho = 1 needs special handling

    # Common global seeds, one per correlation strength tested, 
    # used for all models to get exact same backgrounds
    root_seed = 0x9b197c191e12251f7f430c795ca02c01
    n_mags_tested = correl_range.shape[0]
    seed_generator = np.random.default_rng(root_seed)
    common_seeds = [seed_from_gen(seed_generator, nbits=128) for _ in correl_range]

    # Global test parameters
    new_test_concs = np.asarray([0.5, 1.0])  # to multiply by average whiff c.
    n_runs = 32  # nb of habituation runs, each with a different background
    n_test_times = 5  # nb of late time points at which habituation is tested
    n_back_samples = 4  # nb background samples tested at every time
    n_new_odors = 100  # nb new odors at each test time
    skip_steps = 100
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
    w_alpha_beta = np.asarray([5e-5, 1e-5])  # ower to avoid numerical issues
    projection_arguments = {
        "kc_sparsity": 0.05,
        "adapt_kc": True,
        "n_pn_per_kc": 3 * n_s // 25,
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
    # Add mean concentration to parameters
    desired_mean_concs = mean_turbulent_concs(turbulent_back_params)
    turbulent_back_params.append(desired_mean_concs)
    # Cholesky decomposition of desired covariance -- updated for each simul
    target_cholesky = get_target_cholesky(correl_range[0], n_b)
    turbulent_back_params.append(target_cholesky)

    # Adjust new odor concentrations to average whiff concentration
    avg_whiff_conc = np.mean(truncexp1_average(*turbulent_back_params[4:6]))
    print("Average whiff concentration: {:.4f}".format(avg_whiff_conc))
    new_test_concs *= avg_whiff_conc

    # Compute moments of the background concentration process
    dummy_rgen = np.random.default_rng(0xd915ff2054e9e76dc76983185d14f1f9)
    conc_samples = sample_ss_mixed_concs_powerlaw(
                        *turbulent_back_params, size=int(1e5), rgen=dummy_rgen
                    )
    mean_conc = np.mean(conc_samples)
    moments_conc = np.asarray([
        mean_conc,
        np.var(conc_samples),
        np.mean((conc_samples - mean_conc)**3)
    ])
    assert abs(mean_conc - desired_mean_concs.mean()) < 1e-3, "Discrepancy in average conc."
    print("Computed numerically the concentration moments:", moments_conc)

    ### IBCM RUNS ###
    ibcm_attrs = {
        "model": "IBCM",
        "background": "turbulent_correlation",
        # need to save 128-bit to str, too large for HDF5
        "main_seed": str(common_seeds[0]),  # Will be changed for each sim., 
        "correl_rho": correl_range[0]  # Will be changed for each set of sims
    }
    ibcm_params = {
        "dimensions": dimensions_array,
        "repeats": repeats_array,
        # learnrate, tau_avg, eta, lambda, sat, ktheta, decay_relative
        "m_rates": np.asarray([0.001, 1200.0, 0.6/n_i, 1.0, 50.0, 0.1, 0.005]),
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
    all_ibcm_file_names = {}
    
    for i, rho_i in enumerate(correl_range):
        # Update the correlation and the Cholesky matrix as parameters
        target_cholesky = get_target_cholesky(rho_i, n_b)
        turbulent_back_params[-1] = target_cholesky
        ibcm_params["back_params"] = turbulent_back_params
        # background vectors are appended to the list of parameters
        # in the run function; here, Cholesky matrix is last in the global list. 
        ibcm_attrs["main_seed"] = str(common_seeds[i])
        ibcm_attrs["correl_rho"] = rho_i
        ibcm_file_name = os.path.join(folder, 
                            "ibcm_performance_results_correlation_{}.h5".format(i))
        all_ibcm_file_names[i] = str(ibcm_file_name)
        if do_main_runs:
            print("Starting IBCM simulation for correlation = {}".format(rho_i))
            main_habituation_runs(ibcm_file_name, ibcm_attrs,
                                ibcm_params, ibcm_options, lean=True)
            print("Starting IBCM recognition for correlation = {}".format(rho_i))
            main_recognition_runs(ibcm_file_name, ibcm_attrs, ibcm_params,
                                ibcm_options, projection_arguments, lean=True)

    ### BIOPCA RUNS ###
    # Change number of inhibitory neurons, need less with PCA
    n_i = n_b + 1  # See what an extra neuron will do
    dimensions_array = np.asarray([n_s, n_b, n_i, n_k])
    biopca_attrs = {
        "model": "PCA",
        "background": "turbulent_correlation",
        # Intentionally the same seed to test all models against same backs
        "main_seed": str(common_seeds[0]),  # Updated for each sim
        "correl_rho": correl_range[0]  # Will be changed for each set of sims
    }
    # After a first try, it seems that PCA with Lambda = hs-hn is pretty
    # much like IBCM with Lambda=1, but I have computed a better estimate
    # It is based on uncorrelated odors, but it will approximately do here. 
    ibcm_preds = fixedpoint_thirdmoment_exact(moments_conc, 1, n_b-1)
    hs_and_hn = [max(ibcm_preds[:2]), min(ibcm_preds[:2])]
    lambda_pca = lambda_pca_equivalent(
        hs_and_hn, moments_conc, n_b, w_alpha_beta, verbose=True)
    # learnrate, rel_lrate, lambda_max, lambda_range, xavg_rate
    biopca_rates = np.asarray([1e-4, 2.0, lambda_pca, 0.5, 1e-4])
    biopca_params = {
        "dimensions": dimensions_array,  # updated for each sim
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
        "remove_mean": True,
        "remove_lambda": False
    }
    for i, rho_i in enumerate(correl_range):
        # Update the Cholesky matrix as a parameters
        target_cholesky = get_target_cholesky(rho_i, n_b)
        turbulent_back_params[-1] = target_cholesky
        biopca_params["back_params"] = turbulent_back_params
        biopca_attrs["main_seed"] = str(common_seeds[i])
        biopca_attrs["correl_rho"] = rho_i
        pca_file_name = os.path.join(folder, 
            "biopca_performance_results_correlation_{}.h5".format(i))
        if do_main_runs:
            print("Starting BioPCA recognition for correlation = {}".format(rho_i))
            main_habituation_runs(pca_file_name, biopca_attrs,
                            biopca_params, biopca_options, lean=True)
            print("Starting BioPCA recognition for correlation = {}".format(rho_i))
            main_recognition_runs(pca_file_name, biopca_attrs, biopca_params,
                            biopca_options, projection_arguments, lean=True)

    ### AVERAGE INHIBITION RUNS ###
    # Change number of inhibitory neurons, need less with PCA
    n_i = 1
    dimensions_array = np.asarray([n_s, n_b, n_i, n_k])
    avg_attrs = {
        "model": "AVG",
        "background": "turbulent_correlation",
        # Intentionally the same seed to test all models against same backs
        "main_seed": str(common_seeds[0]),  # updated each sim
        "correl_rho": correl_range[0]
    }
    avg_params = {
        "dimensions": dimensions_array,  # updated each sim
        "repeats": repeats_array,
        "m_rates": [],
        "w_rates": w_alpha_beta,
        "time_params": duration_dt,
        "back_params": turbulent_back_params,
        "snap_times": snapshot_times,
        "new_concs": new_test_concs,
        "moments_conc": moments_conc
    }
    avg_options = {
        "activ_fct": activ_fct_choice
    }
    for i, rho_i in enumerate(correl_range):
        # Update the correlation and the Cholesky matrix as parameters
        target_cholesky = get_target_cholesky(rho_i, n_b)
        turbulent_back_params[-1] = target_cholesky
        ibcm_params["back_params"] = turbulent_back_params
        avg_params["back_params"] = turbulent_back_params
        avg_attrs["main_seed"] = str(common_seeds[i])
        avg_attrs["correl_rho"] = rho_i
        avg_file_name = os.path.join(folder, 
            "avgsub_performance_results_correlation_{}.h5".format(i))
        if do_main_runs:
            print("Starting average sub. simulation for correlation = {}".format(rho_i))
            main_habituation_runs(avg_file_name, avg_attrs,
                                avg_params, avg_options, lean=True)
            print("Starting average sub. recognition for correlation = {}".format(rho_i))
            main_recognition_runs(avg_file_name, avg_attrs, avg_params,
                                avg_options, projection_arguments, lean=True)

    ### IDEAL AND NO INHIBITION ###
    for kind in ["orthogonal", "ideal", "optimal", "none"]:
        for i, rho_i in enumerate(correl_range):
            print("Starting idealized habituation of kind "
                +"{} recognition tests for correlation = {}".format(kind, rho_i))
            ideal_file_name = os.path.join(folder, 
                    kind+"_performance_results_correlation_{}.h5".format(i))
            ibcm_fname = all_ibcm_file_names[i]
            idealized_recognition_from_runs(
                ideal_file_name, ibcm_fname, kind, lean=True)
