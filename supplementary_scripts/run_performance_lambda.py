"""
Scripts to check how the performance of each model (IBCM, BioPCA) changes
as a function of the M weights scale, Lambda.

This requires running multiple simulations, hence we use multiprocessing.
For IBCM and PCA, the dynamics are identical for all Lambda, just rescaled,
 so we can just run the full dynamics once, save the x and c series,
 and re-run W dynamics for various rescaling of M and c.
 There has to be no skip in these series though.

But that was too complicated to code, so we just re-run identical background
simulations for all Lambdas in the end. 

@author: frbourassa
January 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys
if ".." not in sys.path:
    sys.path.insert(1, "..")
    
from modelfcts.ibcm_analytics import (
    fixedpoint_thirdmoment_exact, 
    lambda_pca_equivalent
)

from modelfcts.distribs import truncexp1_average
from modelfcts.backgrounds import sample_ss_conc_powerlaw
from simulfcts.habituation_recognition_lambda import (
    main_habituation_runs_lambda,
    main_performance_lambda
)

if __name__ == "__main__":
    # Results folder
    folder = os.path.join("..", "results", "performance_lambda")

    # Global seed, used for all models to get exact same backgrounds
    common_seed = 0xe4a1f15c70ecc52736db51e441a451dd

    # Dimensionalities
    n_s = 25  # n_R: choose 25 (half of full Drosophila dimensionality)
    n_b = 6   # n_B: check against 6 background odors.
    n_i = 24  # n_I: depends on model choice. Use 24 for IBCM (avg. 4 / odor)
    n_k = 1000 * n_s // 25  # n_K: number of Kenyon cells for neural tag generation
    dimensions_array = np.asarray([n_s, n_b, n_i, n_k])

    # Global test parameters
    new_test_concs = np.asarray([0.5, 1.0])  # to multiply by average whiff c.
    n_lambda_test = 30
    n_test_times = 10  # nb of late time points at which habituation is tested
    n_back_samples = 10  # nb background samples tested at every time
    n_new_odors = 100  # nb new odors at each test time
    skip_steps = 200
    repeats_array = np.asarray([
                        n_lambda_test, n_test_times, n_back_samples,
                        n_new_odors, len(new_test_concs), skip_steps
                    ])

    # Other parameters common to all models
    duration_dt = np.asarray([360000.0, 1.0])
    start_test_t = duration_dt[0] - n_test_times * 2000.0
    snapshot_times = np.linspace(start_test_t, duration_dt[0], n_test_times)
    # Avoid going to exactly the total time, it is not available
    snapshot_times -= duration_dt[1]*skip_steps
    w_alpha_beta = np.asarray([1e-4, 2e-5])
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
    # Adjust new odor concentrations to average whiff concentration
    avg_whiff_conc = np.mean(truncexp1_average(*turbulent_back_params[4:]))
    print("Average whiff concentration: {:.4f}".format(avg_whiff_conc))
    new_test_concs *= avg_whiff_conc

    # Compute moments of the background concentration process
    dummy_rgen = np.random.default_rng(0x51bf7feb1fd2a3f61e1b1b59679f62c6)
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

    ### Run IBCM simulations for each Lambda choice
    ibcm_file_name = os.path.join(folder, f"ibcm_performance_lambda_{n_s}.h5")
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
    print("Running IBCM simulation for various Lambdas and saving to hdf5")
    main_habituation_runs_lambda(ibcm_file_name, ibcm_attrs,
                           ibcm_params, ibcm_options)

    print("Running IBCM performance tests as a function of Lambda")
    # filename, attributes, parameters, model_options
    main_performance_lambda(ibcm_file_name, ibcm_attrs, ibcm_params,
                           ibcm_options, projection_arguments)


    ### Run one BioPCA simulation for each Lambda value
    # Change number of inhibitory neurons, need less with PCA
    n_i = n_b
    dimensions_array = np.asarray([n_s, n_b, n_i, n_k])
    pca_file_name = os.path.join(folder, f"biopca_performance_lambda_{n_s}.h5")
    biopca_attrs = {
        "model": "PCA",
        "background": "turbulent",
        # Intentionally the same seed to test all models against same backs
        "main_seed": str(common_seed)
    }
    # learnrate, rel_lrate, lambda_max, lambda_range, xavg_rate
    # After a first try, it seems that PCA with Lambda = hs-hn is pretty
    # much like IBCM with Lambda=1, but I have computed a slighly better estimate
    hs_and_hn = [max(ibcm_preds[:2]), min(ibcm_preds[:2])]
    lambda_pca = lambda_pca_equivalent(hs_and_hn, moments_conc, n_b, w_alpha_beta, verbose=True)
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
        "remove_mean": True,
        "remove_lambda": False
    }
    print("Starting BioPCA simulations for various Lambdas")
    main_habituation_runs_lambda(pca_file_name, biopca_attrs,
                          biopca_params, biopca_options)
    print("Starting BioPCA performance tests as a function of Lambda")
    main_performance_lambda(pca_file_name, biopca_attrs, biopca_params,
                          biopca_options, projection_arguments)
