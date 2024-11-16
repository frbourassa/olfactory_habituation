"""
Main script to launch habituation and new odor recognition simulations,
using functions coded in simulfcts.habituation_recognition.

@author: frbourassa
September 2023
"""
import numpy as np
import os
import h5py

from simulfcts.habituation_recognition import (
    main_habituation_runs,
    main_recognition_runs
)
from simulfcts.idealized_recognition import idealized_recognition_from_runs
from modelfcts.distribs import truncexp1_average
from modelfcts.backgrounds import sample_ss_conc_powerlaw
from modelfcts.ibcm_analytics import fixedpoint_thirdmoment_exact, lambda_pca_equivalent


if __name__ == "__main__":
    # Results folder
    folder = os.path.join("results", "performance")

    # Global seed, used for all models to get exact same backgrounds
    common_seed = 0xfd31bdee63c5a80e1e0998eafce690cc

    # Dimensionalities
    n_r = 25  # n_R: choose 25 (half of full Drosophila dimensionality)
    n_b = 6   # n_B: check against 6 background odors.
    n_i = 24  # n_I: depends on model choice. Use 24 for IBCM (avg. 4 / odor)
    n_k = 1000  # n_K: number of Kenyon cells for neural tag generation
    dimensions_array = np.asarray([n_r, n_b, n_i, n_k])

    # Global test parameters
    new_test_concs = np.asarray([0.5, 1.0])  # to multiply by average whiff c.
    n_runs = 100  # nb of habituation runs, each with a different background
    n_test_times = 10  # nb of late time points at which habituation is tested
    n_back_samples = 10  # nb background samples tested at every time
    n_new_odors = 100  # nb new odors at each test time
    skip_steps = 20
    repeats_array = np.asarray([
                        n_runs, n_test_times, n_back_samples,
                        n_new_odors, len(new_test_concs), skip_steps
                    ])
    # In total: 10^6 discrimination tests. For each model...
    # 100 x 100 pairs of background vs new odor
    # 1 initial condition per background (random)
    # 10 times where habituation is tested: time averaging over a simulation
    # Each model should be tested against the same backgrounds,
    # at the same time, against the same new odors: use the same random number
    # generator seed for simulations of each model, that should give the same
    # process overall.
    # In the testing phase, save some samples of backgrounds to compare

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

    ### IBCM RUNS ###
    ibcm_file_name = os.path.join(folder, "ibcm_performance_results.h5")
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
    print("Starting IBCM simulations")
    main_habituation_runs(ibcm_file_name, ibcm_attrs,
                          ibcm_params, ibcm_options)
    print("Starting IBCM recognition tests")
    main_recognition_runs(ibcm_file_name, ibcm_attrs, ibcm_params,
                          ibcm_options, projection_arguments)

    ### BIOPCA RUNS ###
    # Change number of inhibitory neurons, need less with PCA
    n_i = n_b
    dimensions_array = np.asarray([n_r, n_b, n_i, n_k])
    pca_file_name = os.path.join(folder, "biopca_performance_results.h5")
    biopca_attrs = {
        "model": "PCA",
        "background": "turbulent",
        # Intentionally the same seed to test all models against same backs
        "main_seed": str(common_seed)
    }
    # After a first try, it seems that PCA with Lambda = hs-hn is pretty
    # much like IBCM with Lambda=1, but I have computed a slighly better estimate
    # Compute prediction of fixed points for IBCM,
    # to estimate the baseline Lambda for BioPCA
    ibcm_preds = fixedpoint_thirdmoment_exact(moments_conc, 1, n_b-1)
    hs_and_hn = [max(ibcm_preds[:2]), min(ibcm_preds[:2])]
    lambda_pca = lambda_pca_equivalent(hs_and_hn, moments_conc, n_b, w_alpha_beta, verbose=True)
    # learnrate, rel_lrate, lambda_max, lambda_range, xavg_rate
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
    print("Starting BioPCA simulations")
    main_habituation_runs(pca_file_name, biopca_attrs,
                          biopca_params, biopca_options)
    print("Starting BioPCA recognition tests")
    main_recognition_runs(pca_file_name, biopca_attrs, biopca_params,
                          biopca_options, projection_arguments)

    ### AVERAGE INHIBITION RUNS ###
    # Change number of inhibitory neurons, need less with PCA
    n_i = 1
    dimensions_array = np.asarray([n_r, n_b, n_i, n_k])
    avg_file_name = os.path.join(folder, "avgsub_performance_results.h5")
    avg_attrs = {
        "model": "AVG",
        "background": "turbulent",
        # Intentionally the same seed to test all models against same backs
        "main_seed": str(common_seed)
    }
    avg_params = {
        "dimensions": dimensions_array,
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
    print("Starting average subtraction simulations")
    main_habituation_runs(avg_file_name, avg_attrs,
                          avg_params, avg_options)
    print("Starting average subtraction recognition tests")
    main_recognition_runs(avg_file_name, avg_attrs, avg_params,
                          avg_options, projection_arguments)


    ### IDEAL AND NO INHIBITION ###
    for kind in ["orthogonal", "ideal", "optimal", "none"]:
        print("Starting idealized habituation of kind "
                +"{} recognition tests".format(kind))
        ideal_file_name = os.path.join(folder, kind+"_performance_results.h5")
        idealized_recognition_from_runs(ideal_file_name, ibcm_file_name, kind)
