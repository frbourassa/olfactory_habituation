"""
Scripts to check how the performance of each model (IBCM, BioPCA) changes
as we increase the magnitude of independent Gaussian noise added to OSNs. 

This requires running multiple simulations for each noise value, 
hence we use multiprocessing. Launch one 
"compare_recognition_performance.py"-like simulation for each 
tested noise magnitude, to be saved to a separate file. 

Need to skip more steps than usual

@author: frbourassa
March 2025
"""
import numpy as np
import os

from simulfcts.habituation_recognition import (
    main_habituation_runs,
    main_recognition_runs
)
from simulfcts.idealized_recognition import (
    idealized_recognition_from_runs, 
    jaccard_between_random_odors
)
from modelfcts.distribs import truncexp1_average
from modelfcts.backgrounds import sample_ss_conc_powerlaw, logof10
from modelfcts.ibcm_analytics import (
    fixedpoint_thirdmoment_exact, lambda_pca_equivalent
)
from utils.statistics import seed_from_gen
import multiprocessing


if __name__ == "__main__":
    # Results folder
    # Need to initialize subprocesses as spawn to avoid collision between
    # processes sharing the same memory. spawn is default on MacOS, so
    # I had no issues locally, but I had lock delays on the Linux server
    # where fork is the default and causes problems in multithreading. 
    # So make sure spawn is default everywhere, I multiprocess at a high
    # level so child processes do not get started often. 
    multiprocessing.set_start_method('spawn')
    folder = os.path.join("results", "performance_noise")

    # Dimensionalities -- will be updated for each launched simulation
    n_s = 25  # n_S: stay small, since we need noise for each OSN
    n_b = 6   # n_B: check against 6 background odors.
    n_i = 24  # n_I: depends on model choice. Use 24 for IBCM (avg. 4 / odor)
    n_k = 1000  # n_K: number of Kenyon cells for neural tag generation
    dimensions_array = np.asarray([n_s, n_b, n_i, n_k])
    noise_range = np.asarray([0.0, 1e-4, 1e-3, 1e-2, 0.01*np.sqrt(10.0), 0.1])
    #noise_range = np.asarray([0.01])

    # Common global seeds, one per dimensionality tested, 
    # used for all models to get exact same backgrounds
    root_seed = 0x9c3508db054c8102665d8bfbed2a9c17
    n_mags_tested = noise_range.shape[0]
    seed_generator = np.random.default_rng(root_seed)
    common_seeds = [seed_from_gen(seed_generator, nbits=128) for _ in noise_range]

    # Global test parameters
    new_test_concs = np.asarray([0.5, 1.0])  # to multiply by average whiff c.
    n_runs = 64  # nb of habituation runs, each with a different background
    n_test_times = 5  # nb of late time points at which habituation is tested
    n_back_samples = 4  # nb background samples tested at every time
    n_new_odors = 100  # nb new odors at each test time
    skip_steps = 100
    repeats_array = np.asarray([
                        n_runs, n_test_times, n_back_samples,
                        n_new_odors, len(new_test_concs), skip_steps
                    ])
    # In total: 2.5x10^5 discrimination tests. For each model...
    # 64 x 100 pairs of background vs new odor
    # 1 initial condition per background (random)
    # 5 times where habituation is tested: time averaging over a simulation
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
        "n_pn_per_kc": 6,  # Needs to be updated to 3/25*N_S for each N_S
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
    # Add the Gaussian noise amplitudes as a parameter
    noise_ampli = np.ones(n_s) * noise_range[0]
    turbulent_back_params.append(noise_ampli)

    # Adjust new odor concentrations to average whiff concentration
    avg_whiff_conc = np.mean(truncexp1_average(*turbulent_back_params[4:6]))
    print("Average whiff concentration: {:.4f}".format(avg_whiff_conc))
    new_test_concs *= avg_whiff_conc

    # Compute moments of the background concentration process in absence of the Gaussian noise
    dummy_rgen = np.random.default_rng(0xcc1e0a471820fd7d4ffd193e6d7ade58)
    conc_samples = sample_ss_conc_powerlaw(
                        *turbulent_back_params[:-1], size=int(1e5), rgen=dummy_rgen
                    )
    mean_conc = np.mean(conc_samples)
    moments_conc = np.asarray([
        mean_conc,
        np.var(conc_samples),
        np.mean((conc_samples - mean_conc)**3)
    ])
    print("Computed numerically the concentration moments:", moments_conc)

    ### IBCM RUNS ###
    ibcm_attrs = {
        "model": "IBCM",
        "background": "turbulent_gaussnoise",
        # need to save 128-bit to str, too large for HDF5
        "main_seed": str(common_seeds[0]),  # Will be changed for each sim., 
        "noise_ampli": noise_range[0]  # Will be changed for each set of sims
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
    all_ibcm_file_names = {}
    for i, n_s_i in enumerate(noise_range):
        # Update the Gaussian noise amplitudes as a parameter
        noise_ampli = np.ones(n_s) * n_s_i
        turbulent_back_params[-1] = noise_ampli
        ibcm_params["back_params"] = turbulent_back_params
        # background vectors are appended to the list of parameters
        # in the run function; here, amplitudes are last in the global list. 
        ibcm_attrs["main_seed"] = str(common_seeds[i])
        ibcm_attrs["noise_ampli"] = n_s_i
        ibcm_file_name = os.path.join(folder, 
                            "ibcm_performance_results_gaussnoise_{}.h5".format(i))
        all_ibcm_file_names[n_s_i] = str(ibcm_file_name)
        print("Starting IBCM simulation for noise level = {}".format(n_s_i))
        main_habituation_runs(ibcm_file_name, ibcm_attrs,
                            ibcm_params, ibcm_options, lean=True)
        print("Starting IBCM recognition for noise_level = {}".format(n_s_i))
        main_recognition_runs(ibcm_file_name, ibcm_attrs, ibcm_params,
                            ibcm_options, projection_arguments, lean=True)

    ### BIOPCA RUNS ###
    # Change number of inhibitory neurons, need less with PCA
    n_i = min(n_b * 2, n_s)  # Some neurons will "over"-fit the noise
    dimensions_array = np.asarray([n_s, n_b, n_i, n_k])
    biopca_attrs = {
        "model": "PCA",
        "background": "turbulent_gaussnoise",
        # Intentionally the same seed to test all models against same backs
        "main_seed": str(common_seeds[0]),  # Updated for each sim
        "noise_ampli": noise_range[0]  # Will be changed for each set of sims
    }
    # After a first try, it seems that PCA with Lambda = hs-hn is pretty
    # much like IBCM with Lambda=1, but I have computed a better estimate
    # Compute prediction of fixed points for IBCM,
    # to estimate the baseline Lambda for BioPCA
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
    for i, n_s_i in enumerate(noise_range):
        # Update the Gaussian noise amplitudes as a parameter
        noise_ampli = np.ones(n_s) * n_s_i
        turbulent_back_params[-1] = noise_ampli
        biopca_params["back_params"] = turbulent_back_params
        biopca_attrs["main_seed"] = str(common_seeds[i])
        biopca_attrs["noise_ampli"] = n_s_i
        pca_file_name = os.path.join(folder, 
            "biopca_performance_results_gaussnoise_{}.h5".format(i))
        print("Starting BioPCA recognition for noise level = {}".format(n_s_i))
        main_habituation_runs(pca_file_name, biopca_attrs,
                          biopca_params, biopca_options, lean=True)
        print("Starting BioPCA recognition for noise_level = {}".format(n_s_i))
        main_recognition_runs(pca_file_name, biopca_attrs, biopca_params,
                          biopca_options, projection_arguments, lean=True)

    ### AVERAGE INHIBITION RUNS ###
    # Change number of inhibitory neurons, need less with PCA
    n_i = 1
    dimensions_array = np.asarray([n_s, n_b, n_i, n_k])
    avg_attrs = {
        "model": "AVG",
        "background": "turbulent_gaussnoise",
        # Intentionally the same seed to test all models against same backs
        "main_seed": str(common_seeds[0]),  # updated each sim
        "noise_ampli": noise_range[0]
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
    for i, n_s_i in enumerate(noise_range):
        noise_ampli = np.ones(n_s) * n_s_i
        turbulent_back_params[-1] = noise_ampli
        avg_params["back_params"] = turbulent_back_params
        avg_attrs["main_seed"] = str(common_seeds[i])
        avg_attrs["noise_ampli"] = n_s_i
        avg_file_name = os.path.join(folder, 
            "avgsub_performance_results_gaussnoise_{}.h5".format(i))
        print("Starting average sub. simulation for noise level = {}".format(n_s_i))
        main_habituation_runs(avg_file_name, avg_attrs,
                            avg_params, avg_options, lean=True)
        print("Starting average sub. recognition for noise level = {}".format(n_s_i))
        main_recognition_runs(avg_file_name, avg_attrs, avg_params,
                            avg_options, projection_arguments, lean=True)

    ### IDEAL AND NO INHIBITION ###
    for kind in ["orthogonal", "ideal", "optimal", "none"]:
        for n_s_i in noise_range:
            print("Starting idealized habituation of kind "
                +"{} recognition tests for noise level = {}".format(kind, n_s_i))
            ideal_file_name = os.path.join(folder, 
                    kind+"_performance_results_gaussnoise_{}.h5".format(i))
            ibcm_fname = all_ibcm_file_names[n_s_i]
            idealized_recognition_from_runs(
                ideal_file_name, ibcm_fname, kind, lean=True)
