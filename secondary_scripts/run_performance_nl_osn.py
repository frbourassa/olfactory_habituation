"""
Scripts to check the performance of each model (IBCM, BioPCA) when OSNs
have the nonlinear response function of Kadakia and Emonet, with odors
defined by the affinity of each type of OR-Orco complex K^*
(without adaptation of the free energy difference for now). 

Also check how this performance changes as the strength of nonlinearity
increases. This is controlled by scaling down exp(epsilon) with respect to K^*, 
and compensating the absolute OSN magnitude by scaling down max_osn_ampli. 

For example runs at different nonlinearity strengths, I will use a separate
Jupyter notebook running and saving relevant details for plotting. 

@author: frbourassa
August 2025
"""
import numpy as np
import os, sys
if ".." not in sys.path:
    sys.path.insert(1, "..")

from simulfcts.habituation_recognition_nonlin_osn import (
    main_habituation_runs_nl_osn,
    main_recognition_runs_nl_osn,
    idealized_recognition_from_runs_nl_osn
)
from modelfcts.distribs import truncexp1_average
from modelfcts.nonlin_adapt_osn import (
    combine_odors_affinities, 
    generate_odor_tanhcdf
)
from modelfcts.backgrounds import sample_ss_conc_powerlaw
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
    folder = os.path.join("..", "results", "performance_nl_osn")
    do_main_runs = True

    # Dimensionalities -- fixed
    n_s = 50  # n_S: stay small, fly size
    n_b = 6   # n_B: 6 background odors
    n_i = 24  # n_I: depends on model choice. Use 24 for IBCM
    n_k = 2000  # n_K: number of Kenyon cells for neural tag generation
    dimensions_array = np.asarray([n_s, n_b, n_i, n_k])
    n_mags_tested = 13
    #unit_scale_range = np.geomspace(1e-6, 5e-3, n_mags_tested)
    epsils_range = np.linspace(2.5, 10.0, n_mags_tested)

    # Common global seeds, one per correlation strength tested, 
    # used for all models to get exact same backgrounds
    root_seed = 0xb0252ee13b08dca462794d94d32e332d
    seed_generator = np.random.default_rng(root_seed)
    common_seeds = [seed_from_gen(seed_generator, nbits=128) 
                    for _ in epsils_range]

    # Global test parameters
    new_test_concs = np.asarray([0.5, 1.0])  # to multiply by average whiff c.
    n_runs = 96  # nb of habituation runs, each with a different background.  CHANGE
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
    w_alpha_beta = np.asarray([5e-5, 1e-5])  # lower to avoid numerical issues
    projection_arguments = {
        "kc_sparsity": 0.05,
        "adapt_kc": True,
        "n_pn_per_kc": 3 * n_s // 25,
        "project_thresh_fact": 0.05
    }
    odor_gen_arguments = {
        "unit_scale": 5e-4   # default, epsilon adjusted with respect to it
    }
    combine_fct = combine_odors_affinities
    activ_fct_choice = "identity"
    turbulent_back_params = [
        np.asarray([1.0] * n_b),        # whiff_tmins
        np.asarray([500.] * n_b),       # whiff_tmaxs
        np.asarray([1.0] * n_b),        # blank_tmins
        np.asarray([800.0] * n_b),      # blank_tmaxs
        np.asarray([0.6] * n_b),        # c0s
        np.asarray([0.5] * n_b),        # alphas
    ]
    # Add OSN amplitude, epsilons to back params, to be updated each simul
    turbulent_back_params.append(3.0 / np.sqrt(n_s))  # Changed for each sim.
    epsils_vec = np.full(n_s, epsils_range[0])  # Changed for each sim.
    turbulent_back_params.append(epsils_vec)

    # Adjust new odor concentrations to average whiff concentration
    avg_whiff_conc = np.mean(truncexp1_average(*turbulent_back_params[4:6]))
    print("Average whiff concentration: {:.4f}".format(avg_whiff_conc))
    new_test_concs *= avg_whiff_conc

    # Compute moments of the background concentration process
    dummy_rgen = np.random.default_rng(0xd915ff2054e9e76dc76983185d14f1f9)
    conc_samples = sample_ss_conc_powerlaw(
                        *turbulent_back_params[:-2], size=int(1e5), rgen=dummy_rgen
                    )
    mean_conc = np.mean(conc_samples)
    moments_conc = np.asarray([
        mean_conc,
        np.var(conc_samples),
        np.mean((conc_samples - mean_conc)**3)
    ])

    # To adjust the OSN amplitude with epsilon, scale the max. activation 
    # at high whiff conc (1.5*average) to a standard value 5.0/sqrt(N_S)
    # This amplitude of strong whiffs affects convergence and is better scaled
    raw_conc_factor = 2.0
    raw_ampli = 2.5
    ampli_statistic = np.mean


    ### IBCM RUNS ###
    ibcm_attrs = {
        "model": "IBCM",
        "background": "turbulent_nl_osn",
        # need to save 128-bit to str, too large for HDF5
        "main_seed": str(common_seeds[0]),  # Will be changed for each sim.
        "epsilon": epsils_range[0]  # Will be changed for each sim.
    }
    ibcm_params = {
        "dimensions": dimensions_array,
        "repeats": repeats_array,
        # learnrate, tau_avg, eta, lambda, sat, ktheta, decay_relative
        "m_rates": np.asarray([0.00075, 1600.0, 0.7/n_i, 1.0, 50.0, 0.1, 0.005]),
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
    max_osn_amplitudes_scales = []
    for i, eps_i in enumerate(epsils_range):
        epsils_vec = np.full(n_s, eps_i)
        # Compute the typical OSN amplitude at this level of nonlinearity
        # scale the max_osn_amplitude to maintain a constant amplitude
        dummy_odors = generate_odor_tanhcdf((n_b, n_s), 
                            dummy_rgen, **odor_gen_arguments)
        dummy_concs = np.full(n_b, raw_conc_factor * avg_whiff_conc)
        raw_osn_activ = ampli_statistic(combine_fct(dummy_concs,
                                    dummy_odors, epsils_vec, fmax=1.0))
        max_osn_ampli = raw_ampli / (raw_osn_activ * np.sqrt(n_s)) 
        max_osn_amplitudes_scales.append(max_osn_ampli)
        turbulent_back_params[-2] = max_osn_ampli
        turbulent_back_params[-1] = epsils_vec
        ibcm_params["back_params"] = turbulent_back_params
        ibcm_attrs["main_seed"] = str(common_seeds[i])
        ibcm_attrs["epsilon"] = eps_i
        ibcm_file_name = os.path.join(folder, 
                            "ibcm_performance_results_nl_osn_{}.h5".format(i))
        all_ibcm_file_names[i] = str(ibcm_file_name)
        if do_main_runs:
            print("Starting IBCM simulation for epsilon = {}".format(eps_i))
            main_habituation_runs_nl_osn(ibcm_file_name, ibcm_attrs,
                ibcm_params, ibcm_options, odor_gen_arguments, lean=True)
            print("Starting IBCM recognition for epsilon = {}".format(eps_i))
            main_recognition_runs_nl_osn(ibcm_file_name, ibcm_attrs, ibcm_params,
                ibcm_options, projection_arguments, odor_gen_arguments, lean=True)

    ### BIOPCA RUNS ###
    # Change number of inhibitory neurons, need less with PCA
    n_i = n_b + 1  # See what an extra neuron will do
    dimensions_array = np.asarray([n_s, n_b, n_i, n_k])
    biopca_attrs = {
        "model": "PCA",
        "background": "turbulent_nl_osn",
        # Intentionally the same seed to test all models against same backs
        "main_seed": str(common_seeds[0]),  # Updated for each sim
        "epsilon": epsils_range[0]  # Will be changed for each sim.
    }
    # Adjust Lambda scale in BioPCA
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
    for i, eps_i in enumerate(epsils_range):
        epsils_vec = np.full(n_s, eps_i)
        max_osn_ampli = max_osn_amplitudes_scales[i] 
        turbulent_back_params[-2] = max_osn_ampli
        turbulent_back_params[-1] = epsils_vec
        biopca_params["back_params"] = turbulent_back_params
        biopca_attrs["main_seed"] = str(common_seeds[i])
        biopca_attrs["epsilon"] = eps_i
        pca_file_name = os.path.join(folder, 
            "biopca_performance_results_nl_osn_{}.h5".format(i))
        if do_main_runs:
            print("Starting BioPCA recognition"
                  + " for epsilon = {}".format(eps_i))
            main_habituation_runs_nl_osn(pca_file_name, biopca_attrs,
                biopca_params, biopca_options, odor_gen_arguments, lean=True)
            print("Starting BioPCA recognition"
                  +" for epsilon = {}".format(eps_i))
            main_recognition_runs_nl_osn(pca_file_name, biopca_attrs, 
                biopca_params, biopca_options, projection_arguments, 
                odor_gen_arguments, lean=True)

    ### AVERAGE INHIBITION RUNS ###
    # Change number of inhibitory neurons, need less with PCA
    n_i = 1
    dimensions_array = np.asarray([n_s, n_b, n_i, n_k])
    avg_attrs = {
        "model": "AVG",
        "background": "turbulent_nl_osn",
        # Intentionally the same seed to test all models against same backs
        "main_seed": str(common_seeds[0]),  # updated each sim
        "epsilon": epsils_range[0]  # updated each sim
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
    for i, eps_i in enumerate(epsils_range):
        epsils_vec = np.full(n_s, eps_i)
        max_osn_ampli = max_osn_amplitudes_scales[i]
        turbulent_back_params[-2] = max_osn_ampli
        turbulent_back_params[-1] = epsils_vec
        avg_params["back_params"] = turbulent_back_params
        avg_attrs["main_seed"] = str(common_seeds[i])
        avg_attrs["epsilon"] = eps_i
        avg_file_name = os.path.join(folder, 
            "avgsub_performance_results_nl_osn_{}.h5".format(i))
        if do_main_runs:
            print("Starting average sub. simulation"
                  + " for epsilon = {}".format(eps_i))
            main_habituation_runs_nl_osn(avg_file_name, avg_attrs,
                avg_params, avg_options, odor_gen_arguments, lean=True)
            print("Starting average sub. recognition "
                + "for epsilon = {}".format(eps_i))
            main_recognition_runs_nl_osn(avg_file_name, avg_attrs, avg_params,
                avg_options, projection_arguments, odor_gen_arguments,lean=True)

    ### OPTIMAL, ORTHOGONAL, AND NO INHIBITION ###
    for kind in ["optimal", "orthogonal", "none"]:
        for i, eps_i in enumerate(epsils_range):
            print("Starting idealized habituation of kind "
                +"{} recognition tests for epsilon = {}".format(kind, eps_i))
            ideal_file_name = os.path.join(folder, 
                    kind+"_performance_results_nl_osn_{}.h5".format(i))
            ibcm_fname = all_ibcm_file_names[i]
            idealized_recognition_from_runs_nl_osn(
                ideal_file_name, ibcm_fname, kind, lean=True)
