""" All-in-one script to run different simulation seeds with OSN adaptation, 
test new odor recognition for each seed, and 
Seeds are launched in parallel using multiprocessing in the main function

@author: frbourassa
August 2025
"""
import numpy as np
from scipy import sparse
from time import perf_counter
import json
from os.path import join as pj
import sys
from threadpoolctl import threadpool_limits
import multiprocessing
if "../" not in sys.path:
    sys.path.insert(1, "../")

from utils.cpu_affinity import count_parallel_cpu, count_threads_per_process
from modelfcts.ibcm import (
    integrate_ibcm_adaptation,
    ibcm_respond_new_odors,  # Unchanged if we give inputs nonlinearized with correct epsilon
)
from modelfcts.biopca import (
    integrate_biopca_adaptation,
    build_lambda_matrix,  
    biopca_respond_new_odors  # Unchanged if we give nonlinearized inputs with correct epsilon
)
# Do not consider average or idealized subtraction here
from modelfcts.ideal import (
    compute_optimal_matrix_fromsamples
)
from utils.metrics import jaccard
from modelfcts.distribs import (
    truncexp1_average,
    powerlaw_cutoff_inverse_transform,
)
# re-use functions for nonlinear OSNs, will need to put 
# updated epsilon in back_params at each step
from modelfcts.nonlin_adapt_osn import (  
    generate_odor_tanhcdf, 
    combine_odors_affinities, 
    update_powerlaw_times_concs_affinities,
)
from modelfcts.backgrounds import (  #
    sample_ss_conc_powerlaw,   # unchanged
)
from modelfcts.tagging import (  # unchanged
    project_neural_tag, 
    create_sparse_proj_mat
)



# Initialization
root_dir = pj("..")
outputs_folder = pj(root_dir, "results", "for_plots", "nonlin_adapt")
params_folder = pj(root_dir, "results", "common_params")

# Global parameters, on which local simulation run functions rely for simplicity

# Initialize common simulation parameters
n_dimensions = 50  # Fly number
n_components = 6  # Number of background odors

# Common parameters for toy and full simulations
inhib_rates = [0.00005, 0.00001]  # alpha, beta  [0.00025, 0.00005]

# Simulation duration
duration = 36000.0
deltat = 1.0

# Simulation skipping, 50 is enough for plots
skp = 50 * int(1.0 / deltat)
tser_common = np.arange(0.0, duration, deltat*skp)

# Common model options
activ_function = "identity"  #"ReLU"

# Background process
combine_fct = combine_odors_affinities
update_fct = update_powerlaw_times_concs_affinities

# Scale of affinity vectors: default
kscale = 5e-4  # default is 5e-4

# OSN target activity and epsilon ranges
target_osn_activ = np.full(n_dimensions, 1.0 / np.sqrt(n_dimensions))
adaptation_params = [
    25.0,  # tau_adapt = 250 ms
    1.0,  # eps_min, allow quite low
    10.0,  # eps_max
    target_osn_activ 
]


### Background initialization functions
def default_background_params(n_comp):
    """ Default time and concentration parameters for the turbulent process"""
    # Turbulent background parameters: same rates and constants for all odors
    back_pms_turbulent = [
        np.asarray([1.0] * n_comp),        # whiff_tmins
        np.asarray([500.] * n_comp),       # whiff_tmaxs
        np.asarray([1.0] * n_comp),        # blank_tmins
        np.asarray([800.0] * n_comp),      # blank_tmaxs
        np.asarray([0.6] * n_comp),        # c0s
        np.asarray([0.5] * n_comp),        # alphas
    ]
    return back_pms_turbulent


def initialize_back_params(adapt_params, rgen, n_comp, n_dim):
    # Turbulent background parameters: same rates and constants for all odors
    back_pms = default_background_params(n_comp)

    tau_eps, eps_min, eps_max, osn_targets = adapt_params
    epsils_vec = np.full(n_dim, 0.5 * (eps_min + eps_max))
    back_comps = generate_odor_tanhcdf((n_comp, n_dim), rgen, unit_scale=kscale)

    # To keep OSN amplitudes comparable to usual simulations, scale down OSN max. ampli
    avg_whiff_conc = np.mean(truncexp1_average(*back_pms[4:6]))

    # Same adjustment of the OSN amplitude as in the performance recognition tests
    raw_conc_factor = 2.5
    raw_ampli = 2.5
    np_statistic = np.mean  # np.mean, np.median, np.amax

    raw_osn_activ = np_statistic(combine_fct(np.full(n_comp, raw_conc_factor * avg_whiff_conc), 
                                        back_comps, epsils_vec, fmax=1.0))
    max_osn_ampli = raw_ampli / (raw_osn_activ * np.sqrt(n_dim))

    # Add these extra parameters to the list of background params
    back_pms.append(max_osn_ampli)
    back_pms.append(epsils_vec)
    back_pms.append(back_comps)

    # Initialization
    # Initial values of background process variables (t, c for each variable)
    init_concs = sample_ss_conc_powerlaw(*back_pms[:-3], size=1, rgen=rgen)
    init_times = powerlaw_cutoff_inverse_transform(
                    rgen.random(size=n_comp), *back_pms[2:4])
    tc_init = np.stack([init_times, init_concs.squeeze()], axis=1)

    # Initial background vector: combine odors with the tc_init concentrations
    init_bkvec = combine_fct(tc_init[:, 1], back_comps, epsils_vec, fmax=max_osn_ampli)
    # nus are first in the list of initial background params
    init_back = [tc_init, init_bkvec]

    return back_pms, init_back


### Function to run and clean a simulation
# Uses global IBCM parameters defined above. 
def run_ibcm_simulation_adapt(adapt_params, ibcm_rates, ibcm_options, 
            lambd, n_comp, n_dim, rgenseed, simseed, skp_local=skp, n_i=None):
    print("Initializing IBCM simul. for adapt_params[:3] =", adapt_params[:3])
    # Initialize background with the random generator with seed rgenseed
    rgen = np.random.default_rng(rgenseed)
    res = initialize_back_params(adapt_params, rgen, n_comp, n_dim)
    back_params_local, init_back = res
    if n_i is None:
        n_i = n_comp * 4
    # Initial synaptic weights: small positive noise
    init_synapses_ibcm = 0.2*rgen.standard_normal(size=[n_i, n_dim])*lambd

    # Run the IBCM simulation
    print("Starting IBCM simulation...")
    tstart = perf_counter()
    sim_results = integrate_ibcm_adaptation(
                init_synapses_ibcm, update_fct, init_back, 
                ibcm_rates, inhib_rates, back_params_local, 
                adapt_params, duration, deltat, seed=simseed, 
                noisetype="uniform",  skp=skp_local, **ibcm_options
    )
    tend = perf_counter()
    print("Finished IBCM simulation in {:.2f} s".format(tend - tstart))

    return back_params_local, sim_results


# Cleaning function
def analyze_clean_ibcm_simul_lean(results_raw, back_pms, t_mix=-1):
    """
    Args:
        results_raw = (tser_ibcm, nuser_ibcm, bkvecser_ibcm, mser_ibcm, 
            cbarser_ibcm, thetaser_ibcm, wser_ibcm, yser_ibcm)
    Returns:
        bkvecser_ibcm, eps_ser, conc_ser
    """
    (_, nuser_ibcm, bkvecser_ibcm, eps_ser, _, _, _, _, _) = results_raw

    # Moments of concentrations
    conc_ser = nuser_ibcm[:, :, 1]
    results_clean = (bkvecser_ibcm, eps_ser, conc_ser)
    return results_clean



### Function to run a BioPCA simulation
# Uses global BioPCA parameters defined above. 

def run_biopca_simulation_adapt(adapt_params, biopca_rates, pca_options, 
                    n_comp, n_dim, n_i, rgenseed, simseed, skp_local=skp):
    print("Initializing BioPCA simulation for adapt_params[:3] =", adapt_params[:3])
    # Initialize background parameters, give same rgenseed as IBCM to have same background
    rgen = np.random.default_rng(rgenseed)
    res = initialize_back_params(adapt_params, rgen, n_comp, n_dim)
    back_params_local, init_back = res

    init_mmat_pca = rgen.standard_normal(size=[n_i, n_dim]) / np.sqrt(n_dim)
    init_lmat_pca = np.eye(n_i, n_i)  # Supposed to be near-identity, start as identity
    ml_inits_pca = [init_mmat_pca, init_lmat_pca]

    # Run the IBCM simulation
    print("Starting BioPCA simulation...")
    tstart = perf_counter()
    sim_results = integrate_biopca_adaptation(
                ml_inits_pca, update_fct, init_back, biopca_rates, 
                inhib_rates, back_params_local, adapt_params, duration, deltat, 
                seed=simseed, noisetype="uniform", skp=skp_local, **pca_options
    )
    tend = perf_counter()
    print("Finished BioPCA simulation in {:.2f} s".format(tend - tstart))

    return back_params_local, sim_results


### Optimal $P$ linear manifold learning matrix
# Compute average across a background time series, 
# using the adapted $\epsilon(t)$ value going with each background sample. 

def mix_new_back_adapt(back_odors, new_odors, cser, newconc, fmax, eps_ser):
    n_new = new_odors.shape[0]
    assert n_new == cser.shape[0]  # one new odor per back sample
    all_mixvecs = []
    for n in range(n_new):
        joint_concs = np.concatenate([cser[n], np.full(1, newconc)])
        joint_components = np.concatenate(
            [back_odors, new_odors[n:n+1]], axis=0)
        mixvecs = combine_fct(joint_concs, 
                    joint_components, eps_ser[n], fmax=fmax)
        all_mixvecs.append(mixvecs)
    mixvecs = np.stack(all_mixvecs, axis=0)
    return mixvecs


def get_optimal_mat_p(bkvecser, concser, eps_ser, back_pms, new_concs_rel, 
                      sd=0xdf8cc55ff9195d82ed83ae87ba4e10fc):
    """ Compute the optimal linear manifold learning matrix P, 
    using a previously simulated background"""
    avg_whiff_conc = np.mean(truncexp1_average(*back_pms[4:6]))
    new_concs = avg_whiff_conc * new_concs_rel
    osn_ampli = back_pms[-3]
    back_comp = back_pms[-1]

    # Compute optimal W matrix for all new odors possible
    # Need samples from the background (use provided bkser)
    # and samples from mixtures of background + new odor
    # (generate from back. conc. series in nuser_ibcm)
    dummy_rgen = np.random.default_rng(sd)
    # New odors, each with a subset of the background samples
    n_samp, n_dims = bkvecser.shape[0], bkvecser.shape[1]
    new_odors_from_distrib = generate_odor_tanhcdf(
        [n_samp, n_dims], dummy_rgen, unit_scale=kscale)

    optimal_matrices = []
    for newconc in new_concs:
        # Mix new odors at newconc with background
        s_new_mix = mix_new_back_adapt(back_comp, new_odors_from_distrib, 
                                 concser, newconc, osn_ampli, eps_ser)
        mat = compute_optimal_matrix_fromsamples(bkvecser, s_new_mix)
        optimal_matrices.append(mat)

    return optimal_matrices


### New odor recognition functions

def find_snap_index(dt, skip, times):
    """ Find nearest multiple of dt*skip to each time in times """
    return np.around(times / (dt*skip)).astype(int)


# For IBCM and BioPCA
def test_odor_recognition_adaptation_lean(
        back_pms, new_od_kmats, ibcm_res, biopca_res, 
        opts_ibcm, opts_biopca, proj_mat, proj_args,
        rates_ibcm, rates_pca, optim_mat, 
        new_conc_rel=1.0, n_test_t=100, n_new=100):
    # Select test times, etc.
    bk_comp = back_pms[-1]
    n_comp, n_dim = bk_comp.shape[0], bk_comp.shape[1]
    n_kc = proj_mat.shape[0]

    # New odors tested
    new_conc = new_conc_rel * np.mean(truncexp1_average(*back_pms[4:6]))

    # Load background and epsilon series, 
    # assert it's the same background for both models
    conc_ser = ibcm_res[1][:, :, 1]  # concentrations
    conc_ser_pca = biopca_res[1][:, :, 1]
    assert np.allclose(conc_ser, conc_ser_pca)
    eps_ser = ibcm_res[3]
    eps_ser_pca = biopca_res[3]
    assert np.allclose(eps_ser, eps_ser_pca)

    # Time series of relevant weights
    mser_ibcm = ibcm_res[4]
    wser_ibcm = ibcm_res[7]
    mser_pca = biopca_res[4]
    lser_pca = biopca_res[5]
    xser_pca = biopca_res[6]
    wser_pca = biopca_res[8]

    # Reference tags are computed for the average epsilon of each OSN
    default_eps = np.mean(eps_ser, axis=0)  # average epsilon for each OSN
    single_mean_eps = np.full(n_dim, np.mean(default_eps))
    osn_ampli = back_pms[-3]

    # OSN response to new odors and back odors at average conc.
    new_odor_responses = np.stack([
        combine_fct(np.asarray([new_conc]), new_od_kmats[i:i+1], 
                    default_eps, fmax=osn_ampli) for i in range(n_new)
    ])

    # Test times, based on global parameters (duration, deltat, skp)
    start_test_t = duration - 30000.0
    test_times = np.linspace(start_test_t, duration, n_test_t)
    test_times -= deltat*skp
    test_idx = find_snap_index(deltat, skp, test_times)

    # Containers for y vectors of each model in response to the mixture
    models = ["none", "adapt", "optimal_adapt", "biopca_adapt", "ibcm_adapt"]
    mixture_yvecs = {a: np.zeros([n_new, n_test_t, n_dim]) 
                    for a in models}
    jaccard_scores = {a: np.zeros([n_new, n_test_t]) 
                      for a in models}

    # Assess recognition of new odors mixed non-linearly
    for i in range(n_new):
        # Compute neural tag of the new odor alone, without inhibition
        new_tag = project_neural_tag(
                        new_odor_responses[i], new_odor_responses[i],
                        proj_mat, **proj_args
                    )

        # Now, loop over snapshots, mix the new odor with the back samples,
        # compute the PN response at each test concentration,
        # compute tags too, and save results
        # Combine background and new odor i's parameters into one joint K matrix
        # to use in the combine_fct. 
        joint_odor_kmats = np.concatenate([bk_comp, new_od_kmats[i:i+1]], axis=0)
        for j in range(n_test_t):
            current_eps = eps_ser[j]
            jj = test_idx[j]
            joint_conc_samples = np.concatenate(
                [conc_ser[j], np.full(1, new_conc)], axis=0)
            mixture = combine_fct(joint_conc_samples, joint_odor_kmats, 
                                   current_eps, fmax=osn_ampli)
            mixture_noadapt = combine_fct(joint_conc_samples, joint_odor_kmats, 
                                   single_mean_eps, fmax=osn_ampli)
            # odors, mlx, wmat, 
            # Compute for each model
            mixture_yvecs["ibcm_adapt"][i, j] = ibcm_respond_new_odors(
                mixture, mser_ibcm[jj], wser_ibcm[jj], 
                rates_ibcm, options=opts_ibcm
            )
            mixture_yvecs["biopca_adapt"][i, j] = biopca_respond_new_odors(
                mixture, [mser_pca[jj], lser_pca[jj], xser_pca[jj]], 
                wser_pca[jj], rates_pca, options=opts_biopca
            )
            mixture_yvecs["adapt"][i, j] = mixture
            mixture_yvecs["none"][i, j] = mixture_noadapt
            mixture_yvecs["optimal_adapt"][i, j] = mixture - optim_mat.dot(mixture)
            #mixture_yvecs["orthogonal"][i, j] = mixtures - mixtures.dot(back_projector.T)
            for mod in mixture_yvecs.keys():
                mix_tag = project_neural_tag(
                    mixture_yvecs[mod][i, j], mixture,
                    proj_mat, **proj_args
                )
                jaccard_scores[mod][i, j] = jaccard(mix_tag, new_tag)
    return jaccard_scores  #, jaccard_backs, mixture_tags, new_odor_tags


### Main simulations 
# Launch several of these main run functions in parallel,
# Repeat for different new odor concentrations if desired. 
def main_one_run_seed(main_seed, simul_seed, new_rel_conc):
    """ Run a habituation simulation and new odor recognition for IBCM, BioPCA,
        optimal P; collect the Jaccard similarities at the end for a quick
        assessment of model performance with predefined model params. 
    """

    ### IBCM habituation and simulation parameters
    # IBCM model parameters, same for each tested epsilon
    n_i_ibcm = 24  # Number of inhibitory neurons for IBCM case

    # Model rates
    learnrate_ibcm = 0.001  #5e-5
    tau_avg_ibcm = 1600  # 2000
    coupling_eta_ibcm = 0.7/n_i_ibcm
    ssat_ibcm = 50.0
    k_c2bar_avg = 0.5
    decay_relative_ibcm = 0.005
    lambd_ibcm = 1.0
    ibcm_rates = [
        learnrate_ibcm, 
        tau_avg_ibcm, 
        coupling_eta_ibcm, 
        lambd_ibcm,
        ssat_ibcm, 
        k_c2bar_avg,
        decay_relative_ibcm 
    ]
    ibcm_options = {
        "activ_fct": activ_function, 
        "saturation": "tanh", 
        "variant": "law", 
        "decay": True
    }

    ### BioPCA habituation and simulation parameters
    # BioPCA model parameters, same for all epsilons
    n_i_pca = n_components * 2  # Number of inhibitory neurons for BioPCA case

    # Model rates
    learnrate_pca = 1e-4  # Learning rate of M
    # Choose Lambda diagonal matrix as advised in Minden et al., 2018
    # but scale it up to counteract W regularization
    lambda_range_pca = 0.5
    lambda_max_pca = 9.0
    # Learning rate of L, relative to learnrate. 
    # Adjusted to Lambda in the integration function
    rel_lrate_pca = 2.0  #  / lambda_max_pca**2 

    xavg_rate_pca = learnrate_pca
    pca_options = {
        "activ_fct": activ_function, 
        "remove_lambda": False, 
        "remove_mean": True
    }
    biopca_rates = [learnrate_pca, rel_lrate_pca, lambda_max_pca, 
                    lambda_range_pca, xavg_rate_pca]

    # IBCM
    back_ibcm, res_ibcm = run_ibcm_simulation_adapt(adaptation_params, 
                    ibcm_rates, ibcm_options, lambd_ibcm, n_components,  
                    n_dimensions, main_seed, simul_seed, n_i=n_i_ibcm)
    res_ibcm_clean = analyze_clean_ibcm_simul_lean(res_ibcm, back_ibcm)
    
    # BioPCA
    _, res_biopca = run_biopca_simulation_adapt(adaptation_params, 
                        biopca_rates, pca_options, n_components, 
                        n_dimensions, n_i_pca, main_seed, simul_seed)

    # Generate new odors, projection matrix, etc. 
    n_new_od = 100
    rgen_od = np.random.default_rng(main_seed.spawn(2).pop(1))
    new_odors_kmats = generate_odor_tanhcdf([n_new_od, n_dimensions], rgen_od, 
                                            unit_scale=kscale)

    # Common parameters
    n_kc = 1000 * n_dimensions // 25
    projection_arguments = {
        "kc_sparsity": 0.05,
        "adapt_kc": True,
        "n_pn_per_kc": 3 * n_dimensions // 25,
        "project_thresh_fact": 0.1
    }
    proj_matrix = create_sparse_proj_mat(n_kc, n_dimensions, rgen_od)


    # Obtain optimal matrix for new concentration = avg whiff
    # cleaned up results: bkvecser_ibcm, eps_ser, conc_ser
    bkser_ibcm = res_ibcm_clean[0]
    eps_ser_ibcm = res_ibcm_clean[1]
    concser_ibcm = res_ibcm_clean[2]
    optimal_matrix = get_optimal_mat_p(bkser_ibcm, concser_ibcm, 
                            eps_ser_ibcm, back_ibcm, np.ones(1))[0]

    # Run odor recognition tests for 
    jaccard_scores = test_odor_recognition_adaptation_lean(
                    back_ibcm, new_odors_kmats, res_ibcm, res_biopca, 
                    ibcm_options, pca_options, proj_matrix, projection_arguments,
                    ibcm_rates, biopca_rates, optimal_matrix, 
                    new_conc_rel=new_rel_conc, n_test_t=100, n_new=n_new_od
    )

    return jaccard_scores


def stack_jaccards_per_model(all_jac_dicts):
    # Check we have at least one simul.
    if len(all_jac_dicts) <= 0: 
        raise ValueError("No simulation saved!")
    # Get the differet model names: each list element is a dict with model 
    # names as keys, and a 2D array of jaccard scores in each item. 
    models = set(all_jac_dicts[0].keys())
    stacked_jacs = {m:[] for m in models}
    # Regroup Jaccards across simulations for each model
    for i in range(len(all_jac_dicts)):
        models_i = set(all_jac_dicts[i].keys())
        # Check that all simuls have the same models
        if models_i != models:
            raise ValueError(f"Simul. {i} and 0 have different models")
        for m in models:
            stacked_jacs[m].append(all_jac_dicts[i].get(m))
    # Turn the lists of 2D arrays into a 3D array for each model
    for m in models:
        stacked_jacs[m] = np.stack(stacked_jacs[m])
    return stacked_jacs
    


def func_wrapper_threadpool(func, threadlim, *args, **kwargs):
    with threadpool_limits(limits=threadlim, user_api='blas'):
        res = func(*args, **kwargs)
    return res


if __name__ == "__main__":
    n_backgrounds = 96
    new_od_rel_conc = 0.5

    # Prepare main seeds for each simulation
    super_seed = 0xe5373126ca83a674edcf704c59de3a1b
    spawn_seed = 0x98792b1f2199475f6058f3ee25e642b4
    seed_seq = np.random.SeedSequence(super_seed, spawn_key=(spawn_seed,))
    main_seed_list = seed_seq.spawn(n_backgrounds)

    # Also simulation seeds 
    super_seed2 = 0x7a3b3b5359122607281a287f69b333ca
    spawn_seed2 = 0x44d0a74961ecfc66a0701d6035e8a4e
    seed_seq2 = np.random.SeedSequence(super_seed2, spawn_key=(spawn_seed2,))
    simul_seed_list = seed_seq2.spawn(n_backgrounds)
    
    # Make each simulation single-core to avoid collisions between them
    n_workers = min(count_parallel_cpu(), n_backgrounds)
    n_threads = count_threads_per_process(n_workers)
    pool = multiprocessing.Pool(n_workers)
    all_processes = []
    for i in range(n_backgrounds):
        # Apply arguments: function to apply, # threads per process,
        # then arguments to the function to apply
        applyargs = (main_one_run_seed, n_threads, 
                      main_seed_list[i], simul_seed_list[i], new_od_rel_conc)
        proc = pool.apply_async(func_wrapper_threadpool, args=applyargs)
        all_processes.append(proc)
    
    # Recover results and combine all Jaccard scores for all models
    all_simuls_jaccards = [p.get() for p in all_processes]
    all_jaccards = stack_jaccards_per_model(all_simuls_jaccards)

    # Save to disk as a npz archive
    fname = "osn_adaptation_odor_recognition_results.npz"
    np.savez_compressed(pj(outputs_folder, fname), **all_jaccards)



