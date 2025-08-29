#!/usr/bin/env python
# coding: utf-8

r""" All-in-one script to run different BioPCA simulation seeds for a variety
of background odor numbers, background concentration scale, and background
whiff and blank time scales, to assess convergence as a function of 
these parameters. Seeds are launched in parallel using multiprocessing. 

We do not explore as much the effect of relative parameters within the BioPCA
model, such as $\mu_L$ (we keep it = 2 by default), the mean averaging rate 
(we keep it equal to $\mu$), or the range of $\Lambda$ values (we keep it 
equal to 0.5). We might explore some of these to quantify how convergence
time scales with these, but here we focus on robustness against turbulence. 

@author: frbourassa
August 2025
"""

import numpy as np
import os, json
from os.path import join as pj
import sys
if ".." not in sys.path:
    sys.path.insert(1, "..")

from utils.cpu_affinity import count_parallel_cpu, count_threads_per_process
import multiprocessing
from simulfcts.idealized_recognition import func_wrapper_threadpool
from modelfcts.biopca import (
    integrate_inhib_biopca_network_skip,
    build_lambda_matrix,
    biopca_respond_new_odors
)
from modelfcts.checktools import (
    analyze_pca_learning, 
    check_conc_samples_powerlaw_exp1
)
from modelfcts.backgrounds import (
    update_powerlaw_times_concs, 
    sample_ss_conc_powerlaw, 
    generate_odorant,
    mean_turbulent_concs
)
from modelfcts.distribs import (
    powerlaw_cutoff_inverse_transform
)
from utils.metrics import l2_norm


### Background initialization functions
def linear_combi(concs, backs):
    """ concs: shaped [..., n_odors]
        backs: 2D array, shaped [n_odors, n_osn]
    """
    return concs.dot(backs)

# Global choice of background and odor mixing functions
update_fct = update_powerlaw_times_concs
combine_fct = linear_combi


# We will later explore the effect of varying these parameters on the
# convergence, but put the default ones in a function
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


# Background initialization, given parameters and a seeded random generator
def initialize_given_background(back_pms, rgen, n_comp, n_dim):
    # Initial values of background process variables (t, c for each variable)
    init_concs = sample_ss_conc_powerlaw(*back_pms[:-1], size=1, rgen=rgen)
    init_times = powerlaw_cutoff_inverse_transform(
                    rgen.random(size=n_comp), *back_pms[2:4])
    tc_init = np.stack([init_times, init_concs.squeeze()], axis=1)

    # Initial background vector: combine odors with the tc_init concentrations
    back_comps = back_pms[-1]
    init_bkvec = combine_fct(tc_init[:, 1], back_comps)
    # background random variables are first in the list of initial values
    init_back = [tc_init, init_bkvec]

    return init_back


### IBCM simulation functions

# Analysis to establish convergence to specific fixed points. 
# Example of analyses to do on BioPCA results
def analyze_clean_biopca_simul(results_raw, pca_options, lambda_mat_diag):
    """
    We do not need to save odor vectors (back_components), 
    since the IBCM simulation will provide them for both models. 
    
    Args:
        results_raw = (tser_pca, nuser_pca, bkvecser_pca, mser_pca, 
            lser_pca, xser_pca, hbarser_pca, wser_pca, yser_pca)
    Returns:
        bkvecser_pca, ysernorm_pca, wser_pca, true_pca, 
            learned_pca, off_diag_l_avg_abs, align_error_ser)
    """
    (tser_pca, nuser_pca, bkvecser_pca, mser_pca, lser_pca, _, 
         _, _, yser_pca) = results_raw
    
    # Analyze versus true offline PCA of the background samples
    res = analyze_pca_learning(bkvecser_pca, mser_pca, lser_pca, 
                           lambda_mat_diag, demean=pca_options["remove_mean"])
    true_pca, learned_pca, _, off_diag_l_avg_abs, align_error_ser = res
    
    # Average alignment error in the last 15 minutes
    transient = int(3 * tser_pca.shape[0] // 4)
    align_error = np.mean(align_error_ser[transient:], axis=0)
    
    # True principal values (all N_S of them for reference) and average learned ones
    pvals_true = true_pca[0]
    pvals_avg_learned = learned_pca[0][transient:].mean(axis=0)
    
    # Variance of the eigenvalues (L diagonal) series
    pvals_vari = np.var(learned_pca[0][transient:], axis=0, ddof=1)

    # For the present parallel simulations, only return summary stats, 
    # not entire time series. True and learned average principal values, 
    # their variance in the last 15 minutes, and the average alignment
    # error (metric for the learned principal vectors), 
    
    results_clean = (pvals_true, pvals_avg_learned, pvals_vari, align_error)
    return results_clean


def run_analyze_biopca_one_back_seed(
        biopca_rates_loc, back_rates, inhib_rates_loc, 
        options_loc, dimensions, seedseq, 
        duration_loc=360000.0, dt_loc=1.0, skp_loc=20, full_returns=False
    ):
    """ Given BioPCA model rates and background parameters except
    background odors (but incl. number odors and c0), and a main seed sequence, 
    run and analyze convergence of BioPCA on the background generated from that seed. 
    The seedseq should itself have been spawned from a root seed to have a distinct
    one per run; this still makes seeds reproducible yet distinct for different runs. 
    The seedseq here is spawned again for a background gen. seed and a simul. seed. 
    
    Args:
        dimensions: gives [n_components, n_dimensions, n_i_ibcm]
    
    Returns:
        iff full_return:
            gaps, specifs, hgamvari, hgammas_ser, sim_results
        else:
            gaps, specifs, hgamvari, None, None
        alignment_gaps: indexed [neuron]
        specif_gammas: indexed [neuron]
        gamma_vari: indexed [neuron, component]
    """
    #print("Initializing IBCM simulation...")
    # Get dimensions
    n_comp, n_dim, n_i_pca = dimensions
    
    # Spawn back. generation seed and simul seed
    initseed, simseed = seedseq.spawn(2)
    
    # Duplicate back params before appending locally-generated odor vectors to them
    back_pms_loc = list(back_rates)
    
    # Create background
    rgen_init = np.random.default_rng(initseed)
    back_comps_loc = generate_odorant((n_comp, n_dim), rgen_init)
    back_comps_loc = back_comps_loc / l2_norm(back_comps_loc, axis=1)[:, None]

    # Add odors to the list of background parameters
    back_pms_loc.append(back_comps_loc)

    # Initialize background with the random generator with seed rgenseed
    rgen_init = np.random.default_rng(initseed)
    init_back = initialize_given_background(back_pms_loc, rgen_init, n_comp, n_dim)

    # Initial synaptic weights: small positive noise
    init_mmat_pca = rgen_init.standard_normal(size=[n_i_pca, n_dim]) / np.sqrt(n_dim)
    init_lmat_pca = np.eye(n_i_pca, n_i_pca)  # Supposed to be near-identity, start as identity
    ml_inits_pca = [init_mmat_pca, init_lmat_pca]
    
    # Run the BioPCA simulation
    sim_results = integrate_inhib_biopca_network_skip(
                ml_inits_pca, update_fct, init_back, 
                biopca_rates_loc, inhib_rates_loc, back_pms_loc,
                duration_loc, dt_loc, seed=simseed, 
                noisetype="uniform",  skp=skp_loc, **options_loc
    )

    # For analysis, rebuild the Lambda matrix
    lambda_max_pca, lambda_range_pca = biopca_rates_loc[2:4]
    lambda_mat_diag = build_lambda_matrix(
                        lambda_max_pca, lambda_range_pca, n_i_pca)

    # Now analyze BioPCA simul for convergence
    results_clean = analyze_clean_biopca_simul(
                    sim_results, options_loc, lambda_mat_diag)
    (pvals_true, pvals_avg_learned, pvals_vari, align_error) = results_clean

    # Doesn't return full series, only the summary statistics of convergence
    analysis_results_ret = (pvals_true, pvals_avg_learned, pvals_vari, align_error)
    if full_returns:
        sim_results_ret = sim_results
    else:
        sim_results_ret = None
    
    return analysis_results_ret, sim_results_ret


### Multiprocessed main functions

def combine_seed_results(res_dict, n_seeds):
    """ Combine convergence analysis results of simulation seeds
    in res_dict. """
    combi_true, combi_learned, combi_varis, combi_err = [], [], [], []
    for i in range(n_seeds):
        (pvals_true, pvals_learn, pvals_vari, align_err) = res_dict[i][0]
        combi_true.append(pvals_true)
        combi_learned.append(pvals_learn)
        combi_varis.append(pvals_vari)
        combi_err.append(align_err)
    return [np.stack(a) for a in 
            [combi_true, combi_learned, combi_varis, combi_err]]


def adjust_alpha(back_pms, back_pms_ref, alpha_ref):
    r""" The magnitude of h scales as 1/c_0 where c_0 is the concentration
    scale, so we need to adjust $\alpha$ in the $W$ equation to prevent
    numerical instabilities: keep $\alpha/c_0$ constant. """
    c0ref = back_pms_ref[4].mean()
    c0 = back_pms[4].mean()
    alpha = alpha_ref * c0 / c0ref
    return alpha


def main_convergence_vs_background_ampli(orig_seedseq, n_seeds, n_comp_sim=3):
    """ Run n_seeds IBCM simulations for each combination of mu learning rate
    and background amplitude, show that we get excellent compensation by 
    keeping mu * <c>^2 constant, so mu*c0**2 constant here. Make sure
    to include this quadratic diagonal in the grid of tested values. 

    Since this is a main function, the rates grid and other model parameters
    are defined within. 

    Args:
        orig_seedseq (np.random.SeedSequence): fresh SeedSequence from which
            all other simulation seeds will be spawned. 
        n_comp_sim (int): number of background odor components

    Returns:
        learnrate_c0_grid (np.ndarray): indexed [2, mu_idx, c0_idx]
            i.e. the first array along axis 0 contains the 2d grid of mu vals, 
            the 2nd array contains the 2d grid of c0 values, and in each grid,
            mu varies along axis 0 (rows), c0 varies along axis 1 (columns); 
            the result of np.meshgrid(murange, c0range, indexing="ij"
        all_gaps (np.ndarray): indexed [mu, c0, seed, neuron]
        all_specifs (np.ndarray): indexed [mu, c0, seed, neuron]
        all_varis (np.ndarray): indexed [mu, c0, seed, neuron, component]
    """
    # Grid of c0, this only affects convergence logarithmically, 
    # so vary both c_0 and the learning rate a lot and independently. 
    c0range = np.geomspace(0.6/16, 0.6*16, 5)
    c0_default = 0.6
    mu_default = 1e-4
    # For mu range, use constant mu*c0**2. 
    learnrate_range = np.geomspace(1e-4/25, 1e-4*25, 9)
    learnrate_c0_grid = np.stack(
        np.meshgrid(learnrate_range, c0range, indexing="ij"), axis=0)
    # learnrate varies on axis 0 (y, rows), c0 on axis 1 (x, columns)

    # Define simulation and model parameters
    # For simplicity, let number of neurons = number of components
    n_i_pca_sim = n_comp_sim
    n_dims_sim = 25
    dimensions_sim = [n_comp_sim, n_dims_sim, n_i_pca_sim]

    # Default model rates
    learnrate_pca_ref = 1e-4
    learnrate_pca_sim = learnrate_pca_ref  # Learning rate of M, will change
    # Choose Lambda diagonal matrix as advised in Minden et al., 2018
    lambda_range_pca = 0.3
    lambda_max_pca = 9.0
    # Learning rate of L, relative to learnrate. Adjusted to Lambda in the integration function
    rel_lrate_pca = 3.0
    lambda_mat_diag = build_lambda_matrix(lambda_max_pca, 
                            lambda_range_pca, n_i_pca_sim)

    xavg_rate_pca = learnrate_pca_ref  # will not change
    pca_options_sim = {
        "activ_fct": "identity", 
        "remove_lambda": False, 
        "remove_mean": True
    }
    biopca_rates_sim = [learnrate_pca_sim, rel_lrate_pca, 
                lambda_max_pca, lambda_range_pca, xavg_rate_pca]
    
    # default turbulent background parameters
    back_rates_sim = default_background_params(n_comp_sim)
    # Will vary c0, which is the element at index 4 of this param list
    
    # Default alpha, beta. 
    # Need lower alpha for larger c0, to prevent numerical instabilities.
    back_rates_ref = default_background_params(n_comp_sim)  # reference
    alpha_def = 5e-5  # needs to be adjusted to h scaling as 1/c_0
    inhib_rates_sim = [alpha_def, 1e-5]  # alpha, beta

    # Time parameters
    duration_sim = 36000.0
    deltat_sim = 1.0
    skp_sim = 20

    # Containers for true PVs, learned PVs, PV variance, and align error
    # that will be stacked arrays indexed [mu, c0, seed, ...]
    all_true, all_learn, all_varis, all_err = [], [], [], []

    # Spawn simulation seeds, reused at each combination on the IBCM rates grid
    simul_seeds = orig_seedseq.spawn(n_seeds)
    n_workers = min(count_parallel_cpu(), n_seeds)
    n_threads = count_threads_per_process(n_workers)
    pool = multiprocessing.Pool(n_workers)

    # Treat one rate combination at a time
    for i in range(learnrate_range.shape[0]):
        mu = learnrate_range[i]
        i_true, i_learn, i_varis, i_err = [], [], [], []
        for j in range(c0range.shape[0]):
            c0 = c0range[j]
            # x averaging kept constant, allow to be different from M rate mu
            biopca_rates_sim[0] = mu
            back_rates_sim[4][:] = c0
            # Adjust alpha to h magnitude scaling as 1/c_0
            alpha_sim = adjust_alpha(back_rates_sim, back_rates_ref, alpha_def)
            inhib_rates_sim[0] = alpha_sim
            # Launch multiple seeds for the current (mu, c0) combination
            all_procs_muc0 = {}
            for k in range(n_seeds):
                apply_args = (run_analyze_biopca_one_back_seed, n_threads, 
                              biopca_rates_sim, back_rates_sim, inhib_rates_sim,  
                              pca_options_sim, dimensions_sim, simul_seeds[k])
                apply_kwds = dict(duration_loc=duration_sim, dt_loc=deltat_sim, 
                                    skp_loc=skp_sim, full_returns=False)
                all_procs_muc0[k] = pool.apply_async(func_wrapper_threadpool, 
                                 args=apply_args, kwds=apply_kwds)

            # Collect convergence analysis results for this mu, c0
            res_seeds_muc0 = {k:all_procs_muc0[k].get() for k in 
                                all_procs_muc0.keys()}
            combined_seed_res = combine_seed_results(res_seeds_muc0, n_seeds)
            i_true.append(combined_seed_res[0])
            i_learn.append(combined_seed_res[1])
            i_varis.append(combined_seed_res[2])
            i_err.append(combined_seed_res[3])  # alignment error

            print("Finished mu i = {}, c0 j = {}".format(i, j))

        # Stack arrays over j (c0) for the current i value (mu)
        all_true.append(np.stack(i_true))
        all_learn.append(np.stack(i_learn))
        all_varis.append(np.stack(i_varis))
        all_err.append(np.stack(i_err))

    # Stack arrays over i (mu)
    all_true = np.stack(all_true)
    all_learn = np.stack(all_learn)
    all_varis = np.stack(all_varis)
    all_err = np.stack(all_err)

    # Reuse pool for each (mu, c0) but close them at the end
    pool.close()
    pool.join()

    return learnrate_c0_grid, all_true, all_learn, all_varis, all_err


def main_convergence_vs_turbulence_strength(
        orig_seedseq, n_seeds, n_comp_sim=3, do_adjust_mu=False):
    """ Run n_seeds IBCM simulations for each combination of whiff and
    blank max duration. Optionally adjust the learning rate mu to compensate
    for the change in background magnitude, keeping mu * <c>**2 constant. 

    We change tw, tb for all odors the same; exploring different time
    statistics for different odors could be follow-up work. 

    Since this is a main function, the rates grid and other model parameters
    are defined within. 

    Args:
        orig_seedseq (np.random.SeedSequence): fresh SeedSequence from which
            all other simulation seeds will be spawned. 
        n_seeds (int): number of seeds
        n_comp_sim (int): number of background odor components
        do_adjust_mu (bool, default True): if True, adjust learning
            rate for each t_b, t_w combination to keep mu * <c>**2 constant

    Returns:
        tw_tb_grid (np.ndarray): indexed [2, tw_idx, tb_idx],
            the first value is the whiff max duration tw, the other
            is the blank max duration tb;
            the result of np.meshgrid(twrange, tbrange, indexing="ij"
        all_gaps (np.ndarray): indexed [tw, tb, seed, neuron]
        all_specifs (np.ndarray): indexed [tw, tb, seed, neuron]
        all_varis (np.ndarray): indexed [tw, tb, seed, neuron, component]
    """
    raise NotImplementedError()
    # Grid of tw and tb durations
    tw_range = 2.0 * np.logspace(1, 4, 7) # From 0.2 s to 200 s
    tb_range = 2.0 * np.logspace(1, 4, 7) # From 0.2 s to 200 s
    tw_tb_grid = np.stack(
        np.meshgrid(tw_range, tb_range, indexing="ij"), axis=0)
    # tw varies on axis 0 (y, rows), tb on axis 1 (x, columns)

    # Define simulation and model parameters
    n_i_ibcm_sim = 24
    n_dims_sim = 25
    dimensions_sim = [n_comp_sim, n_dims_sim, n_i_ibcm_sim]

    # Default IBCM model rates
    mu_ref = 1.25e-3  # reference
    mu = mu_ref  # will be adjusted if do_adjust_mu
    tau_avg_ibcm_sim = 2000
    coupling_eta_ibcm_sim = 0.6/n_i_ibcm_sim
    ssat_ibcm_sim = 50.0
    k_c2bar_avg_sim = 0.1
    decay_relative_ibcm_sim = 0.005
    lambd_ibcm_sim = 1.0
    ibcm_rates_sim = [
        mu, 
        tau_avg_ibcm_sim, 
        coupling_eta_ibcm_sim, 
        lambd_ibcm_sim,
        ssat_ibcm_sim, 
        k_c2bar_avg_sim,
        decay_relative_ibcm_sim
    ]
    ibcm_options_sim = {
        "activ_fct": "identity",
        "saturation": "tanh", 
        "variant": "law",   # maybe we will want to test "intrator" later?
        "decay": True
    }
    # default turbulent background parameters
    back_rates_ref = default_background_params(n_comp_sim)  # reference
    back_rates_sim = default_background_params(n_comp_sim)
    
    # Need lower alpha for larger c0, to prevent numerical instabilities.
    alpha_def = 5e-5  # needs to be adjusted to h scaling as 1/c_0
    inhib_rates_sim = [alpha_def, 1e-5]  # alpha, beta

    # Time parameters
    duration_sim = 360000.0
    deltat_sim = 1.0
    skp_sim = 20

    # Containers for alignment gaps, specificities, and hgammas variances
    # that will be stacked arrays indexed [mu, tau, seed, ...]
    all_gaps, all_specifs, all_varis = [], [], []
    all_gaps_th = []  # theoretical gaps, analytical prediction for h_sp - h_ns

    # Spawn simulation seeds, reused at each combination on the IBCM rates grid
    simul_seeds = orig_seedseq.spawn(n_seeds)
    n_workers = min(count_parallel_cpu(), n_seeds)
    n_threads = count_threads_per_process(n_workers)
    pool = multiprocessing.Pool(n_workers)

    # Treat one rate combination at a time
    for i in range(tw_range.shape[0]):
        tw = tw_range[i]
        i_gaps, i_specifs, i_varis, i_gaps_th = [], [], [], []
        for j in range(tb_range.shape[0]):
            tb = tb_range[j]
            back_rates_sim[1][:] = tw  # max whiff durations
            back_rates_sim[3][:] = tb  # max back durations
            # Adjust alpha to h magnitude scaling as 1/c_0
            alpha_sim = adjust_alpha(back_rates_sim, back_rates_ref, alpha_def)
            inhib_rates_sim[0] = alpha_sim
            if do_adjust_mu:
                mu = adjust_learnrate(back_rates_sim, back_rates_ref, mu_ref)
            else:
                mu = mu_ref
            ibcm_rates_sim[0] = mu
            # Launch multiple seeds for the current (tw, tb) combination
            all_procs_twtb = {}
            for k in range(n_seeds):
                apply_args = (run_analyze_ibcm_one_back_seed, n_threads, 
                              ibcm_rates_sim, back_rates_sim, inhib_rates_sim,  
                              ibcm_options_sim,dimensions_sim, simul_seeds[k])
                apply_kwds = dict(duration_loc=duration_sim, dt_loc=deltat_sim, 
                                  skp_loc=skp_sim, full_returns=True)
                all_procs_twtb[k] = pool.apply_async(func_wrapper_threadpool, 
                                 args=apply_args, kwds=apply_kwds)

            # Collect convergence analysis results for this tw, tb
            res_seeds_twtb = {k:all_procs_twtb[k].get() 
                              for k in all_procs_twtb.keys()}
            combined_seed_res = combine_seed_results(res_seeds_twtb, n_seeds)
            i_gaps.append(combined_seed_res[0])
            i_specifs.append(combined_seed_res[1])
            i_varis.append(combined_seed_res[2])

            # Theoretical gap for this c0, use seeds to estimate conc. moments
            hs_hn_hd_u2 = compute_preds_from_seeds(res_seeds_twtb, n_seeds)
            gap_th = hs_hn_hd_u2[0] - hs_hn_hd_u2[1]
            i_gaps_th.append(gap_th)

            print("Finished tw i = {}, tb j = {}".format(i, j))

        # Stack arrays over j (tb) for the current i value (mu)
        all_gaps.append(np.stack(i_gaps))
        all_specifs.append(np.stack(i_specifs))
        all_varis.append(np.stack(i_varis))
        all_gaps_th.append(np.stack(i_gaps_th))

    # Stack arrays over i (tw)
    all_gaps = np.stack(all_gaps)
    all_specifs = np.stack(all_specifs)
    all_varis = np.stack(all_varis)
    all_gaps_th = np.stack(all_gaps_th)

    # Reuse pool for each (tw, tb) but close them at the end
    pool.close()
    pool.join()

    return tw_tb_grid, all_gaps, all_specifs, all_varis, all_gaps_th



if __name__ == "__main__":
    do_nodors_scale = True
    do_turbulence_strength = False
    n_simuls = 2

    # Simulations as a function of the number of odors, 
    # # background amplitude c0 and mu
    # Save results to disk for further analysis and plotting
    # For simplicity, we set N_I = N_B, extra neurons would break
    # alignment and be annoying. 
    root_dir = pj("..")
    outputs_folder = pj(root_dir, "results", "for_plots", "convergence")
    topseed = np.random.SeedSequence(0xaf1208a9561f8a3d3cda1f8da0bbd795)
    nrange = [3, 4, 5, 6, 8]
    if do_nodors_scale:
        spawned_seeds = [topseed, *topseed.spawn(len(nrange)-1)]
        for n in nrange:
            print("Starting simulation for N_B = {}...".format(n))
            res = main_convergence_vs_background_ampli(
                spawned_seeds.pop(0), n_simuls, n_comp_sim=n)
            muc0_grid, true_pvs, learn_pvs, vari_pvs, align_errs = res
            fname = "biopca_convergence_vs_background_ampli_{}odors.npz".format(n)
            fname = pj(outputs_folder, fname)
            np.savez_compressed(fname, 
                muc0_grid=muc0_grid, 
                true_pvs=true_pvs,
                learn_pvs=learn_pvs,
                vari_pvs=vari_pvs,
                align_errs=align_errs
            )
    # End here for the moment
    exit()
    raise NotImplementedError()
    
    
    # Simulations as a function of turbulence strength
    root_dir = pj("..")
    outputs_folder = pj(root_dir, "results", "for_plots", "convergence")
    topseed3 = np.random.SeedSequence(0x1e2da14713c78569488488616d2758e6)
    n_odors = 3
    if do_turbulence_strength:
        print("Starting simulations vs whiff and blank durations...")
        res = main_convergence_vs_turbulence_strength(
            topseed3, n_simuls, n_comp_sim=n_odors, do_adjust_mu=True)
        twtb_grid, align_gaps, specifs, varis, gaps_th = res
        fname = f"biopca_convergence_vs_turbulence_strength_{n_odors}odors.npz"
        fname = pj(outputs_folder, fname)
        np.savez_compressed(fname, 
            twtb_grid=twtb_grid, 
            align_gaps=align_gaps,
            gamma_specifs=specifs,
            hgamma_varis=varis,
            gaps_th=gaps_th
        )

        