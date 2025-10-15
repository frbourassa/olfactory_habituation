#!/usr/bin/env python
# coding: utf-8

""" All-in-one script to run different IBCM simulation seeds for a variety
of learning and averaging rates, background odor numbers, and background
whiff and blank time scales, to assess convergence as a function of 
these parameters. Seeds are launched in parallel using multiprocessing. 

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
from modelfcts.ibcm import (
    integrate_inhib_ibcm_network_options,
    compute_mbars_hgammas_hbargammas
)
from modelfcts.ibcm_analytics import fixedpoint_thirdmoment_exact
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
def analyze_ibcm_simulation(sim_results, ibcm_rates_loc, back_pms, 
                            skp_loc=20, dt=1.0, duration_loc=360000.0):
    """
    Args:
        sim_results = (tser_ibcm, nuser_ibcm, bkvecser_ibcm, mser_ibcm, 
            hbarser_ibcm, thetaser_ibcm, wser_ibcm, yser_ibcm)
        ibcm_rates_loc: learnrate_ibcm, tau_avg_ibcm, coupling_eta_ibcm, ...

    Returns:
        alignment_gaps, indexed [neuron]
        specif_gammas, indexed [neuron]
        gamma_vari, indexed [neuron, component]
    """
    coupling = ibcm_rates_loc[2]
    (_, _, _, mser_ibcm, _, _, _, _) = sim_results
    # Calculate hgammas_bar and mbars
    transient = int(5/6*duration_loc / dt) // skp_loc
    basis = back_pms[-1]

    # Dot products \bar{c}_{\gamma} = \bar{\vec{m}} \cdot \vec{x}_{\gamma}
    _, _, hbgam = compute_mbars_hgammas_hbargammas(mser_ibcm, coupling, basis)
    hbgam_mean = np.mean(hbgam[transient:], axis=0)
    n_i = hbgam_mean.shape[0]
    # Sorted odor indices, from min to max, of odor alignments for each neuron
    aligns_idx_sorted = np.argsort(hbgam_mean, axis=1) 
    specif_gammas = np.argmax(hbgam_mean, axis=1)
    # If argmax and last argsort indices differ, must be because some
    # elements are equal; check that specif and argsort max are equal
    if not np.all(specif_gammas == aligns_idx_sorted[:, -1]):
        assert np.allclose(hbgam_mean[np.arange(n_i), aligns_idx_sorted[:, -1]], 
                           hbgam_mean[np.arange(n_i), specif_gammas])


    # Gap between first and second largest alignments for each neuron
    alignment_gaps = (hbgam_mean[np.arange(n_i), specif_gammas]
                     - hbgam_mean[np.arange(n_i), aligns_idx_sorted[:, -2]])

    # Variance (fluctuations) of hbars gamma in the last 20 minutes of the sim.
    # Increases when the learning rate increases
    last_steps = int(2.0*duration_loc/3.0 / dt) // skp_loc
    hbgam_vari = np.var(hbgam[last_steps:], axis=0)

    return hbgam, alignment_gaps, specif_gammas, hbgam_vari


def run_analyze_ibcm_one_back_seed(
        ibcm_rates_loc, back_rates, inhib_rates_loc, 
        options_loc, dimensions, seedseq, minit_scale=0.2,
        duration_loc=360000.0, dt_loc=1.0, skp_loc=20, full_returns=False
    ):
    """ Given IBCM model rates and background parameters except
    background odors (but incl. number odors, c0), and a main seed sequence, 
    run and analyze convergence of IBCM on the background generated from 
    that seed. The seedseq should itself have been spawned from a root seed 
    to have a distinct one per run; this still makes seeds reproducible yet 
    distinct for different runs. The seedseq here is spawned again for 
    a background gen. seed and a simul. seed. 

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
    n_comp, n_dim, n_i = dimensions

    # Spawn back. generation seed and simul seed
    initseed, simseed = seedseq.spawn(2)

    # Duplicate back params before appending locally-generated odors
    back_pms_loc = list(back_rates)

    # Create background
    rgen_init = np.random.default_rng(initseed)
    back_comps_loc = generate_odorant((n_comp, n_dim), rgen_init)
    back_comps_loc = back_comps_loc / l2_norm(back_comps_loc, axis=1)[:, None]

    # Add odors to the list of background parameters
    back_pms_loc.append(back_comps_loc)

    # Initialize background with the random generator with seed rgenseed
    rgen_init = np.random.default_rng(initseed)
    init_back = initialize_given_background(
                    back_pms_loc, rgen_init, n_comp, n_dim)

    # Initial synaptic weights: small positive noise
    lambd_loc = ibcm_rates_loc[3]
    init_synapses_ibcm = minit_scale*rgen_init.standard_normal(
                                    size=[n_i, n_dim])*lambd_loc

    # Run the IBCM simulation
    sim_results = integrate_inhib_ibcm_network_options(
                init_synapses_ibcm, update_fct, init_back, 
                ibcm_rates_loc, inhib_rates_loc, back_pms_loc, 
                duration_loc, dt_loc, seed=simseed, 
                noisetype="uniform",  skp=skp_loc, **options_loc
    )

    # Now analyze IBCM simul for convergence
    analysis_res = analyze_ibcm_simulation(sim_results, ibcm_rates_loc, 
                    back_pms_loc, skp_loc=skp_loc, duration_loc=duration_loc)
    hgammas_ser, gaps, specifs, hgamvari = analysis_res
    #print("Finished analyzing IBCM simulation")

    # Doesn't return full h gamma series, only the summary statistics 
    if full_returns:
        hgammas_ser_ret = hgammas_ser
        sim_results_ret = sim_results
    else:
        hgammas_ser_ret = None
        sim_results_ret = None

    return gaps, specifs, hgamvari, hgammas_ser_ret, sim_results_ret



### Multiprocessed main functions

def combine_seed_results(res_dict, n_seeds):
    """ Combine convergence analysis results of simulation seeds
    in res_dict. """
    combi_gaps, combi_specifs, combi_varis = [], [], []
    for i in range(n_seeds):
        gaps, specifs, hgamvari, _, _ = res_dict[i]
        combi_gaps.append(gaps)
        combi_specifs.append(specifs)
        combi_varis.append(hgamvari)
    return [np.stack(a) for a in [combi_gaps, combi_specifs, combi_varis]]


def main_convergence_vs_ibcm_rates(orig_seedseq, n_seeds, n_comp_sim=3):
    """ Run n_seeds IBCM simulations for each combination of mu learning rate
    and tau_theta averaging time, collect convergence statistics for each. 
    The same simulation seeds are tested at each combinatino of model rates, 
    to assess the convergence vs these rates while keeping the set of
    backgrounds tested the same, for a more direct comparison. 

    Since this is a main function, the rates grid and other model parameters
    are defined within. 

    Args:
        orig_seedseq (np.random.SeedSequence): fresh SeedSequence from which
            all other simulation seeds will be spawned. 
        n_comp_sim (int): number of background odor components

    Returns:
        learnrate_tautheta_grid (np.ndarray): indexed [2, mu_idx, tau_idx]
            i.e. the first array along axis 0 contains the 2d grid of mu vals, 
            the 2nd array contains the 2d grid of tau values, and in each grid,
            mu varies along axis 0 (rows), tau varies along axis 1 (columns); 
            the result of np.meshgrid(murange, taurange, indexing="ij"
        all_gaps (np.ndarray): indexed [mu, tau, seed, neuron]
        all_specifs (np.ndarray): indexed [mu, tau, seed, neuron]
        all_varis (np.ndarray): indexed [mu, tau, seed, neuron, component]
    """
    # Grid of IBCM rates mu, tau_theta in a range going to either side
    # of the region where we get convergence for N_B = 3 odors
    # Approximately geomspace, clustered around usual rates (0.75e-3 - 1.25e-3)
    learnrate_range = np.asarray([5e-5, 2e-4, 5e-4, 7.5e-4, 
                                  1.25e-3, 2e-3, 5e-3, 2e-2])
    # Also a somewhat geometric progression, clustered around 
    # good ones (800-1200-1600)
    tautheta_range = np.asarray([100, 200, 400, 800, 1200, 1600, 2000, 3000])
    learnrate_tautheta_grid = np.stack(
        np.meshgrid(learnrate_range, tautheta_range, indexing="ij"), axis=0)
    # learnrate varies on axis 0 (y, rows), tautheta on axis 1 (x, columns)

    # Define simulation and model parameters
    n_i_ibcm_sim = 24
    n_dims_sim = 25
    dimensions_sim = [n_comp_sim, n_dims_sim, n_i_ibcm_sim]

    # Default IBCM model rates
    learnrate_ibcm_sim = 0.00125  # will vary
    tau_avg_ibcm_sim = 1200  # will vary
    coupling_eta_ibcm_sim = 0.6/n_i_ibcm_sim
    ssat_ibcm_sim = 50.0
    k_c2bar_avg_sim = 0.1
    decay_relative_ibcm_sim = 0.005
    lambd_ibcm_sim = 1.0
    ibcm_rates_sim = [
        learnrate_ibcm_sim, 
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
        "variant": "law",
        "decay": True
    }
    # default turbulent background parameters
    back_rates_sim = default_background_params(n_comp_sim)
    # Default alpha, beta
    inhib_rates_sim = [0.00005, 0.00001]  # alpha, beta

    # Time parameters
    duration_sim = 360000.0
    deltat_sim = 1.0
    skp_sim = 20

    # Containers for alignment gaps, specificities, and hgammas variances
    # that will be stacked arrays indexed [mu, tau, seed, ...]
    all_gaps, all_specifs, all_varis = [], [], []

    # Spawn simulation seeds, reused at each combination on the IBCM rates grid
    simul_seeds = orig_seedseq.spawn(n_seeds)
    n_workers = min(count_parallel_cpu(), n_seeds)
    n_threads = count_threads_per_process(n_workers)
    pool = multiprocessing.Pool(n_workers)

    # Treat one rate combination at a time
    for i in range(learnrate_range.shape[0]):
        mu = learnrate_range[i]
        i_gaps, i_specifs, i_varis = [], [], []
        for j in range(tautheta_range.shape[0]):
            tau = tautheta_range[j]
            ibcm_rates_sim[0] = mu
            ibcm_rates_sim[1] = tau
            # Launch multiple seeds for the current (mu, tau) combination
            all_procs_mutau = {}
            #res_seeds_mutau = {}
            for k in range(n_seeds):
                apply_args = (run_analyze_ibcm_one_back_seed, n_threads, 
                              ibcm_rates_sim, back_rates_sim, inhib_rates_sim,  
                              ibcm_options_sim,dimensions_sim, simul_seeds[k])
                apply_kwds = dict(duration_loc=duration_sim, dt_loc=deltat_sim, 
                                  skp_loc=skp_sim, full_returns=False)
                all_procs_mutau[k] = pool.apply_async(func_wrapper_threadpool, 
                                 args=apply_args, kwds=apply_kwds)


            # Collect convergence analysis results for this mu, tau
            res_seeds_mutau = {k:all_procs_mutau[k].get() for k in 
                                all_procs_mutau.keys()}
            combined_seed_res = combine_seed_results(res_seeds_mutau, n_seeds)
            i_gaps.append(combined_seed_res[0])
            i_specifs.append(combined_seed_res[1])
            i_varis.append(combined_seed_res[2])
            print("Finished mu i = {}, tau j = {}".format(i, j))

        # Stack arrays over j (tau_theta) for the current i value (mu)
        all_gaps.append(np.stack(i_gaps))
        all_specifs.append(np.stack(i_specifs))
        all_varis.append(np.stack(i_varis))

    # Stack arrays over i (mu)
    all_gaps = np.stack(all_gaps)
    all_specifs = np.stack(all_specifs)
    all_varis = np.stack(all_varis)

    # Reuse pool for each (mu, tau) but close them at the end
    pool.close()
    pool.join()

    return learnrate_tautheta_grid, all_gaps, all_specifs, all_varis


def adjust_alpha(back_pms, back_pms_ref, alpha_ref):
    r""" The magnitude of h scales as 1/c_0 where c_0 is the concentration
    scale, so we need to adjust $\alpha$ in the $W$ equation to prevent
    numerical instabilities: keep $\alpha/c_0$ constant. """
    c0ref = back_pms_ref[4].mean()
    c0 = back_pms[4].mean()
    alpha = alpha_ref * c0 / c0ref
    return alpha


def compute_preds_from_seeds(res_dict, n_seeds):
    """ Combine concentration series from seeds in res_dict and
    use them to compute analytical predictions for h_sp, h_ns, h_d, u2.
    """
    # Collect all concentration time series.
    combi_conc_sers = []
    for i in range(n_seeds):
        sim_res = res_dict[i][4]
        conc_ser = sim_res[1][:, :, 1]  # tc_ser: [all times, all odors, conc]
        combi_conc_sers.append(conc_ser)
    combi_conc_sers = np.stack(combi_conc_sers)  # indexed [seed, time, odor]
    
    # All samples are iid, can average over all
    mean_conc = np.mean(combi_conc_sers)
    moments_conc = [
        mean_conc,
        np.var(combi_conc_sers),
        np.mean((combi_conc_sers - mean_conc)**3.0)
    ]
    n_components = combi_conc_sers.shape[2]

    # Compute analytical predictions for alignments
    try:
        preds = fixedpoint_thirdmoment_exact(moments_conc, 1, n_components-1)
    except:  # some issue with simulations, no convergence possible, align=0
        preds = np.full((4,), np.nan)
    return preds


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
    # Grid of c0, explore a linear range as this affects convergence as c0**2
    c0range = np.geomspace(0.6/3, 0.6*9, 7)
    c0_default = 0.6
    mu_default = 1.25e-3
    # For mu range, use constant mu*c0**2. 
    learnrate_range = mu_default * (c0_default/c0range)**2
    learnrate_c0_grid = np.stack(
        np.meshgrid(learnrate_range, c0range, indexing="ij"), axis=0)
    # learnrate varies on axis 0 (y, rows), c0 on axis 1 (x, columns)

    # Define simulation and model parameters
    n_i_ibcm_sim = 24
    n_dims_sim = 25
    dimensions_sim = [n_comp_sim, n_dims_sim, n_i_ibcm_sim]

    # Default IBCM model rates
    learnrate_ibcm_sim = mu_default  # will vary
    tau_avg_ibcm_sim = 2000
    coupling_eta_ibcm_sim = 0.6/n_i_ibcm_sim
    ssat_ibcm_sim = 50.0
    k_c2bar_avg_sim = 0.1
    decay_relative_ibcm_sim = 0.005
    lambd_ibcm_sim = 1.0
    ibcm_rates_sim = [
        learnrate_ibcm_sim, 
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
        "variant": "law",
        "decay": True
    }
    # default turbulent background parameters
    back_rates_sim = default_background_params(n_comp_sim)
    # Will vary c0, which is the element at index 4 of this param list
    # Initial M scale
    mscale_def = 0.2
    mscale_sim = mscale_def
    
    # Default alpha, beta. 
    # Need lower alpha for larger c0, to prevent numerical instabilities.
    back_rates_ref = default_background_params(n_comp_sim)  # reference
    alpha_def = 5e-5  # needs to be adjusted to h scaling as 1/c_0
    inhib_rates_sim = [alpha_def, 1e-5]  # alpha, beta

    # Time parameters
    duration_sim = 360000.0
    deltat_sim = 1.0
    skp_sim = 20

    # Containers for alignment gaps, specificities, and hgammas variances
    # that will be stacked arrays indexed [mu, c0, seed, ...]
    all_gaps, all_specifs, all_varis = [], [], []
    all_gaps_th = []  # theoretical gaps, analytical prediction for h_sp - h_ns

    # Spawn simulation seeds, reused at each combination on the IBCM rates grid
    simul_seeds = orig_seedseq.spawn(n_seeds)
    n_workers = min(count_parallel_cpu(), n_seeds)
    n_threads = count_threads_per_process(n_workers)
    pool = multiprocessing.Pool(n_workers)

    # Treat one rate combination at a time
    for i in range(learnrate_range.shape[0]):
        mu = learnrate_range[i]
        i_gaps, i_specifs, i_varis, i_gaps_th = [], [], [], []
        for j in range(c0range.shape[0]):
            c0 = c0range[j]
            ibcm_rates_sim[0] = mu
            back_rates_sim[4][:] = c0
            # Adjust alpha to h magnitude scaling as 1/c_0
            alpha_sim = adjust_alpha(back_rates_sim, back_rates_ref, alpha_def)
            inhib_rates_sim[0] = alpha_sim
            # Launch multiple seeds for the current (mu, c0) combination
            all_procs_muc0 = {}
            for k in range(n_seeds):
                apply_args = (run_analyze_ibcm_one_back_seed, n_threads, 
                              ibcm_rates_sim, back_rates_sim, inhib_rates_sim,  
                              ibcm_options_sim,dimensions_sim, simul_seeds[k])
                apply_kwds = dict(duration_loc=duration_sim, dt_loc=deltat_sim, 
                                    minit_scale=mscale_sim, skp_loc=skp_sim, 
                                    full_returns=True)
                all_procs_muc0[k] = pool.apply_async(func_wrapper_threadpool, 
                                 args=apply_args, kwds=apply_kwds)

            # Collect convergence analysis results for this mu, c0
            res_seeds_muc0 = {k:all_procs_muc0[k].get() for k in 
                                all_procs_muc0.keys()}
            combined_seed_res = combine_seed_results(res_seeds_muc0, n_seeds)
            i_gaps.append(combined_seed_res[0])
            i_specifs.append(combined_seed_res[1])
            i_varis.append(combined_seed_res[2])

            # Theoretical gap for this c0, use seeds to estimate conc. moments
            hs_hn_hd_u2 = compute_preds_from_seeds(res_seeds_muc0, n_seeds)
            gap_th = hs_hn_hd_u2[0] - hs_hn_hd_u2[1]
            i_gaps_th.append(gap_th)

            print("Finished mu i = {}, c0 j = {}".format(i, j))

        # Stack arrays over j (c0) for the current i value (mu)
        all_gaps.append(np.stack(i_gaps))
        all_specifs.append(np.stack(i_specifs))
        all_varis.append(np.stack(i_varis))
        all_gaps_th.append(np.stack(i_gaps_th))

    # Stack arrays over i (mu)
    all_gaps = np.stack(all_gaps)
    all_specifs = np.stack(all_specifs)
    all_varis = np.stack(all_varis)
    all_gaps_th = np.stack(all_gaps_th)

    # Reuse pool for each (mu, c0) but close them at the end
    pool.close()
    pool.join()

    return learnrate_c0_grid, all_gaps, all_specifs, all_varis, all_gaps_th


def adjust_learnrate(back_pms, back_pms_ref, mu_ref):
    """ Return mu to keep mu * <c>**2 constant compared
    to the average concentration entailed by back_pms_ref
    and the learning rate reference mu_ref. """
    avg_conc_ref = mean_turbulent_concs(back_pms_ref).mean()
    avg_conc_new = mean_turbulent_concs(back_pms).mean()
    return mu_ref * (avg_conc_ref / avg_conc_new)**2.0


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
        "variant": "law",
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
    do_nodors_rates = True
    do_conc_scale = True  # vs c0, compensate with mu, mu * c0**2 = cst
    do_turbulence_strength = True
    n_simuls = 32

    # Simulations as a function of the number of odors
    # and the IBCM learning and averaging rates. 
    # Save results to disk for further analysis and plotting
    root_dir = pj("..")
    outputs_folder = pj(root_dir, "results", "for_plots", "convergence")
    topseed = np.random.SeedSequence(0xf44f62d0818452d631061e695b75c517)
    nrange = [3, 4, 5, 6, 8]
    if do_nodors_rates:
        spawned_seeds = [topseed, *topseed.spawn(len(nrange)-1)]
        for n in nrange:
            print("Starting simulation for N_B = {}...".format(n))
            res = main_convergence_vs_ibcm_rates(
                spawned_seeds.pop(0), n_simuls, n_comp_sim=n)
            mutau_grid, align_gaps, specifs, varis = res
            fname = "convergence_vs_ibcm_rates_results_{}odors.npz".format(n)
            fname = pj(outputs_folder, fname)
            np.savez_compressed(fname, 
                mutau_grid=mutau_grid, 
                align_gaps=align_gaps,
                gamma_specifs=specifs,
                hgamma_varis=varis
            )
    
    # Simulations as a function of background amplitude c0 and mu compensation
    root_dir = pj("..")
    outputs_folder = pj(root_dir, "results", "for_plots", "convergence")
    topseed2 = np.random.SeedSequence(0x8f380db086f6b8c843bd7b9969a7a2c0)
    n_odors = 3
    if do_conc_scale:
        print("Starting simulations vs mu and c0 conc. scale...")
        res = main_convergence_vs_background_ampli(
            topseed2, n_simuls, n_comp_sim=n_odors)
        muc0_grid, align_gaps, specifs, varis, gaps_th = res
        fname = f"convergence_vs_background_ampli_results_{n_odors}odors.npz"
        fname = pj(outputs_folder, fname)
        np.savez_compressed(fname, 
            muc0_grid=muc0_grid, 
            align_gaps=align_gaps,
            gamma_specifs=specifs,
            hgamma_varis=varis, 
            gaps_th=gaps_th
        )


    # Simulations as a function of turbulence strength
    root_dir = pj("..")
    outputs_folder = pj(root_dir, "results", "for_plots", "convergence")
    topseed3 = np.random.SeedSequence(0xa41cb5e74887125e2281840c8d1a6b12)
    n_odors = 3
    if do_turbulence_strength:
        print("Starting simulations vs whiff and blank durations...")
        res = main_convergence_vs_turbulence_strength(
            topseed3, n_simuls, n_comp_sim=n_odors, do_adjust_mu=True)
        twtb_grid, align_gaps, specifs, varis, gaps_th = res
        fname = f"convergence_vs_turbulence_strength_results_{n_odors}odors.npz"
        fname = pj(outputs_folder, fname)
        np.savez_compressed(fname, 
            twtb_grid=twtb_grid, 
            align_gaps=align_gaps,
            gamma_specifs=specifs,
            hgamma_varis=varis,
            gaps_th=gaps_th
        )

        