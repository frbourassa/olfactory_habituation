#!/usr/bin/env python
# coding: utf-8

""" All-in-one script to run different simulation seeds with OSN adaptation, 
test new odor recognition for each seed, and 
Seeds are launched in parallel using multiprocessing in the main function

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
    compute_mbars_cgammas_cbargammas
)
from modelfcts.backgrounds import (
    update_powerlaw_times_concs, 
    sample_ss_conc_powerlaw, 
    generate_odorant
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
            cbarser_ibcm, thetaser_ibcm, wser_ibcm, yser_ibcm)
        ibcm_rates_loc: learnrate_ibcm, tau_avg_ibcm, coupling_eta_ibcm, ...

    Returns:
        alignment_gaps, indexed [neuron]
        specif_gammas, indexed [neuron]
        gamma_vari, indexed [neuron, component]
    """
    coupling = ibcm_rates_loc[2]
    (_, _, _, mser_ibcm, _, _, _, _) = sim_results
    # Calculate cgammas_bar and mbars
    transient = int(5/6*duration_loc / dt) // skp_loc
    basis = back_pms[-1]

    # Dot products \bar{c}_{\gamma} = \bar{\vec{m}} \cdot \vec{x}_{\gamma}
    _, _, cbgam = compute_mbars_cgammas_cbargammas(mser_ibcm, coupling, basis)
    cbgam_mean = np.mean(cbgam[transient:], axis=0)
    # Sorted odor indices, from min to max, of odor alignments for each neuron
    aligns_idx_sorted = np.argsort(cbgam_mean, axis=1) 
    specif_gammas = np.argmax(cbgam_mean, axis=1)
    assert np.all(specif_gammas == aligns_idx_sorted[:, -1])


    # Gap between first and second largest alignments for each neuron
    n_i = cbgam_mean.shape[0]
    alignment_gaps = (cbgam_mean[np.arange(n_i), specif_gammas]
                     - cbgam_mean[np.arange(n_i), aligns_idx_sorted[:, -2]])

    # Variance (fluctuations) of cbars gamma in the last 20 minutes of the sim.
    # Increases when the learning rate increases
    last_steps = int(2.0*duration_loc/3.0 / dt) // skp_loc
    cbgam_vari = np.var(cbgam[last_steps:], axis=0)

    return cbgam, alignment_gaps, specif_gammas, cbgam_vari


def run_analyze_ibcm_one_back_seed(
        ibcm_rates_loc, back_rates, inhib_rates_loc, 
        options_loc, dimensions, seedseq, 
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
            gaps, specifs, cgamvari, cgammas_ser, sim_results
        else:
            gaps, specifs, cgamvari, None, None
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
    init_synapses_ibcm = 0.2*rgen_init.standard_normal(
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
    cgammas_ser, gaps, specifs, cgamvari = analysis_res
    #print("Finished analyzing IBCM simulation")

    # Doesn't return full c gamma series, only the summary statistics 
    if full_returns:
        cgammas_ser_ret = cgammas_ser
        sim_results_ret = sim_results
    else:
        cgammas_ser_ret = None
        sim_results_ret = None

    return gaps, specifs, cgamvari, cgammas_ser_ret, sim_results_ret



### Multiprocessed main functions

def combine_seed_results(res_dict, n_seeds):
    """ Combine convergence analysis results of simulation seeds
    in res_dict. """
    combi_gaps, combi_specifs, combi_varis = [], [], []
    for i in range(n_seeds):
        gaps, specifs, cgamvari, _, _ = res_dict[i]
        combi_gaps.append(gaps)
        combi_specifs.append(specifs)
        combi_varis.append(cgamvari)
    return [np.stack(a) for a in [combi_gaps, combi_specifs, combi_varis]]


def main_convergence_vs_ibcm_rates(orig_seedseq, n_seeds):
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
    n_comp_sim = 3
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
        "variant": "law",   # maybe we will want to test "intrator" later?
        "decay": True
    }
    # default turbulent background parameters
    back_rates_sim = default_background_params(n_comp_sim)
    # Default alpha, beta
    inhib_rates_sim = [0.0001, 0.00002]  # alpha, beta

    # Time parameters
    duration_sim = 360000.0
    deltat_sim = 1.0
    skp_sim = 20

    # Containers for alignment gaps, specificities, and cgammas variances
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
            res_seeds_mutau = {}
            for k in range(n_seeds):
                apply_args = (run_analyze_ibcm_one_back_seed, n_threads, 
                              ibcm_rates_sim, back_rates_sim, inhib_rates_sim,  
                              ibcm_options_sim,dimensions_sim, simul_seeds[k])
                apply_kwds = dict(duration_loc=duration_sim, dt_loc=deltat_sim, 
                                  skp_loc=skp_sim, full_returns=False)
                #all_procs_mutau[k] = pool.apply_async(func_wrapper_threadpool, 
                #                 args=apply_args, kwds=apply_kwds)
                res_seeds_mutau[k] = func_wrapper_threadpool(
                                        *apply_args, **apply_kwds)


            # Collect convergence analysis results for this mu, tau
            #res_seeds_mutau = {k:all_procs_mutau[k].get() for k in 
            # all_procs_mutau.keys()}. Stack them over seeds
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




if __name__ == "__main__":

    mutau_grid, align_gaps, specifs, varis = main_convergence_vs_ibcm_rates(
        np.random.SeedSequence(0xf44f62d0818452d631061e695b75c517), 32)
    
    # Save to disk for further analysis and plotting
    root_dir = pj("..")
    outputs_folder = pj(root_dir, "results", "for_plots", "convergence")
    fname = pj(outputs_folder, "convergence_vs_ibcm_rates_results.npz")
    np.savez_compressed(fname, 
            mutau_grid=mutau_grid, 
            align_gaps=align_gaps,
            gamma_specifs=specifs,
            cgamma_varis=varis
    )
    