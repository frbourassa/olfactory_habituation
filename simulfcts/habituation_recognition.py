"""
This is the module with functions to compare the performance of the IBCM model
to online PCA, average background subtraction, and no habituation for
new odor recongition.

We test three different background statistics: weakly non-gaussian, log-normal,
and turbulent.

We test with random odor components (n_R = 25 dimensions), maybe in the future
with components from the Hallem and Carson 2006 dataset (n_R=25 dimensions).
The number of Kenyon cells (generating neural tags) are scaled appropriately.

We collect statistics across multiple parameters:
    - We test at least 10 different background odor compositions
    - We test at least 10 random initializations for each background
      (includes different initial weights and seed for the background process)
    --- Never mind, we test 100 random backgrounds, one random initialization
        for each. Gives better stats on backgrounds, same level of stats
        on the ability of IBCM neurons to distribute properly.
This represents 10^2 runs of habituation. Then, for each such run:
    - We test at least 100 background samples from each run (by taking
        10 different time points spaced enough from each other
        and 1+9 background samples at each time point)
    - We test 20 % and 50 % background composition
    - We test at least 100 random new odors for each run

This means habituating 10^2 times for each model and each background
distribution, then 10^4 tests per run. Each habituation run taking about 10 s,
and each test a few tenths of s, the total can be pretty long: run
on a computational cluster and multiprocess the runs as much as possible.

We will use the same KC wiring throughout for all models. In other words,
the PN to neural tag mapping is the same for all models and trials.
In fact, we will only save PN values and map them to KC during analysis.

The results are analyzed and plotted elsewhere.
For each run, I save:
    - The background vectors for that run (we do one run per background)
    - Snapshots of the network state at each test time,
        including the background concentration state
    - The extra background samples tested at each snapshot time
    - The model parameters and background concentration parameters
    - Each set of background components tried
    - The initial weights and background process seed
    - The state of the weights at each new odor introduction point
    - The new odors introduced in each run at each point
    - The response to each mixture (s)

On top of that, I save global parameters in the hdf file:
    - Dimensions
    - snapshot times, total duration, skip step size

For detailed documentation on the HDF results file structure, see the file
results/performance_hdf_file_structure_doc.txt
All HDF results files are saved in the results/performance/ folder.

@author: frbourassa
July 2023
"""
import numpy as np
from scipy import sparse
import h5py
import pandas as pd
import multiprocessing
import gc
import time
from threadpoolctl import threadpool_limits
from utils.profiling import DeltaTimer, IterationProfiler
from numba import njit


# Local imports
from modelfcts.ideal import relu_inplace
from modelfcts.ibcm import (
    integrate_inhib_ibcm_network_options,
    ibcm_respond_new_odors
)
from modelfcts.biopca import (
    integrate_inhib_ifpsp_network_skip,
    biopca_respond_new_odors
)
from modelfcts.average_sub import (
    integrate_inhib_average_sub_skip,
    average_sub_respond_new_odors
)
from modelfcts.distribs import (
    truncexp1_inverse_transform,  # whiff concs
    powerlaw_cutoff_inverse_transform
)
from modelfcts.backgrounds import (
    update_ou_kinputs,
    update_alternating_inputs,
    update_thirdmoment_kinputs,
    update_logou_kinputs,
    update_powerlaw_times_concs,
    sample_background_powerlaw,
    sample_ss_conc_powerlaw,
    generate_odorant
)
from modelfcts.tagging import (
    project_neural_tag,
    create_sparse_proj_mat,
    SparseNDArray
)
from utils.statistics import seed_from_gen
from utils.cpu_affinity import count_parallel_cpu, count_threads_per_process
from utils.metrics import l2_norm, jaccard
from utils.export import (
    dict_to_hdf5,
    csr_matrix_to_hdf5,
    add_to_npz
)

def select_sampling_functions(attrs):
    """ Select the right background sampling function, based on the background
    process in attrs"""
    back_conc_map = {
        "turbulent": sample_ss_conc_powerlaw
    }
    back_vec_map = {
        "turbulent": sample_background_powerlaw
    }
    try:
        sample_conc = back_conc_map[attrs["background"]]
        sample_vec = back_vec_map[attrs["background"]]
    except KeyError:
        raise KeyError("Unknown background: {}".format(attrs["background"]))
    return sample_conc, sample_vec


def appropriate_response(attrs, params, mix, snaps, j, options):
    if attrs["model"] == "IBCM":
        resp = ibcm_respond_new_odors(
                    mix, snaps["m"][j], snaps["w"][j], params["m_rates"],
                    options=options
                )
    elif attrs["model"] == "PCA":
        mlx_snaps = []
        for k in "mlx":
            if snaps[k] is not None: mlx_snaps.append(snaps[k][j])
            else: mlx_snaps.append(None)

        resp = biopca_respond_new_odors(
                    mix, mlx_snaps, snaps["w"][j], params["m_rates"],
                    options=options
                )
    elif attrs["model"] == "AVG":
        resp = average_sub_respond_new_odors(
                    mix, snaps["w"][j],
                    options=options
                )
    else:
        raise ValueError("Unknown model: {}".format(attrs["background"]))
    return resp


def test_new_odor_recognition(snaps, attrs, params, sim_odors, test_params):
    """ snaps (dict):
            "m": snapshots of M
            "w": snapshots of W
            "l": snapshots of L (if applicable)
            "conc": snapshots of background concentrations
            "back": snapshots of the simulated background x_b.
        attrs (dict): "model", "background" are relevant here
        params (dict): dimensions, repeats, m_rates, back_params,
            new_concs are relevant and same for all simulations.
        sim_odors (dict):
            "back": the background odors of that simulation
                (not all of them like int he HDF file), shape [n_b, n_r]
            "new": all new odors, shape [n_new_odors, n_r]
        test_params (dict):
            test_seed_seq (np.random.SeedSequence): child SeedSequence,
                needed for background sampling.
            pmat (sp.sparse.csr_matrix): sparse projection mat. of this simul.
            proj_kwargs (dict): projection function kwargs (sparsity, etc.)
            model_options (dict): model options
    """
    n_times = params["repeats"][1]
    n_back_samples = params["repeats"][2]
    activ_fct = test_params.get("model_options", {}).get("activ_fct", "ReLU")
    conc_sampling, vec_sampling = select_sampling_functions(attrs)
    # Generate new background samples (stationary)
    test_rgen = np.random.default_rng(test_params["test_seed_seq"])
    conc_samples = conc_sampling(
                        *params["back_params"],
                        size=n_times*(n_back_samples-1), rgen=test_rgen
                    )  # Shaped [sample, component]
    back_samples = conc_samples.dot(sim_odors["back"])
    back_samples = back_samples.reshape([n_times, n_back_samples-1, -1])
    conc_samples = conc_samples.reshape([n_times, n_back_samples-1, -1])
    # Append the background snapshots to them
    # These samples will be returned and saved to disk afterwards
    # Conc samples not immediately useful here, but nice for future analysis
    back_samples = np.concatenate([snaps["back"][:, None, :], back_samples], 
                                  axis=1)
    conc_samples = np.concatenate([snaps["conc"][:, None, :], conc_samples], 
                                  axis=1)
    # Loop over new odors first
    n_new_odors = params['repeats'][3]
    n_new_concs = params['repeats'][4]
    mixture_svecs = np.zeros([n_new_odors, n_times, n_new_concs,
                            n_back_samples, params['dimensions'][0]])
    n_kc = params['dimensions'][3]
    assert n_kc == test_params["pmat"].shape[0], "Inconsistent KC number"
    mixture_tags = SparseNDArray((n_new_odors, n_times, n_new_concs,
                                    n_back_samples, n_kc), dtype=bool)
    new_odor_tags = sparse.lil_array((n_new_odors, n_kc), dtype=bool)
    jaccard_scores = np.zeros(mixture_tags.ndshape[:4])

    for i in range(n_new_odors):
        # Compute neural tag of the new odor alone, without inhibition
        new_tag = project_neural_tag(
                        sim_odors["new"][i], sim_odors["new"][i],
                        test_params["pmat"], **test_params["proj_kwargs"]
                    )
        new_odor_tags[i, list(new_tag)] = True
        # Now, loop over snapshots, mix the new odor with the back samples,
        # compute the PN response at each test concentration,
        # compute tags too, and save results
        for j in range(n_times):
            for k in range(n_new_concs):
                mixtures = (back_samples[j]
                    + params["new_concs"][k] * sim_odors["new"][i])
                mixture_svecs[i, j, k] = appropriate_response(
                                            attrs, params, mixtures, snaps,
                                            j, test_params["model_options"]
                                        )
                # We actually don't want to apply ReLU to really understand
                # what happens if a s vector is zero.
                #if str(activ_fct).lower() == "relu":
                #    mixture_svecs[i,j,k] = relu_inplace(mixture_svecs[i,j,k])
                for l in range(n_back_samples):
                    mix_tag = project_neural_tag(
                        mixture_svecs[i, j, k, l], mixtures[l],
                        test_params['pmat'], **test_params['proj_kwargs']
                    )
                    try:
                        mixture_tags[i, j, k, l, list(mix_tag)] = True
                    except ValueError as e:
                        print(mix_tag)
                        print(mixture_svecs[i, j, k, l])
                        print(test_params["pmat"].dot(mixture_svecs[i, j, k, l]))
                        raise e
                    jaccard_scores[i, j, k, l] = jaccard(mix_tag, new_tag)
    # Prepare simulation results dictionary
    new_odor_tags = new_odor_tags.tocsr()
    test_results = {
        "conc_samples": conc_samples,
        "back_samples": back_samples,
        "new_odor_tags": new_odor_tags,
        "mixture_svecs": mixture_svecs,
        "mixture_tags": mixture_tags,
        "jaccard_scores": jaccard_scores
    }
    return test_results

# Version of test_new_odor_recognition with lower memory usage
@njit(parallel=False)
def test_new_odor_recognition_lean(snaps, attrs, params, sim_odors, test_params):
    """ Version of test_new_odor_recognition which only saves jaccard scores
    and distances between new odor and mixture y vectors, without storing
    in memory the full matrices of y vectors and neural tags, only the
    ones relevant to the current iteration. 
    """
    n_times = params["repeats"][1]
    n_back_samples = params["repeats"][2]
    activ_fct = test_params.get("model_options", {}).get("activ_fct", "ReLU")
    conc_sampling, vec_sampling = select_sampling_functions(attrs)
    # Generate new background samples (stationary)
    test_rgen = np.random.default_rng(test_params["test_seed_seq"])
    conc_samples = conc_sampling(
                        *params["back_params"],
                        size=n_times*(n_back_samples-1), rgen=test_rgen
                    )  # Shaped [sample, component]
    conc_samples = conc_samples.reshape([n_times, n_back_samples-1, -1])
    back_samples = conc_samples.dot(sim_odors["back"])
    # Append the background snapshots to them
    back_samples = np.concatenate([snaps["back"][:, None, :], back_samples], 
                                  axis=1)
    # The conc. samples will be returned and saved to disk afterwards
    # to be able to reconstruct back samples for optimal recognition models
    conc_samples = np.concatenate([snaps["conc"][:, None, :], conc_samples], 
                                  axis=1)
    # Loop over new odors first
    n_new_odors = params['repeats'][3]
    n_new_concs = params['repeats'][4]
    n_kc = params['dimensions'][3]
    assert n_kc == test_params["pmat"].shape[0], "Inconsistent KC number"
    jaccard_scores = np.zeros((n_new_odors, n_times, 
                               n_new_concs, n_back_samples))
    y_l2_distances = np.zeros((n_new_odors, n_times, 
                               n_new_concs, n_back_samples))
    
    # Profiling
    switch = False
    profiler = IterationProfiler(str(test_params["test_seed_seq"].spawn_key))
    for i in range(n_new_odors):
        # Compute neural tag of the new odor alone, without inhibition
        new_odor = sim_odors["new"][i]
        if i % 25 == 0:  # Turn on switch to profile one iteration every 25 odors
            switch = True
            profiler.start(f"i=={i}")
        # TODO: This has issues
        new_tag = project_neural_tag(
            new_odor, new_odor,test_params["pmat"], **test_params["proj_kwargs"]
        )
        if switch: profiler.addpoint("new odor tagging")
        # Now, loop over snapshots, mix the new odor with the back samples,
        # compute the PN response at each test concentration,
        # compute tags too, and save results
        for j in range(n_times):
            for k in range(n_new_concs):
                # mixtures is shaped [back_sample, OSN dimension]
                # so the new_odor broadcasts against it well. 
                mixtures = back_samples[j] + params["new_concs"][k] * new_odor
                if switch: profiler.addpoint("compute mixtures")
                # TODO: This has issues
                mixture_yvecs_ijk = appropriate_response(
                                attrs, params, mixtures, snaps,
                                j, test_params["model_options"]
                )
                if str(activ_fct).lower() == "relu":
                    mixture_yvecs_ijk = relu_inplace(mixture_yvecs_ijk)
                if switch: profiler.addpoint("compute yvecs response to mixtures")
                # Compute L2 distance between response to mixtures 
                # with back_samples and new odor vector at the current conc. 
                # ydiffs shaped [n_back_samples, n_s], compute norm along axis 1
                ydiffs = mixture_yvecs_ijk - params["new_concs"][k] * new_odor
                if switch: profiler.addpoint("compute ydiffs")
                y_l2_distances[i, j, k] = l2_norm(ydiffs, axis=1)
                if switch: profiler.addpoint("computing l2_norm of ydiffs")
                # Compute Jaccard similarity between new odor tag
                # and tags of responses to mixtures with back_samples
                for l in range(n_back_samples):
                    # TODO: This has issues
                    mix_tag = project_neural_tag(
                        mixture_yvecs_ijk[l], mixtures[l],
                        test_params['pmat'], **test_params['proj_kwargs']
                    )
                    if switch: profiler.addpoint("computing mixture tag")
                    jaccard_scores[i, j, k, l] = jaccard(mix_tag, new_tag)
                    if switch: profiler.addpoint("computing jaccard")
                    if switch:
                        profiler.end_iter()
                        switch = False  # Stop profiling after first pass
    # Prepare simulation results dictionary
    test_results = {
        "conc_samples": conc_samples,
        "jaccard_scores": jaccard_scores, 
        "y_l2_distances": y_l2_distances
    }
    return test_results


# Label a function call with some id (typically, int or tuple of ints)
def func_wrapper_with_id(func, id, *args, **kwargs):
    return id, func(*args, **kwargs)

# Launch in a pool of threads limited to the number of threads per CPU
# since we are already running one process per CPU core. 
def func_wrapper_with_id_threadpool(func, id, threadlim, *args, **kwargs):
    with threadpool_limits(limits=threadlim, user_api='blas'):
        res = func(*args, **kwargs)
    return id, res

def find_snap_index(dt, skip, times):
    """ Find nearest multiple of dt*skip to each time in times """
    return np.around(times / (dt*skip)).astype(int)

def id_to_simkey(id):
    """ Decide on a group name format for each simulation based on its id """
    return f"sim{id:04}"

def select_model_functions(attrs):
    # Select integration function
    if attrs["model"] == "IBCM":
        integrate = integrate_inhib_ibcm_network_options
    elif attrs["model"] == 'PCA':
        integrate = integrate_inhib_ifpsp_network_skip
    elif attrs["model"] == "AVG":
        integrate = integrate_inhib_average_sub_skip
    else:
        raise ValueError("Unknown model: {}".format(attrs["model"]))

    # Select background update function
    back_function_map = {
        "turbulent": (update_powerlaw_times_concs, "uniform"),
        "log-normal": (update_logou_kinputs, "normal"),
        "third_moment": (update_thirdmoment_kinputs, "normal"),
        "gaussian": (update_ou_kinputs, "normal"),
        "alternating": (update_alternating_inputs, "uniform")
    }
    try:
        update_bk, noise_dist = back_function_map[attrs["background"]]
    except KeyError:
        raise ValueError("Unknown back. process: " + attrs["background"])

    return integrate, update_bk, noise_dist


def save_simul_results(id, res, attrs, gp, snap_i, full_file=None, lean=False):
    result_items = {
        "IBCM": ["tser", "back_conc_snaps", "back_vec_snaps", "m_snaps",
                 "cbar_snaps", "theta_snaps", "w_snaps", "s_snaps"],
        "PCA": ["tser", "back_conc_snaps", "back_vec_snaps", "m_snaps",
                 "l_snaps", "x_snaps", "cbar_snaps", "w_snaps", "s_snaps"],
        "AVG": ["tser", "back_conc_snaps", "back_vec_snaps",
                 "w_snaps", "s_snaps"]
    }
    drops = ["tser"]
    if lean: drops += ["cbar_snaps", "theta_snaps", "s_snaps"]
    try:
        result_items[attrs["model"]]
    except KeyError:
        raise NotImplementedError("Treat output of other models")
    for i, lbl in enumerate(result_items[attrs["model"]]):
        if lbl in drops or res[i] is None: continue  # don't save this one
        elif lbl == "back_conc_snaps" and res[i].ndim == 3:
            dset = res[i][snap_i, :, -1]  # Keep only concentrations
        else:
            dset = res[i][snap_i]
        gp.create_dataset(lbl, data=dset.copy())
    if full_file is not None:
        full_results = dict(zip(result_items[attrs["model"]], res))
        back_results = {k:full_results[k] 
            for k in ["back_vec_snaps", "s_snaps"]}
        np.savez_compressed(full_file, **back_results)
    return gp


def error_callback(excep):
    print()
    print(excep)
    print()
    return -1


def initialize_background(attrs, dimensions, back_params, back_vecs, rng):
    if attrs["background"] == "turbulent":
        i_concs = sample_ss_conc_powerlaw(*back_params, size=1, rgen=rng)
        i_times = powerlaw_cutoff_inverse_transform(
                rng.random(size=dimensions[1]), *back_params[2:4])
        back_tc = np.stack([i_times, i_concs.squeeze()], axis=1)
        back_vec = back_tc[:, 1].dot(back_vecs)
        back_init = [back_tc, back_vec]
    else:
        raise NotImplementedError("Did not implement background type"
                + "{} initialization".format(attrs["background"]))
    return back_init


def initialize_weights(attrs, dims, rgen, params):
    # IBCM: only m_init
    if attrs["model"] == "IBCM":
        lambd = params["m_rates"][3]
        all_init_weights = 0.2*rgen.standard_normal(
                                size=[dims[2], dims[0]]) * lambd
        init_weight_names = "m_init"
    # BioPCA: list of [m_inits, l_inits]
    elif attrs["model"] == "PCA":
        lambd = params["m_rates"][2]
        init_mmat = rgen.standard_normal(size=[dims[2], dims[0]]) * lambd
        init_mmat /= np.sqrt(dims[0])
        init_lmat = np.eye(dims[2], dims[2])  # Supposed to be near-identity
        all_init_weights = [init_mmat, init_lmat]
        init_weight_names = ["m_init", "l_init"]
    elif attrs["model"] == "AVG":
        # dummy, M is useless and will stay equal to m_init throughout.
        # Used to infer n_orn, so it should have shape [1, n_orn].
        all_init_weights = np.zeros([1, dims[0]], dtype=bool)
        init_weight_names = "m_init"
    return all_init_weights, init_weight_names


def initialize_integration(
        id, nwork, gp, attrs, params, modopt, back, rgen, spseed
    ):
    """ Create a list of args and dictionary of kwargs to launch
    the main habituation parallel runs
    nwork (int): number of Pool workers, to determine the
        allowable number of threads per pool in numpy
    """
    # Select appropriate functions for the model and background
    # Yes we could do that outside of the loop, once for all repeats
    # but it's easier to do it here for code clarity
    integrate, back_update, noise_type = select_model_functions(attrs)
    init_weights, init_weight_names = initialize_weights(
                                attrs, params["dimensions"], rgen, params)
    if isinstance(id, int):
        back_id = back[id]
    else:
        back_id = back
    back_init_sim = initialize_background(
                    attrs, params["dimensions"], params["back_params"],
                    back_id, rgen
                )
    back_params_sim = params["back_params"] + [back_id]
    threads_per_process = count_threads_per_process(nwork)
    apply_args = (integrate,
                    id,
                    threads_per_process,
                    init_weights,
                    back_update,
                    back_init_sim,
                    params["m_rates"],
                    params["w_rates"],
                    back_params_sim,
                    *params["time_params"]
                )
    apply_kwargs = {
        "seed": spseed,
        "noisetype": noise_type,
        "skp": params["repeats"][5]
    }
    apply_kwargs.update(modopt)
    # Save initialization to HDF file. Move this to a function
    # treating different models
    if type(init_weights) == list:  # M and L inits
        for i, lbl in enumerate(init_weight_names):
            gp.create_dataset(lbl, data=init_weights[i])
    else:
        gp.create_dataset(init_weight_names, data=init_weights)
    bk_init_gp = gp.create_group("bk_init")
    bk_init_gp.create_dataset("bk_vari_init", data=back_init_sim[0])
    bk_init_gp.create_dataset("bk_vec_init", data=back_init_sim[1])
    return apply_args, apply_kwargs


def main_habituation_runs(
    filename, attributes, parameters, model_options,
    save_fct=save_simul_results, full_example_file=None, lean=False
):
    """
    Args: everything in the .attrs and parameters group of the HDF file that
    will be saved. Also the path and file name.
        filename (str): path and file name of the HDF results file to create.
        attributes (dict): attributes dictionary as in the HDF file, contains
            model (str): either "IBCM", "PCA", "AVG"
            background (str): typically "turbulent"
            main_seed (int): 128-bit integer entropy of main SeedSequence
        parameters (dict): dictionary containing the following arrays
            model_options (dict): model options, saved as attrs
            dimensions (np.ndarray): n_r, n_b, n_i, n_k
            repeats (np.ndarray): [n_runs, n_test_times, n_back_samples,
                                    n_new_odors, n_new_concs, skip_steps]
            m_rates (np.ndarray): depends on the model, should be consistent
            w_rates (np.ndarray): alpha, beta
            time_params (np.ndarray): tmax, dt
            back_params (np.ndarray): background process parameters
            snap_times (np.ndarray): times at which snapshots are taken
        model_options (dict): model options, will be saved
            in the parameters group attrs in HDF5 file.
        save_fct (callable): default is save_simul_results, but can be changed
            if we need to save different things for other kinds of simulations
        full_example_file (str): if a filename is passed, save one full 
            habituation run in that location, for illustration purposes. 
        lean (bool): if True, save only the bare minimum results (necessary
            for simulations in high dimensions). 
    Returns:
        Saves an HDF file.
        0
    """
    # Some consistency checks
    assert parameters['snap_times'].size == parameters['repeats'][1]
    # 0. Setting up the simulations
    # main random generator
    main_seed_seq = np.random.SeedSequence(int(attributes["main_seed"]))
    main_rgen = np.random.default_rng(main_seed_seq)
    repeats, dimensions = parameters["repeats"], parameters['dimensions']

    # Create background odors; new odors not needed here.
    # Dimension of all background components array: [n_runs, n_b, n_r]
    back_odors = generate_odorant(
                            [repeats[0], dimensions[1], dimensions[0]],
                            main_rgen, lambda_in=0.1
                            )
    back_odors /= l2_norm(back_odors).reshape(*back_odors.shape[:-1], 1)

    # Create HDF file that will contain all simulation results
    results_file = h5py.File(filename, "w")
    dict_to_hdf5(results_file.attrs, attributes)
    param_group = results_file.create_group("parameters")
    param_group = dict_to_hdf5(param_group, parameters)
    # Also save model options in the param_group.attrs
    dict_to_hdf5(param_group.attrs, model_options)

    # Save the background and new odors in an 'odors' group
    odors_group = results_file.create_group("odors")
    odors_group.create_dataset("back_odors", data=back_odors)

    # Find indices of snapshot times, based on skip steps, snap times and dt
    snap_idx = find_snap_index(
                    parameters["time_params"][1], repeats[5],
                    parameters["snap_times"]
                )

    # 1. Run habituation against each background
    # Use a callback function to save only snapshots to disk
    # Keep new odor recognition tests separate (launch another multiprocess)
    def callback(result):
        # Get the result being returned: id, [all results]
        sim_id, sim_results = result
        # Get the group created at launch time for this simulation id
        sim_gp = results_file.get(id_to_simkey(sim_id))
        # Save results, depending on model type the structure changes a bit
        full_file = full_example_file if sim_id == 0 else None
        save_fct(sim_id, sim_results, attributes, sim_gp, snap_idx, 
                 full_file=full_file, lean=lean)
        print("Habituation run {} saved".format(sim_id))
        del sim_results, result
        return sim_id

    # Initialize all simulations, save to disk
    # Then loop again to launch simulations in parallel.

    spawned_seeds = main_seed_seq.spawn(repeats[0])
    n_workers = min(count_parallel_cpu(), repeats[0])
    pool = multiprocessing.Pool(n_workers)
    for sim_id in range(repeats[0]):
        # Package simulation arguments and save initialization
        # to a new group for this simulation
        sim_gp = results_file.create_group(id_to_simkey(sim_id))
        apply_args, apply_kwargs = initialize_integration(
            sim_id, n_workers, sim_gp, attributes, parameters, model_options,
            back_odors, main_rgen, spawned_seeds[sim_id]
        )
        pool.apply_async(
            func_wrapper_with_id_threadpool, args=apply_args, kwds=apply_kwargs, 
            callback=callback, error_callback=error_callback
        )

    # No need to .get() results: the callback takes care of it
    # and in fact we don't want to get them, else they get stuck in the
    # parent process memory and fill it up...
    pool.close()
    pool.join()
    results_file.close()
    # Finish here and code the tests in a separate main function.
    # All relevant results can be reloaded from the HDF file.
    return 0


def new_rng_from_old_seed(old_seed, jumps=0):
    """ From a previously used seed and main random generator,
    create a new seed. Use jumps if the old_seed needs to be reused
    more than once.
    """
    seed_seq = np.random.SeedSequence(int(old_seed))
    # Jump the generator created with old_seed.
    rgen = np.random.Generator(np.random.default_rng(seed_seq)
                        .bit_generator.jumped(jumps=jumps))
    seed2 = seed_from_gen(rgen)
    # Create a new main random number generator (different seed!)
    seed_seq2 = np.random.SeedSequence(int(seed2))
    rgen2 = np.random.default_rng(seed_seq2)
    return seed_seq2, rgen2


def get_data(g, k):
    gp = g.get(k)
    if gp is not None:
        gp = gp[()]
    return gp


def initialize_recognition(id, nwork, gp, odors_gp, 
    attrs, params, modopt, rgen, spseed, projkw, lean=False):
    """ Function to arrange the arguments of multiprocessing pool apply
    for new odor detection tests
    """
    p_matrix = create_sparse_proj_mat(
                    params["dimensions"][3], params["dimensions"][0], rgen
                )
    # lean or not, need this matrix to compare tags in runs of optimal models
    try:
        csr_matrix_to_hdf5(gp.create_group("kc_proj_mat"), p_matrix)
    except ValueError:
        pass  # Matrix already exists
    # Load relevant snapshots.
    snaps_dict = {
        "m": get_data(gp, "m_snaps"),
        "w": get_data(gp, "w_snaps"),
        "l": get_data(gp, "l_snaps"),
        "x": get_data(gp, "x_snaps"),
        "conc": get_data(gp, "back_conc_snaps"),
        "back": get_data(gp, "back_vec_snaps"),
    }
    # Load odors locally, argument to pass to each test process
    sim_odors_dict = {
        "back": get_data(odors_gp, "back_odors")[id],
        "new": get_data(odors_gp, "new_odors"),
    }
    # Construct dictionaries to pass to test_new_odor_recognition
    test_params = {
        "test_seed_seq": spseed,
        "pmat": p_matrix,
        "proj_kwargs": projkw,
        "model_options": modopt,
    }
    test_fct = (test_new_odor_recognition_lean if lean 
                else test_new_odor_recognition)
    n_threads = count_threads_per_process(nwork)
    apply_args = (test_fct, id, n_threads, snaps_dict,
                    attrs, params, sim_odors_dict, test_params)
    return apply_args


def main_recognition_runs(
        filename, attrs, params, model_options, proj_kwargs, 
        full_example_file=None, lean=False
    ):
    """ After having performed several habituation runs and saved these
    results to HDF, load the results and test new odor recognition.

    Args: same as main_habituation_runs
    Be careful to create a new random number generator, not re-creating
    the generator with the same seed used in the previous main simulation.

    full_example_file (str): .npz file name in which to save
        all the PN response y vectors to mixtures in the first sim_id. 
    lean (bool): if True, save only the essential results. 
    """
    # Some consistency checks
    assert params['snap_times'].size == params['repeats'][1]
    # Check that the result file can be loaded
    results_file = h5py.File(filename, "r+")
    assert attrs["main_seed"] == results_file.attrs["main_seed"]
    # Use the previous random number generator to create a new seed
    main_seed_seq, main_rgen = new_rng_from_old_seed(attrs["main_seed"])
    repeats, dimensions = params["repeats"], params['dimensions']

    # Create new odors and save to odors group
    new_odors = generate_odorant(
                        [repeats[3], dimensions[0]], main_rgen, lambda_in=0.1
                    )
    new_odors /= l2_norm(new_odors).reshape(*new_odors.shape[:-1], 1)
    odors_group = results_file.get("odors")

    # Save the new odors and the projection kwargs to disk
    try:
        odors_group.create_dataset("new_odors", data=new_odors)
        dict_to_hdf5(results_file.create_group("proj_kwargs"), proj_kwargs)
    except ValueError:
        pass  # already exists (for now)

    # Define callback functions
    def callback(result):
        sim_id, sim_results = result
        sim_gp = results_file.get(id_to_simkey(sim_id))
        if sim_gp.get("test_results") is None:
            if not lean:
                csr_matrix_to_hdf5(sim_gp.create_group("new_odor_tags"),
                            sim_results.pop("new_odor_tags"))
                sim_results.pop("mixture_tags").to_hdf(
                                sim_gp.create_group("mixture_tags"))
            dict_to_hdf5(sim_gp.create_group("test_results"), sim_results)
        else:  # Group already exists, skip
            print("test_results group for sim {}".format(sim_id)
                    + " already exists; not saving")
        # Save sample response to new odors
        if sim_id == 0 and full_example_file is not None and not lean:
            add_to_npz(full_example_file, 
                {"mixture_svecs": sim_results["mixture_svecs"]})
            
        print("New odor recognition tested for simulation {}".format(sim_id))
        return sim_id

    # 2. For each background and run, test new odor recognition at snap times.
    # Create a projection matrix, save to HDF file.
    # Against n_back_samples backgrounds, including the simulation one.
    # and test at 20 % or 50 % concentration
    all_seeds = main_seed_seq.spawn(repeats[0])
    n_workers = min(count_parallel_cpu(), repeats[0])
    pool = multiprocessing.Pool(n_workers)
    for sim_id in range(repeats[0]):
        # Retrieve relevant results of that simulation,
        # then create and s ave the proj. mat., and initialize arguments
        sim_gp = results_file.get(id_to_simkey(sim_id))
        apply_args = initialize_recognition(
                    sim_id, n_workers, sim_gp, odors_group, attrs, params,
                    model_options, main_rgen, all_seeds[sim_id], 
                    proj_kwargs, lean=lean
        )
        pool.apply_async(
            func_wrapper_with_id_threadpool, args=apply_args, 
            callback=callback, error_callback=error_callback
        )

    # Close and join the Pool to finish processes, then close the file
    pool.close()
    pool.join()
    results_file.close()
    return 0
