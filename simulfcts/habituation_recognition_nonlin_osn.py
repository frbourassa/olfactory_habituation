"""
Mdule with functions to compare the performance of the IBCM model
to online PCA, average background subtraction, and no habituation for
new odor recongition, when OSNs have a nonlinear response function. 

Similar to the regular habituation_recognition.py module, but with the
appropriate odor generation, background update, and odor mixing functions. 

For detailed documentation on the HDF results file structure, see the file
results/performance_hdf_file_structure_doc.txt
All HDF results files are saved in the results/performance/ folder.

@author: frbourassa
July 2023
"""
import numpy as np
import h5py
import multiprocessing
from utils.profiling import IterationProfiler


# Local imports
from modelfcts.ideal import relu_inplace
from modelfcts.nonlin_adapt_osn import (
    combine_odors_affinities,
    combine_odors_compet, 
    generate_odor_tanhcdf
)
from modelfcts.backgrounds import (
    generate_odorant, 
)
from modelfcts.tagging import (
    project_neural_tag,
    create_sparse_proj_mat
)
from simulfcts.habituation_recognition import (
    select_sampling_functions,
    appropriate_response,  # TODO: will need new version to add OSN adaptation
    get_data,
    new_rng_from_old_seed,
    func_wrapper_with_id_threadpool,
    find_snap_index,
    id_to_simkey,
    save_simul_results,  # TODO: will need new version to add OSN adaptation
    error_callback,
    initialize_integration
)
from utils.cpu_affinity import count_parallel_cpu, count_threads_per_process
from utils.metrics import l2_norm, jaccard
from utils.export import (
    dict_to_hdf5,
    save_params_individually,
    csr_matrix_to_hdf5,
    add_to_npz,
    hdf5_to_dict
)


### UTILITY FUNCTIONS ###
def select_osn_response(attrs):
    response_fct_map = {
        "turbulent_nl_osn": combine_odors_affinities,
        "turbulent_compet_osn": combine_odors_compet
    }
    try:
        response_fct = response_fct_map[attrs["background"]]
    except KeyError:
        raise KeyError("Unknown background: {}".format(attrs["background"]))
    return response_fct


def select_odor_gen_fct(attrs):
    generation_fct_map = {
        "turbulent_nl_osn": generate_odor_tanhcdf,
        "turbulent_compet_osn": generate_odorant
    }
    try:
        generation_fct = generation_fct_map[attrs["background"]]
    except KeyError:
        raise KeyError("Unknown background: {}".format(attrs["background"]))
    return generation_fct


### MAIN HABITUATION SIMULATION FUNCTION ###
def main_habituation_runs_nl_osn(
    filename, attributes, parameters, model_options, odor_gen_kwargs,
    save_fct=save_simul_results, full_example_file=None, lean=True
):
    """
    Args: everything in the .attrs and parameters group of the HDF file that
    will be saved. Also the path and file name.
        filename (str): path and file name of the HDF results file to create.
        attributes (dict): attributes dictionary as in the HDF file, contains
            model (str): either "IBCM", "PCA", "AVG"
            background (str): typically "turbulent_nl_osn"
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
        odor_gen_kwargs (dict): odor generation function options. 
        save_fct (callable): default is save_simul_results, but can be changed
            if we need to save different things for other kinds of simulations
        full_example_file (str): if a filename is passed, save one full 
            habituation run in that location, for illustration purposes. 
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

    # Create background odor affinities; new odors not needed here.
    # Dimension of all background components array: [n_runs, n_b, n_r]
    odor_gen_fct = select_odor_gen_fct(attributes)
    back_odors_shape = [repeats[0], dimensions[1], dimensions[0]]
    back_odors = odor_gen_fct(back_odors_shape, main_rgen, **odor_gen_kwargs)

    # Create HDF file that will contain all simulation results
    results_file = h5py.File(filename, "w")
    dict_to_hdf5(results_file.attrs, attributes)
    param_group = results_file.create_group("parameters")
    #param_group = dict_to_hdf5(param_group, parameters)
    param_group = save_params_individually(param_group, parameters)
    # Also save model options in the param_group.attrs
    dict_to_hdf5(param_group.attrs, model_options)
    # And save odor generation kwargs
    dict_to_hdf5(results_file.create_group("odor_gen_kwargs"), odor_gen_kwargs)

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


### ODOR RECOGNITION TEST FUNCTIONS ###
def test_new_odor_recognition_nl_osn(
        snaps, attrs, params, sim_odors, test_params):
    """ Version of test_new_odor_recognition for non-linear OSN responses.
    It only saves jaccard scores and distances between new odor and mixture 
    y vectors, without storing in memory the full matrices of y vectors and 
    neural tags, only the ones relevant to the current iteration.  
    
    Args:
        snaps (dict):
            "m": snapshots of M
            "w": snapshots of W
            "l": snapshots of L (if applicable)
            "x": snapshots of background moving average, if applicable
            "conc": snapshots of background concentrations
            "back": snapshots of the simulated background x_b.
        attrs (dict): "model", "background" are relevant here
        params (dict): dimensions, repeats, m_rates, back_params,
            new_concs are relevant and same for all simulations.
        sim_odors (dict):
            "back": the background odors of that simulation
                (not all of them like in the HDF file), shape [n_aff, n_b, n_r]
                The first axis indexes the n_aff affinities describing an odor
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
    activ_fct = test_params.get("model_options",{}).get("activ_fct","identity")
    _, vec_sampling = select_sampling_functions(attrs)
    response_fct = select_osn_response(attrs)
    
    # TODO: for adaptation, epsilons will be dynamical variables in snapshots
    epsils_vec = params["back_params"][-1]  
    max_osn_ampli = params["back_params"][-2]
    
    # Generate new background samples (stationary)
    test_rgen = np.random.default_rng(test_params["test_seed_seq"])
    back_samples, conc_samples = vec_sampling(
                        sim_odors["back"], 
                        *params["back_params"],
                        size=n_times*(n_back_samples-1), rgen=test_rgen
                    )  # Shaped [sample, component]
    back_samples = back_samples.reshape([n_times, n_back_samples-1, -1])
    conc_samples = conc_samples.reshape([n_times, n_back_samples-1, -1])
    # Append the background snapshots to them
    back_samples = np.concatenate(
                    [snaps["back"][:, None, :], back_samples], axis=1)
    conc_samples = np.concatenate(
                    [snaps["conc"][:, None, :], conc_samples], axis=1)
    # Loop over new odors first
    n_new_odors = params['repeats'][3]
    n_new_concs = params['repeats'][4]
    n_kc = params['dimensions'][3]
    n_back_dims = params["dimensions"][1]
    assert n_kc == test_params["pmat"].shape[0], "Inconsistent KC number"
    jaccard_scores = np.zeros((n_new_odors, n_times, 
                               n_new_concs, n_back_samples))
    y_l2_distances = np.zeros((n_new_odors, n_times, 
                               n_new_concs, n_back_samples))
    
    # Also compute similarity to background. First, need tags of back odors
    jaccard_scores_back = np.zeros(jaccard_scores.shape)
    jaccard_backs_indiv = np.zeros(n_back_dims)
    back_tags = []
    # For single-odor responses, typical concentration
    standard_conc = np.asarray([np.mean(params["new_concs"])])
    for b in range(n_back_dims):
        # Compute the OSN response at a standard concentration
        odor_vec = sim_odors["back"][b:b+1]
        back_response = response_fct(standard_conc, odor_vec, 
                                     epsils_vec, fmax=max_osn_ampli)
        back_tags.append(
            project_neural_tag(back_response, back_response,
                test_params['pmat'], **test_params['proj_kwargs'])
        )

    # Profiling only one simulation, to keep track of time without
    # producing too much text. 
    switch = False
    sim_id = int(test_params["test_seed_seq"].spawn_key[0])
    if sim_id % 32 == 0:
        profiler = IterationProfiler(sim_id)
    for i in range(n_new_odors):
        # Get new odor i's affinitiy parameters, and compute neural tag of the
        # nonlinear OSN response to this new odor alone, at standard conc.
        new_odor = sim_odors["new"][i:i+1]
        # Turn on switch to profile one iteration every 50 odors
        if i % 25 == 1 and sim_id % 32 == 0:  
            switch = True
            profiler.start(f"i=={i}")
        # Apply the OSN response function to it for a standard conc.
        new_odor_response = response_fct(standard_conc, new_odor, 
                                         epsils_vec, fmax=max_osn_ampli)
        new_tag = project_neural_tag(
            new_odor_response, new_odor_response, test_params["pmat"], 
            **test_params["proj_kwargs"]
        )

        # Prepare joint affinity matrices for the background + new odor
        joint_kmats = np.concatenate([sim_odors["back"], new_odor])
        
        # Now, loop over snapshots, mix the new odor with the back samples,
        # compute the PN response at each test concentration,
        # compute tags too, and save results
        for j in range(n_times):
            for k in range(n_new_concs):
                new_conc = params["new_concs"][k]
                joint_concs = np.concatenate([conc_samples[j], 
                        np.full((n_back_samples, 1), new_conc)], axis=1)
                # mixtures is shaped [back_sample, OSN dimension]
                # so the new_odor broadcasts against it well. 
                mixtures = response_fct(joint_concs, joint_kmats, 
                                        epsils_vec, fmax=max_osn_ampli)
                if switch: profiler.addpoint("compute mixtures")
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
                ydiffs = mixture_yvecs_ijk - params["new_concs"][k] * new_odor_response
                if switch: profiler.addpoint("compute ydiffs")
                y_l2_distances[i, j, k] = l2_norm(ydiffs, axis=1)
                if switch: profiler.addpoint("computing l2_norm of ydiffs")
                # Compute Jaccard similarity between new odor tag
                # and tags of responses to mixtures with back_samples
                for l in range(n_back_samples):
                    mix_tag = project_neural_tag(
                        mixture_yvecs_ijk[l], mixtures[l],
                        test_params['pmat'], **test_params['proj_kwargs']
                    )
                    if switch: profiler.addpoint("computing mixture tag")
                    jaccard_scores[i, j, k, l] = jaccard(mix_tag, new_tag)
                    if switch: profiler.addpoint("computing jaccard")
                     # Also save similarity to the most similar background odor
                    for b in range(n_back_dims):
                        jaccard_backs_indiv[b] = jaccard(mix_tag, back_tags[b])
                    jaccard_scores_back[i, j, k, l] = np.amax(jaccard_backs_indiv)
                    if switch: profiler.addpoint("computing jaccards with back")
                    if switch:
                        profiler.end_iter()
                        switch = False  # Stop profiling after first pass
    # Prepare simulation results dictionary
    test_results = {
        "conc_samples": conc_samples,
        "jaccard_scores": jaccard_scores, 
        "jaccard_scores_back": jaccard_scores_back,
        "y_l2_distances": y_l2_distances
    }
    return test_results


def initialize_recognition_nl_osn(id, nwork, gp, odors_gp, 
    attrs, params, modopt, rgen, spseed, projkw, lean=True):
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
        # TODO: add epsilons snaps here later for adaptation
        "conc": get_data(gp, "back_conc_snaps"),
        "back": get_data(gp, "back_vec_snaps"),
    }
    # Load odors locally, argument to pass to each test process
    sim_odors_dict = {
        "back": get_data(odors_gp, "back_odors")[id],
        "new": get_data(odors_gp, "new_odors"),
    }
    # Construct dictionaries to pass to test_new_odor_recognition_nl_osn
    test_params = {
        "test_seed_seq": spseed,
        "pmat": p_matrix,
        "proj_kwargs": projkw,
        "model_options": modopt,
    }
    gp.attrs["spawn_seed"] = str(hex(spseed.entropy))
    test_fct = test_new_odor_recognition_nl_osn
    n_threads = count_threads_per_process(nwork)
    apply_args = (test_fct, id, n_threads, snaps_dict,
                    attrs, params, sim_odors_dict, test_params)
    return apply_args


def main_recognition_runs_nl_osn(
        filename, attrs, params, model_options, proj_kwargs, 
        odor_gen_kwargs, full_example_file=None, lean=True
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

    # Check the odor generation kwargs are the same
    gen_kwargs2 = hdf5_to_dict(results_file.get("odor_gen_kwargs"))
    for k in gen_kwargs2.keys():
        assert odor_gen_kwargs.get(k) == gen_kwargs2.get(k)

    # Create new odors and save to odors group
    # Dimension of all new odors array: [n_runs, n_b, n_r]
    odor_gen_fct = select_odor_gen_fct(attrs)
    new_odors_shape = [repeats[3], dimensions[0]]
    new_odors = odor_gen_fct(new_odors_shape, main_rgen, **odor_gen_kwargs)
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
                {"mixture_yvecs": sim_results["mixture_yvecs"]})
            
        print("New odor recognition tested for simulation {}".format(sim_id))
        return sim_id

    # 2. For each background and run, test new odor recognition at snap times.
    # Create a projection matrix, save to HDF file.
    # Against n_back_samples backgrounds, including the simulation one.
    all_seeds = main_seed_seq.spawn(repeats[0])
    n_workers = min(count_parallel_cpu(), repeats[0])
    pool = multiprocessing.Pool(n_workers)
    for sim_id in range(repeats[0]):
        # Retrieve relevant results of that simulation,
        # then create and s ave the proj. mat., and initialize arguments
        sim_gp = results_file.get(id_to_simkey(sim_id))
        apply_args = initialize_recognition_nl_osn(
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
