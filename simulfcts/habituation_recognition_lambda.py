"""
Functions similar to habituation_recognition.py, but for evaluation of
model performance as a function of the scaling factor Lambda. 
"""
import numpy as np
import h5py
import multiprocessing

# Local imports
from modelfcts.backgrounds import generate_odorant
from modelfcts.tagging import (
    create_sparse_proj_mat,
    SparseNDArray
)
from utils.statistics import seed_from_gen
from utils.cpu_affinity import count_parallel_cpu, count_threads_per_process
from utils.metrics import l2_norm, jaccard
from utils.export import (
    dict_to_hdf5,
    csr_matrix_to_hdf5,
)

from simulfcts.habituation_recognition import (
    test_new_odor_recognition,
    test_new_odor_recognition_lean, 
    func_wrapper_with_id_threadpool, 
    find_snap_index, 
    id_to_simkey, 
    select_model_functions, 
    error_callback, 
    initialize_background, 
    initialize_weights, 
    get_data, 
    new_rng_from_old_seed
)


def initialize_integration_lambda(
        id,  nwork, lambd, gp, attrs, params, modopt, back, inits, spseed
    ):
    """ Create a list of args and dictionary of kwargs to launch
    the main habituation parallel runs.
    """
    # Select appropriate functions for the model and background
    # Yes we could do that outside of the loop, once for all repeats
    # but it's easier to do it here for code clarity
    integrate, back_update, noise_type = select_model_functions(attrs)
    back_init, [init_weights, init_weight_names] = inits
    back_params_sim = params["back_params"] + [back]
    # Adjust Lambda parameter in the list of parameters
    # Also adjust initial m weights
    adjusted_m_rates = list(params["m_rates"])
    if attrs["model"] == "IBCM":
        adjusted_m_rates[3] = lambd
        assert init_weight_names == "m_init", "Account for new weight names"
        adjusted_inits = init_weights * lambd
    elif attrs["model"] == "PCA":
        adjusted_m_rates[2] = lambd
        assert init_weight_names[0] == "m_init", "Account for new weight names"
        adjusted_inits = [lambd*init_weights[0], *init_weights[1:]]
    else:
        raise ValueError("Lambda not in model {}".format(attrs["model"]))

    # Threads per process
    threads_per_process = count_threads_per_process(nwork)
    apply_args = (integrate,
                    id,
                    threads_per_process,
                    adjusted_inits,
                    back_update,
                    back_init,
                    adjusted_m_rates,
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
            gp.create_dataset(lbl, data=adjusted_inits[i])
    else:
        gp.create_dataset(init_weight_names, data=adjusted_inits)
    bk_init_gp = gp.create_group("bk_init")
    bk_init_gp.create_dataset("bk_vari_init", data=back_init[0])
    bk_init_gp.create_dataset("bk_vec_init", data=back_init[1])
    return apply_args, apply_kwargs


def save_simul_results_lambda(id, res, attrs, gp, snap_i):
    result_items = {
        "IBCM": ["tser", "back_conc_snaps", "back_vec_snaps", "m_snaps",
                 "cbar_snaps", "theta_snaps", "w_snaps", "s_snaps"],
        "PCA": ["tser", "back_conc_snaps", "back_vec_snaps", "m_snaps",
                 "l_snaps", "x_snaps", "cbar_snaps", "w_snaps", "s_snaps"],
        "AVG": ["tser", "back_conc_snaps", "back_vec_snaps",
                 "w_snaps", "s_snaps"]
    }
    try:
        item_names = result_items.get(attrs["model"])
    except KeyError:
        raise NotImplementedError("Treat output of other models")
    for i, lbl in enumerate(result_items[attrs["model"]]):
        if lbl == "tser" or res[i] is None: continue  # don't save this one
        elif lbl == "back_conc_snaps" and res[i].ndim == 3:
            dset = res[i][snap_i, :, -1]  # Keep only concentrations
        elif lbl == "s_snaps":
            dset = res[i][snap_i]
            transient = 8 * res[i].shape[0] // 10
            snorm = l2_norm(res[i][transient:], axis=1)
            s_stats = np.asarray([
                np.mean(snorm), np.var(snorm),
                np.mean((snorm - np.mean(snorm))**3)
            ])
            gp.create_dataset("s_stats", data=s_stats)
        elif lbl == "cbar_snaps":
            # Save maximum cbar encountered, to trace back blowups to
            # large cbar causing instability of the W numerical integrator
            dset = res[i][snap_i]
            # Take the maximum norm; this could either over or under-estimate
            # the stability threshold, depending on 1) how long the whiff 
            # causing max hnorm lasted, and 2) whether a larger even has been
            # discarded when skipping steps in saved time courses. 
            cbar_max_norm = np.max(l2_norm(res[i], axis=1))
            gp.create_dataset("cbar_max_norm", data=np.asarray([cbar_max_norm]))
        else:
            dset = res[i][snap_i]
        gp.create_dataset(lbl, data=dset.copy())
    return gp


# Just save the bare minimum, don't save all tags etc. for now.
def main_habituation_runs_lambda(filename, attributes, parameters, model_options):
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
                                    n_new_odors, skip_steps]
            m_rates (np.ndarray): depends on the model, should be consistent
            w_rates (np.ndarray): alpha, beta
            time_params (np.ndarray): tmax, dt
            back_params (np.ndarray): background process parameters
            snap_times (np.ndarray): times at which snapshots are taken
        model_options (dict): model options, will be saved
            in the parameters group attrs in HDF5 file.
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
                            [dimensions[1], dimensions[0]],
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

    # Define the range of new lambdas
    if attributes["model"] == "IBCM":
        lambda_0 = parameters["m_rates"][3]
    elif attributes["model"] == "PCA":
        lambda_0 = parameters["m_rates"][2]
    else:
        raise ValueError("Lambda not in model {}".format(attributes["model"]))
    lambd_range = np.geomspace(0.1, 4.0, repeats[0]) * lambda_0
    param_group.create_dataset("lambd_range", data=lambd_range)

    # Initial weights, same for all
    # init_weights, init_weight_names
    init_weights_info = initialize_weights(
                            attributes, parameters["dimensions"],
                            main_rgen, parameters
                        )
    # Initialize background, same for all simulations too
    back_init_sim = initialize_background(
                    attributes, parameters["dimensions"],
                    parameters["back_params"], back_odors, main_rgen
                )
    all_inits = [back_init_sim, init_weights_info]

    # 1. Run habituation against each background
    # Use a callback function to save only snapshots to disk
    # Keep new odor recognition tests separate (launch another multiprocess)
    def callback(result):
        # Get the result being returned: id, [all results]
        sim_id, sim_results = result
        # Get the group created at launch time for this simulation id
        sim_gp = results_file.get(id_to_simkey(sim_id))
        # Save results, depending on model type the structure changes a bit
        # TODO: Save statistics of inhibition in sser as well
        save_simul_results_lambda(sim_id, sim_results, attributes, 
                                  sim_gp, snap_idx)
        print("Habituation run {} saved".format(sim_id))
        del sim_results, result
        return sim_id

    # Initialize and launch all simulations
    # Use the same seed for all Lambdas.
    spawned_seed = main_seed_seq.spawn(1)[0]
    n_workers = min(count_parallel_cpu(), repeats[0])
    pool = multiprocessing.Pool(n_workers)
    for sim_id in range(repeats[0]):
        # Package simulation arguments and save initialization
        # to a new group for this simulation
        sim_gp = results_file.create_group(id_to_simkey(sim_id))
        lambd = lambd_range[sim_id]
        sim_gp.attrs["lambd"] = lambd
        # id, lambd, gp, attrs, params, modopt, back, inits, spseed
        apply_args, apply_kwargs = initialize_integration_lambda(
                    sim_id, n_workers, lambd, sim_gp, attributes, parameters,
                    model_options, back_odors, all_inits, spawned_seed
        )
        pool.apply_async(
                    func_wrapper_with_id_threadpool, args=apply_args,
                    kwds=apply_kwargs, callback=callback,
                    error_callback=error_callback
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


def initialize_recognition_lambda(id, nwork, gp, odors_gp, 
    attrs, params, modopt, p_matrix, spseed, projkw):
    """ Function to arrange the arguments of multiprocessing pool apply
    for new odor detection tests
    """
    # Adjust Lambda parameter in the list of parameters
    lambd = gp.attrs["lambd"]
    adjusted_m_rates = list(params["m_rates"])
    if attrs["model"] == "IBCM":
        adjusted_m_rates[3] = lambd
    elif attrs["model"] == "PCA":
        adjusted_m_rates[2] = lambd
    else:
        raise ValueError("Lambda not in model {}".format(attrs["model"]))
    params_adjusted = {l:p for l, p in params.items()}
    params_adjusted["m_rates"] = adjusted_m_rates

    # Create projection matrix
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
        "back": get_data(odors_gp, "back_odors"),
        "new": get_data(odors_gp, "new_odors"),
    }
    # Construct dictionaries to pass to test_new_odor_recognition
    test_params = {
        "test_seed_seq": spseed,
        "pmat": p_matrix,
        "proj_kwargs": projkw,
        "model_options": modopt,
    }
    n_threads = count_threads_per_process(nwork)
    apply_args = (test_new_odor_recognition, id, n_threads, snaps_dict,
                    attrs, params, sim_odors_dict, test_params)
    return apply_args


def main_performance_lambda(filename, attrs, params, model_options, proj_kwargs):
    """ After having performed several habituation runs and saved these
    results to HDF, load the results and test new odor recognition.

    Args: same as main_habituation_runs
    Be careful to create a new random number generator, not re-creating
    the generator with the same seed used in the previous main simulation.
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

    # Create a unique KC projection matrix for all Lambda values 
    p_matrix = create_sparse_proj_mat(
                    params["dimensions"][3], params["dimensions"][0], main_rgen
                )
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
        csr_matrix_to_hdf5(sim_gp.create_group("new_odor_tags"),
                            sim_results.pop("new_odor_tags"))
        sim_results.pop("mixture_tags").to_hdf(
                                sim_gp.create_group("mixture_tags"))
        dict_to_hdf5(sim_gp.create_group("test_results"), sim_results)
        print("New odor recognition tested for simulation {}".format(sim_id))
        return sim_id

    # 2. For each background and run, test new odor recognition at snap times.
    # Create a projection matrix, save to HDF file.
    # Against n_back_samples backgrounds, including the simulation one.
    # and test at 20 % or 50 % concentration
    # We want the same test seed for all Lambdas, for better direct comparison
    testing_seed = main_seed_seq.spawn(1)[0]
    n_workers = min(count_parallel_cpu(), repeats[0])
    pool = multiprocessing.Pool()
    for sim_id in range(repeats[0]):
        # Retrieve relevant results of that simulation,
        # then create and save the proj. mat., and initialize arguments
        sim_gp = results_file.get(id_to_simkey(sim_id))
        # Lambda parameter gets adjusted in the initialization
        apply_args = initialize_recognition_lambda(
                    sim_id, n_workers, sim_gp, odors_group, attrs, params,
                    model_options, p_matrix, testing_seed, proj_kwargs
                    )
        pool.apply_async(func_wrapper_with_id_threadpool, args=apply_args,
                        callback=callback, error_callback=error_callback)

    # Close and join the Pool to finish processes, then close the file
    pool.close()
    pool.join()
    results_file.close()
    return 0
