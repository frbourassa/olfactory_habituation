"""
Functions to test the ideal habituation scenarios.
We save some of the information, in particular:
    - s vectors (after ideal habituation)
    - jaccard scores
    - tags of the orthogonal components of the new odor

We reuse all background samples and new odors from an existing simulation, to
make sure we compare ideal inhibition to habituation models on fair grounds.

@author: frbourassa
October 2023
"""
import numpy as np
from scipy import sparse
import h5py
import multiprocessing
from modelfcts.ideal import (
    find_projector,
    find_parallel_component,
    ideal_linear_inhibitor
)
from modelfcts.tagging import (
    project_neural_tag,
    SparseNDArray
)
from utils.export import (
    hdf5_to_dict,
    dict_to_hdf5,
    hdf5_to_csr_matrix,
    csr_matrix_to_hdf5
)
from utils.metrics import jaccard
from utils.cpu_affinity import count_parallel_cpu
from simulfcts.habituation_recognition import (
    error_callback,
    id_to_simkey
)
from modelfcts.backgrounds import generate_odorant


def no_habituation_one_sim(sim_id, filename_ref):
    """ Load new odors, background samples and projection matrix
    for a given simulation in ref_file, and test odor recognition
    without any habituation.
    """
    ref_file = h5py.File(filename_ref, "r")
    sim_gp = ref_file.get(id_to_simkey(sim_id))
    projmat = hdf5_to_csr_matrix(sim_gp.get("kc_proj_mat"))
    new_odors = ref_file.get("odors").get("new_odors")[()]
    back_odors = ref_file.get("odors").get("back_odors")[sim_id, :, :]
    new_concs = ref_file.get("parameters").get("new_concs")[()]

    # Dimensions, etc.
    n_r, n_b, _, n_kc = ref_file.get("parameters").get("dimensions")
    (n_times, n_back_samples, n_new_odors,
        n_new_concs) = ref_file.get("parameters").get("repeats")[1:5]
    proj_kwargs = hdf5_to_dict(ref_file.get("proj_kwargs"))
    activ_fct = ref_file.get("parameters").attrs.get("activ_fct", "ReLU")

    # Get the average of background samples to set the KC thresholds
    back_samples = sim_gp.get("test_results").get("back_samples")[()]

    # Containers for the results
    jaccard_scores = np.zeros([n_new_odors, n_times,
                                n_new_concs, n_back_samples])
    new_odor_tags = sparse.lil_array((n_new_odors, n_kc), dtype=bool)
    mixture_tags = SparseNDArray( (n_new_odors, n_times, n_new_concs,
                                n_back_samples, n_kc), dtype=bool)
    mixture_svecs = np.zeros([n_new_odors, n_times,
                                n_new_concs, n_back_samples, n_r])

    # Treat one new odor at a time, one new conc. at a time, etc.
    for i in range(n_new_odors):
        new_tag = project_neural_tag(
                    new_odors[i], new_odors[i], projmat, **proj_kwargs)
        new_odor_tags[i, list(new_tag)] = True
        for k in range(n_new_concs):
            mixtures = back_samples + new_concs[k]*new_odors[i]
            mixture_svecs[i, :, k] = mixtures
            # We actually don't want to apply ReLU to understand what
            # happens if some svec is zero.
            #if str(activ_fct).lower() == "relu":
            #    mixture_svecs[i, :, k] = relu_inplace(mixture_svecs[i, :, k])
            for j in range(n_times):
                for l in range(n_back_samples):
                    mix_tag = project_neural_tag(
                        mixture_svecs[i, j, k, l], mixtures[j, l],
                        projmat, **proj_kwargs
                    )
                    try:
                        mixture_tags[i, j, k, l, list(mix_tag)] = True
                    except ValueError as e:
                        print(mix_tag)
                        print(mixture_svecs[i, j, k, l])
                        print(projmat.dot(mixture_svecs[i, j, k, l]))
                        raise e
                    jaccard_scores[i, j, k, l] = jaccard(mix_tag, new_tag)
    ref_file.close()

    # Prepare simulation results dictionary
    new_odor_tags = new_odor_tags.tocsr()
    test_results = {
        "new_odor_tags": new_odor_tags,
        "mixture_svecs": mixture_svecs,
        "mixture_tags": mixture_tags,
        "jaccard_scores": jaccard_scores
    }
    return sim_id, test_results, projmat


def orthogonal_recognition_one_sim(sim_id, filename_ref):
    """ Load new odors, background samples and projection matrix
    for a given simulation in ref_file, and test odor recognition
    after ideal habituation where all that is left is the new odor component
    perpendicular to the background subspace.

    We don't need background samples: we assume the entire background
    is removed anyways. This changes a little bit dimensionalities:
        - We don't need back_samples or back_concs
        - mixture_svecs has shape [n_new_odors, 1,  n_new_concs, 1, n_r]
            as it is simply the orthogonal component of the new odor
        - mixture_tags has shape [n_new_odors, 1,  n_new_concs, 1, n_kc]
        - jaccard_scores has shape [n_new_odors, 1, n_new_concs, 1]
    new_odor_tags still has shape [n_new_odors, n_kc]

    This is actually not the best method at very low concentrations,
    since the orthogonal component can get lost in the KC threshold.
    """
    ref_file = h5py.File(filename_ref, "r")
    sim_gp = ref_file.get(id_to_simkey(sim_id))
    projmat = hdf5_to_csr_matrix(sim_gp.get("kc_proj_mat"))
    new_odors = ref_file.get("odors").get("new_odors")[()]
    back_odors = ref_file.get("odors").get("back_odors")[sim_id, :, :]
    new_concs = ref_file.get("parameters").get("new_concs")[()]

    # Dimensions, etc.
    n_r, n_b, _, n_kc = ref_file.get("parameters").get("dimensions")
    n_new_odors, n_new_concs = ref_file.get("parameters").get("repeats")[3:5]
    proj_kwargs = hdf5_to_dict(ref_file.get("proj_kwargs"))
    activ_fct = ref_file.get("parameters").attrs.get("activ_fct", "ReLU")

    # Get the average of background samples to set the KC thresholds
    average_back = np.mean(sim_gp.get("test_results").get("back_samples"),
                        axis=(0, 1))

    # Projector to subtract the parallel component
    projector = find_projector(back_odors.T)

    # Containers for the results
    jaccard_scores = np.zeros([n_new_odors, 1, n_new_concs, 1])
    new_odor_tags = sparse.lil_array((n_new_odors, n_kc), dtype=bool)
    mixture_tags = SparseNDArray(
                    (n_new_odors, 1, n_new_concs, 1, n_kc), dtype=bool)
    mixture_svecs = np.zeros([n_new_odors, 1, n_new_concs, 1, n_r])
    # Treat one new odor at a time, one new conc. at a time, etc.
    for i in range(n_new_odors):
        x_par = find_parallel_component(new_odors[i], None, projector)
        new_tag = project_neural_tag(
                    new_odors[i], new_odors[i], projmat, **proj_kwargs)
        new_odor_tags[i, list(new_tag)] = True
        for j in range(n_new_concs):
            # Project new odor, project its perpendicular component
            svec = new_concs[j]*(new_odors[i] - x_par)
            # We actually don't want to apply ReLU to understand what
            # happened if some s vector is zero
            #if str(activ_fct).lower() == "relu":
            #    svec = relu_inplace(svec)
            mixture_svecs[i, 0, j, 0] = svec
            x_mix = average_back + new_concs[j]*new_odors[i]
            # TODO: the threshold scale should be determined from the
            # actual mixture of background + conc*new_odor,
            # so I should load background samples after all.
            # Unless I start using a fixed, small threshold always.
            perp_tag = project_neural_tag(
                        svec, x_mix, projmat, **proj_kwargs
                        )
            mixture_tags[i, 0, j, 0, list(perp_tag)] = True
            jaccard_scores[i, 0, j, 0] = jaccard(new_tag, perp_tag)
    ref_file.close()

    # Prepare simulation results dictionary
    new_odor_tags = new_odor_tags.tocsr()
    test_results = {
        "new_odor_tags": new_odor_tags,
        "mixture_svecs": mixture_svecs,
        "mixture_tags": mixture_tags,
        "jaccard_scores": jaccard_scores
    }
    return sim_id, test_results, projmat


def ideal_recognition_one_sim(sim_id, filename_ref):
    """ Load new odors, background samples and projection matrix
    for a given simulation in ref_file, and test odor recognition
    after ideal habituation where the background and parallel
    component are reduced by the optimal factor.

    This is not the simplest idealized inhibition, but probably the best,
    especially at low concentrations where KC thresholds can be a problem.
    """
    ref_file = h5py.File(filename_ref, "r")
    sim_gp = ref_file.get(id_to_simkey(sim_id))
    projmat = hdf5_to_csr_matrix(sim_gp.get("kc_proj_mat"))
    new_odors = ref_file.get("odors").get("new_odors")[()]
    back_odors = ref_file.get("odors").get("back_odors")[sim_id, :, :]
    new_concs = ref_file.get("parameters").get("new_concs")[()]
    moments_conc = ref_file.get("parameters").get("moments_conc")[()]
    w_rates = ref_file.get("parameters").get("w_rates")
    # IBCM factor: this can be too much reduction compared to the optimum.
    #factor = w_rates[1] / (2*w_rates[0] + w_rates[1])

    # Dimensions, etc.
    n_r, n_b, _, n_kc = ref_file.get("parameters").get("dimensions")
    (n_times, n_back_samples, n_new_odors,
        n_new_concs) = ref_file.get("parameters").get("repeats")[1:5]
    proj_kwargs = hdf5_to_dict(ref_file.get("proj_kwargs"))
    activ_fct = ref_file.get("parameters").attrs.get("activ_fct", "ReLU")

    # Get the average of background samples to set the KC thresholds
    back_samples = sim_gp.get("test_results").get("back_samples")[()]

    # Projector to subtract the parallel component
    projector = find_projector(back_odors.T)

    # Compute optimal factor for each new concentration
    dummy_rgen = np.random.default_rng(0x6e3e2886c30163741daaaf7c8b8a00e6)
    factors = [compute_optimal_factor(c, moments_conc[:2], [n_b, n_r],
                    generate_odorant, (dummy_rgen,)) for c in new_concs]

    # Containers for the results
    jaccard_scores = np.zeros([n_new_odors, n_times,
                                n_new_concs, n_back_samples])
    new_odor_tags = sparse.lil_array((n_new_odors, n_kc), dtype=bool)
    mixture_tags = SparseNDArray( (n_new_odors, n_times, n_new_concs,
                                n_back_samples, n_kc), dtype=bool)
    mixture_svecs = np.zeros([n_new_odors, n_times,
                                n_new_concs, n_back_samples, n_r])

    # Treat one new odor at a time, one new conc. at a time, etc.
    for i in range(n_new_odors):
        x_par = find_parallel_component(new_odors[i], None, projector)
        x_ort = new_odors[i] - x_par
        new_tag = project_neural_tag(
                    new_odors[i], new_odors[i], projmat, **proj_kwargs)
        new_odor_tags[i, list(new_tag)] = True
        for k in range(n_new_concs):
            mixtures = back_samples + new_concs[k]*new_odors[i]
            mixture_svecs[i, :, k] = (factors[k]*back_samples
                        + new_concs[k] * (factors[k]*x_par + x_ort))
            # We actually don't want to apply ReLU to understand what
            # happens if some svec is zero.
            #if str(activ_fct).lower() == "relu":
            #    mixture_svecs[i, :, k] = relu_inplace(mixture_svecs[i, :, k])
            for j in range(n_times):
                for l in range(n_back_samples):
                    mix_tag = project_neural_tag(
                        mixture_svecs[i, j, k, l], mixtures[j, l],
                        projmat, **proj_kwargs
                    )
                    try:
                        mixture_tags[i, j, k, l, list(mix_tag)] = True
                    except ValueError as e:
                        print(mix_tag)
                        print(mixture_svecs[i, j, k, l])
                        print(projmat.dot(mixture_svecs[i, j, k, l]))
                        raise e
                    jaccard_scores[i, j, k, l] = jaccard(mix_tag, new_tag)
    ref_file.close()

    # Prepare simulation results dictionary
    new_odor_tags = new_odor_tags.tocsr()
    test_results = {
        "new_odor_tags": new_odor_tags,
        "mixture_svecs": mixture_svecs,
        "mixture_tags": mixture_tags,
        "jaccard_scores": jaccard_scores,
        "ideal_factors": factors
    }
    return sim_id, test_results, projmat


def idealized_recognition_from_runs(filename, filename_ref, kind="none"):
    """ After having performed several habituation runs and tested them
    for recognition with some model, compute the ideal inhibition
    we could have achieved in these runs.

    Args:
        filename (str): name of the file to contain the ideal results
        filename_ref (str): name of the file with other simulation results
            All necessary information is extracted from there.
        kind (str): the kind of idealized habituation considered,
            either "orthogonal", "none", or "ideal".

    Args: same as main_habituation_runs
    Be careful to create a new random number generator, not re-creating
    the generator with the same seed used in the previous main simulation.
    """
    ref_file = h5py.File(filename_ref, "r")
    res_file = h5py.File(filename, "w")

    # Extract parameters from simulation results
    repeats = ref_file.get("parameters").get("repeats")[()]

    # Copy some params to the results file, in case the ref_file changes
    for k in ref_file.attrs.keys():
        if k == "model":
            res_file.attrs[k] = kind
        else:
            res_file.attrs[k] = ref_file.attrs[k]
    param_group = res_file.create_group("parameters")
    for p in ["dimensions", "repeats", "new_concs", "moments_conc"]:
        param_group.create_dataset(p, data=ref_file.get("parameters").get(p)[()])
    options = ref_file.get("parameters").attrs
    param_group.attrs["activ_fct"] = (ref_file.get("parameters")
                                        .attrs.get("activ_fct"))

    # Background and new odors separately
    odors_group = res_file.create_group("odors")
    for k in ["back_odors", "new_odors"]:
        odors_group.create_dataset(k, data=ref_file.get("odors").get(k))

    # Projection kwargs
    proj_gp = res_file.create_group("proj_kwargs")
    dict_to_hdf5(ref_file.get("proj_kwargs"), proj_gp)

    # Close the file: each sub-process should reopen it to access
    # the mixture samples (we avoid loading all of those in memory)
    # This is safer to avoid hanging threads
    ref_file.close()

    # Choose the right idealized inhibition function
    func_choices = {
        "none": no_habituation_one_sim,
        "orthogonal": orthogonal_recognition_one_sim,
        "ideal": ideal_recognition_one_sim
    }
    recognition_one_sim = func_choices.get(kind, no_habituation_one_sim)

    # Define callback functions
    def callback(result):
        sim_id, sim_results, projmat = result
        sim_gp = res_file.create_group(id_to_simkey(sim_id))
        csr_matrix_to_hdf5(sim_gp.create_group("kc_proj_mat"), projmat)
        # Different for each simulation with its own proj. matrix
        # This is identical to what is saved in the reference file
        # but convenient to have it in case the reference file changes.
        csr_matrix_to_hdf5(sim_gp.create_group("new_odor_tags"),
                            sim_results.pop("new_odor_tags"))
        # After ideal inhibition
        sim_results.pop("mixture_tags").to_hdf(
                                sim_gp.create_group("mixture_tags"))
        dict_to_hdf5(sim_gp.create_group("test_results"), sim_results)
        print("Ideal recognition tested for simulation {}".format(sim_id))
        return sim_id

    # 2. For each background and run, test new odor recognition at snap times.
    # Create a projection matrix, save to HDF file.
    # Against n_back_samples backgrounds, including the simulation one.
    # and test at 20 % or 50 % concentration
    all_returns = []
    pool = multiprocessing.Pool(min(count_parallel_cpu(), repeats[0]))
    for sim_id in range(repeats[0]):
        # Retrieve relevant results of that simulation,
        # then create and save the proj. mat., and initialize arguments
        sim_gp = res_file.get(id_to_simkey(sim_id))
        apply_args = (sim_id, filename_ref)
        res = pool.apply_async(recognition_one_sim, args=apply_args,
                        callback=callback, error_callback=error_callback)
        all_returns.append(res)

    # Get results, callback is applied
    for p in all_returns:
        try: p.get()
        except RuntimeError: pass

    # Finally, close the results file
    res_file.close()
    pool.close()
    return 0
