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
from threadpoolctl import threadpool_limits
from scipy.spatial.distance import pdist, squareform

from modelfcts.ideal import (
    find_projector,
    find_parallel_component,
    ideal_linear_inhibitor,
    compute_ideal_factor, 
    relu_inplace, 
    compute_optimal_matrices
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
from utils.metrics import jaccard, l2_norm
from utils.cpu_affinity import count_parallel_cpu, count_threads_per_process
from simulfcts.habituation_recognition import (
    error_callback,
    id_to_simkey
)
from modelfcts.backgrounds import generate_odorant


def recover_back_samples(gp, backvecs, backtype, reffile, lean):
    if lean:
        nu_samples = gp.get("test_results").get("conc_samples")[()]
        # Add Gaussian noise
        if backtype == "turbulent_gaussnoise":
            n_osn = backvecs.shape[1]
            n_odors = backvecs.shape[0]
            n_missing = n_osn - (nu_samples.shape[-1] - n_odors)
            paramgp = reffile.get("parameters")
            noise_amplis = paramgp["back_params_6"]
            # Need to create a new random generator with unique spawn key
            mainseed = reffile.attrs["main_seed"]
            spseed = int(gp.attrs["spawn_seed"], 16)
            seed_seq = np.random.SeedSequence(int(mainseed), spawn_key=(spseed, n_osn))
            rng = np.random.default_rng(seed_seq)
            extra_gauss = rng.normal(size=(nu_samples.shape[0], 
                                nu_samples.shape[1], n_missing))
            noises = np.concatenate([nu_samples[:, :, n_odors:], extra_gauss], axis=2)
            conc_samples = nu_samples[:, :, :n_odors]
            back_samples = conc_samples.dot(backvecs) + noises*noise_amplis
        else:
            conc_samples = nu_samples
            back_samples = conc_samples.dot(backvecs)
    else:
        back_samples = gp.get("test_results").get("back_samples")[()]
    return back_samples


def no_habituation_one_sim(sim_id, filename_ref, lean=False):
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
    bkname = ref_file.attrs["background"]

    # Dimensions, etc.
    n_r, n_b, _, n_kc = ref_file.get("parameters").get("dimensions")
    (n_times, n_back_samples, n_new_odors,
        n_new_concs) = ref_file.get("parameters").get("repeats")[1:5]
    proj_kwargs = hdf5_to_dict(ref_file.get("proj_kwargs"))
    activ_fct = ref_file.get("parameters").attrs.get("activ_fct", "ReLU")

    # Get the background samples
    # gp, backvecs, backtype, paramgp, lean
    back_samples = recover_back_samples(sim_gp, back_odors, bkname, ref_file, lean)

    # Containers for the results
    jaccard_scores = np.zeros([n_new_odors, n_times,
                            n_new_concs, n_back_samples])
    # Also compute similarity to background. First, need tags of back odors
    jaccard_scores_back = np.zeros(jaccard_scores.shape)
    jaccard_backs_indiv = np.zeros(n_b)
    back_tags = []
    for b in range(n_b):
        back_tags.append(
            project_neural_tag(back_odors[b], back_odors[b],
                projmat, **proj_kwargs
        ))

    if lean:  # only save scores and L2 distances 
        y_l2_distances = np.zeros((n_new_odors, n_times, 
                               n_new_concs, n_back_samples))
    else:  # save the whole y vectors, jaccard scores, etc.
        new_odor_tags = sparse.lil_array((n_new_odors, n_kc), dtype=bool)
        mixture_tags = SparseNDArray( (n_new_odors, n_times, n_new_concs,
                                    n_back_samples, n_kc), dtype=bool)
        mixture_yvecs = np.zeros([n_new_odors, n_times,
                                    n_new_concs, n_back_samples, n_r])

    # Treat one new odor at a time, one new conc. at a time, etc.
    for i in range(n_new_odors):
        new_tag = project_neural_tag(
                    new_odors[i], new_odors[i], projmat, **proj_kwargs)
        if not lean:
            new_odor_tags[i, list(new_tag)] = True
        for k in range(n_new_concs):
            mixtures_ik = back_samples + new_concs[k]*new_odors[i]
            if str(activ_fct).lower() == "relu":
                mixtures_ik = relu_inplace(mixtures_ik)
            if not lean:
                mixture_yvecs[i, :, k] = mixtures_ik
            else:
                # Compute L2 distance between response to mixtures 
                # with back_samples and new odor vector at the current conc. 
                # ydiffs shaped [n_back_samples, n_s], compute norm along axis 1
                ydiffs = mixtures_ik - new_concs[k] * new_odors[i]
                y_l2_distances[i, :, k] = l2_norm(ydiffs, axis=2)
            for j in range(n_times):
                for l in range(n_back_samples):
                    mix_tag = project_neural_tag(
                        mixtures_ik[j, l], mixtures_ik[j, l],
                        projmat, **proj_kwargs
                    )
                    jaccard_scores[i, j, k, l] = jaccard(mix_tag, new_tag)
                    # Also save similarity to the most similar background odor
                    for b in range(n_b):
                        jaccard_backs_indiv[b] = jaccard(mix_tag, back_tags[b])
                    jaccard_scores_back[i, j, k, l] = np.amax(jaccard_backs_indiv)
                    if not lean:
                        try:
                            mixture_tags[i, j, k, l, list(mix_tag)] = True
                        except ValueError as e:
                            print(mix_tag)
                            print(mixture_yvecs[i, j, k, l])
                            print(projmat.dot(mixture_yvecs[i, j, k, l]))
                            raise e
                    
    ref_file.close()

    # Prepare simulation results dictionary
    test_results = {
        "jaccard_scores": jaccard_scores, 
        "jaccard_scores_back": jaccard_scores_back
    }
    if lean:
        test_results["y_l2_distances"] = y_l2_distances
    else:
        new_odor_tags = new_odor_tags.tocsr()
        test_results.update({
            "new_odor_tags": new_odor_tags,
            "mixture_yvecs": mixture_yvecs,
            "mixture_tags": mixture_tags,
        })
    return sim_id, test_results, projmat


def orthogonal_recognition_one_sim_gaussnoise(sim_id, filename_ref, lean=False):
    """ Load new odors, background samples and projection matrix
    for a given simulation in ref_file, and test odor recognition
    after ideal habituation where all that is left is the new odor component
    perpendicular to the background subspace.

    We don't need background samples: we assume the entire background
    is removed anyways. This changes a little bit dimensionalities:
        - We don't need back_samples or back_concs
        - mixture_yvecs has shape [n_new_odors, 1,  n_new_concs, 1, n_r]
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
    bkname = ref_file.attrs["background"]

    # Dimensions, etc.
    n_r, n_b, _, n_kc = ref_file.get("parameters").get("dimensions")
    (n_times, n_back_samples, n_new_odors,
        n_new_concs) = ref_file.get("parameters").get("repeats")[1:5]
    proj_kwargs = hdf5_to_dict(ref_file.get("proj_kwargs"))
    activ_fct = ref_file.get("parameters").attrs.get("activ_fct", "ReLU")

    # Get the average of background samples to set the KC thresholds
    back_samples = recover_back_samples(sim_gp, back_odors, bkname, ref_file, lean)
    average_back = np.mean(back_samples, axis=(0, 1))

    # Projector to subtract the parallel component
    projector = find_projector(back_odors.T)

    # Containers for the results
    jaccard_scores = np.zeros([n_new_odors, n_times, 
                            n_new_concs, n_back_samples])
    # Also compute similarity to background. First, need tags of back odors
    jaccard_scores_back = np.zeros(jaccard_scores.shape)
    jaccard_backs_indiv = np.zeros(n_b)
    back_tags = []
    for b in range(n_b):
        back_tags.append(
            project_neural_tag(back_odors[b], back_odors[b],
                projmat, **proj_kwargs
        ))
    # Since there is orthogonal noise unique to each simulation, 
    # we need to loop over background samples and test times now
    if lean:
        y_l2_distances = np.zeros((n_new_odors, n_times, 
                                n_new_concs, n_back_samples))
    else:
        new_odor_tags = sparse.lil_array((n_new_odors, n_kc), dtype=bool)
        mixture_tags = SparseNDArray((n_new_odors, n_times, n_new_concs,
                                    n_back_samples, n_kc), dtype=bool)
        mixture_yvecs = np.zeros([n_new_odors, n_times,
                                n_new_concs, n_back_samples, n_r])
    # Treat one new odor at a time, one new conc. at a time, etc.
    for i in range(n_new_odors):
        new_tag = project_neural_tag(
                    new_odors[i], new_odors[i], projmat, **proj_kwargs)
        if not lean:
            new_odor_tags[i, list(new_tag)] = True
        for k in range(n_new_concs):
            mixtures_ik = back_samples + new_concs[k]*new_odors[i]
            # Project the mixture, subtract the projection 
            # to keep the orthogonal component
            yvecs_ik = mixtures_ik - mixtures_ik.dot(projector.T)
            if str(activ_fct).lower() == "relu":
                yvecs_ik = relu_inplace(yvecs_ik)
            # Now tag each mixture
            for j in range(n_times):
                for l in range(n_back_samples):
                    perp_tag = project_neural_tag(
                                yvecs_ik[j, l], mixtures_ik[j, l], projmat, **proj_kwargs
                                )
                    jaccard_scores[i, j, k, l] = jaccard(new_tag, perp_tag)
                    # Also save similarity to the most similar background odor
                    for b in range(n_b):
                        jaccard_backs_indiv[b] = jaccard(perp_tag, back_tags[b])
                    jaccard_scores_back[i, j, k, l] = np.amax(jaccard_backs_indiv)
                    # Distance to new odor
                    ydiff = yvecs_ik[j, l] - new_concs[k]*new_odors[i]
                    if lean:
                        y_l2_distances[i, j, k, l] = l2_norm(ydiff)
                    else:
                        mixture_yvecs[i, j, k, l] = yvecs_ik[j, l]
                        mixture_tags[i, j, k, l, list(perp_tag)] = True

    ref_file.close()

    # Prepare simulation results dictionary
    test_results = {
        "jaccard_scores": jaccard_scores,
        "jaccard_scores_back":jaccard_scores_back
    }
    if lean:
        test_results["y_l2_distances"] = y_l2_distances
    else:
        new_odor_tags = new_odor_tags.tocsr()
        test_results.update({
            "new_odor_tags": new_odor_tags,
            "mixture_yvecs": mixture_yvecs,
            "mixture_tags": mixture_tags,
        })
    return sim_id, test_results, projmat


def orthogonal_recognition_one_sim(sim_id, filename_ref, lean=False):
    """ Load new odors, background samples and projection matrix
    for a given simulation in ref_file, and test odor recognition
    after ideal habituation where all that is left is the new odor component
    perpendicular to the background subspace.

    We don't need background samples: we assume the entire background
    is removed anyways. This changes a little bit dimensionalities:
        - We don't need back_samples or back_concs
        - mixture_yvecs has shape [n_new_odors, 1,  n_new_concs, 1, n_r]
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
    bkname = ref_file.attrs["background"]

    # Dimensions, etc.
    n_r, n_b, _, n_kc = ref_file.get("parameters").get("dimensions")
    n_new_odors, n_new_concs = ref_file.get("parameters").get("repeats")[3:5]
    proj_kwargs = hdf5_to_dict(ref_file.get("proj_kwargs"))
    activ_fct = ref_file.get("parameters").attrs.get("activ_fct", "ReLU")

    # Get the average of background samples to set the KC thresholds
    back_samples = recover_back_samples(sim_gp, back_odors, bkname, ref_file, lean)
    average_back = np.mean(back_samples, axis=(0, 1))

    # Projector to subtract the parallel component
    projector = find_projector(back_odors.T)

    # Containers for the results
    jaccard_scores = np.zeros([n_new_odors, 1, n_new_concs, 1])
    # Also compute similarity to background. First, need tags of back odors
    jaccard_scores_back = np.zeros(jaccard_scores.shape)
    jaccard_backs_indiv = np.zeros(n_b)
    back_tags = []
    for b in range(n_b):
        back_tags.append(
            project_neural_tag(back_odors[b], back_odors[b],
                projmat, **proj_kwargs
        ))

    if lean:
        y_l2_distances = np.zeros((n_new_odors, 1, n_new_concs, 1))
    else:
        new_odor_tags = sparse.lil_array((n_new_odors, n_kc), dtype=bool)
        mixture_tags = SparseNDArray(
                    (n_new_odors, 1, n_new_concs, 1, n_kc), dtype=bool)
        mixture_yvecs = np.zeros([n_new_odors, 1, n_new_concs, 1, n_r])
    # Treat one new odor at a time, one new conc. at a time, etc.
    for i in range(n_new_odors):
        x_par = find_parallel_component(new_odors[i], None, projector)
        new_tag = project_neural_tag(
                    new_odors[i], new_odors[i], projmat, **proj_kwargs)
        if not lean:
            new_odor_tags[i, list(new_tag)] = True
        for j in range(n_new_concs):
            # Project new odor, project its perpendicular component
            yvec = new_concs[j]*(new_odors[i] - x_par)
            if str(activ_fct).lower() == "relu":
                yvec = relu_inplace(yvec)
            x_mix = average_back + new_concs[j]*new_odors[i]
            perp_tag = project_neural_tag(
                        yvec, x_mix, projmat, **proj_kwargs
                        )
            jaccard_scores[i, 0, j, 0] = jaccard(new_tag, perp_tag)
            # Also save similarity to the most similar background odor
            for b in range(n_b):
                jaccard_backs_indiv[b] = jaccard(perp_tag, back_tags[b])
            jaccard_scores_back[i, 0, j, 0] = np.amax(jaccard_backs_indiv)
            if lean:
                y_l2_distances[i, 0, j, 0] = l2_norm(new_concs[j]*x_par)
            else:
                mixture_yvecs[i, 0, j, 0] = yvec
                mixture_tags[i, 0, j, 0, list(perp_tag)] = True

    ref_file.close()

    # Prepare simulation results dictionary
    test_results = {
        "jaccard_scores": jaccard_scores,
        "jaccard_scores_back":jaccard_scores_back
    }
    if lean:
        test_results["y_l2_distances"] = y_l2_distances
    else:
        new_odor_tags = new_odor_tags.tocsr()
        test_results.update({
            "new_odor_tags": new_odor_tags,
            "mixture_yvecs": mixture_yvecs,
            "mixture_tags": mixture_tags,
        })
    return sim_id, test_results, projmat


def ideal_recognition_one_sim(sim_id, filename_ref, lean=False):
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
    bkname = ref_file.attrs["background"]

    # Dimensions, etc.
    n_r, n_b, _, n_kc = ref_file.get("parameters").get("dimensions")
    (n_times, n_back_samples, n_new_odors,
        n_new_concs) = ref_file.get("parameters").get("repeats")[1:5]
    proj_kwargs = hdf5_to_dict(ref_file.get("proj_kwargs"))
    activ_fct = ref_file.get("parameters").attrs.get("activ_fct", "ReLU")

    # Get the background samples
    back_samples = recover_back_samples(sim_gp, back_odors, bkname, ref_file, lean)
    
    # Projector to subtract the parallel component
    projector = find_projector(back_odors.T)

    # Compute optimal factor for each new concentration
    dummy_rgen = np.random.default_rng(0x6e3e2886c30163741daaaf7c8b8a00e6)
    factors = [compute_ideal_factor(c, moments_conc[:2], [n_b, n_r],
                generate_odorant, (dummy_rgen,), reps=100) for c in new_concs]

    # Containers for the results
    jaccard_scores = np.zeros([n_new_odors, n_times,
                                n_new_concs, n_back_samples])
    # Also compute similarity to background. First, need tags of back odors
    jaccard_scores_back = np.zeros(jaccard_scores.shape)
    jaccard_backs_indiv = np.zeros(n_b)
    back_tags = []
    for b in range(n_b):
        back_tags.append(
            project_neural_tag(back_odors[b], back_odors[b],
                projmat, **proj_kwargs
        ))

    if lean:
        y_l2_distances = np.zeros([n_new_odors, n_times,
                                n_new_concs, n_back_samples])
    else:
        new_odor_tags = sparse.lil_array((n_new_odors, n_kc), dtype=bool)
        mixture_tags = SparseNDArray((n_new_odors, n_times, n_new_concs,
                                    n_back_samples, n_kc), dtype=bool)
        mixture_yvecs = np.zeros([n_new_odors, n_times,
                                n_new_concs, n_back_samples, n_r])
    
    # Orthogonal components of the background samples, if any
    if bkname.endswith("gaussnoise"):
        back_ort_comp = back_samples - back_samples.dot(projector.T)
    else:
        back_ort_comp = 0.0

    # Treat one new odor at a time, one new conc. at a time, etc.
    for i in range(n_new_odors):
        x_par = find_parallel_component(new_odors[i], None, projector)
        x_ort = new_odors[i] - x_par
        new_tag = project_neural_tag(
                    new_odors[i], new_odors[i], projmat, **proj_kwargs)
        if not lean:
            new_odor_tags[i, list(new_tag)] = True
        for k in range(n_new_concs):
            mixtures_ik = back_samples + new_concs[k]*new_odors[i]
            for j in range(n_times):
                for l in range(n_back_samples):
                    yvec = (factors[k]*back_samples[j, l]
                        + new_concs[k] * (factors[k]*x_par + x_ort))
                    # Add back the orthogonal part of the background, if any
                    yvec += (1.0 - factors[k]) * back_ort_comp[j, l]
                    if str(activ_fct).lower() == "relu":
                        yvec = relu_inplace(yvec)
                    mix_tag = project_neural_tag(
                        yvec, mixtures_ik[j, l],
                        projmat, **proj_kwargs
                    )
                    jaccard_scores[i, j, k, l] = jaccard(mix_tag, new_tag)
                    # Also save similarity to the most similar background odor
                    for b in range(n_b):
                        jaccard_backs_indiv[b] = jaccard(mix_tag, back_tags[b])
                    jaccard_scores_back[i, j, k, l] = np.amax(jaccard_backs_indiv)
                    if lean:
                        ydiff = yvec - new_concs[k]*new_odors[i]
                        y_l2_distances[i, j, k, l] = l2_norm(ydiff)
                    else:
                        mixture_yvecs[i, j, k, l] = yvec
                        try:
                            mixture_tags[i, j, k, l, list(mix_tag)] = True
                        except ValueError as e:
                            print(mix_tag)
                            print(mixture_yvecs[i, j, k, l])
                            print(projmat.dot(mixture_yvecs[i, j, k, l]))
                            raise e
                    
    ref_file.close()

    # Prepare simulation results dictionary
    test_results = {
        "jaccard_scores": jaccard_scores,
        "jaccard_scores_back": jaccard_scores_back,
        "ideal_factors": factors,
        "projector": projector
    }
    if lean:
        test_results["y_l2_distances"] = y_l2_distances
    else:
        new_odor_tags = new_odor_tags.tocsr()
        test_results.update({
            "new_odor_tags": new_odor_tags,
            "mixture_yvecs": mixture_yvecs,
            "mixture_tags": mixture_tags,
        })
    return sim_id, test_results, projmat


def optimal_recognition_one_sim(sim_id, filename_ref, lean=False):
    """Load new odors, background samples and projection matrix
    for a given simulation in ref_file, and test odor recognition
    after optimal habituation where the projection matrix W
    is the optimum to minimize the distance between the 
    background and the subtracted vector, <||b - W(b + s_n)||>^2. 

    This is the true optimum for manifold learning, 
    it should be in principle even better than the "ideal" case
    above, which simplified the process to targeting the parallel
    component of the new odor only. 
    """
    ref_file = h5py.File(filename_ref, "r")
    sim_gp = ref_file.get(id_to_simkey(sim_id))
    projmat = hdf5_to_csr_matrix(sim_gp.get("kc_proj_mat"))
    new_odors = ref_file.get("odors").get("new_odors")[()]
    back_odors = ref_file.get("odors").get("back_odors")[sim_id, :, :]
    new_concs = ref_file.get("parameters").get("new_concs")[()]
    moments_conc = ref_file.get("parameters").get("moments_conc")[()]
    bkname = ref_file.attrs["background"]
    # IBCM factor: this can be too much reduction compared to the optimum.
    #factor = w_rates[1] / (2*w_rates[0] + w_rates[1])

    # Dimensions, etc.
    n_r, n_b, _, n_kc = ref_file.get("parameters").get("dimensions")
    (n_times, n_back_samples, n_new_odors,
        n_new_concs) = ref_file.get("parameters").get("repeats")[1:5]
    proj_kwargs = hdf5_to_dict(ref_file.get("proj_kwargs"))
    activ_fct = ref_file.get("parameters").attrs.get("activ_fct", "ReLU").lower()

    # Get the background samples
    back_samples = recover_back_samples(sim_gp, back_odors, bkname, ref_file, lean)

    # Compute optimal W matrix for all new odors possible
    dummy_rgen = np.random.default_rng(0x773989f92ef489f1ae6b044e2ebdcb36)
    n_samp = int(1e4)
    new_odors_from_distrib = generate_odorant([n_samp, n_r], 
                                        dummy_rgen, lambda_in=0.1)
    new_odors_from_distrib /= l2_norm(new_odors_from_distrib)[:, None]
    optimal_ws = compute_optimal_matrices(back_odors, 
                    new_odors_from_distrib, moments_conc, new_concs)

    # Containers for the results
    jaccard_scores = np.zeros([n_new_odors, n_times,
                                n_new_concs, n_back_samples])
    # Also compute similarity to background. First, need tags of back odors
    jaccard_scores_back = np.zeros(jaccard_scores.shape)
    jaccard_backs_indiv = np.zeros(n_b)
    back_tags = []
    for b in range(n_b):
        back_tags.append(
            project_neural_tag(back_odors[b], back_odors[b],
                projmat, **proj_kwargs
        ))

    if lean:
        y_l2_distances = np.zeros([n_new_odors, n_times,
                                n_new_concs, n_back_samples])
    else:
        new_odor_tags = sparse.lil_array((n_new_odors, n_kc), dtype=bool)
        mixture_tags = SparseNDArray( (n_new_odors, n_times, n_new_concs,
                                    n_back_samples, n_kc), dtype=bool)
        mixture_yvecs = np.zeros([n_new_odors, n_times,
                                    n_new_concs, n_back_samples, n_r])

    # Treat one new odor at a time, one new conc. at a time, etc.
    for i in range(n_new_odors):
        new_tag = project_neural_tag(
                    new_odors[i], new_odors[i], projmat, **proj_kwargs)
        if not lean:
            new_odor_tags[i, list(new_tag)] = True
        for k in range(n_new_concs):
            mixtures_ik = back_samples + new_concs[k]*new_odors[i]
            # Can compute all mixture responses at once for this i, k
            yvecs_ik = mixtures_ik - mixtures_ik.dot(optimal_ws[k].T)
            for j in range(n_times):
                for l in range(n_back_samples):
                    yvec = yvecs_ik[j, l]
                    if activ_fct == "relu":
                        yvec = relu_inplace(yvec)
                    mix_tag = project_neural_tag(
                        yvec, mixtures_ik[j, l], projmat, **proj_kwargs
                    )
                    jaccard_scores[i, j, k, l] = jaccard(mix_tag, new_tag)
                    # Also save similarity to the most similar background odor
                    for b in range(n_b):
                        jaccard_backs_indiv[b] = jaccard(mix_tag, back_tags[b])
                    jaccard_scores_back[i, j, k, l] = np.amax(jaccard_backs_indiv)
                    if lean:
                        ydiff = yvec - new_concs[k]*new_odors[i]
                        y_l2_distances[i, j, k, l] = l2_norm(ydiff)
                    else:
                        mixture_yvecs[i, j, k, l] = yvec
                        try:
                            mixture_tags[i, j, k, l, list(mix_tag)] = True
                        except ValueError as e:
                            print(mix_tag)
                            print(mixture_yvecs[i, j, k, l])
                            print(projmat.dot(mixture_yvecs[i, j, k, l]))
                            raise e
    ref_file.close()

    # Prepare simulation results dictionary
    test_results = {
        "jaccard_scores": jaccard_scores,
        "jaccard_scores_back": jaccard_scores_back,
        "optimal_ws": optimal_ws
    }
    if lean:
        test_results["y_l2_distances"] = y_l2_distances
    else:
        new_odor_tags = new_odor_tags.tocsr()
        test_results.update({
            "new_odor_tags": new_odor_tags,
            "mixture_yvecs": mixture_yvecs,
            "mixture_tags": mixture_tags,
        })
    return sim_id, test_results, projmat


def save_examples(fname, kind, sim_res):
    arrs = {"mixture_yvecs": sim_res["mixture_yvecs"]}
    if kind == "ideal": 
        arrs["ideal_factors"] = sim_res["ideal_factors"]
        arrs["projector"] = sim_res["projector"]
    elif kind == "optimal": 
        arrs["optimal_ws"] = sim_res["optimal_ws"]
    else: 
        pass
    np.savez_compressed(fname, **arrs)
    return None


# If using these functions, we are most likely performing
# many parallel simulations, so we need to limit the
# multithreading of matrix operations to prevent collision
# between parallel processes. 
def func_wrapper_threadpool(func, nthreads, *args, **kwargs):
    with threadpool_limits(limits=nthreads, user_api='blas'):
        res = func(*args, **kwargs)
    return res


def idealized_recognition_from_runs(
        filename, filename_ref, kind="none", example_file=None, lean=False
    ):
    """ After having performed several habituation runs and tested them
    for recognition with some model, compute the ideal inhibition
    we could have achieved in these runs.

    Args:
        filename (str): name of the file to contain the ideal results
        filename_ref (str): name of the file with other simulation results
            All necessary information is extracted from there.
        kind (str): the kind of idealized habituation considered,
            either "orthogonal", "none", "optimal", or "ideal".
        example_file (str): if a npz filename is provided, save
            the optimal matrix W or ideal factors are saved

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
        "ideal": ideal_recognition_one_sim, 
        "optimal": optimal_recognition_one_sim
    }
    recognition_one_sim = func_choices.get(kind, no_habituation_one_sim)
    # Special choice depending on the background
    if res_file.attrs["background"] == "turbulent_gaussnoise" and kind == "orthogonal":
        recognition_one_sim = orthogonal_recognition_one_sim_gaussnoise

    # Define callback functions
    def callback(result):
        sim_id, sim_results, projmat = result
        sim_gp = res_file.create_group(id_to_simkey(sim_id))
        if sim_gp.get("test_results") is None:
            if not lean:
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
        else:  # Group already exists, skip
            print("test_results group for sim {}".format(sim_id)
                    + " already exists; not saving")
        # Save parts of the first sim. separately as an example to plot
        if sim_id == 0 and example_file is not None and not lean:
            save_examples(example_file, kind, sim_results)
        print("Ideal recognition tested for simulation {}".format(sim_id))
        return sim_id

    # 2. For each background and run, test new odor recognition at snap times.
    # Create a projection matrix, save to HDF file.
    # Against n_back_samples backgrounds, including the simulation one.
    # and test at 20 % or 50 % concentration
    n_workers = min(count_parallel_cpu(), repeats[0])
    threads_per_proc = count_threads_per_process(n_workers)
    pool = multiprocessing.Pool(n_workers)
    for sim_id in range(repeats[0]):
        # Retrieve relevant results of that simulation,
        # then create and save the proj. mat., and initialize arguments
        apply_args = (recognition_one_sim, threads_per_proc, 
                      sim_id, filename_ref)
        apply_kwargs = dict(lean=lean)
        pool.apply_async(func_wrapper_threadpool, args=apply_args,
            kwds=apply_kwargs, callback=callback, error_callback=error_callback
        )
    
    # Close and join the pool
    # No need to .get() results: the callback takes care of it
    # and in fact we don't want to get them, else they get stuck in the
    # parent process memory and fill it up...
    pool.close()
    pool.join()
    # Finally, close the results file
    res_file.close()
    return 0


### Functions to compute similarity between random odors ###
def flattened_tri(j, k, n):  # index in flattened array of j, k
    """ Return the index of the upper triangular nxn matrix element (j, k)
    in a flattened array. If j > k, assume the matrix is symmetric and
    return element (k, j). 
    """
    # Swap indices if j > k
    if j > k:  j, k = k, j
    elif j == k: raise ValueError("Should not have diagonal terms")
    # Rows above contain N(N-1)/2 - (N-j)(N-j-1)/2 terms, then
    # add (k-j-1) = number of elements right of the diagonal
    i = n*j - (j+1)*(j+2)//2 + k
    return i


# The following functions are called in parallel 
# for the projection matrix of each run
def tag_all_odors(projmat, odors, proj_kwargs, nthreads=None):
    n_odors = odors.shape[0]
    tags = []
    with threadpool_limits(limits=nthreads, user_api="blas"):
        for j in range(n_odors):
            odj = odors[j]
            tag = project_neural_tag(odj, odj, projmat, **proj_kwargs)
            tags.append(tag)
    return tags


def similarity_all_pairs(all_tags, proc_id):
    n_odors = len(all_tags)
    jaccard_scores = np.zeros([(n_odors * (n_odors - 1)) // 2])
    for j in range(n_odors):
        tagj = all_tags[j]
        for k in range(j+1, n_odors):
            tagk = all_tags[k]
            jaccard_scores[flattened_tri(j, k, n_odors)] = jaccard(tagj, tagk)
    print("Finished to compare tags of run {}".format(proc_id))
    return jaccard_scores


def jaccard_between_random_odors(filename, filename_ref):
    """ Estimate what is the Jaccard similarity between two random
    odors, as a function of the distance between them. 
    Take a results file from an IBCM or PCA simulation (filename_ref), take
    all new odors and background odors to form one pool of iid odors, 
    then compute Jaccard similarity and L2 distance between each, 
    using each projection matrix generated in these simulations. 
    Save the results to an npz archive (filename). 
    """
    ref_file = h5py.File(filename_ref, "r")
    n_runs = ref_file.get("parameters").get("repeats")[0]
    n_s, n_b, _, _ = ref_file.get("parameters").get("dimensions")
    new_odors = ref_file.get("odors").get("new_odors")[()]
    back_odors = ref_file.get("odors").get("back_odors")[()]
    # Concatenate backgrounds of all simulations
    back_odors = back_odors.reshape([n_runs*n_b, n_s])
    # Concatenate all odors
    all_odors = np.concatenate([back_odors, new_odors])
    n_odors = all_odors.shape[0]
    all_projmats = []
    proj_kwargs = hdf5_to_dict(ref_file.get("proj_kwargs"))
    for si in range(n_runs):
        sim_gp = ref_file.get("sim{:04d}".format(si))
        all_projmats.append(hdf5_to_csr_matrix(sim_gp.get("kc_proj_mat")))
    ref_file.close()
        
    # First, tag all odors. Store as list of lists of sets
    n_workers = min(count_parallel_cpu(), n_runs)
    apply_kwargs = {"nthreads": count_threads_per_process(n_workers)}
    pool = multiprocessing.Pool(n_workers)
    all_processes = []
    for i in range(n_runs):
        projmat = all_projmats[i]
        apply_args = (projmat, all_odors, proj_kwargs)
        proc = pool.apply_async(
            tag_all_odors, args=apply_args, kwds=apply_kwargs
        )
        all_processes.append(proc)
    all_tags = [res.get() for res in all_processes]
    print("Finished tagging odors: {} odors x {} runs".format(n_odors, n_runs))
    # Close pool
    pool.close()
    pool.join()
    del all_processes, all_results
    
    # Then, compute the Jaccard between each pair of tags
    # Containers for the results and start new pool
    # Store only the flattened upper triangular parts, elements (j, k), j > k
    jaccard_scores = []
    all_processes = []
    pool = multiprocessing.Pool(n_workers)
    for i in range(n_runs):
        apply_args = (all_tags[i], i)
        proc = pool.apply_async(similarity_all_pairs, args=apply_args)
        all_processes.append(proc)
    jaccard_scores = [proc.get() for proc in all_processes]
    # Stack the results in appropriate arrays
    # In the end, this matrix of Jaccard similarities is indexed
    # [run, pair of iid odors], for a given N_S
    jaccard_scores = np.stack(jaccard_scores, axis=0)
    
    # Close pool
    pool.close()
    pool.join()
    del all_processes, all_results
    
    # Also, distances between odors: same for all runs, so compute once
    # Results returned as a flattened upper triangular matrix, indexed
    # [pair of iid odors], for a given N_S
    y_l2_distances = pdist(all_odors, metric="euclidean")

    # Prepare simulation results dictionary
    test_results = {
        "jaccard_scores": jaccard_scores,
        "y_l2_distances": y_l2_distances
    }
    # Save results
    np.savez_compressed(filename, **test_results)
    return 0
