"""
Reanalyze the saved performance comparison results to compute odor
recognition performance based on alternate metrics than the Jaccard
similarity. In particular, we compute the fraction of tag elements
that distinguish the new odor from the background (z_new / z_back)
that are also captured in the tag of the mixture. 

@author: frbourassa
August 2025
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import h5py
import os, sys
from os.path import join as pj
if not ".." in sys.path:
    sys.path.insert(1, "../")

from utils.metrics import jaccard_distinct
from utils.export import hdf5_to_csr_matrix, hdf5_to_dict
from modelfcts.tagging import (
    project_neural_tag,
    SparseNDArray
)
from simulfcts.habituation_recognition import id_to_simkey
from multiprocessing import Pool
from utils.cpu_affinity import count_parallel_cpu


# Function to apply to each simulation seed (i.e. one background)
# Load stored mixture and new odor tags, recompute background tags
def analyze_distinct_tag_elements(back_new_mix_tags, sim_params, test_params):
    """ 
    Args:
        back_new_mix_tags (dict): contains new_odor_tags (dataset), 
            mixture_tags (group), back_samples (dataset)
        sim_params (dict): dimensions, repeats, m_rates, back_params,
            new_concs are relevant and same for all simulations.
        test_params (dict):
            pmat (sp.sparse.csr_matrix): sparse projection mat. of this simul.
            proj_kwargs (dict): projection function kwargs (sparsity, etc.)
    """
    n_times = sim_params["repeats"][1]
    n_back_samples = sim_params["repeats"][2]
    n_new_odors = sim_params['repeats'][3]
    n_new_concs = sim_params['repeats'][4]
    n_kc = sim_params['dimensions'][3]
    assert n_kc == test_params["pmat"].shape[0], "Inconsistent KC number"

    # Get the background samples, shape: n_times, n_back_samples, N_S
    back_samples = back_new_mix_tags["back_samples"][:]  
    # mixture tags: n_new_odors, n_times, n_new_concs, n_back_samples, n_kc
    mixture_tags = SparseNDArray.read_from_hdf(None,
        back_new_mix_tags["mixture_tags"]).todense() 
    # Get new odor tags, shape: n_new_odors, n_kc
    new_odor_tags = hdf5_to_csr_matrix(back_new_mix_tags["new_odor_tags"]).toarray()   # we don't want a matrix

    # Compute all background tags
    background_tags = {}  # dict of sets, indexed by (j, l) tuples
    for j in range(n_times):
        for l in range(n_back_samples):
            back_tag = project_neural_tag(
                        back_samples[j, l], back_samples[j, l],
                        test_params['pmat'], **test_params['proj_kwargs']
                    )
            background_tags[(j, l)] = set(back_tag)

    # For the orthogonal habituation, all the background component
    # is removed, so the remaining yvec is the same for all background
    # vectors, hence we only save 1 score instead of n_times x n_back_samples
    # The sign for this is that mixture_tags has length 1 in axis 3
    n_back_samples_adj = min(n_back_samples, mixture_tags.shape[3])
    n_times_adj =  min(n_times, mixture_tags.shape[1])
    # Loop over new odors first
    frac_tag_scores = np.zeros(
        [n_new_odors, n_times_adj, n_new_concs, n_back_samples_adj])

    for i in range(n_new_odors):
        # Compute neural tag of the new odor alone, without inhibition
        new_tag = set(new_odor_tags[i].nonzero()[0])
        # Now, loop over snapshots, mix the new odor with the back samples,
        # compute the PN response at each test concentration,
        # compute tags too, and save results
        for j in range(n_times_adj):
            for k in range(n_new_concs):
                for l in range(n_back_samples_adj):
                    mix_tag = set(mixture_tags[i, j, k, l].nonzero()[0])
                    back_tag = background_tags[(j, l)]
                    frac_tag_scores[i, j, k, l] = jaccard_distinct(
                                            mix_tag, new_tag, back_tag)
                    
    return frac_tag_scores, background_tags


def main_reanalyze_model_distinct_tag(fname, fullfile=None):
    # Apply to each model and each simulation seed
    fpath = pj("results", "performance", fname)
    hfile = h5py.File(fpath, "r")
    if fullfile is not None:  # for idealized models, they don't save background samples
        fpath2 = pj("results", "performance", fullfile)
        hfile2 = h5py.File(fpath2, "r")
    else:
        hfile2 = hfile
    sim_params = hdf5_to_dict(hfile.get("parameters"))
    n_runs = sim_params.get("repeats")[0]
    proj_kwargs = hdf5_to_dict(hfile.get("proj_kwargs"))
    all_tag_scores = []
    for s in range(n_runs):
        simkey = id_to_simkey(s)
        proj_mat = hdf5_to_csr_matrix(hfile.get(simkey).get("kc_proj_mat"))
        test_params = {"pmat": proj_mat, "proj_kwargs": proj_kwargs}
        back_new_mix_tags = {
            "back_samples": (hfile2.get(simkey)
                            .get("test_results").get("back_samples")),
            "new_odor_tags": hfile2.get(simkey).get("new_odor_tags"),
            "mixture_tags": hfile.get(simkey).get("mixture_tags")
        }

        tags_s, _ = analyze_distinct_tag_elements(
            back_new_mix_tags, sim_params, test_params)
        all_tag_scores.append(tags_s)
    # indexed n_runs, n_new_odors, n_times, n_new_concs, n_back_samples
    all_tag_scores = np.asarray(all_tag_scores)  
    print("For file {}".format(fname))
    print("For new conc = 0.5<c>: ", np.mean(all_tag_scores[0][:, :, 0, :]))
    print("For new conc = <c>: ", np.mean(all_tag_scores[0][:, :, 1, :]))
    hfile.close()
    if fullfile is not None and fullfile != fname:
        hfile2.close()
    return all_tag_scores


if __name__ == "__main__":
    models = ["ibcm", "biopca", "avgsub", 
              "optimal", "ideal", "orthogonal", "none"]
    results_files = {m: "{}_performance_results.h5".format(m) for m in models}
    all_tag_scores_models = {}
    n_workers = min(count_parallel_cpu(), len(models))
    pool = Pool(n_workers)
    all_procs = {}
    for m in models:
        print("Starting simulation for model", m)
        fullfile = (results_files[m] if m in ["ibcm", "biopca", "avgsub"]
                    else results_files["ibcm"])
        all_procs[m] = pool.apply_async(main_reanalyze_model_distinct_tag, 
                        args=(results_files[m],), kwds={"fullfile":fullfile})
    all_tag_scores_models = {m:all_procs[m].get() for m in models}
    pool.close()
    pool.join()
    # Save all to npz archive for further plotting of histogram, statistics, etc.
    # versus distances to background (which are already saved 
    # in new_mix_distances_identity.npz)
    plotfile = pj("results", "for_plots",
                   "frac_distinct_tag_elements_identity.npz")
    np.savez_compressed(plotfile, **all_tag_scores_models)