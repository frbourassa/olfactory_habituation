"""
Function for analysis of simulation results, comparison of habituation
models, etc.

@author: frbourassa
October 2023
"""
import numpy as np
import h5py
from simulfcts.habituation_recognition import id_to_simkey


# Function to check background reduction
def compute_back_reduction_stats(bkser, sser, trans=0):
    back_stats = dict(
        avg_bk = np.mean(bkser[trans:]),
        avg_s = np.mean(sser[trans:]),
        std_bk = np.std(bkser[trans:]),
        std_s = np.std(sser[trans:])
    )
    back_stats["avg_reduction"] = back_stats['avg_s'] / back_stats['avg_bk']
    back_stats["std_reduction"] = back_stats['std_s'] / back_stats['std_bk']
    return back_stats


def concat_jaccards(f, k="jaccard_scores"):
    all_jacs = []
    for i in range(f.get("parameters").get("repeats")[0]):
        all_jacs.append(f.get(id_to_simkey(i)).get("test_results").get(k)[()])
    all_jacs = np.stack(all_jacs)
    return all_jacs


def concat_new_mix_distances(f):
    all_dists = []
    for i in range(f.get("parameters").get("repeats")[0]):
        all_dists.append(f.get(id_to_simkey(i)).get("test_results")
                            .get("y_l2_distances")[()])
    all_dists = np.stack(all_dists)
    return all_dists

def concat_mixtures(f):
    all_mixes = []
    for i in range(f.get("parameters").get("repeats")[0]):
        all_mixes.append(f.get(id_to_simkey(i)).get("test_results")
                            .get("mixture_yvecs")[()])
    all_mixes = np.stack(all_mixes)
    return all_mixes


def concat_sstats(f):
    all_stats = []
    for i in range(f.get("parameters").get("repeats")[0]):
        all_stats.append(f.get(id_to_simkey(i)).get("s_stats")[()])
    all_stats = np.stack(all_stats)
    return all_stats


def concat_wmats(f):
    all_mats = []
    for i in range(f.get("parameters").get("repeats")[0]):
        all_mats.append(f.get(id_to_simkey(i)).get("w_snaps")[()])
    all_mats = np.stack(all_mats)
    return all_mats


def concat_mmats(f):
    all_mats = []
    for i in range(f.get("parameters").get("repeats")[0]):
        all_mats.append(f.get(id_to_simkey(i)).get("m_snaps")[()])
    all_mats = np.stack(all_mats)
    return all_mats


def concat_lmats(f, model="PCA"):
    n_r, n_b, n_i, n_k = f.get("parameters").get("dimensions")
    if model == "IBCM":
        eta = f.get("parameters").get("m_rates")[2]
        lmat = np.full([n_i, n_i], -eta)
        lmat[np.diag_indices(n_i)] = 1.0
        all_mats = np.expand_dims(lmat, axis=0)
    elif model == "PCA":
        all_mats = []
        for i in range(f.get("parameters").get("repeats")[0]):
            lmat = f.get(id_to_simkey(i)).get("l_snaps")[()]
            all_mats.append(np.linalg.inv(lmat))
        all_mats = np.stack(all_mats)
    return all_mats