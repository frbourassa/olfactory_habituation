import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from simulfcts.habituation_recognition import id_to_simkey
from simulfcts.plotting import hist_outline
from modelfcts.ideal import find_projector, find_parallel_component
from utils.metrics import l2_norm
import os
import json

from analyze_comparison_results import (
    concat_jaccards,
    concat_sstats
)
from modelfcts.backgrounds import sample_background_powerlaw
from simulfcts.habituation_recognition import (
    appropriate_response,
    get_data
)


def ij_from_name(fname):
    """ Returns strings """
    return (fname.split(".")[0]).split("_")[-2:]


def compute_norm_stats(vecs):
    norm = l2_norm(vecs, axis=1)
    norm_stats = np.asarray([
        np.mean(norm),
        np.var(norm),
        np.mean((norm - np.mean(norm))**3)
    ])
    return norm_stats


def s_stats_from_snaps(res_file):
    # Table of sample functions depending on the model
    sample_fcts = {
        "turbulent": sample_background_powerlaw  # (vecs_nu, *args, size=1, rgen=None)
    }
    sample_fct = sample_fcts.get(res_file.attrs["background"])

    # Get model parameters and appropriate snapshots
    all_params = res_file.get("parameters")
    back_params = all_params.get("back_params")[()]
    n_snaps = params.get("repeats")[1]
    # Random generator to sample many x vectors
    rng = np.random.default_rng(
            np.random.SeedSequence(int(res_file.attrs["main_seed"])).spawn()
    )
    all_s_stats_dict = {}
    all_x_stats_dict = {}

    # loop over simulations in this file. For each, compute x and s stats
    for sim_id in res_file.keys():
        if not sim_id.startswith("sim"): continue
        gp = res_file.get(sim_id)
        snaps_dict = {
            "m": get_data(gp, "m_snaps"),
            "w": get_data(gp, "w_snaps"),
            "l": get_data(gp, "l_snaps"),
            "x": get_data(gp, "x_snaps"),
            "conc": get_data(gp, "back_conc_snaps"),
            "back": get_data(gp, "back_vec_snaps"),
        }
        # Generate a bunch of background samples at each snap time
        all_svecs = []
        all_xvecs = []
        for i in range(n_snaps):
            back_samples = sample_fct(back_comps, *back_params, size=100, rgen=rng)
            # Compute response to each sample at each snapshot: gives s vectors
            svecs = appropriate_response(
                res_file.attrs, all_params, back_samples, snaps_dict,
                i, all_params.attrs
            )
            all_svecs.append(svecs)
            all_xvecs.append(back_samples)
        # Compute s vector statistics
        x_stats = compute_norm_stats(np.concatenate(all_xvecs))
        all_x_stats_dict[sim_id] = x_stats
        s_stats = compute_norm_stats(np.concatenate(all_svecs))
        all_s_stats_dict[sim_id] = s_stats
    return all_x_stats_dict, all_s_stats_dict


def aggregate_result_files(folder, model):
    # Load main tables mapping simulation indices to (p, q) and (alpha, beta)
    with open(os.path.join(folder, "table_i_pnorm-qnorm.json"), "r") as f:
        table_i_pq = json.load(f)
    with open(os.path.join(folder, "table_ij_alpha-beta.json"), "r") as f:
        table_ij_ab = json.load(f)

    # Find all available simulation files for (p, q, alpha, beta) choices
    model_fname_prefixes = {
        "IBCM": "ibcm",
        "PCA": "biopca"
    }
    sim_files = [
        a for a in os.listdir(folder)
            if a.startswith(model_fname_prefixes.get(model))
            and a.endswith(".h5")
    ]
    sim_indices = list(map(ij_from_name, sim_files))

    # Prepare DataFrame to store simulation performance statistics
    wanted_stats = pd.Index([
        "pnorm", "qnorm", "alpha", "beta",  # will be made index levels
        "s_norm_mean_reduction",
        "s_norm_variance_reduction",
        "s_norm_thirdmoment_reduction",
        "jaccard_mean",
        "jaccard_median",
        "jaccard_variance"
    ], name="statistics")
    wanted_index = pd.MultiIndex.from_tuples(sim_indices, names=["i", "j"])
    df = pd.DataFrame(
            np.zeros([len(wanted_index), len(wanted_stats)]),
            index=wanted_index, columns=wanted_stats
    )

    # For each simulation file, compute performance statistics
    for fname in sim_files:
        print("Starting to process file {}".format(fname))
        si, sj = ij_from_name(fname)
        pnorm, qnorm = table_i_pq.get(si)
        alpha, beta = table_ij_ab.get("{}_{}".format(si, sj))
        df.loc[(si, sj), "pnorm":"beta"] = (pnorm, qnorm, alpha, beta)

        res_file = h5py.File(fname, "r")
        # Get jaccard scores of all seeds
        all_jacs = concat_jaccards(res_file)

        # Get x and s stats for each seed in each simulation
        x_stats, s_stats = s_stats_from_snaps(res_file)
        # Then compute reduction in stats for each seed, and average over seeds
        reds = []
        for sd in x_stats.keys():
            reds.append(s_stats[sd] / x_stats[sd])
        reds = np.stack(reds).mean(axis=0)
        # Store these aggregate habituation statistics in the DataFrame
        df.loc[(si, sj),
                "s_norm_mean_reduction":"s_norm_thirdmoment_reduction"] = reds
        res_file.close()

    return df


if __name__ == "__main__":
    df_stats_ibcm = aggregate_result_files(
        os.path.join("results", "performance_w"), "IBCM"
    )
    #df_stats_pca = aggregate_result_files(
    #    os.path.join("results", "performance_w"), "PCA"
    #)
    
