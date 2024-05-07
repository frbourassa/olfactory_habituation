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


def simkey_to_id(sk):
    return int(sk.strip("sim"))


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
    n_snaps = all_params.get("repeats")[1]
    # Random generator to sample many x vectors
    rng = np.random.default_rng(
        np.random.SeedSequence(int(res_file.attrs["main_seed"])).spawn(2)[1]
    )
    all_s_stats_dict = {}
    all_x_stats_dict = {}

    # loop over simulations in this file. For each, compute x and s stats
    for sim_id in res_file.keys():
        if not sim_id.startswith("sim"): continue
        gp = res_file.get(sim_id)
        back_comps = (res_file.get("odors")
                    .get("back_odors")[simkey_to_id(sim_id)])
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

        res_file = h5py.File(os.path.join(folder, fname), "r")
        # Get jaccard scores of all seeds
        all_jacs = concat_jaccards(res_file)
        df.loc[(si, sj), "jaccard_mean"] = all_jacs.mean()
        df.loc[(si, sj), "jaccard_median"] = np.median(all_jacs)
        df.loc[(si, sj), "jaccard_variance"] = all_jacs.var()

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


def create_or_load(f, model):
    if os.path.isfile(f):
        df_stats = pd.read_hdf(f, key="df")
        print("Loaded existing {}".format(f))
    else:
        print("Aggregating stats for {}".format(f))
        df_stats = aggregate_result_files(
            os.path.join("results", "performance_w"), model
        )
        df_stats.to_hdf(f, key="df")
    return df_stats


def main_plot_w_norms(df_ibcm, df_pca):
    # Line plots: x axis is (i, j), y axis is some statistic.
    # One line for PCA, one line for IBCM
    # As a way to identify best/worst cases first, and see variability
    fig, axes = plt.subplots(2, 3, sharex=True)
    size_inches = fig.get_size_inches()
    fig.set_size_inches(size_inches[0]*2, size_inches[1]*0.75)
    model_colors = {
        "ibcm": "xkcd:turquoise",
        "biopca": "xkcd:orangey brown",
        "avgsub": "xkcd:navy blue",
        "ideal": "xkcd:powder blue",
        "orthogonal": "xkcd:pale rose",
        "none": "grey"
    }
    metrics = [
        "s_norm_mean_reduction",
        "s_norm_variance_reduction",
        "s_norm_thirdmoment_reduction",
        "jaccard_mean",
        "jaccard_median",
        "jaccard_variance"
    ]
    xaxis = np.arange(df_ibcm.index.size)

    for i, m in enumerate(metrics):
        df_ibcm_plot = df_ibcm.loc[:, m].values
        df_pca_plot = df_pca.loc[df_ibcm.index, m].values
        axes.flat[i].plot(xaxis, df_pca_plot,
                    label="BioPCA", color=model_colors.get("biopca"))
        axes.flat[i].plot(xaxis, df_ibcm_plot,
                    label="IBCM", color=model_colors.get("ibcm"))
        axes.flat[i].set(ylabel=m, xlabel="Grid search index")
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "detection", "line_plots_w_norm.pdf"),
                transparent=True, bbox_inches="tight")
    return None

if __name__ == "__main__":
    # Create or load statistics dfs
    ibcm_df_f = os.path.join("results", "performance_w", "df_w_stats_ibcm.h5")
    df_stats_ibcm = create_or_load(ibcm_df_f, "IBCM")
    pca_df_f = os.path.join("results", "performance_w", "df_w_stats_biopca.h5")
    df_stats_biopca = create_or_load(pca_df_f, "PCA")

    # Now, plot results
    print("Starting to plot results...")
    main_plot_w_norms(df_stats_ibcm, df_stats_biopca)

    print("Done!")
