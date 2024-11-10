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
import itertools

from analyze_comparison_results import (
    concat_jaccards,
    concat_sstats,
    concat_wmats,
    concat_mmats, 
    concat_lmats
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
    norm[norm > 1e1] = np.nan
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


def load_fnames_indices(folder, model):
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

    return table_i_pq, table_ij_ab, sim_files, sim_indices


def aggregate_result_files(folder, model):
    # Load main tables mapping simulation indices to (p, q) and (alpha, beta)
    info = load_fnames_indices(folder, model)
    table_i_pq, table_ij_ab, sim_files, sim_indices = info

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
        df.loc[(si, sj), "jaccard_mean"] = np.mean(all_jacs)
        df.loc[(si, sj), "jaccard_median"] = np.median(all_jacs)
        df.loc[(si, sj), "jaccard_variance"] = np.var(all_jacs)

        # Get x and s stats for each seed in each simulation
        x_stats, s_stats = s_stats_from_snaps(res_file)
        # Then compute reduction in stats for each seed, and average over seeds
        reds = []
        for sd in x_stats.keys():
            reds.append(s_stats[sd] / x_stats[sd])
        reds = np.mean(np.stack(reds), axis=0)  # Drop simulations with NaNs
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
    # Prevent large but non-NaNs divergences from being plotted.
    xaxis = np.arange(df_ibcm.index.size)
    df_ibcm2 = df_ibcm.copy()
    df_pca2 = df_pca.copy()
    for lbl in ["pnorm", "qnorm"]:
        df_ibcm2[lbl] = df_ibcm2[lbl].astype(int)
        df_pca2[lbl] = df_pca2[lbl].astype(int)
    df_pca2.loc[df_pca2["s_norm_mean_reduction"] > 10.0] = pd.NA
    df_ibcm2.loc[df_ibcm2["s_norm_mean_reduction"] > 10.0] = pd.NA

    for i, m in enumerate(metrics):
        df_ibcm_plot = df_ibcm2.loc[:, m].values
        df_pca_plot = df_pca2.loc[df_ibcm2.index, m].values
        axes.flat[i].plot(xaxis, df_pca_plot,
                    label="BioPCA", color=model_colors.get("biopca"))
        axes.flat[i].plot(xaxis, df_ibcm_plot,
                    label="IBCM", color=model_colors.get("ibcm"))
        axes.flat[i].set(ylabel=m, xlabel="Grid search index")
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "noise_struct", "line_plots_w_norm.pdf"),
                transparent=True, bbox_inches="tight")
    #plt.show()
    plt.close()

    # Make a clearer plot now. Take the best (alpha, beta) combination
    # for each p, q and make barplots both models? and annotate
    # optimal alpha, beta for each bar.
    fig, axes = plt.subplots(2, 3)
    size_inches = fig.get_size_inches()
    fig.set_size_inches(size_inches[0]*2, size_inches[1]*0.75)
    pq_combi = list(itertools.product(
        df_pca2["pnorm"].unique(), df_pca2["qnorm"].unique()
    ))
    qstr = "pnorm == @p and qnorm == @q"
    idx_lvls = ["alpha", "beta"]
    for i, m in enumerate(metrics):
        for x, (p, q) in enumerate(pq_combi):
            y_ibcm = df_ibcm2.query(qstr).set_index(idx_lvls)
            y_pca = df_pca2.query(qstr).set_index(idx_lvls)
            if m.startswith("s_norm"):
                yidx_ibcm = y_ibcm["s_norm_mean_reduction"].idxmin()
                yidx_pca = y_pca["s_norm_mean_reduction"].idxmin()
            else:
                yidx_ibcm = y_ibcm["jaccard_mean"].idxmax()
                yidx_pca = y_pca["jaccard_mean"].idxmax()
            axes.flat[i].bar(x-0.125, y_pca.loc[yidx_pca, m], width=0.25,
                            color=model_colors.get("biopca"))
            axes.flat[i].annotate(
                ", ".join(("{:.0e}".format(a) for a in yidx_pca)),
                xy=(x-0.125,y_pca.loc[yidx_pca, m]*1.05), fontsize=6,
                rotation=90, ha="center"
            )
            axes.flat[i].bar(x+0.125, y_ibcm.loc[yidx_ibcm, m],
                            width=0.25, color=model_colors.get("ibcm"))
            axes.flat[i].annotate(
                ", ".join(("{:.0e}".format(a) for a in yidx_ibcm)),
                xy=(x+0.125,y_ibcm.loc[yidx_ibcm, m]*1.05), fontsize=6,
                rotation=90, ha="center"
            )
        axes.flat[i].set_xticks(range(len(pq_combi)))
        axes.flat[i].set_xticklabels([(int(p), int(q)) for (p, q) in pq_combi])
        axes.flat[i].set(ylabel=m, xlabel="(p, q) combination")
        for s in ["top", "right"]:
            axes.flat[i].spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join("figures", "noise_struct", "bar_graphs_pq_w_norms.pdf"),
        transparent=True, bbox_inches="tight"
    )
    #plt.show()
    plt.close()

    return None


def main_compare_lm_w_magnitudes(folder, model):
    info = load_fnames_indices(folder, model)
    table_i_pq, table_ij_ab, sim_files, sim_indices = info

    wanted_stats = pd.Index([
        "pnorm", "qnorm", "alpha", "beta",  # will be made index levels
        "mean_m_magnitude",
        "vari_m_magnitude",
        "mean_w_magnitude",
        "vari_w_magnitude"
    ], name="statistics")
    wanted_index = pd.MultiIndex.from_tuples(sim_indices, names=["i", "j"])
    df = pd.DataFrame(
            np.zeros([len(wanted_index), len(wanted_stats)]),
            index=wanted_index, columns=wanted_stats
    )

    for fname in sim_files:
        print("Starting to process file {}".format(fname))
        si, sj = ij_from_name(fname)
        pnorm, qnorm = table_i_pq.get(si)
        alpha, beta = table_ij_ab.get("{}_{}".format(si, sj))
        df.loc[(si, sj), "pnorm":"beta"] = (pnorm, qnorm, alpha, beta)

        res_file = h5py.File(os.path.join(folder, fname), "r")
        # Get jaccard scores of all seeds
        all_mmats = np.abs(concat_mmats(res_file))
        all_wmats = np.abs(concat_wmats(res_file))
        all_lmats = np.abs(concat_lmats(res_file, model))
        non_na_rows = [np.all(all_wmats[i] < 100)
                        for i in range(all_wmats.shape[0])]
        all_wmats = all_wmats[non_na_rows]
        all_projmats = np.einsum("...ij,...jk", all_lmats, all_mmats)
        df.loc[(si, sj), "mean_m_magnitude"] = np.mean(all_mmats)
        df.loc[(si, sj), "vari_m_magnitude"] = np.var(all_mmats)
        df.loc[(si, sj), "mean_w_magnitude"] = np.mean(all_wmats)
        df.loc[(si, sj), "vari_w_magnitude"] = np.var(all_wmats)
        df.loc[(si, sj), "mean_proj_magnitude"] = np.mean(all_projmats)
        df.loc[(si, sj), "vari_proj_magnitude"] = np.var(all_projmats)
        res_file.close()
    return df


def main_plot_m_w_magnitudes(df_ibcm, df_pca):
    # Line plots: x axis is (i, j), y axis is some statistic.
    # One line for PCA, one line for IBCM
    # As a way to identify best/worst cases first, and see variability
    fig, axes = plt.subplots(1, 2)
    fig_size = fig.get_size_inches()
    fig.set_size_inches(fig_size[0]*1.5, fig_size[1]*0.75)
    axes = axes.flatten()
    model_colors = {
        "ibcm": "xkcd:turquoise",
        "biopca": "xkcd:orangey brown",
        "avgsub": "xkcd:navy blue",
        "ideal": "xkcd:powder blue",
        "orthogonal": "xkcd:pale rose",
        "none": "grey"
    }
    metrics = [
        "mean_m_magnitude",
        "mean_w_magnitude",
        "mean_proj_magnitude",
        "vari_m_magnitude",
        "vari_w_magnitude",
        "vari_proj_magnitude"
    ]
    # 2D scatter plot of M vs W mean magnitudes, all points of each model
    ax = axes[0]
    # TODO: adding jitter is stupid, realize that all M are identical
    # in these simulations and playing with Lambda would make more sense.
    jit1 = 0.05*np.random.normal(size=df_ibcm.shape[0])
    jit2 = 0.05*np.random.normal(size=df_ibcm.shape[0])
    ax.plot(
        df_ibcm[metrics[0]] + jit1, df_ibcm[metrics[1]],
        marker="o", ms=4, ls="none",
        color=model_colors.get("ibcm"), label="IBCM"
    )
    ax.plot(
        df_pca[metrics[0]] + jit2, df_pca[metrics[1]],
        marker="s", ms=4, ls="none",
        color=model_colors.get("biopca"), label="BioPCA"
    )
    ax.set(xlabel="LM weights average magnitude",
            ylabel="W weights average magnitude", yscale="log")
    ax.legend()

    # Second plot for magnitude stdev
    ax = axes[1]
    ax.plot(
        df_ibcm[metrics[2]], df_ibcm[metrics[3]], marker="o", ms=4, ls="none",
        color=model_colors.get("ibcm"), label="IBCM"
    )
    ax.plot(
        df_pca[metrics[2]], df_pca[metrics[3]], marker="s", ms=4, ls="none",
        color=model_colors.get("biopca"), label="BioPCA"
    )
    ax.set(xlabel="LM weights magnitude st.dev.",
            ylabel="W weights magnitude st.dev.", yscale="log")

    fig.tight_layout()
    fig.savefig(
        os.path.join("figures", "noise_struct", "scatter_m_w_magnitudes_pq.pdf"),
        transparent=True, bbox_inches="tight"
    )
    plt.show()
    plt.close()

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

    # Compare W and M magnitudes in IBCM and PCA, see if one more realistic
    folder = os.path.join("results", "performance_w")
    df_m_w_ibcm = main_compare_lm_w_magnitudes(folder, "IBCM")
    df_m_w_pca = main_compare_lm_w_magnitudes(folder, "PCA")
    main_plot_m_w_magnitudes(df_m_w_ibcm, df_m_w_pca)

    print("Done!")
