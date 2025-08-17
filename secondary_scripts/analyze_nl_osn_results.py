""" Analyze and save to disk a summary of the simulations results
for various levels of OSN nonlinearity, controlled by the free energy
difference, epsilon. Similar script to analyze_dimensionality_results. 

@author: frbourassa
August 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os, sys
pj = os.path.join
if ".." not in sys.path:
    sys.path.insert(1, "..")

from simulfcts.analysis import (
    concat_jaccards, 
    concat_new_mix_distances,
)


def epsil_from_file(fname):
    """ Returns the free energy difference, epsilon """
    f = h5py.File(fname, "r")
    #sd = int(f.attrs["main_seed"])
    epsilon = f.attrs["epsilon"]
    f.close()
    return epsilon


def get_epsil_range_from_files(fold, models):
    eps_ranges = []
    for m in models:
        model_us = [epsil_from_file(pj(fold, a)) for a in os.listdir(fold) 
                    if (a.startswith(m) and a.endswith(".h5"))]
        eps_ranges.append(model_us)
    # Check all these lists are equal, i.e. we tested the same
    # epsilons for all models
    assert sum([sum([eps_ranges[j][i] == eps_ranges[0][i] 
                     for i in range(len(eps_ranges[0]))]) 
                     for j in range(len(eps_ranges))])
    eps_range = np.sort(np.asarray(eps_ranges[0]))
    for i in range(len(eps_range)):
        assert eps_range[i] == epsil_from_file(
            pj(fold, f"{models[0]}_performance_results_nl_osn_{i}.h5"))
    return eps_range


def main_plot_perf_vs_epsilon():
    # Compare all algorithms
    folder = pj("results", "performance_nl_osn")
    models = ["none", "avgsub", "orthogonal", "biopca", 
              "ibcm", "optimal"]
    model_nice_names = {
        "ibcm": "IBCM",
        "biopca": "BioPCA",
        "avgsub": "Average",
        "optimal": "Manifold W",
        "orthogonal": "Orthogonal",
        "none": "None"
    }
    model_colors = {
        "ibcm": "xkcd:turquoise",
        "biopca": "xkcd:orangey brown",
        "avgsub": "xkcd:navy blue",
        "optimal": "xkcd:powder blue",
        "orthogonal": "xkcd:pale rose",
        "none": "grey"
    }

    try:
        example_file_ibcm = [a for a in os.listdir(folder) 
            if a.startswith("ibcm") and a.endswith(".h5")][0]
    except IndexError:
        raise FileNotFoundError(f"No results file found for IBCM in {folder}")

    # Get new odor concentrations
    # Assume it's the same for all models: it should!
    with h5py.File(pj(folder, example_file_ibcm), "r") as f:
        n_new_concs = f.get("parameters").get("repeats")[4]
        new_concs = f.get("parameters").get("new_concs")[()]
        activ_fct = f.get("parameters").attrs.get("activ_fct")
    
    # Get the range of N_S tested for each model
    epsil_range = get_epsil_range_from_files(folder, models)
    all_jacs = {}
    median_jaccard_ranges = {}
    mean_jaccard_ranges = {}
    #ci_595_jaccard_ranges = {}
    # The 5-95 CI is too wide
    std_jaccard_ranges = {}
    for m in models:
        jacs_m = []
        for i, ns in enumerate(epsil_range):
            fname = f"{m}_performance_results_nl_osn_{i}.h5"
            f = h5py.File(pj(folder, fname), "r")
            # Isolate new odor concentration axis, bunch other replicates
            jacs = np.moveaxis(concat_jaccards(f), 3, 0)
            jacs_m.append(jacs.reshape(jacs.shape[0], -1))
            f.close()
        jacs_m = np.stack(jacs_m, axis=0)  # indexed [n_s, new_conc, replicate]
        all_jacs[m] = jacs_m
        median_jaccard_ranges[m] = np.median(jacs_m, axis=2)
        # ci_595_jaccard_ranges[m] = np.quantile(jacs_m, [0.05, 0.95], axis=2)
        std_jaccard_ranges[m] = np.std(jacs_m, axis=2)
        mean_jaccard_ranges[m] = np.mean(jacs_m, axis=2)
    # One plot per new odor concentration
    fig, axes = plt.subplots(1, n_new_concs, sharex=True, sharey=True)
    fig.set_size_inches(3.5*n_new_concs, 4)
    axes = axes.flatten()
    for m in models:  # Plot IBCM last
        for i in range(n_new_concs):
            axes[i].fill_between(epsil_range, 
                mean_jaccard_ranges[m][:, i] - std_jaccard_ranges[m][:, i], 
                mean_jaccard_ranges[m][:, i] + std_jaccard_ranges[m][:, i], 
                color=model_colors.get(m), alpha=0.4
            )
            axes[i].plot(epsil_range, mean_jaccard_ranges[m][:, i],
                label=model_nice_names.get(m, m),
                color=model_colors.get(m), alpha=1.0, marker="o"
            )
    
    # Labeling the graphs
    for i in range(n_new_concs):
        axes[i].set_title("New conc. = {:.1f}".format(new_concs[i]))
        axes[i].set_xlabel(
            r"Free energy difference $\epsilon$ ($k_\mathrm{B} T$)")
        axes[i].set_ylabel("Mean Jaccard similarity")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(pj("figures", "nonlin_adapt", 
                f"compare_models_nl_osn.pdf"),
                transparent=True, bbox_inches="tight")
    #plt.show()
    plt.close()
    return None


def stats_df_from_samples(samp, eps_range, new_concs):
    """ 
    Take a multidimensional array of similarity metric samples, 
    aggregate them into a statistics DataFrame with index levels
    [N_S, new conc.] and columns for mean, median, etc. 
    samp: 3d array, indexed [N_S, new_conc, replicate]
    """
    df_idx = pd.MultiIndex.from_product(
        [eps_range, new_concs], names=["epsilon", "new_conc"])
    df_cols = pd.Index(["mean", "median", "var", 
        "quantile_05", "quantile_95"], name="stats")
    df = pd.DataFrame(0.0, index=df_idx, columns=df_cols)
    df["mean"] = np.mean(samp, axis=2).flatten()
    df["median"] = np.median(samp, axis=2).flatten()
    df["var"] = np.var(samp, axis=2).flatten()
    quantiles = np.quantile(samp, [0.05, 0.95], axis=2)
    df["quantile_05"] = quantiles[0].flatten()
    df["quantile_95"] = quantiles[1].flatten()
    return df


def main_export_jaccard_stats(dest_name, k='jaccard_scores'):
    # Compare all algorithms
    folder = pj("results", "performance_nl_osn")
    models = ["ibcm", "biopca", "avgsub", 
              "optimal", "orthogonal", "none"]
    # Get the range of epsilons tested for each model
    epsil_range = get_epsil_range_from_files(folder, models)
    try:
        example_file_ibcm = [a for a in os.listdir(folder) 
            if a.startswith("ibcm") and a.endswith(".h5")][0]
    except IndexError:
        raise FileNotFoundError(f"No results file found for IBCM in {folder}")

    # Get new odor concentrations
    # Assume it's the same for all models: it should!
    with h5py.File(pj(folder, example_file_ibcm), "r") as f:
        n_new_concs = f.get("parameters").get("repeats")[4]
        new_concs = f.get("parameters").get("new_concs")[()]
        assert len(new_concs) == n_new_concs
        activ_fct = f.get("parameters").attrs.get("activ_fct")

    # For each model, extract the Jaccard similarities array for all epsils,
    # concatenate, compute statistics, then save Jaccard stats for all models
    # into one npz archive file.
    all_jacs = {}
    for m in models:
        jacs_m = []
        for i, ns in enumerate(epsil_range):
            fname = f"{m}_performance_results_nl_osn_{i}.h5"
            f = h5py.File(pj(folder, fname), "r")
            jacs_m.append(concat_jaccards(f, k=k))
            f.close()
        jacs_m = np.stack(jacs_m, axis=0)  
        # currently indexed [epsil, run, new_odor, test_time, new_conc, back_sample] 
        # Reshape to flatten last dimensions and 
        # be indexed [epsil, new_conc, replicate]
        jacs_m = np.moveaxis(jacs_m, source=4, destination=1)
        jacs_m = jacs_m.reshape(jacs_m.shape[0], jacs_m.shape[1], -1)
        all_jacs[m] = stats_df_from_samples(jacs_m, epsil_range, new_concs)

    # Concatenate all models
    all_jacs = pd.concat(all_jacs, names=["Model"])

    # Save the jacs
    print(all_jacs.shape)
    dest_name_full = dest_name + ".h5"
    all_jacs.to_hdf(dest_name_full, key="df")
    # and the information about the epsil range in a separate Series
    # in the same file, with key "epsil_range"
    epsil_ser = pd.Series(epsil_range, name="epsilon", 
        index=pd.Index(np.arange(len(epsil_range)), name="epsil_index"))
    epsil_ser.to_hdf(dest_name_full, key="epsil_range")
    return None


def main_export_new_mix_distance_stats(dest_name):
    # Compare all algorithms
    folder = pj("results", "performance_nl_osn")
    models = ["ibcm", "biopca", "avgsub", 
              "optimal", "orthogonal", "none"]
    # Get the range of N_S tested for each model
    epsil_range = get_epsil_range_from_files(folder, models)
    try:
        example_file_ibcm = [a for a in os.listdir(folder) 
            if a.startswith("ibcm") and a.endswith(".h5")][0]
    except IndexError:
        raise FileNotFoundError(f"No results file found for IBCM in {folder}")
   
    # Get new odor concentrations
    # Assume it's the same for all models: it should!
    with h5py.File(pj(folder, example_file_ibcm), "r") as f:
        n_new_concs = f.get("parameters").get("repeats")[4]
        new_concs = f.get("parameters").get("new_concs")[()]
        assert len(new_concs) == n_new_concs
        activ_fct = f.get("parameters").attrs.get("activ_fct")

    # Check that all models were exposed to the same background indeed
    backs, news = {}, {}
    for n, ns in enumerate(epsil_range):
        backs[n] = {}
        news[n] = {}
        for m in models:
            fname = pj(folder, f"{m}_performance_results_nl_osn_{n}.h5")
            with h5py.File(fname, "r") as f:
                backs[n][m] = f.get("odors").get("back_odors")[()]
                news[n][m] = f.get("odors").get("new_odors")[()]
                activ_fct = f.get("parameters").attrs.get("activ_fct")
        assert np.all([backs[n]["ibcm"] == backs[n][m] 
                       for m in backs[n]]), "Different backs"
        assert np.all([news[n]["ibcm"] == news[n][m] 
                       for m in news[n]]), "Different news"
        backs[n] = backs[n]["ibcm"]
        news[n] = news[n]["ibcm"]

    # For each model, extract the matrix of new odor - mixture distances 
    # for all N_S, concatenate, then save concatenated distances for all models
    # into one npz archive file.
    all_dists = {}
    for m in models:
        dists_m = []
        for n, ns in enumerate(epsil_range):
            fname = f"{m}_performance_results_nl_osn_{n}.h5"
            f = h5py.File(pj(folder, fname), "r")
            dists_m.append(concat_new_mix_distances(f))
            f.close()
        dists_m = np.stack(dists_m, axis=0)
        # currently indexed [n_s, run, new_odor, test_time, new_conc, back_sample] 
        # Reshape to flatten last dimensions and 
        # be indexed [n_s, new_conc, replicate]
        dists_m = np.moveaxis(dists_m, source=4, destination=1)
        dists_m = dists_m.reshape(dists_m.shape[0], dists_m.shape[1], -1)
        all_dists[m] = stats_df_from_samples(dists_m, epsil_range, new_concs)
    
    # Concatenate all models
    all_dists = pd.concat(all_dists, names=["Model"])

    # Save the results
    print(all_dists.shape)
    dest_name_full = dest_name + "_" + activ_fct + ".h5"
    all_dists.to_hdf(dest_name_full, key="df")
    # and the information about the epsilon range in a separate Series
    # in the same file, with key "epsil_range"
    epsil_ser = pd.Series(epsil_range, name="epsilon", 
        index=pd.Index(np.arange(len(epsil_range)), name="epsil_index"))
    epsil_ser.to_hdf(dest_name_full, key="epsil_range")

    return None

if __name__ == "__main__":
    print("Starting up analysis script...")

    main_plot_perf_vs_epsilon()
    print("Finished plotting performance vs nonlinearity threshold (epsilon)")
    
    # Export Jaccard similarities to new odors
    main_export_jaccard_stats(pj("results", 
       "for_plots", "nonlin_adapt", "jaccard_similarities_stats_nl_osn")
    )
    print("Finished exporting Jaccard similarity stats")

    # Also export Jaccard similarities to background
    # The resulting file will have Jaccard similiarities in key "df"
    # and the epsilons corresponding to the 
    # integer indices in key "epsil_range"
    main_export_jaccard_stats(
        os.path.join("results", "for_plots", "nonlin_adapt", 
            "jaccard_similarities_back_nl_osn"), k='jaccard_scores_back'
    )

    main_export_new_mix_distance_stats(pj("results",  
       "for_plots", "nonlin_adapt", "new_mix_distances_stats_nl_osn")
    )
    print("Finished exporting distances between"
          + " model responses to mixtures and new odors"
    )
