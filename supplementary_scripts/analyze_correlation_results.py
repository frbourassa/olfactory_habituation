""" Analyze and save to disk a summary of the simulations results
for correlated odors. Similar script to analyze_dimensionality_results. 
Run from main folder as 
>>> python supplementary_scripts/analyze_correlation_results.py

@author: frbourassa
July 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os, sys
pj = os.path.join
if ".." not in sys.path:
    sys.path.insert(1, "..")

from modelfcts.ideal import find_projector, find_parallel_component
from utils.metrics import l2_norm
from simulfcts.analysis import (
    concat_jaccards, 
    concat_new_mix_distances,
)


def correl_from_file(fname):
    """ Returns the correlation rho amplitude """
    f = h5py.File(fname, "r")
    #sd = int(f.attrs["main_seed"])
    rho = f.attrs["correl_rho"]
    f.close()
    return rho


def get_correl_range_from_files(fold, models):
    rho_ranges = []
    for m in models:
        model_rho = [correl_from_file(pj(fold, a)) for a in os.listdir(fold) 
                    if (a.startswith(m) and a.endswith(".h5"))]
        rho_ranges.append(model_rho)
    # Check all these lists are equal, i.e. we tested the same
    # N_S for all models
    assert sum([sum([rho_ranges[j][i] == rho_ranges[0][i] 
                     for i in range(len(rho_ranges[0]))]) 
                     for j in range(len(rho_ranges))])
    rho_range = np.sort(np.asarray(rho_ranges[0]))
    for i in range(len(rho_range)):
        assert rho_range[i] == correl_from_file(
            pj(fold, f"{models[0]}_performance_results_correlation_{i}.h5"))
    return rho_range


def main_plot_perf_vs_correl():
    # Compare all algorithms
    folder = pj("results", "performance_correlation")
    models = ["none", "avgsub", "orthogonal", "biopca", 
              "ibcm", "ideal", "optimal"]
    model_nice_names = {
        "ibcm": "IBCM",
        "biopca": "BioPCA",
        "avgsub": "Average",
        "ideal": "Ideal",
        "optimal": "Manifold W",
        "orthogonal": "Orthogonal",
        "none": "None"
    }
    model_colors = {
        "ibcm": "xkcd:turquoise",
        "biopca": "xkcd:orangey brown",
        "avgsub": "xkcd:navy blue",
        "ideal": "xkcd:light green",
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
    
    # Get the range of rho tested for each model
    correl_range = get_correl_range_from_files(folder, models)
    all_jacs = {}
    median_jaccard_ranges = {}
    mean_jaccard_ranges = {}
    #ci_595_jaccard_ranges = {}
    # The 5-95 CI is too wide
    std_jaccard_ranges = {}
    for m in models:
        jacs_m = []
        for i, rho in enumerate(correl_range):
            fname = f"{m}_performance_results_correlation_{i}.h5"
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
            axes[i].fill_between(correl_range, 
                mean_jaccard_ranges[m][:, i] - std_jaccard_ranges[m][:, i], 
                mean_jaccard_ranges[m][:, i] + std_jaccard_ranges[m][:, i], 
                color=model_colors.get(m), alpha=0.4
            )
            axes[i].plot(correl_range, mean_jaccard_ranges[m][:, i],
                label=model_nice_names.get(m, m),
                color=model_colors.get(m), alpha=1.0, marker="o"
            )
    
    # Labeling the graphs
    for i in range(n_new_concs):
        axes[i].set_title("New conc. = {:.1f}".format(new_concs[i]))
        axes[i].set_xlabel(r"Odor correlation, $\rho$")
        axes[i].set_ylabel("Mean Jaccard similarity")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(pj("figures", "correlation", 
                f"compare_models_turbulent_correlation.pdf"),
                transparent=True, bbox_inches="tight")
    #plt.show()
    plt.close()
    return None


def stats_df_from_samples(samp, rho_range, new_concs):
    """ 
    Take a multidimensional array of similarity metric samples, 
    aggregate them into a statistics DataFrame with index levels
    [correl_rho, new conc.] and columns for mean, median, etc. 
    samp: 3d array, indexed [correl_rho, new_conc, replicate]
    """
    df_idx = pd.MultiIndex.from_product(
        [rho_range, new_concs], names=["correl_rho", "new_conc"])
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
    folder = pj("results", "performance_correlation")
    models = ["ibcm", "biopca", "avgsub", "ideal", 
              "optimal", "orthogonal", "none"]
    # Get the range of correlation coefficients tested for each model
    correl_range = get_correl_range_from_files(folder, models)
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

    # For each model, extract the Jaccard similarities array for all correls,
    # concatenate, compute statistics, then save Jaccard stats for all models
    # into one npz archive file.
    all_jacs = {}
    for m in models:
        jacs_m = []
        for i, rho in enumerate(correl_range):
            fname = f"{m}_performance_results_correlation_{i}.h5"
            f = h5py.File(pj(folder, fname), "r")
            jacs_m.append(concat_jaccards(f, k=k))
            f.close()
        jacs_m = np.stack(jacs_m, axis=0)  
        # currently indexed [correl, run, new_odor, test_time, new_conc, back_sample] 
        # Reshape to flatten last dimensions and 
        # be indexed [correl, new_conc, replicate]
        jacs_m = np.moveaxis(jacs_m, source=4, destination=1)
        jacs_m = jacs_m.reshape(jacs_m.shape[0], jacs_m.shape[1], -1)
        all_jacs[m] = stats_df_from_samples(jacs_m, correl_range, new_concs)

    # Concatenate all models
    all_jacs = pd.concat(all_jacs, names=["Model"])

    # Save the jacs
    print(all_jacs.shape)
    dest_name_full = dest_name + "_" + activ_fct + ".h5"
    all_jacs.to_hdf(dest_name_full, key="df")
    # and the information about the correlation range in a separate Series
    # in the same file, with key "correl_range"
    rho_ser = pd.Series(correl_range, name="correl_rho", 
        index=pd.Index(np.arange(len(correl_range)), name="correl_index"))
    rho_ser.to_hdf(dest_name_full, key="correl_range")
    return None


def main_export_new_back_distances(dest_name):
    """ The concatenated array of background-new odor distances
    will have shape [len(correl_range), n_backs, n_news]"""
    # Compare all algorithms
    folder = pj("results", "performance_correlation")
    models = ["ibcm", "biopca", "avgsub", "ideal", 
              "optimal", "orthogonal", "none"]
    # Get the range of correlation coefficients tested for each model
    correl_range = get_correl_range_from_files(folder, models)
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
    for n, rho in enumerate(correl_range):
        backs[n] = {}
        news[n] = {}
        for m in models:
            fname = pj(folder, f"{m}_performance_results_correlation_{n}.h5")
            with h5py.File(fname, "r") as f:
                backs[n][m] = f.get("odors").get("back_odors")[()]
                news[n][m] = f.get("odors").get("new_odors")[()]
                activ_fct = f.get("parameters").attrs.get("activ_fct")
                n_backs, n_news = f.get("parameters").get("repeats")[[0, 3]]  # type: ignore
        assert np.all([backs[n]["ibcm"] == backs[n][m] 
                       for m in backs[n]]), "Different backs"
        assert np.all([news[n]["ibcm"] == news[n][m] 
                       for m in news[n]]), "Different news"
        backs[n] = backs[n]["ibcm"]
        news[n] = news[n]["ibcm"]

    # n_runs, n_test_times, n_back_samples, n_new_odors, n_new_concs, skp
    new_back_distances = np.zeros([len(correl_range), n_backs, n_news])
    for n, rho in enumerate(correl_range):
        for i in range(n_backs):
            back_proj = find_projector(backs[n][i].T)
            for j in range(n_news):
                new_par = find_parallel_component(
                    news[n][j], backs[n][i], back_proj)
                new_ort = news[n][j] - new_par
                new_back_distances[n, i, j] = l2_norm(new_ort)
    np.savez_compressed(
        dest_name + "_" + str(activ_fct) + ".npz",
        new_back_distances=new_back_distances
    )
    return None


def main_export_new_mix_distance_stats(dest_name):
    # Compare all algorithms
    folder = pj("results", "performance_correlation")
    models = ["ibcm", "biopca", "avgsub", "ideal", 
              "optimal", "orthogonal", "none"]
    # Get the range of N_S tested for each model
    correl_range = get_correl_range_from_files(folder, models)
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
    for n, rho in enumerate(correl_range):
        backs[n] = {}
        news[n] = {}
        for m in models:
            fname = pj(folder, f"{m}_performance_results_correlation_{n}.h5")
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
        for n, rho in enumerate(correl_range):
            fname = f"{m}_performance_results_correlation_{n}.h5"
            f = h5py.File(pj(folder, fname), "r")
            dists_m.append(concat_new_mix_distances(f))
            f.close()
        dists_m = np.stack(dists_m, axis=0)
        # currently indexed [n_s, run, new_odor, test_time, new_conc, back_sample] 
        # Reshape to flatten last dimensions and 
        # be indexed [n_s, new_conc, replicate]
        dists_m = np.moveaxis(dists_m, source=4, destination=1)
        dists_m = dists_m.reshape(dists_m.shape[0], dists_m.shape[1], -1)
        all_dists[m] = stats_df_from_samples(dists_m, correl_range, new_concs)
    
    # Concatenate all models
    all_dists = pd.concat(all_dists, names=["Model"])

    # Save the results
    print(all_dists.shape)
    dest_name_full = dest_name + "_" + activ_fct + ".h5"
    all_dists.to_hdf(dest_name_full, key="df")
    # and the information about the correlation range in a separate Series
    # in the same file, with key "correl_range"
    rho_ser = pd.Series(correl_range, name="correl_rho", 
        index=pd.Index(np.arange(len(correl_range)), name="correl_index"))
    rho_ser.to_hdf(dest_name_full, key="correl_range")

    return None

if __name__ == "__main__":
    print("Starting up analysis script...")

    main_plot_perf_vs_correl()
    print("Finished plotting performance vs correlation coefficient")
    
    # Export Jaccard similarities to new odors
    main_export_jaccard_stats(pj("results", 
       "for_plots", "correlation", "jaccard_similarities_stats_correlation")
    )
    print("Finished exporting Jaccard similarity stats")

    # Also export Jaccard similarities to background
    # The resulting file will have Jaccard similiarities in key "df"
    # and the correlation coefficient corresponding to the 
    # integer indices in key "correl_range"
    main_export_jaccard_stats(
        os.path.join("results", "for_plots", "correlation", "jaccard_similarities_back_correlation"),
        k='jaccard_scores_back'
    )

    # With correlated odors, it will be interesting
    # to plot ymix - ynew distance, and ynew - s_back distance too. 
    # These distances will increase. 
    main_export_new_mix_distance_stats(pj("results",  
       "for_plots", "correlation", "new_mix_distances_stats_correlation")
    )
    print("Finished exporting distances between"
          + " model responses to mixtures and new odors"
    )
    
    main_export_new_back_distances(pj("results",  
       "for_plots", "correlation", "new_back_distances_correlation")
    )
    print("Finished exporting distances between new odors and backgrounds")
