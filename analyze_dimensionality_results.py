import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from simulfcts.plotting import hist_outline
from modelfcts.ideal import find_projector, find_parallel_component
from utils.metrics import l2_norm
import os

from simulfcts.analysis import (
    concat_jaccards, 
    concat_new_mix_distances,
    concat_sstats, 
    concat_wmats, 
    concat_mmats, 
    concat_lmats
)


def ns_from_name(fname):
    """ Returns the integer number of OSNs, N_S """
    return int((fname.split(".")[0]).split("_")[-1])

def get_ns_range_from_files(fold, models):
    ns_ranges = []
    for m in models:
        model_ns = [ns_from_name(a) for a in os.listdir(fold) 
                    if (a.startswith(m) and a.endswith(".h5"))]
        ns_ranges.append(model_ns)
    # Check all these lists are equal, i.e. we tested the same
    # N_S for all models
    assert sum([sum([ns_ranges[j][i] == ns_ranges[0][i] 
                     for i in range(len(ns_ranges[0]))]) 
                     for j in range(len(ns_ranges))])
    ns_range = np.sort(np.asarray(ns_ranges[0]))
    return ns_range


def main_plot_perf_vs_dimension():
    # Compare all algorithms
    folder = os.path.join("results", "performance_ns")
    models = ["none", "avgsub", "ideal", "orthogonal", "biopca", "ibcm", "optimal"]
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
    with h5py.File(os.path.join(folder, example_file_ibcm), "r") as f:
        n_new_concs = f.get("parameters").get("repeats")[4]
        new_concs = f.get("parameters").get("new_concs")[()]
        activ_fct = f.get("parameters").attrs.get("activ_fct")
    
    # Get the range of N_S tested for each model
    ns_range = get_ns_range_from_files(folder, models)
    all_jacs = {}
    median_jaccard_ranges = {}
    mean_jaccard_ranges = {}
    #ci_595_jaccard_ranges = {}
    # The 5-95 CI is too wide
    std_jaccard_ranges = {}
    for m in models:
        jacs_m = []
        for ns in ns_range:
            fname = f"{m}_performance_results_ns_{ns}.h5"
            f = h5py.File(os.path.join(folder, fname), "r")
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
    
    # Also, compare to Jaccard between random odors
    jacs_random = []
    for ns in ns_range:
        fname = f"similarity_random_odors_ns_{ns}.npz"
        jacs = np.load(os.path.join(folder, fname))["jaccard_scores"]
        # shaped [n_runs, pairs of odors], flatten because all
        # runs/projection matrices should be concatenated as samples
        jacs_random.append(jacs.flatten())
    # Reshape to have a dummy "new odor" axis first, then N_S, then samples
    jacs_random = np.stack(jacs_random, axis=0).reshape(1, len(ns_range), -1)
    all_jacs["random"] = jacs_random
    std_jaccard_ranges["random"] = np.std(jacs_random, axis=2)
    mean_jaccard_ranges["random"] = np.mean(jacs_random, axis=2)
    
    # One plot per new odor concentration
    fig, axes = plt.subplots(1, n_new_concs, sharex=True, sharey=True)
    fig.set_size_inches(3.5*n_new_concs, 4)
    axes = axes.flatten()
    for m in models:  # Plot IBCM last
        for i in range(n_new_concs):
            axes[i].fill_between(ns_range, 
                mean_jaccard_ranges[m][:, i] - std_jaccard_ranges[m][:, i], 
                mean_jaccard_ranges[m][:, i] + std_jaccard_ranges[m][:, i], 
                color=model_colors.get(m), alpha=0.4
            )
            axes[i].plot(ns_range, mean_jaccard_ranges[m][:, i],
                label=model_nice_names.get(m, m),
                color=model_colors.get(m), alpha=1.0
            )
    
    # Labeling the graphs, adding similarity between random odors, etc.
    for i in range(n_new_concs):
        # Add similarity between random odors
        axes[i].plot(ns_range, mean_jaccard_ranges["random"][0], 
                     label="Random", color="k", ls="--"
        )
        axes[i].fill_between(ns_range, 
            mean_jaccard_ranges["random"][0] - std_jaccard_ranges["random"][0], 
            mean_jaccard_ranges["random"][0] + std_jaccard_ranges["random"][0],
            color="k", alpha=0.4
        )
        axes[i].set_title("New conc. = {:.1f}".format(new_concs[i]))
        axes[i].set_xlabel(r"OSN space dimensionality, $N_S$")
        axes[i].set_ylabel("Mean Jaccard similarity")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "detection", 
                f"compare_models_dimensionality_{activ_fct}.pdf"),
                transparent=True, bbox_inches="tight")
    plt.show()
    plt.close()
    return None


def stats_df_from_samples(samp, ns_range, new_concs):
    """ 
    Take a multidimensional array of similarity metric samples, 
    aggregate them into a statistics DataFrame with index levels
    [N_S, new conc.] and columns for mean, median, etc. 
    samp: 3d array, indexed [N_S, new_conc, replicate]
    """
    df_idx = pd.MultiIndex.from_product(
        [ns_range, new_concs], names=["N_S", "new_conc"])
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


def main_export_jaccard_stats(dest_name):
    # Compare all algorithms
    folder = os.path.join("results", "performance_ns")
    models = ["ibcm", "biopca", "avgsub", "ideal", 
              "optimal", "orthogonal", "none"]
    # Get the range of N_S tested for each model
    ns_range = get_ns_range_from_files(folder, models)
    try:
        example_file_ibcm = [a for a in os.listdir(folder) 
            if a.startswith("ibcm") and a.endswith(".h5")][0]
    except IndexError:
        raise FileNotFoundError(f"No results file found for IBCM in {folder}")

    # Get new odor concentrations
    # Assume it's the same for all models: it should!
    with h5py.File(os.path.join(folder, example_file_ibcm), "r") as f:
        n_new_concs = f.get("parameters").get("repeats")[4]
        new_concs = f.get("parameters").get("new_concs")[()]
        assert len(new_concs) == n_new_concs
        activ_fct = f.get("parameters").attrs.get("activ_fct")

    # For each model, extract the matrix of Jaccard similarities for all N_S,
    # concatenate, compute statistics, then save Jaccard stats for all models
    # into one npz archive file.
    all_jacs = {}
    for m in models:
        jacs_m = []
        for ns in ns_range:
            fname = f"{m}_performance_results_ns_{ns}.h5"
            f = h5py.File(os.path.join(folder, fname), "r")
            jacs_m.append(concat_jaccards(f))
            f.close()
        jacs_m = np.stack(jacs_m, axis=0)  
        # currently indexed [n_s, run, new_odor, test_time, new_conc, back_sample] 
        # Reshape to flatten last dimensions and 
        # be indexed [n_s, new_conc, replicate]
        jacs_m = np.moveaxis(jacs_m, source=4, destination=1)
        jacs_m = jacs_m.reshape(jacs_m.shape[0], jacs_m.shape[1], -1)
        all_jacs[m] = stats_df_from_samples(jacs_m, ns_range, new_concs)
    
    # Also, compare to Jaccard between random odors
    jacs_random = []
    for ns in ns_range:
        fname = f"similarity_random_odors_ns_{ns}.npz"
        jacs = np.load(os.path.join(folder, fname))["jaccard_scores"]
        # shaped [n_runs, pairs of odors], flatten because all
        # runs/projection matrices should be concatenated as samples
        jacs_random.append(jacs.flatten())
    # Reshape to be indexed [N_S, new odor conc. (dummy), sample]
    jacs_random = np.stack(jacs_random, axis=0).reshape(len(ns_range), 1, -1)
    jacs_random = np.tile(jacs_random, (1, n_new_concs, 1))
    all_jacs["random"] = stats_df_from_samples(jacs_random, ns_range, new_concs)

    # Concatenate all models
    all_jacs = pd.concat(all_jacs, names=["Model"])

    # Save, the information about new concentrations and N_S dimensions
    # is saved in the DataFrame index. 
    print(all_jacs.shape)
    all_jacs.to_hdf(dest_name + "_" + activ_fct + ".hdf", key="df")
    return None


def main_export_new_back_distances(dest_name):
    """ The concatenated array of background-new odor distances
    will have shape [len(ns_range), n_backs, n_news]"""
    # Compare all algorithms
    folder = os.path.join("results", "performance_ns")
    models = ["ibcm", "biopca", "avgsub", "ideal", 
              "optimal", "orthogonal", "none"]
    # Get the range of N_S tested for each model
    ns_range = get_ns_range_from_files(folder, models)
    try:
        example_file_ibcm = [a for a in os.listdir(folder) 
            if a.startswith("ibcm") and a.endswith(".h5")][0]
    except IndexError:
        raise FileNotFoundError(f"No results file found for IBCM in {folder}")

    # Get new odor concentrations
    # Assume it's the same for all models: it should!
    with h5py.File(os.path.join(folder, example_file_ibcm), "r") as f:
        n_new_concs = f.get("parameters").get("repeats")[4]
        new_concs = f.get("parameters").get("new_concs")[()]
        assert len(new_concs) == n_new_concs
        activ_fct = f.get("parameters").attrs.get("activ_fct")

    # Check that all models were exposed to the same background indeed
    backs, news = {}, {}
    for ns in ns_range:
        backs[ns] = {}
        news[ns] = {}
        for m in models:
            fname = os.path.join(folder, f"{m}_performance_results_ns_{ns}.h5")
            with h5py.File(fname, "r") as f:
                backs[ns][m] = f.get("odors").get("back_odors")[()]
                news[ns][m] = f.get("odors").get("new_odors")[()]
                activ_fct = f.get("parameters").attrs.get("activ_fct")
                n_backs, n_news = f.get("parameters").get("repeats")[[0, 3]]  # type: ignore
        assert np.all([backs[ns]["ibcm"] == backs[ns][m] 
                       for m in backs[ns]]), "Different backs"
        assert np.all([news[ns]["ibcm"] == news[ns][m] 
                       for m in news[ns]]), "Different news"
        backs[ns] = backs[ns]["ibcm"]
        news[ns] = news[ns]["ibcm"]

    # n_runs, n_test_times, n_back_samples, n_new_odors, n_new_concs, skp
    new_back_distances = np.zeros([len(ns_range), n_backs, n_news])
    for n, ns in enumerate(ns_range):
        for i in range(n_backs):
            back_proj = find_projector(backs[ns][i].T)
            for j in range(n_news):
                new_par = find_parallel_component(
                    news[ns][j], backs[ns][i], back_proj)
                new_ort = news[ns][j] - new_par
                new_back_distances[n, i, j] = l2_norm(new_ort)
    np.savez_compressed(
        dest_name + "_" + str(activ_fct) + ".npz",
        new_back_distances=new_back_distances
    )
    return None


def main_export_new_mix_distance_stats(dest_name):
    # Compare all algorithms
    folder = os.path.join("results", "performance_ns")
    models = ["ibcm", "biopca", "avgsub", "ideal", 
              "optimal", "orthogonal", "none"]
    # Get the range of N_S tested for each model
    ns_range = get_ns_range_from_files(folder, models)
    try:
        example_file_ibcm = [a for a in os.listdir(folder) 
            if a.startswith("ibcm") and a.endswith(".h5")][0]
    except IndexError:
        raise FileNotFoundError(f"No results file found for IBCM in {folder}")
   
    # Get new odor concentrations
    # Assume it's the same for all models: it should!
    with h5py.File(os.path.join(folder, example_file_ibcm), "r") as f:
        n_new_concs = f.get("parameters").get("repeats")[4]
        new_concs = f.get("parameters").get("new_concs")[()]
        assert len(new_concs) == n_new_concs
        activ_fct = f.get("parameters").attrs.get("activ_fct")

    # Check that all models were exposed to the same background indeed
    backs, news = {}, {}
    for ns in ns_range:
        backs[ns] = {}
        news[ns] = {}
        for m in models:
            fname = os.path.join(folder, f"{m}_performance_results_ns_{ns}.h5")
            with h5py.File(fname, "r") as f:
                backs[ns][m] = f.get("odors").get("back_odors")[()]
                news[ns][m] = f.get("odors").get("new_odors")[()]
                activ_fct = f.get("parameters").attrs.get("activ_fct")
        assert np.all([backs[ns]["ibcm"] == backs[ns][m] 
                       for m in backs[ns]]), "Different backs"
        assert np.all([news[ns]["ibcm"] == news[ns][m] 
                       for m in news[ns]]), "Different news"
        backs[ns] = backs[ns]["ibcm"]
        news[ns] = news[ns]["ibcm"]

    # For each model, extract the matrix of new odor - mixture distances 
    # for all N_S, concatenate, then save concatenated distances for all models
    # into one npz archive file.
    all_dists = {}
    for m in models:
        dists_m = []
        for ns in ns_range:
            fname = f"{m}_performance_results_ns_{ns}.h5"
            f = h5py.File(os.path.join(folder, fname), "r")
            dists_m.append(concat_new_mix_distances(f))
            f.close()
        dists_m = np.stack(dists_m, axis=0)
        # currently indexed [n_s, run, new_odor, test_time, new_conc, back_sample] 
        # Reshape to flatten last dimensions and 
        # be indexed [n_s, new_conc, replicate]
        dists_m = np.moveaxis(dists_m, source=4, destination=1)
        dists_m = dists_m.reshape(dists_m.shape[0], dists_m.shape[1], -1)
        all_dists[m] = stats_df_from_samples(dists_m, ns_range, new_concs)
    
    # Also add distance between random odors
    dists_random = []
    for ns in ns_range:
        fname = f"similarity_random_odors_ns_{ns}.npz"
        dists_m = np.load(os.path.join(folder, fname))["y_l2_distances"]
        # Shaped [pairs of iid odors], 1d axis, one per N_S: already flattened
        dists_random.append(dists_m)
    dists_random = np.stack(dists_random, axis=0).reshape(len(ns_range), 1, -1)
    dists_random = np.tile(dists_random, (1, n_new_concs, 1))
    all_dists["random"] = stats_df_from_samples(
                            dists_random, ns_range, new_concs)
    
    # Concatenate all models
    all_dists = pd.concat(all_dists, names=["Model"])

    # Save, the information about new concentrations and N_S dimensions
    # is saved in the DataFrame index. 
    print(all_dists.shape)
    all_dists.to_hdf(dest_name + "_" + activ_fct + ".hdf", key="df")
    return None

if __name__ == "__main__":
    print("Starting up analysis script...")
    
    main_plot_perf_vs_dimension()
    print("Finished plotting performance vs OSN dimensionality")
    
    main_export_jaccard_stats(os.path.join("results", 
       "for_plots", "jaccard_similarities_stats_dimensionality")
    )
    print("Finished exporting Jaccard similarity stats")
    
    main_export_new_mix_distance_stats(os.path.join("results",  
       "for_plots", "new_mix_distances_stats_dimensionality")
    )
    print("Finished exporting distances between"
          + " model responses to mixtures and new odors"
    )
 
    main_export_new_back_distances(os.path.join("results",  
       "for_plots", "new_back_distances_dimensionality")
    )
    print("Finished exporting distances between new odors and backgrounds")
