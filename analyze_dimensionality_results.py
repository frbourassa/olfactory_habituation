import numpy as np
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
    models = ["none", "avgsub", "orthogonal", "biopca", "ibcm", "optimal"]
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
    # Labeling the graphs, etc.
    for i in range(n_new_concs):
        ax = axes[i]
        axes[i].set_title("New conc. = {:.1f}".format(new_concs[i]))
        axes[i].set_xlabel(r"OSN space dimensionality, $N_S$")
        axes[i].set_ylabel("Median Jaccard similarity")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "detection", 
                f"compare_models_dimensionality_{activ_fct}.pdf"),
                transparent=True, bbox_inches="tight")
    plt.show()
    plt.close()
    return None


def main_export_jaccards(dest_name):
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
    # concatenate, then save concatenated Jaccards for all models
    # into one npz archive file.
    all_jacs = {}
    for m in models:
        jacs_m = []
        for ns in ns_range:
            fname = f"{m}_performance_results_ns_{ns}.h5"
            f = h5py.File(os.path.join(folder, fname), "r")
            jacs_m.append(concat_jaccards(f))
            f.close()
        jacs_m = np.stack(jacs_m, axis=0)  # indexed [n_s, new_conc, replicate]
        all_jacs[m] = jacs_m

    # Save, with also some extra information
    all_jacs["new_concs"] = new_concs
    all_jacs["ns_range"] = ns_range
    np.savez_compressed(
        dest_name + "_" + activ_fct + ".npz",
        **all_jacs
    )
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


def main_export_new_mix_distances(dest_name):
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
        dists_m = np.stack(dists_m, axis=0)  # indexed [n_s, new_conc, replicate]
        all_dists[m] = dists_m

    # Save, with also some extra information
    all_dists["new_concs"] = new_concs
    all_dists["ns_range"] = ns_range
    np.savez_compressed(
        dest_name + "_" + activ_fct + ".npz",
        **all_dists
    )
    return None

if __name__ == "__main__":
    main_plot_perf_vs_dimension()
    main_export_jaccards(os.path.join("results", 
       "for_plots", "jaccard_similarities_dimensionality")
    )
    main_export_new_back_distances(os.path.join("results",  
       "for_plots", "new_back_distances_dimensionality")
    )
    main_export_new_mix_distances(os.path.join("results",  
       "for_plots", "new_mix_distances_dimensionality")
    )