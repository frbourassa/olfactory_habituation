import numpy as np
import matplotlib.pyplot as plt
import h5py
from simulfcts.plotting import hist_outline
from modelfcts.ideal import find_projector, find_parallel_component
from utils.metrics import l2_norm
import os

from simulfcts.analysis import (
    concat_jaccards, 
    concat_mixtures, 
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
    ns_range = np.asarray(ns_ranges[0])
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
        "ideal": "xkcd:powder blue",
        "optimal": "xkcd:light green",
        "orthogonal": "xkcd:pale rose",
        "none": "grey"
    }
    # Get new odor concentrations
    # Assume it's the same for all models: it should!
    with h5py.File(example_file_ibcm, "r") as f:
        n_new_concs = f.get("parameters").get("repeats")[4]
        new_concs = f.get("parameters").get("new_concs")[()]
        activ_fct = f.get("parameters").attrs.get("activ_fct")
    
    # Get the range of N_S tested for each model
    ns_range = get_ns_range_from_files(folder, models)
    all_jacs = {}
    median_jaccard_ranges = {}
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
    
    try:
        example_file_ibcm = [a for a in os.listdir(folder) 
            if a.startswith("ibcm") and a.endswith(".h5")][0]
    except IndexError:
        raise FileNotFoundError(f"No results file found for IBCM in {folder}")

    # One plot per new odor concentration
    fig, axes = plt.subplots(1, n_new_concs, sharex=True)
    fig.set_size_inches(9.5, 4)
    axes = axes.flatten()
    for m in models:  # Plot IBCM last
        for i in range(n_new_concs):
            axes[i].plot(ns_range, median_jaccard_ranges[m][i],
                label=model_nice_names.get(m, m),
                color=model_colors.get(m), alpha=1.0
            )
    # Labeling the graphs, etc.
    for i in range(n_new_concs):
        ax = axes[i]
        axes[i].set_title("New conc. = {:.1f}".format(new_concs[i]))
        axes[i].set_xlabel("Jaccard similarity (higher is better)")
        axes[i].set_ylabel("Probability density")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig("figures/detection/compare_models_dimensionality.pdf",
                transparent=True, bbox_inches="tight")

    plt.show()
    plt.close()
    return None


def main_export_jaccards(dest_name):
    raise NotImplementedError()
    # Compare all algorithms
    folder = os.path.join("results", "performance")
    models = ["ibcm", "biopca", "avgsub", "ideal", "optimal", "orthogonal", "none"]
    model_file_choices = {
        a:os.path.join(folder, a+"_performance_results.h5")
        for a in models
    }
    # Get new odor concentrations
    # Assume it's the same for all models: it should!
    # TODO: check it is indeed the same
    with h5py.File(model_file_choices["ibcm"], "r") as f:
        n_new_concs = f.get("parameters").get("repeats")[4]
        new_concs = f.get("parameters").get("new_concs")[()]
        activ_fct = f.get("parameters").attrs.get("activ_fct")
    # For each model, extract the matrix of Jaccard similarities,
    # then save them all to one npz archive file.
    jac_file = {"new_concs": new_concs}
    for m in models[::-1]:  # Plot IBCM last
        f = h5py.File(model_file_choices[m], "r")
        all_jacs = concat_jaccards(f)
        f.close()
        jac_file[m] = all_jacs
    # Save
    np.savez_compressed(
        dest_name + "_" + activ_fct + ".npz",
        **jac_file
    )
    return None


def main_export_new_back_distances(dest_name):
    raise NotImplementedError()
    # Compare all algorithms
    folder = os.path.join("results", "performance")
    models = ["ibcm", "biopca", "avgsub", "ideal", "optimal", "orthogonal", "none"]
    model_file_choices = {
        a:os.path.join(folder, a+"_performance_results.h5")
        for a in models
    }
    # Check that all models were exposed to the same background indeed
    backs, news = {}, {}
    for m in model_file_choices.keys():
        with h5py.File(model_file_choices[m], "r") as f:
            backs[m] = f.get("odors").get("back_odors")[()]
            news[m] = f.get("odors").get("new_odors")[()]
            activ_fct = f.get("parameters").attrs.get("activ_fct")
            n_backs, n_news = f.get("parameters").get("repeats")[[0, 3]]  # type: ignore
    assert np.all([backs["ibcm"] == backs[m] for m in backs]), "Different backs"
    assert np.all([news["ibcm"] == news[m] for m in news]), "Different news"
    backs = backs["ibcm"]
    news = news["ibcm"]

    # n_runs, n_test_times, n_back_samples, n_new_odors, n_new_concs, skp
    new_back_distances = np.zeros([n_backs, n_news])
    for i in range(n_backs):
        back_proj = find_projector(backs[i].T)
        for j in range(n_news):
            new_par = find_parallel_component(news[j], backs[i], back_proj)
            new_ort = news[j] - new_par
            new_back_distances[i, j] = l2_norm(new_ort)
    np.savez_compressed(
        dest_name + "_" + str(activ_fct) + ".npz",
        new_back_distances=new_back_distances
    )
    return None


def main_export_new_mix_distances(dest_name):
    raise NotImplementedError()
    # Compare all algorithms
    folder = os.path.join("results", "performance")
    models = ["ibcm", "biopca", "avgsub", "ideal", "optimal", "orthogonal", "none"]
    model_file_choices = {
        a:os.path.join(folder, a+"_performance_results.h5")
        for a in models
    }
    # Check that all models were exposed to the same background indeed
    # Get all PN responses to mixtures: each array in mixes will have shape
    # [sim_id, n_new_odors, n_times, n_new_concs, n_back_samples, n_r]
    backs, news, mixes = {}, {}, {}
    for m in model_file_choices.keys():
        with h5py.File(model_file_choices[m], "r") as f:
            backs[m] = f.get("odors").get("back_odors")[()]
            news[m] = f.get("odors").get("new_odors")[()]
            mixes[m] = concat_mixtures(f)  # concat simulations
            activ_fct = f.get("parameters").attrs.get("activ_fct")
            new_concs = f.get("parameters").get("new_concs")[()]
            # n_runs, n_test_times, n_back_samples, n_new_odors, n_new_concs, skp
            n_news, n_concs = f.get("parameters").get("repeats")[[3, 4]]  # type: ignore
    assert np.all([backs["ibcm"] == backs[m] for m in backs]), "Different backs"
    assert np.all([news["ibcm"] == news[m] for m in news]), "Different news"
    backs = backs["ibcm"]
    news = news["ibcm"]

    # Compute the distance between the response to the mixture
    # and the new odor for each trial, each background, etc.
    new_mix_distances = {"new_concs": new_concs}
    for m in models[::-1]:  # Plot IBCM last
        new_mix_distances[m] = np.zeros(mixes[m].shape[:5])
        # One distance for each vector in mixes[m]
        # Do one new odor at a time, news[i] broadcasts 
        # against the OSN dimension, the last (5th) axis in mixes[m]
        for i in range(n_news):
            for j in range(n_concs):
                new_mix_distances[m][:, i, :, j] = l2_norm(
                    mixes[m][:, i, :, j] - new_concs[j]*news[i], axis=3)
    np.savez_compressed(
        dest_name + "_" + str(activ_fct) + ".npz",
        **new_mix_distances
    )
    return None

if __name__ == "__main__":
    main_plot_perf_vs_dimension()
    #main_export_jaccards(
    #    os.path.join("results", "for_plots", "jaccard_similarities")
    #)
    #main_export_new_back_distances(
    #    os.path.join("results", "for_plots", "new_back_distances")
    #)
    #main_export_new_mix_distances(
    #    os.path.join("results", "for_plots", "new_mix_distances")
    #)