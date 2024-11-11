import numpy as np
import matplotlib.pyplot as plt
import h5py
from simulfcts.habituation_recognition import id_to_simkey
from simulfcts.plotting import hist_outline
from modelfcts.ideal import find_projector, find_parallel_component
from utils.metrics import l2_norm
import os

from analyze_comparison_results import (
    concat_jaccards,
    concat_sstats
)


def get_lambda0(f):
    if f.attrs.get("model") == "IBCM":
        lambd_0 = f.get("parameters").get("m_rates")[3]
    elif f.attrs.get("model") == "PCA":
        lambd_0 = f.get("parameters").get("m_rates")[2]
    else:
        raise KeyError("Model {} not available for Lambda"
                        .format(f.attrs.get("model")))
    return lambd_0


def main_plot_performance():
    # Compare IBCM and PCA as a function of Lambda
    # Plot s statistics vs lambda, plot Jaccard statistics vs Lambda
    folder = os.path.join("results", "performance")
    models = ["ibcm", "biopca"]
    model_nice_names = {
        "ibcm": "IBCM",
        "biopca": "BioPCA",
        "avgsub": "Average",
        "ideal": "Ideal",
        "optimal": "Optimal", 
        "orthogonal": "Orthogonal",
        "none": "None"
    }
    model_file_choices = {
        a:os.path.join(folder, a+"_performance_lambda.h5")
        for a in models
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
    # Get new odor concentrations and activation function
    # Assume it's the same for all models: it should!
    with h5py.File(model_file_choices["ibcm"], "r") as f:
        n_new_concs = f.get("parameters").get("repeats")[4]
        new_concs = f.get("parameters").get("new_concs")[()]
        activ_fct = f.get("parameters").attrs.get("activ_fct")

    # First, plots of statistics vs lambda
    fig, axes = plt.subplots(1, 3, sharex=True)
    fig.set_size_inches(9.5, 3)
    axes = axes.flatten()
    for m in models[::-1]:
        f = h5py.File(model_file_choices[m], "r")
        all_stats = concat_sstats(f)
        lambd_axis = f.get("parameters").get("lambd_range")[()]
        mask = all_stats > 100.0  # Clip excessively large values
        masked_stats = np.ma.masked_array(all_stats, mask)
        # Rescale lambda axis by the default value? 1 for IBCM, 10 for PCA
        lambd_0 = get_lambda0(f)
        f.close()
        for i in range(all_stats.shape[1]):
            axes[i].plot(lambd_axis / lambd_0, masked_stats[:, i],
                label=model_nice_names.get(m, m),
                color=model_colors.get(m), lw=2.0
            )
    # Labeling the graphs, etc.
    stat_names = ["Mean", "Variance", "3rd moment"]
    for i in range(all_stats.shape[1]):
        ax = axes[i]
        axes[i].set_title(stat_names[i])
        axes[i].set_xlabel(r"Scale $\Lambda / \Lambda_0$")
        axes[i].set_ylabel(r"Statistic of $\| \vec{s} \|$")
        axes[i].set_xscale("log")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("figures/detection/s_stats_vs_lambda.pdf", transparent=True,
                bbox_inches="tight")
    plt.close()

    # Second, plots of Jaccard median and mean similarities, for each new conc.
    # One plot per new odor concentration
    fig, axes = plt.subplots(2, n_new_concs, sharex=True)
    fig.set_size_inches(9.5, 7.5)
    # Top row: median, bottom row: mean
    for m in models[::-1]:  # Plot IBCM last
        f = h5py.File(model_file_choices[m], "r")
        lambd_axis = f.get("parameters").get("lambd_range")[()]
        all_jacs = concat_jaccards(f)
        median_jacs = np.median(all_jacs, axis=(1, 2, 4))
        mean_jacs = np.mean(all_jacs, axis=(1, 2, 4))
        # Rescale lambda axis by the default value? 1 for IBCM, ~12 for PCA
        lambd_0 = get_lambda0(f)
        f.close()
        for i in range(n_new_concs):
            axes[0, i].plot(lambd_axis / lambd_0, median_jacs[:, i],
                        label=model_nice_names.get(m, m),
                        color=model_colors.get(m), lw=2.0)
            axes[1, i].plot(lambd_axis / lambd_0, mean_jacs[:, i], 
                        label=model_nice_names.get(m, m),
                        color=model_colors.get(m), lw=2.0)
    # Labeling the graphs, etc.
    for i in range(n_new_concs):
        axes[0, i].set_title("Medians, new conc. = {:.1f}".format(new_concs[i]))
        axes[0, i].set_xlabel(r"Scale $\Lambda / \Lambda_0$")
        axes[0, i].set_ylabel("Median Jaccard similarity")
        axes[0, i].set_xscale("log")
        axes[1, i].set_title("Means, new conc. = {:.1f}".format(new_concs[i]))
        axes[1, i].set_xlabel(r"Scale $\Lambda / \Lambda_0$")
        axes[1, i].set_ylabel("Mean Jaccard similarity")
        axes[1, i].set_xscale("log")
    axes[0, 0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig("figures/detection/jaccard_vs_lambda.pdf", transparent=True,
                bbox_inches="tight")
    plt.close()
    return None


def main_export_jaccards(dest_name):
    # Compare all algorithms
    folder = os.path.join("results", "performance")
    models = ["ibcm", "biopca"]
    model_file_choices = {
        a:os.path.join(folder, a+"_performance_results.h5")
        for a in models
    }
    # Get new odor concentrations
    # Assume it's the same for all models: it should!
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
    # Compare all algorithms
    folder = os.path.join("results", "performance")
    models = ["ibcm", "biopca", "avgsub", "ideal", "orthogonal", "none"]
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

if __name__ == "__main__":
    main_plot_performance()
    #main_export_jaccards(
    #    os.path.join("results", "for_plots", "jaccard_similarities")
    #)
    #main_export_new_back_distances(
    #    os.path.join("results", "for_plots", "new_back_distances")
    #)
