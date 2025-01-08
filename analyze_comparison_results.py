import numpy as np
import matplotlib.pyplot as plt
import h5py
from simulfcts.habituation_recognition import id_to_simkey
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


def main_plot_histograms():
    # Compare all algorithms
    folder = os.path.join("results", "performance")
    models = ["ibcm", "biopca", "avgsub", "ideal", "optimal", "orthogonal", "none"]
    model_nice_names = {
        "ibcm": "IBCM",
        "biopca": "BioPCA",
        "avgsub": "Average",
        "ideal": "Ideal",
        "optimal": "Manifold W",
        "orthogonal": "Orthogonal",
        "none": "None"
    }
    model_file_choices = {
        a:os.path.join(folder, a+"_performance_results.h5")
        for a in models
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
    with h5py.File(model_file_choices["ibcm"], "r") as f:
        n_new_concs = f.get("parameters").get("repeats")[4]
        new_concs = f.get("parameters").get("new_concs")[()]
        activ_fct = f.get("parameters").attrs.get("activ_fct")
    # One plot per new odor concentration
    fig, axes = plt.subplots(1, n_new_concs, sharex=True)
    fig.set_size_inches(9.5, 4)
    axes = axes.flatten()
    for m in models[::-1]:  # Plot IBCM last
        f = h5py.File(model_file_choices[m], "r")
        all_jacs = concat_jaccards(f)
        f.close()
        for i in range(n_new_concs):
            hist_outline(
                axes[i], all_jacs[:, :, :, i, :].flatten(),
                bins="doane", density=True, label=model_nice_names.get(m, m),
                color=model_colors.get(m), alpha=1.0
            )
            #axes[i].axvline(
            #    np.median(all_jacs[:, :, :, i, :]), ls="--",
            #    color=model_colors.get(m)
            #)
    # Labeling the graphs, etc.
    for i in range(n_new_concs):
        ax = axes[i]
        axes[i].set_title("New conc. = {:.1f}".format(new_concs[i]))
        axes[i].set_xlabel("Jaccard similarity (higher is better)")
        axes[i].set_ylabel("Probability density")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig("figures/detection/compare_models_{}.pdf".format(activ_fct),
                transparent=True, bbox_inches="tight")

    plt.show()
    plt.close()
    return None


def main_export_jaccards(dest_name, k='jaccard_scores'):
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
        all_jacs = concat_jaccards(f, k=k)
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
    #main_plot_histograms()
    main_export_jaccards(
        os.path.join("results", "for_plots", "jaccard_similarities")
    )
    # Also export Jaccard similarities to background
    main_export_jaccards(
        os.path.join("results", "for_plots", "jaccard_similarities_back"),
        k='jaccard_scores_back'
    )
    #main_export_new_back_distances(
    #    os.path.join("results", "for_plots", "new_back_distances")
    #)
    main_export_new_mix_distances(
        os.path.join("results", "for_plots", "new_mix_distances")
    )