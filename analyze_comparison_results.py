import numpy as np
import matplotlib.pyplot as plt
import h5py
from simulfcts.habituation_recognition import id_to_simkey
from simulfcts.plotting import hist_outline
import os


def concat_jaccards(f):
    all_jacs = []
    for i in range(f.get("parameters").get("repeats")[0]):
        all_jacs.append(f.get(id_to_simkey(i)).get("test_results")
                            .get("jaccard_scores")[()])
    all_jacs = np.stack(all_jacs)
    return all_jacs


def main_plot_histograms():
    # Compare all algorithms
    folder = os.path.join("results", "performance")
    models = ["ibcm", "biopca", "avgsub", "ideal", "orthogonal", "none"]
    model_nice_names = {
        "ibcm": "IBCM",
        "biopca": "BioPCA",
        "avgsub": "Average",
        "ideal": "Ideal",
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


def main_export_jaccards(dest_name):
    # Compare all algorithms
    folder = os.path.join("results", "performance")
    models = ["ibcm", "biopca", "avgsub", "ideal", "orthogonal", "none"]
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


if __name__ == "__main__":
    main_plot_histograms()
    main_export_jaccards(
        os.path.join("results", "for_plots", "jaccard_similarities")
    )
