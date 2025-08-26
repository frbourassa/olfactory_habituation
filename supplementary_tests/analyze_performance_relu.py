import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import os, sys, json
from os.path import join as pj
if not ".." in sys.path:
    sys.path.insert(1, "..")

from simulfcts.plotting import hist_outline
from modelfcts.ideal import find_projector, find_parallel_component
from utils.metrics import l2_norm
from simulfcts.analysis import (
    concat_jaccards, 
    concat_mixtures, 
)

do_save_plots = True

root_dir = pj("..")
params_folder = pj(root_dir, "results", "common_params")

# rcParams
plt.rcParams["figure.figsize"] = (4.5, 3.0)
with open(pj(params_folder, "olfaction_rcparams.json"), "r") as f:
    new_rcParams = json.load(f)
plt.rcParams.update(new_rcParams)

# color maps
with open(pj(params_folder, "back_colors.json"), "r") as f:
    all_back_colors = json.load(f)
back_color = all_back_colors["back_color"]
back_color_samples = all_back_colors["back_color_samples"]
back_palette = all_back_colors["back_palette"]

with open(pj(params_folder, "orn_colors.json"), "r") as f:
    orn_colors = json.load(f)
    
with open(pj(params_folder, "inhibitory_neuron_two_colors.json"), "r") as f:
    neuron_colors = np.asarray(json.load(f))
with open(pj(params_folder, "inhibitory_neuron_full_colors.json"), "r") as f:
    neuron_colors_full24 = np.asarray(json.load(f))
# Here, 32 neurons, need to make a new palette with same parameters
neuron_colors_full = np.asarray(sns.husl_palette(n_colors=32, h=0.01, s=0.9, l=0.4, as_cmap=False))

with open(pj(params_folder, "model_colors.json"), "r") as f:
    model_colors = json.load(f)
with open(pj(params_folder, "model_nice_names.json"), "r") as f:
    model_nice_names = json.load(f)

models = list(model_colors.keys())


def main_plot_histograms():
    # Compare all algorithms
    folder = os.path.join("..", "results", "performance_ReLU")
    models = ["ibcm", "biopca", "avgsub", "optimal", "orthogonal", "none"]

    model_file_choices = {
        a:os.path.join(folder, a+"_performance_results_relu.h5")
        for a in models
    }

    # Get new odor concentrations
    # Assume it's the same for all models: it should!
    with h5py.File(model_file_choices["ibcm"], "r") as f:
        n_new_concs = f.get("parameters").get("repeats")[4]
        new_concs = f.get("parameters").get("new_concs")[()]
        activ_fct = f.get("parameters").attrs.get("activ_fct")
    # One plot per new odor concentration
    fig, axes = plt.subplots(1, n_new_concs, sharex=True)
    fig.set_size_inches(plt.rcParams["figure.figsize"][0]*n_new_concs*0.9, 
                        plt.rcParams["figure.figsize"][1])
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
            axes[i].axvline(
                np.median(all_jacs[:, :, :, i, :]), ls="--",
                color=model_colors.get(m)
            )
    # Labeling the graphs, etc.
    for i in range(n_new_concs):
        ax = axes[i]
        axes[i].set_title("New conc. = {:.1f}".format(new_concs[i]))
        axes[i].set_xlabel("Jaccard similarity (higher is better)")
        axes[i].set_ylabel("Probability density")
    
    fig.tight_layout()    
    leg = axes[-1].legend(loc="upper left", bbox_to_anchor=(0.75, 1.0), 
                   frameon=False)
    fig_name = "compare_models_{}.pdf".format(activ_fct)
    fig.savefig(pj(root_dir, "figures", "detection", fig_name),
                transparent=True, bbox_inches="tight", 
                bbox_extra_artists=(leg,))

    plt.show()
    plt.close()
    return None


def main_export_jaccards(dest_name, k='jaccard_scores'):
    # Compare all algorithms
    folder = os.path.join("..", "results", "performance_ReLU")
    models = ["ibcm", "biopca", "avgsub", "ideal", "optimal", "orthogonal", "none"]
    model_file_choices = {
        a:os.path.join(folder, a+"_performance_results_relu.h5")
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
        print("Exporting {} of file {}".format(k, model_file_choices[m]))
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
    folder = os.path.join("..", "results", "performance_ReLU")
    models = ["ibcm", "biopca", "avgsub", "ideal", "optimal", "orthogonal", "none"]
    model_file_choices = {
        a:os.path.join(folder, a+"_performance_results_relu.h5")
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
    folder = os.path.join("..", "results", "performance_ReLU")
    models = ["ibcm", "biopca", "avgsub", "ideal", "optimal", "orthogonal", "none"]
    model_file_choices = {
        a:os.path.join(folder, a+"_performance_results_relu.h5")
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
    main_plot_histograms()
    main_export_jaccards(
        os.path.join("..", "results", "for_plots", "jaccard_similarities")
    )
    # Also export Jaccard similarities to background
    #main_export_jaccards(
    #    os.path.join("..", "results", "for_plots", "jaccard_similarities_back"),
    #    k='jaccard_scores_back'
    #)
    #main_export_new_back_distances(
    #    os.path.join("..", "results", "for_plots", "new_back_distances")
    #)
    #main_export_new_mix_distances(
    #    os.path.join("..", "results", "for_plots", "new_mix_distances")
    #)