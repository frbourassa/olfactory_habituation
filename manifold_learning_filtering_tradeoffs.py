"""
Plots of the loss function for optimal new odor recognition in fluctuating
olfactory backgrounds combining predictive filtering and manifold learning.

@author: frbourassa
September 2024
"""
import numpy as np
import matplotlib.pyplot as plt


def loss_vw_ou(tau, n_r, p):
    """
    Loss function with both W and v strategies, for a special case
    where background odors are orthogonal and iid, new odors are a
    concentration iid to the background times a vector uniformly
    sampled on the unit hypersphere.

    tau (float, np.ndarray): time constant of the exponentially decaying
        autocorrelation function (O-U process' autocorrelation) of each odor.
    n_r (float, np.ndarray): N_R, number of olfactory receptor dimensions
    p (dict): parameters,
        sigma^2: variance of odor concentrations
        K: number of background odors
    """
    expo = 1.0 - np.exp(-2.0/tau)
    loss = p["sigma^2"] * p["K"] * expo / (1.0 + n_r*expo)
    return loss


def loss_v_ou(tau, n_r, p):
    """ Loss with only predictive filtering v.
    """
    loss = p["K"] * p["sigma^2"] * (1.0 - np.exp(-2.0/tau))
    if hasattr(n_r, "shape"):
        loss = np.ones(n_r.shape) * loss  # Ensure full array shape
    return loss   


def loss_w_ou(tau, n_r, p):
    """ Loss with only manifold learning W.
    """
    loss = p["K"] * p["sigma^2"] / (n_r+ 1.0)
    if hasattr(tau, "shape"):
        loss = np.ones(tau.shape) * loss  # Ensure full array shape
    return loss 

def phase_boundary_v_w(tau_range):
    n_r = 1.0 / (np.exp(2.0/tau_range) - 1.0)
    return n_r

def plot_loss_vs_1_param(tau, n_r, p, figax=None):
    # Compute losses
    loss_lines = {}
    loss_fcts = {"vw":loss_vw_ou, "w":loss_w_ou, "v":loss_v_ou}
    norm_fact = p["K"] * p["sigma^2"]
    for strat in loss_fcts.keys():
        loss_lines[strat] = loss_fcts[strat](tau, n_r, p)

    # Plot
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
    strat_names = {"vw":r"Combined, $\mathcal{L}_{v, W}$", 
        "w":r"$W$ only, $\mathcal{L}_W$", "v":r"$v$ only, $\mathcal{L}_v$"}
    strat_styles = {"vw":"-", "w":"--", "v":":"}
    strat_clrs = {"vw":"tab:purple", "w":"tab:red", "v":"tab:blue"}
    if isinstance(tau, np.ndarray):
        xrange = tau
        xlbl = r"Autocorrelation time, $\tau$"
        annot_lbl = r"Fixed $\frac{\sigma^2}{\sigma_x^2} N_R = " 
        annot_lbl += "{:d}$".format(n_r) 
    else:
        xrange = n_r
        xlbl = r"Scaled olfactory dimension, $\frac{\sigma^2}{\sigma_x^2} N_R$"
        annot_lbl = r"Fixed $\tau = {:d}$".format(tau) 

    for strat in ["v", "w", "vw"]:
        ax.plot(
            xrange, loss_lines[strat] / norm_fact, 
            label=strat_names[strat], ls=strat_styles[strat],
            lw=2.5, color=strat_clrs[strat]
        )
    ax.set_title(annot_lbl, fontsize=ax.xaxis.label.get_size())
    ax.set(
        xlabel=xlbl,
        ylabel=r"Normalized minimized loss, $\mathcal{L} \, / K \sigma^2$", 
        yscale="log", xscale="log"
    )
    ax.legend()
    return [fig, ax]

def phase_w_v_heatmap(tau_range, n_r_range, p, figax=None):
    # First, compute the costs at each grid point. 
    loss_maps = {}
    loss_fcts = {"vw":loss_vw_ou, "w":loss_w_ou, "v":loss_v_ou}
    norm_fact = p["K"] * p["sigma^2"]
    tau_grid, n_r_grid = np.meshgrid(tau_range, n_r_range, indexing="xy")
    for strat in loss_fcts.keys():
        loss_maps[strat] = loss_fcts[strat](tau_grid, n_r_grid, p) / norm_fact
    loss_ratio_v = np.log(loss_maps["v"] - loss_maps["vw"])
    loss_ratio_w = np.log(loss_maps["w"] - loss_maps["vw"])
    phase = loss_ratio_v - loss_ratio_w
    max_ampli = np.amax(np.abs(phase))
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
    xtent = (tau_grid.min(), tau_grid.max(), n_r_grid.min(), n_r_grid.max())
    im = ax.imshow(phase, cmap="RdBu_r", vmin=-max_ampli, vmax=max_ampli,
        extent=xtent, origin="lower", 
    )
    # Analytical boundary
    nr_boundary = phase_boundary_v_w(tau_range)
    ax.plot(tau_range, nr_boundary, ls="--", color="k")
    ax.set_xlim(xtent[0], xtent[1])
    ax.set_ylim(xtent[2], xtent[3])
    ax.set_xlabel(r"Autocorrelation time, $\tau$")
    ax.set_ylabel(r"Scaled olfactory dimension, $\frac{\sigma^2}{\sigma_x^2} N_R$")

    cbar = fig.colorbar(im, ax=ax, 
        label=(r"Regime: $\Phi = \log(\mathcal{L}_v - \mathcal{L}_{v,W})"
            + r"- \log(\mathcal{L}_W - \mathcal{L}_{v,W})$"), 
        shrink=0.7
    )
    return [fig, ax], cbar, loss_maps
    

if __name__ == "__main__":
    loss_params = {
        "sigma^2": 0.16,  # Useless, just an overall scale in the end
        "K": 5  # number of odors, useless in the end, 
                # since general loss = k * 1-d back. loss
    }
    do_save = False

    fig, axes = plt.subplots(1, 2, sharey="row")
    fig.set_size_inches(6, 3.25)
    # Plot as a function of tau, fixed N_R
    tau_range = np.geomspace(1.0, 200.0, 400)
    n_r_fix = 50
    figax = (fig, axes.flat[0])
    plot_loss_vs_1_param(tau_range, n_r_fix, loss_params, figax=figax)

    # Plot as a function of N_R, tau fixed. 
    n_r_range = np.geomspace(1, 200, 400)
    tau_fix = 100
    figax = (fig, axes.flat[1])
    plot_loss_vs_1_param(tau_fix, n_r_range, loss_params, figax=figax)

    fig.tight_layout()
    if do_save:
        fig.savefig(
            "figures/noise_struct/loss_lineplots_manifold_vs_predictive.pdf",
            transparent=True, bbox_inches="tight"
        )
    plt.show()
    plt.close()

    # Now, phase diagram. Plot min(log(L_v/L_{v,W}), log(L_W/L_{v,W}))
    # with a different color depending on which term is smallest. 
    tau_range = np.linspace(4.0, 200.0, 400)
    n_r_range = np.linspace(4, 200, 400)
    figax = plt.subplots()
    figax[0].set_size_inches(4.25, 4.25)
    [fig, ax], cbar, lossgrids = phase_w_v_heatmap(tau_range, n_r_range, loss_params, figax)
    tau_mid = tau_range[tau_range.shape[0] // 2]
    ax.annotate(r"$\tau \sim 2\left(\frac{\sigma^2}{\sigma_x^2} N_R + 1\right)$", 
                (tau_mid, tau_mid/2.0 - 15.0), rotation=25)
    ax.annotate("Manifold learning", #r"$W$ dominates", 
        (tau_range[tau_range.shape[0] // 4],
          n_r_range[n_r_range.shape[0]//4 * 3]))
    ax.annotate("Predictive filtering", #r"$v$ dominates", 
        (tau_range[-10], n_r_range[n_r_range.shape[0]//20]), 
        ha="right", va="bottom")
    fig.tight_layout()
    if do_save:
        fig.savefig(
            "figures/noise_struct/loss_strategy_phase_diagram_heatmap.pdf",
            transparent=True, bbox_inches="tight"
        )
    plt.show()
    plt.close()

    # Also save the heatmap data for final figure plotting
    np.savez(
        "results/for_plots/manifold_learning_heatmap_data.npz",
        **lossgrids, 
        tau_range=tau_range,
        n_r_range=n_r_range
    )