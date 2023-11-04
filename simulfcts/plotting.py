""" Various functions to plot the results of simulating
habituation by a
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.metrics import l2_norm

# Improved histogram
def hist_outline(ax, values, bins=None, density=True, **kwargs):
    """ kwargs passed directly to ax.plot """
    hist, bin_edges = np.histogram(values, bins=bins, density=density)
    # Double the y values
    plot_hist = np.stack([hist, hist], axis=1).flatten()
    plot_edges = np.stack([bin_edges[:-1], bin_edges[1:]], axis=1).flatten()
    ax.plot(plot_edges, plot_hist, **kwargs)
    ax.fill_between(plot_edges, 0.0, plot_hist,
                    color=kwargs.get("color"), alpha=0.3)
    return ax, hist, bin_edges


### GENERIC PLOTS OF INHIBITORY NETWORK
def plot_3d_series(vs, dim_idx=[0, 1, 2], skp=1000, transient=None):
    """ # Function to plot three dimensions of vectors of synaptic weights,
    e.g. $\vec{m}^j$ or $\vec{w}^j$. Let the user label the plot.

    Args:
        vs (np.ndarray): time series of vector of each neuron,
            indexed [time, neuron, dimension].
        dims_idx (list of 3 ints): list of the three dimensions to plot.
            Defaults to the first three.
        transient (int): first time step to plot
        skp (int): plot only every skp time points
    """
    # Plot a sample of points for each neuron
    if len(dim_idx) != 3:
        raise ValueError("dim_idx should contain exactly three integers. ")
    if transient is None:
        transient = vs.shape[0] // 2
    tslice = slice(transient, None, skp)
    n_neu = vs.shape[1]
    neurons_palette = sns.color_palette("magma", n_colors=n_neu)

    # 3D plot of the synaptic weight vectors of the different neurons
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(0, 0, 0, color="k", marker="o", ls="none", ms=12)
    for i in range(n_neu):
        ax.scatter(vs[tslice, i, dim_idx[0]], vs[tslice, i, dim_idx[1]], vs[tslice, i, dim_idx[2]],
                   alpha=0.5, color=neurons_palette[i], label="Neuron {}".format(i))
    return fig, ax


def plot_w_matrix(ts, ws, skp=500, **kwargs):
    """ Function to plot the synaptic weight vectors in the W matrix.
    Makes one plot per inhibitory neuron, i.e. per column in W.

    Args:
        ts (np.ndarray): time points
        ws (np.ndarray): W matrix over time, indexed [time, orn, inhib. neuron]
        skp (int): plot only every skp time point

    Extra kwargs are passed to axes.plot

    Returns:
        fig, axes_mat
    """
    n_neu = ws.shape[2]
    n_dim = ws.shape[1]

    n_cols = int(np.ceil(np.sqrt(n_neu)))
    n_rows = n_neu // n_cols + min(1, n_neu % n_cols)
    n_plots = n_rows * n_cols
    assert n_plots >= n_neu, "Need to increase the number of plots!"

    # Number of entries in legend limited to 4
    skp_lbl = max(1, n_dim // 4)
    tfactor = 1000

    # Prepare color palette
    cpal = sns.color_palette(n_colors=n_dim)

    fig, axes_mat = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    fig.set_size_inches(max(n_cols*2.5, 3.), n_rows*1.75)
    if n_rows == 1 and n_cols == 1:
        axes_mat = np.asarray([[axes_mat]])
    if axes_mat.ndim == 1:  # only one row
        axes_mat = axes_mat.reshape(1, -1)
    axes = axes_mat.flatten()
    for i in range(n_dim):
        lbl = "i={}".format(i) if i % skp_lbl == 0 else ""
        for j in range(n_neu):
            axes[j].plot(ts[::skp]/tfactor, ws[::skp, i, j], color=cpal[i], label=lbl, **kwargs)
    for j in range(n_neu):  # Legend and title
        txt = axes[j].annotate("Neuron j={}".format(j), xy=(0.05, 0.95), xycoords="axes fraction",
                        ha="left", va="top")
        txt.set_bbox(dict(facecolor='w', alpha=0.7, edgecolor='w', pad=0.0))
        axes[j].legend(fontsize=8)
    for j in range(n_cols):
        axes_mat[-1, j].set_xlabel("Time (x{})".format(tfactor))
    for i in range(n_rows):
        axes_mat[i, 0].set_ylabel(r"$W^{ij}$")

    # Remove unnecessary plots
    for j in range(n_neu, n_plots):
        axes[j].set_axis_off()
        # Label x axis of axis above
        if j - n_cols >= 0:
            axes[j - n_cols].set_xlabel("Time (x{})".format(tfactor))
            axes[j - n_cols].set_xticklabels(axes[n_plots - n_neu].get_xticklabels())

    fig.tight_layout()

    return fig, axes_mat


def plot_m_matrix(ts, ms, skp=500, **kwargs):
    """ Function to plot the synaptic weight vectors in the M^T matrix.
    Makes one plot per inhibitory neuron, i.e. per row in M^T.

    Args:
        ts (np.ndarray): time points
        ms (np.ndarray): M^T matrix over time,
            indexed [time, inhib. neuron, orn]
        skp (int): plot only every skp time point

    Extra kwargs are passed to axes.plot

    Returns:
        fig, axes_mat
    """
    n_neu = ms.shape[1]
    n_dim = ms.shape[2]

    n_cols = int(np.ceil(np.sqrt(n_neu)))
    n_rows = n_neu // n_cols + min(1, n_neu % n_cols)
    n_plots = n_rows * n_cols
    assert n_plots >= n_neu, "Need to increase the number of plots!"

    # Number of entries in legend limited to 4
    skp_lbl = max(1, n_dim // 4)
    tfactor = 1000

    # Prepare color palette
    cpal = sns.color_palette(n_colors=n_dim)

    fig, axes_mat = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    fig.set_size_inches(max(n_cols*2.5, 3.), n_rows*1.75)
    if axes_mat.ndim == 1:
        axes_mat = axes_mat.reshape(-1, 1)
    axes = axes_mat.flatten()
    for i in range(n_dim):
        lbl = "i={}".format(i) if i % skp_lbl == 0 else ""
        for j in range(n_neu):
            axes[j].plot(ts[::skp]/tfactor, ms[::skp, j, i], color=cpal[i],
                label=lbl, **kwargs)
    for j in range(n_neu):  # Legend and title
        txt = axes[j].annotate("Neuron j={}".format(j), xy=(0.05, 0.95),
            xycoords="axes fraction", ha="left", va="top")
        txt.set_bbox(dict(facecolor='w', alpha=0.7, edgecolor='w', pad=0.0))
        axes[j].legend(fontsize=8)
    for j in range(n_cols):
        axes_mat[-1, j].set_xlabel("Time (x{})".format(tfactor))
    for i in range(n_rows):
        axes_mat[i, 0].set_ylabel(r"$M^{ij}$")

    # Remove unnecessary plots
    for j in range(n_neu, n_plots):
        axes[j].set_axis_off()
        # Label x axis of axis above
        if j - n_cols >= 0:
            axes[j - n_cols].set_xlabel("Time (x{})".format(tfactor))
            axes[j - n_cols].set_xticklabels(axes[n_plots - n_neu].get_xticklabels())

    fig.tight_layout()

    return fig, axes_mat



def plot_background_norm_inhibition(ts, bks, ss, norm_fct=l2_norm, skp=100, colors=[]):
    """ Plot the norm of the ORN activity vector
    (background before inhibition) and of the PN activity vector
    (background after inhibition) over time to illustrate
    noise reduction by the inhibitory network.

    Let the user add analytical values and a legend.

    Args:
        ts (np.ndarray): time points
        bks (np.ndarray): time series of background input vectors,
            indexed [time, orn].
        ss (np.ndarray): time series of background post-inhibition
            at the projection neuron layer, indexed [time, pn].
        norm_fct (callable): function computing the norm of vectors,
            call signature (vectors, axis=-1)
        skp (int): plot only every skp time point
        colors (list of 2 colors): color of un-inhibited and inhibited
            backgrounds, respectively.

    Returns:
        fig, ax
        bks_norm (np.ndarray): time series of activity
            vector norm before inhibition
        ss_norm (np.ndarray): time series of activity
            vector norm after inhibition
    """
    if colors == []:
        colors = ["grey", "k"]
    time_factor = 1000

    # Compute norms
    bks_norm = norm_fct(bks, axis=1)
    ss_norm = norm_fct(ss, axis=1)

    # Plot both norms on the same graph
    fig, ax = plt.subplots()
    ax.plot(ts[::skp]/time_factor, bks_norm[::skp], lw=1.0, alpha=0.8,
            color=colors[0], label="Before inhibition (ORNs)")
    ax.plot(ts[::skp]/time_factor, ss_norm[::skp], lw=1.0, alpha=0.8,
            color=colors[1], label="After inhibition (PNs)")
    ax.set(xlabel="Time (x{})".format(time_factor), ylabel="Activity vector norm")

    fig.set_size_inches(4.5, 3)

    return fig, ax, bks_norm, ss_norm

def plot_background_neurons_inhibition(ts, bks, ss, skp=100, colors=[]):
    """ Plot the activity of each ORN (background before inhibition)
    and of the PN activity vector (background after inhibition)
    over time to illustrate noise reduction by the inhibitory network.

    Makes one small plot per element of the ORN/PN vector.

    Let the user add a legend to the desired plot.

    Args:
        bks (np.ndarray): time series of background input vectors,
            indexed [time, orn].
        ss (np.ndarray): time series of background post-inhibition
            at the projection neuron layer, indexed [time, pn].

    Returns:
        fig, ax
    """
    if colors == []:
        colors = ["grey", "k"]
    time_factor = 1000

    # Prepare the right number of plots, one per ORN or PN
    n_neu = bks.shape[1]

    n_cols = int(np.ceil(np.sqrt(n_neu)))
    n_rows = n_neu // n_cols + min(1, n_neu % n_cols)
    n_plots = n_rows * n_cols
    assert n_plots >= n_neu, "Need to increase the number of plots!"

    tfactor = 1000

    # Plot both norms on the same graph
    fig, axes_mat = plt.subplots(n_rows, n_cols, sharex=True, sharey=False)
    if n_rows == 1:
        axes_mat = axes_mat.reshape(1, n_cols)
    elif n_cols == 1:
        axes_mat = axes_mat.reshape(n_rows, 1)
    fig.set_size_inches(max(n_cols*2.5, 3.), n_rows*1.75)
    axes = axes_mat.flatten()
    for i in range(n_neu):
        axes[i].plot(ts[::skp]/time_factor, bks[::skp, i], lw=0.7, alpha=0.8,
            color=colors[0], label="Before inhibition (ORNs)")
        axes[i].plot(ts[::skp]/time_factor, ss[::skp, i], lw=0.7, alpha=0.8,
            color=colors[1], label="After inhibition (PNs)")

    # Label relevant axes
    for j in range(n_neu):  # Legend and title
        txt = axes[j].annotate("Neuron {}".format(j), xy=(0.05, 0.95), xycoords="axes fraction",
                        ha="left", va="top")
        txt.set_bbox(dict(facecolor='w', alpha=0.7, edgecolor='w', pad=0.0))
    for j in range(n_cols):
        axes_mat[-1, j].set_xlabel("Time (x{})".format(tfactor))
    for i in range(n_rows):
        axes_mat[i, 0].set_ylabel("Neuron activity")

    # Remove unnecessary plots
    for j in range(n_neu, n_plots):
        axes[j].set_axis_off()
        # Label x axis of axis above
        if j - n_cols >= 0:
            axes[j - n_cols].set_xlabel("Time (x{})".format(tfactor))
            axes[j - n_cols].set_xticklabels(axes[n_plots - n_neu].get_xticklabels())

    for j in range(n_neu, n_plots):  # Remove from the axes list.
        axes[j].pop(-1)
    fig.tight_layout()

    return fig, axes_mat, axes


### PLOTS SPECIFIC TO IBCM MODEL
def plot_cbars_gammas_sums(ts, ser_sums_cbg, ser_sums_cbg2, skp=200, skp_lbl=1):
    """ Plot the time course of sums of dot products
        $$ \sum_{\gamma} \bar{c}_{\gamma} $$
    and
        $$ \sum_{\gamma} \bar{c}_{\gamma}^2 $$
    where
        $$ \bar{c}_{|gamma} = \vec{\bar{m}} \cdot \vec{x}_{\gamma} $$

    For a gaussian process, theoretical values are 1/nu, 1/sigma^2, resp.
    Let users plot those analytical values and add the legend.

    Args:
        ts (np.ndarray): time points
        ser_sums_cbg (np.ndarray): time series of the sum of cbar_gamma
            of each neuron, indexed [time, neuron]
        ser_sums_cbg2 (np.ndarray): time series of the sum of cbar_gamma^2
            of each neuron, indexed [time, neuron]
        skp (int): plot only every skp time step
        skp_lbl (int): put only every skp_lbl line in the legend

    Returns:
        fig, axes
    """
    # Plotting the time course of the dot products
    fig, axes = plt.subplots(2)
    n_neu = ser_sums_cbg.shape[1]
    tfactor = 1000
    neurons_palette = sns.color_palette("gray", n_colors=n_neu)
    for i in range(n_neu):
        lbl = "Neuron {}".format(i) if i % skp_lbl == 0 else ""
        axes[0].plot(ts[::skp]/tfactor, ser_sums_cbg[::skp, i], color=neurons_palette[i], lw=1.0,
                     alpha=1.0 - 0.4*i/n_neu, label=lbl)
        axes[1].plot(ts[::skp]/tfactor, ser_sums_cbg2[::skp, i], color=neurons_palette[i], lw=1.0,
                     alpha=1.0 - 0.4*i/n_neu, label=lbl)

    for i in range(2):
        ylim = axes[i].get_ylim()
        axes[i].set_ylim(min(ylim[0], 0.0 - 0.05*(ylim[1]-ylim[0])), ylim[1])
        axes[i].legend(loc="lower right")

    axes[0].set(ylabel=r"$\sum_{\gamma} \bar{c}_{\gamma}$")
    axes[1].set(xlabel="Time (x{})".format(tfactor),
                ylabel=r"$\sum_{\gamma} \bar{c}_{\gamma}^2$")

    return fig, axes


def plot_cbars_gamma_series(ts, cbg_ser, skp=200, transient=None):
    """ Plotting the time course of dot products of mbar with odor components.

    Args:
        ts (np.ndarray): time points
        cbg_ser (np.ndarray): cbar_gamma time series,
            indexed [time, neuron, gamma]
        skp (int): plot only every skp time step
        transient (int): time step where steady-state is reached

    Returns: fig, ax
    """
    n_comp = cbg_ser.shape[2]
    n_neu = cbg_ser.shape[1]
    if transient is None:
        transient = tser.size // 2
    tfactor = 1000

    # Count the number of cgammas above and below the average.
    # In non-degenerate cases, should have, for each neuron,
    # one c_gamma above, others below.
    cbg_means = np.mean(cbg_ser[transient:], axis=0)
    cbg_thresh = np.mean(cbg_means)
    cbg_counts = {
        "above": np.count_nonzero(cbg_means >= cbg_thresh),
        "below": np.count_nonzero(cbg_means < cbg_thresh)
    }

    # Aesthetical parameters
    components_palettes = [sns.color_palette(c, n_colors=n_neu) for c in
                ["Blues", "Oranges", "Greens", "Reds", "Purples", "Greys"]]
    while len(components_palettes) < n_comp:
        components_palettes += components_palettes

    # Plot the time series
    fig, ax = plt.subplots()
    for j in range(n_comp):
        palette = components_palettes[j]
        for i in range(n_neu - 1):
            ax.plot(ts[::skp]/tfactor, cbg_ser[::skp, i, j], color=palette[i],
                    alpha=0.8, lw=1.0)
        ax.plot(ts[::skp]/tfactor, cbg_ser[::skp, -1, j], color=palette[-1],
                label=r"$\bar{c}_{(\gamma="+str(j)+ r")}$", alpha=0.8, lw=1.0)
    # Count
    ax.annotate(cbg_counts["above"], ha="left",
        xy=(ts[-1]*1.005/tfactor, cbg_means.max()*1.1))
    ax.annotate(cbg_counts["below"], ha="left",
        xy=(ts[-1]*1.005/tfactor, cbg_means.min()*1.2),)
    ax.set(xlabel="Time (x{} steps)".format(tfactor),
           ylabel=r"Reduced dot product $\bar{c}_{\gamma} ="
                    + r" \vec{\bar{m}} \cdot \vec{x}_{\gamma}$")
    ax.legend()

    return fig, ax, cbg_counts

### PLOTS SPECIFIC TO ONLINE PCA ###
def plot_pca_results(tser, truepca, learntpca, alignerrser, off_diag_l_ser,
                     palette1=None, palette2=None):
    """
    Args:
        tser (np.ndarray): time steps
        truepca (list of 2 arrays): principal values, principal components [dim, component]
        learntpca (list of 2 arrays): principal values, indexed [time, value],
            and principal components, indexed [time, dim, component]
        alignerrser (np.ndarray): subspace alignment error over time
        off_diag_l_ser (np.ndarray): average absolute value of off-diagonal L matrix
            elements over time
        palette1 (list of colors): colors for the various principal values
        palette2 (list of 2 colors): palette for the plots of error vs time
            and L elements vs time.

    Returns:
        fig, axes (list of fig, ax): fig and axes of principal values vs time,
            align error vs time, and L elements magnitude over time.
    """
    if palette1 is None:
        palette1 = sns.color_palette(n_colors=learntpca[0].shape[1])
    if palette2 is None:
        palette2 = ["k", "grey"]
    fig, axes = plt.subplots(3)
    ax = axes[0]
    default_size = fig.get_size_inches()
    fig.set_size_inches(default_size[0], default_size[1]*1.33)
    n_comp = learntpca[0].shape[1]
    for i in range(n_comp):
        li, = ax.plot(tser, learntpca[0][:, i], label="Learnt {}".format(i),
                      lw=1.0, zorder=10-i, color=palette1[i])
        if truepca[0][i] / truepca[0].max() > 1e-12:
            ax.axhline(truepca[0][i], ls="--", color=palette1[i], lw=1.0 - i/n_comp,
                label="True {}".format(i), zorder=n_comp-i)
    ax.set(ylabel="Principal values", yscale="log")
    leg = ax.legend(ncol=2)
    leg.set_zorder(30)

    ax = axes[1]
    ax.plot(tser, alignerrser, color=palette2[0])
    ax.set(yscale="log", ylabel="Subspace alignment error")

    ax = axes[2]
    ax.plot(tser, np.mean(learntpca[0], axis=1), label="Diagonal", color=palette2[0])
    ax.plot(tser, off_diag_l_ser, label="Off-diagonal", color=palette2[1])
    ax.set(yscale="log", ylabel=r"Average $L_{ij}$ magnitude", xlabel="Time (steps)")
    ax.legend()
    fig.tight_layout()

    return fig, axes
