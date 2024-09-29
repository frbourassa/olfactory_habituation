"""
Plots of the cost function for optimal new odor recognition in fluctuating
olfactory backgrounds combining predictive filtering and manifold learning.

@author: frbourassa
September 2024
"""
import numpy as np
import matplotlib.pyplot as plt


def cost_vw_ou(tau, p):
    """
    Cost function with both W and v strategies, for a special case
    where background odors are orthogonal and iid, new odors are a
    concentration iid to the background times a vector uniformly
    sampled on the unit hypersphere.

    tau (float, np.ndarray): time constant of the exponentially decaying
        autocorrelation function (O-U process' autocorrelation) of each odor.
    p (dict): parameters,
        N_R: number of olfactory receptor dimensions
        sigma^2: variance of odor concentrations
        K: number of background odors
    """
    expo = 1.0 - np.exp(-2.0/tau)
    cost = p["sigma^2"] * p["K"] * expo / (1.0 + p["N_R"]*expo)
    return cost


def cost_v_ou(tau, p):
    """ Cost with only predictive filtering v.
    """
    cost = p["K"] * p["sigma^2"] * (1.0 - np.exp(-2.0/tau))
    return cost


def cost_w_ou(tau, p):
    """ Cost with only manifold learning W.
    """
    cost = p["K"] * p["sigma^2"] / (p["N_R"] + 1.0)
    return cost * np.ones(tau.shape)


if __name__ == "__main__":
    cost_params = {
        "N_R": 50,
        "sigma^2": 0.16,  # Useless, just an overall scale in the end
        "K": 5  # number of odors, useless in the end, cost = k * 1-d back. cost
    }

    tau_range = np.linspace(0.01, 100.0, 100)
    cost_lines = {}
    cost_fcts = {"vw":cost_vw_ou, "w":cost_w_ou, "v":cost_v_ou}
    for strat in cost_fcts.keys():
        cost_lines[strat] = cost_fcts[strat](tau_range, cost_params)

    fig, ax = plt.subplots()
    strat_names = {"vw":r"Combined $v, W$", "w":r"$W$ only", "v":r"$v$ only"}
    for strat in cost_fcts.keys():
        ax.plot(tau_range, cost_lines[strat], label=strat_names[strat])
    ax.set(xlabel=r"Autocorrelation time, $\tau$",
        ylabel=r"Minimized cost, $\mathcal{L}$", yscale="log")
    ax.legend()
    plt.show()
    plt.close()
