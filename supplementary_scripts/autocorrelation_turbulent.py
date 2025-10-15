import numpy as np
from os.path import join as pj
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Estimator of autocorrelation based on Sokal 1995 and emcee. 
# Autocorrelation analysis. Code from emcee's documentation:
# https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))
    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm and acf[0] != 0.0:
        acf /= acf[0]

    return acf


# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def autocorr_avg(y, c=5.0):
    """ y is shaped [n_walkers, n_samples] """
    # First compute the integrated autocorrelation time
    f = np.zeros(y.shape[1])
    for yy in y:  # loop over walkers
        f += autocorr_func_1d(yy)
    f /= len(y)
    # Use the automatic windowing described by Sokal, stop the
    # sum to compute tau_int at the auto_window position
    # The sume extends from -time to +time, here use symmetry
    # to rewrite t_int = 2*sum_1^{time} + 1
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    # This returns the autocorrelation time and the
    # integrated autocorrelation time,
    # equal to 1/2 + correlation time if the decay is exponential
    return f, taus[window]

def scipy_avg_correlate(sers):
    # sers is shaped [odor, time]
    s0 = sers - np.mean(sers, axis=1).reshape(-1, 1)
    corr_full = np.zeros(2*s0.shape[1]-1)
    nser = sers.shape[0]
    for i in range(nser):
        corr_full += correlate(s0[i], s0[i])
    # normalize
    corr_full /= (nser * sers.shape[1])
    corr_full /= np.amax(corr_full)
    corr = corr_full[corr_full.shape[0]//2:]
    return corr
    


if __name__ == "__main__":
    # Import pre-saved turbulent time series of odor concentration
    conc_ser = np.load(pj("..", "results", "for_plots", 
        "sample_turbulent_background.npz"))["nuser"]
    # Extract concentration series and reshape as [odor, time]
    conc_ser = conc_ser[:, :, 1].T
    
    dt = 1.0  #steps  #10.0 / 1000.0 # s
    dt_units = "steps"

    # Compute autocorrelation function, average over odors since they are iid
    autocorr_fct, autocorr_tau = autocorr_avg(conc_ser)
    trange_show = min(1000, autocorr_fct.shape[0])
    print("autocorrelation time:", autocorr_tau*dt, dt_units)

    # Compare to a scipy autocorrelation function
    scipy_fct = scipy_avg_correlate(conc_ser)

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(np.arange(trange_show)*dt, autocorr_fct[:trange_show], label="Own implem.")
    ax.plot(np.arange(trange_show)*dt, scipy_fct[:trange_show], label="Scipy", ls="--")
    ax.axvline(autocorr_tau*dt, ls="--", color="k", label=r"$\tau_{\mathrm{int}}$ Sokal")
    ax.set(xlabel=f"Time difference ({dt_units})", ylabel="Turbulent odor conc. autocorrelation")
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()

    
