"""
Test sampling from a power law.
"""
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

cmyk_blue = "#3E529F"
cmyk_red = "#DA3833"
cmyk_green = "#307F54"

# Transform U(0, 1) samples into power law samples
def powerlaw_inverse_transform(unif, tmin):
    return tmin / ((1.0 - unif)**2)

def powerlaw_cutoff_inverse_transform(unif, tmin, tmax):
    factor = (1.0 - np.sqrt(tmin/tmax))
    return tmin / ((1.0 - unif * factor)**2)


if __name__ == "__main__":
    rgen = np.random.default_rng(seed=0x850aaa01e9f9fd65ebbcd20537551e65)
    unif_samples = rgen.random(size=int(1e6))

    # 1. Test the power law with unlimited upper value
    thresh_min = 1.0

    start_time = perf_counter()
    print("Starting to convert {} samples".format(unif_samples.size))
    times_samples = powerlaw_inverse_transform(unif_samples, thresh_min)

    end_time = perf_counter()
    print("Finished converting {} samples".format(unif_samples.size))
    time_per_sample = (end_time - start_time) / unif_samples.size
    print("This took on average {} s per sample".format(time_per_sample))

    fig, ax = plt.subplots()
    # Note: To get a -3/2 power law pdf in a log-log histogram, we need to
    # use logarithmic bins, i.e. bin the log of t_d, but then, to get
    # the density of t_d plotted in log, rather than the density ot log(t_d),
    # we need to normalize each bin by its linear width, not its log width.
    # Indeed, because of the jacobian, p_l(log(t_d)) as a slope of -1/2 while
    # p_t(t_d) has a slope of -3/2. So either make a histogram of log(t_d)
    # and normalize density with log width of bins, and then expect slope -1/2
    # or normalize bin counts with the linear bin width (10^(log bin width))
    # and then expect a -3/2 slope.
    # I do the latter here because intuitively we want
    # to see the -3/2 and hide the normalization details in the background.
    counts, binseps = np.histogram(np.log10(times_samples), bins="doane")
    binwidths = np.diff(10**binseps)
    # Center of bins on a log scale, but given in linear coordinates
    bin_centers_forlog = 10**((binseps[1:] + binseps[:-1])/2)
    pdf = counts / binwidths / times_samples.size
    cdf = np.cumsum(pdf)
    ax.bar(x=10**binseps[:-1], align="edge", height=pdf, width=binwidths,
            color=cmyk_blue, label="Samples")
    ax.set(xlabel=r"$t_b$", ylabel=r"$p_t(t_d)$", yscale="log", xscale="log")

    dens = (bin_centers_forlog/thresh_min)**(-3/2) / (2.0 * thresh_min)
    ax.plot(bin_centers_forlog, dens, color=cmyk_red, lw=3.,
        label=r"$p(t_d) \sim t_d^{-3/2}$")
    ax.legend(ncol=2)
    plt.show()
    plt.close()

    # 2. Test the power law with an upper cutoff
    thresh_max = 1e3 * thresh_min
    times_samples = powerlaw_cutoff_inverse_transform(unif_samples, thresh_min, thresh_max)

    fig, ax = plt.subplots()
    counts, binseps = np.histogram(np.log10(times_samples), bins="doane")
    binwidths = np.diff(10**binseps)
    # Center of bins on a log scale, but given in linear coordinates
    bin_centers_forlog = 10**((binseps[1:] + binseps[:-1])/2)
    pdf = counts / binwidths / times_samples.size
    cdf = np.cumsum(pdf)
    ax.bar(x=10**binseps[:-1], align="edge", height=pdf, width=binwidths,
            color=cmyk_blue, label="Samples")
    ax.set(xlabel=r"$t_b$", ylabel=r"$p_t(t_d)$", yscale="log", xscale="log")

    # Analytical density with an upper cutoff
    norm_factor = 2 * thresh_min * (1.0 - np.sqrt(thresh_min / thresh_max))
    dens = (bin_centers_forlog/thresh_min)**(-3/2) / norm_factor
    ax.plot(bin_centers_forlog, dens, color=cmyk_red, lw=3.,
        label=r"$p(t_d) \sim t_d^{-3/2}$")
    ax.legend(ncol=2)
    fig.tight_layout()
    #fig.savefig("../figures/tests/powerlaw_waiting_time_test.pdf",
    #    transparent=True, bbox_inches="tight")
    plt.show()
    plt.close()
