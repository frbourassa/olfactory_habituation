""" To add concentration moments to results files with having to re-run all
"""
import numpy as np
from modelfcts.backgrounds import sample_ss_conc_powerlaw
import h5py
import os

if __name__ == "__main__":
    # Compute moments of the background concentration process
    dummy_rgen = np.random.default_rng(0x51bf7feb1fd2a3f61e1b1b59679f62c6)
    conc_samples = sample_ss_conc_powerlaw(
                        *turbulent_back_params, size=int(1e5), rgen=dummy_rgen
                    )
    mean_conc = np.mean(conc_samples)
    moments_conc = np.asarray([
        mean_conc,
        np.var(conc_samples),
        np.mean((conc_samples - mean_conc)**3)
    ])
    print("Computed numerically the concentration moments:", moments_conc)

    # Add to every results file
    folder = os.path.join("results", "performance")
    for model in ["none", "ideal", "orthogonal", "ibcm", "biopca", "avgsub"]:
        file_name = os.path.join(folder, kind+"_performance_results.h5")
        with h5py.File(file_name, "a") as f:
            f.get("parameters").create_dataset("moments_conc", data=moments_conc)
        print("Treated {} successfully".format(model))
