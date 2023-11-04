"""
Function for analysis of simulation results, comparison of habituation
models, etc.

@author: frbourassa
October 2023
"""
import numpy as np


# Function to check background reduction
def compute_back_reduction_stats(bkser, sser, trans=0):
    back_stats = dict(
        avg_bk = np.mean(bkser[trans:]),
        avg_s = np.mean(sser[trans:]),
        std_bk = np.std(bkser[trans:]),
        std_s = np.std(sser[trans:])
    )
    back_stats["avg_reduction"] = back_stats['avg_s'] / back_stats['avg_bk']
    back_stats["std_reduction"] = back_stats['std_s'] / back_stats['std_bk']
    return back_stats
