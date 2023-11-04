""" Finding the maximum number of multiprocessing.Process that can
run in parallel on a computer.

On Mac OS or Windows, exclude virtual CPUs, use the number of physical CPUs.

On Linux, use len(psutil.Process().cpu_affinity()) which takes into
account multithreading.

@author: frbourassa
May 19, 2023
"""
from sys import platform as sys_pf
import psutil

def count_parallel_cpu():
    """
    Find the maximal number of processes we can actually run
    in parallel on separate CPUs in a multiprocessing task.
    """
    # Number of CPUs we can actually use in multiprocessing
    if sys_pf == "linux":
        n_cpu = len(psutil.Process().cpu_affinity())
    # On Mac OS, cpu_affinity isn't available
    # I don't know on Windows so I include that case here
    else:
        n_cpu = psutil.cpu_count(logical=False)
    return n_cpu
