""" Custom classes to profile the duration of steps in iterative functions. 

@author: frbourassa
November 2024
"""

from time import perf_counter

# Small utility class to compute time difference since last call 
# with a single command
class DeltaTimer():
    def __init__(self):
        self.t0 = 0.0  # Start time
        self.t1 = 0.0  # Second to last latest time for delta
        self.t2 = 0.0  # Latest time
        
    def start(self):
        self.t0 = perf_counter()
        self.t1 = self.t0

    def delta(self):
        self.t2 = perf_counter()
        dt = self.t2 - self.t1
        self.t1 = self.t2
        return dt
    
    def total_time(self):
        return self.t2 - self.t0

    def reset(self):
        self.t0 = 0.0
        self.t1 = 0.0
        self.t2 = 0.0


class IterationProfiler():
    """
    Usage:
        - Initialize outside of the loop (optional: give a prof_name)
        - Start at the beginning of the iteration block (self.start)
            (self.reset is executed at the beginning of self.start)
        - After each command (or group of commands) in the iteration, 
            add a time point (self.addpoint)
        - At the end of the iteration block, end the profiling and
            print the results (self.end_iter)

    TODO: include checking a defined list of iterations as part of the class, 
        avoids having to add "if k in ktests" kinds of statements in code. 
    """
    def __init__(self, prof_name=""):
        self.timer = DeltaTimer()
        self.events = []
        self.iter_name = ""
        self.prof_name = prof_name
        self.started = False
    
    def start(self, iter_name=""):
        self.reset()
        self.events = []  # Flush events
        self.iter_name = iter_name
        self.started = True
        self.timer.start()

    def addpoint(self, name):
        if not self.started:
            raise ValueError("Start IterationProfiler before ending it!")
        self.events.append((name, self.timer.delta()))
        return self.events[-1][1]  # Return the time
    
    def end_iter(self):
        # Print outputs
        if not self.started:
            raise ValueError("Start IterationProfiler before ending it!")
        iter_time = self.timer.total_time()
        print("*** Iteration {0.iter_name} profile, sim {0.prof_name} ***".format(self))
        for pt in self.events:
            print("Time to {0[0]}: {0[1]:.3e} s".format(pt))
        print("Total iteration time:", iter_time)
        print("*** End profile iteration {} ***".format(self.iter_name))
        self.started = False
        return iter_time
    
    def reset(self):
        self.timer.reset()
        self.events = []
        self.iter_name = ""
        self.started = False