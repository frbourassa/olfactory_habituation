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