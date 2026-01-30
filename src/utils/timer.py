import time

class Timer:
    def __init__(self):
        self.times = {}

    def tic(self, name):
        self._t0 = time.perf_counter_ns()
        self._name = name

    def toc(self):
        dt = time.perf_counter_ns() - self._t0
        self.times[self._name] = self.times.get(self._name, 0.0) + dt

    def report(self):
        total = sum(self.times.values())
        for k, v in sorted(self.times.items(), key=lambda x: -x[1]):
            print(f"{k:30s}: {v*1e-6:7.2f} ms  ({100*v/total:5.1f}%)")
        print(f"{'TOTAL':30s}: {total*1e-6:7.2f} ms")
    
    def get_times(self):
        return self.times

def print_timing_report(timings):
    total = sum(timings.values())
    for k, v in sorted(timings.items(), key=lambda x: -x[1]):
        print(f"{k:30s}: {v*1e-6:7.2f} ms  ({100*v/total:5.1f}%)")
    print(f"{'TOTAL':30s}: {total*1e-6:7.2f} ms")