"""torch-lite torch.utils.benchmark: Timer / Measurement on time.perf_counter."""

import statistics
import time
import timeit as _timeit


class Measurement:
    def __init__(self, number_per_run, raw_times, label=None, sub_label=None, description=None, stmt=""):
        self.number_per_run = number_per_run
        self.raw_times = raw_times
        self.times = [t / number_per_run for t in raw_times]
        self.label, self.sub_label, self.description, self.stmt = label, sub_label, description, stmt

    @property
    def mean(self):
        return statistics.mean(self.times)

    @property
    def median(self):
        return statistics.median(self.times)

    @property
    def iqr(self):
        if len(self.times) < 2:
            return 0.0
        q = statistics.quantiles(self.times, n=4)
        return q[2] - q[0]

    def __repr__(self):
        unit, scale = ("us", 1e6) if self.median < 1e-3 else ("ms", 1e3) if self.median < 1 else ("s", 1)
        head = f"{self.label}: {self.sub_label}" if self.label else self.stmt
        return (f"<torch.utils.benchmark.utils.common.Measurement object>\n{head}\n"
                f"  Median: {self.median * scale:.2f} {unit}\n  {len(self.times)} measurements, {self.number_per_run} runs per measurement")


class Timer:
    def __init__(self, stmt="pass", setup="pass", globals=None, label=None, sub_label=None, description=None, num_threads=1,
                 timer=time.perf_counter):
        self.stmt, self.setup, self.label, self.sub_label, self.description = stmt, setup, label, sub_label, description
        self.num_threads = num_threads
        self._t = _timeit.Timer(stmt, setup, timer=timer, globals=dict(globals or {}))

    def timeit(self, number=1000000):
        self._t.timeit(number=1)  # warmup
        return Measurement(number, [self._t.timeit(number)], self.label, self.sub_label, self.description, self.stmt)

    def blocked_autorange(self, callback=None, min_run_time=0.2):
        number, t = 1, self._t.timeit(1)
        while t < 0.01 and number < 10 ** 6:
            number *= 10
            t = self._t.timeit(number)
        raw, total = [], 0.0
        while total < min_run_time or not raw:
            t = self._t.timeit(number)
            raw.append(t)
            total += t
            if callback:
                callback(number, t)
        return Measurement(number, raw, self.label, self.sub_label, self.description, self.stmt)

    adaptive_autorange = blocked_autorange


class Compare:
    def __init__(self, results):
        self.results = list(results)

    def print(self):
        for r in self.results:
            print(f"{r.description or r.sub_label or r.stmt}: {r.median * 1e6:.1f} us")

    def __str__(self):
        return "\n".join(f"{r.description or r.sub_label or r.stmt}: {r.median * 1e6:.1f} us" for r in self.results)
