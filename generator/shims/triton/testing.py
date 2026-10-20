"""triton-lite triton.testing."""

import statistics
import time


def do_bench(fn, warmup=5, rep=20, grad_to_none=None, quantiles=None, return_mode="mean"):
    fn()
    times = []
    for _ in range(max(warmup // 5, 1)):
        fn()
    for _ in range(max(rep // 5, 3)):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1e3)
    if quantiles:
        qs = statistics.quantiles(times, n=100)
        return [qs[min(max(int(q * 100) - 1, 0), 98)] for q in quantiles]
    return {"mean": statistics.mean, "median": statistics.median, "min": min, "max": max}[return_mode](times)


def assert_close(x, y, atol=None, rtol=None, err_msg=""):
    import torch
    torch.testing.assert_close(x, y, atol=atol, rtol=rtol)
