#!/usr/bin/env python3
"""Quick benchmark: verify free-threaded Python gives parallel speedup.

Run on login node (keep workers low, e.g. 4):

    source activate jannes
    python scripts/test_free_threading.py --workers 4

Expected with GIL disabled: threaded run noticeably faster than serial.
With GIL enabled: serial and threaded times will be similar.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def _gil_status():
    if hasattr(sys, "_is_gil_enabled"):
        return sys._is_gil_enabled()
    return True


def _cpu_work(seed, size=4000, repeats=8):
    """CPU-bound numpy work (similar load to per-country farmer updates)."""
    rng = np.random.default_rng(seed)
    a = rng.random((size, size))
    b = rng.random((size, size))
    out = 0.0
    for _ in range(repeats):
        out += float(np.sum(a @ b))
    return out


def _run_serial(n_countries, work_size, work_repeats):
    t0 = time.perf_counter()
    for i in range(n_countries):
        _cpu_work(i, work_size, work_repeats)
    return time.perf_counter() - t0


def _run_threaded(n_countries, n_workers, work_size, work_repeats):
    def task(i):
        return _cpu_work(i, work_size, work_repeats)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(task, range(n_countries)))
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Thread pool size (default: min(4, cpu_count))",
    )
    parser.add_argument(
        "--countries",
        type=int,
        default=8,
        help="Number of fake countries to simulate (default: 8)",
    )
    parser.add_argument(
        "--work-size",
        type=int,
        default=3500,
        help="Matrix size for CPU work per country",
    )
    parser.add_argument(
        "--work-repeats",
        type=int,
        default=6,
        help="Matrix multiplies per country",
    )
    args = parser.parse_args()

    gil = _gil_status()
    print(f"Python: {sys.version.split()[0]}")
    print(f"GIL enabled: {gil}")
    print(f"cpu_count(): {os.cpu_count()}")
    print(f"SLURM_CPUS_ON_NODE: {os.environ.get('SLURM_CPUS_ON_NODE', '(not set)')}")
    print(f"Simulating {args.countries} countries with {args.workers} workers")
    print()

    serial_t = _run_serial(args.countries, args.work_size, args.work_repeats)
    threaded_t = _run_threaded(
        args.countries, args.workers, args.work_size, args.work_repeats
    )

    speedup = serial_t / threaded_t if threaded_t > 0 else float("inf")
    print(f"Serial:   {serial_t:.2f}s")
    print(f"Threaded: {threaded_t:.2f}s")
    print(f"Speedup:  {speedup:.2f}x")

    if gil:
        print("\nNote: GIL is ON — threading will not speed up CPU-bound work.")
    elif speedup < 1.2:
        print("\nWarning: low speedup. Try more countries or check login-node CPU limits.")
    else:
        print("\nOK: free-threading appears to be working.")


if __name__ == "__main__":
    main()
