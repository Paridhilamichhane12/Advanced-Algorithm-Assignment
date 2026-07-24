import time
import random
import threading
import numpy as np
import matplotlib.pyplot as plt
 
random.seed(42)
np.random.seed(42)
 
# =====================================================================
# 1. SEQUENTIAL BASELINE (NumPy-vectorised merge, single thread)
# =====================================================================
 
def np_merge(arr, lo, mid, hi):
   
    left = arr[lo:mid]
    right = arr[mid:hi]
    n_left, n_right = len(left), len(right)
    left_positions = np.arange(n_left) + np.searchsorted(right, left, side='right')
    # Symmetric count for `right` relative to `left`.
    right_positions = np.arange(n_right) + np.searchsorted(left, right, side='left')
 
    merged = np.empty(n_left + n_right, dtype=arr.dtype)
    merged[left_positions] = left
    merged[right_positions] = right
 
    arr[lo:hi] = merged
 
 
def sequential_merge_sort(arr, lo, hi):
    if hi - lo <= 1:
        return
    mid = lo + (hi - lo) // 2
    sequential_merge_sort(arr, lo, mid)
    sequential_merge_sort(arr, mid, hi)
    np_merge(arr, lo, mid, hi)
 
 
# =====================================================================
# 2. CONCURRENT (threading + mutex) MERGE SORT
# =====================================================================
 
class ParallelMergeSort:
    SEQ_CUTOFF = 20000  
 
    def __init__(self, max_threads=4):
        self.max_threads = max_threads
        self.active_threads = 0
        self.lock = threading.Lock()  
 
    def _try_reserve_thread(self):
        with self.lock:                      
            if self.active_threads < self.max_threads:
                self.active_threads += 1
                return True
            return False
                                                
 
    def _release_thread(self):
        with self.lock:                     
            self.active_threads -= 1
                                              
 
    def _parallel_sort(self, arr, lo, hi):
        if hi - lo <= 1:
            return
        if hi - lo <= self.SEQ_CUTOFF:
            sequential_merge_sort(arr, lo, hi)
            return
 
        mid = lo + (hi - lo) // 2
 
        if self._try_reserve_thread():
            
            left_thread = threading.Thread(
                target=self._parallel_sort, args=(arr, lo, mid)
            )
            left_thread.start()
 
            self._parallel_sort(arr, mid, hi)   
 
            left_thread.join()                  
            self._release_thread()
        else:
            self._parallel_sort(arr, lo, mid)
            self._parallel_sort(arr, mid, hi)
 
        np_merge(arr, lo, mid, hi)
 
    def sort(self, arr):
        self.active_threads = 0
        self._parallel_sort(arr, 0, len(arr))
 
 
# =====================================================================
# 3. EXPERIMENT RUNNER (1, 2, 4, 8 threads; median of several repeats)
# =====================================================================
 
def run_sequential_once(data):
    arr = np.array(data, dtype=np.float64)
    t0 = time.perf_counter()
    sequential_merge_sort(arr, 0, len(arr))
    t1 = time.perf_counter()
    assert np.all(arr[:-1] <= arr[1:]), "Sequential result not sorted!"
    return t1 - t0
 
 
def run_parallel_once(data, num_threads):
    arr = np.array(data, dtype=np.float64)
    sorter = ParallelMergeSort(max_threads=num_threads)
    t0 = time.perf_counter()
    sorter.sort(arr)
    t1 = time.perf_counter()
    assert np.all(arr[:-1] <= arr[1:]), "Parallel result not sorted! (race condition or bug)"
    return t1 - t0
 
 
if __name__ == "__main__":
    print("=" * 80)
    print("      TASK 5: CONCURRENT PROGRAMMING BENCHMARK (Python threading + NumPy)")
    print("=" * 80)
 
    DATASET_SIZE = 500_000  # increase this on a real multi-core machine
                              # (e.g. 2-5 million) for a clearer speedup signal
    REPEATS = 5
    THREAD_CONFIGS = [1, 2, 4, 8]
 
    original_data = [random.random() for _ in range(DATASET_SIZE)]
    print(f"Dataset Initialized: Sorting {DATASET_SIZE:,} elements.")
    print(f"Running benchmarks across {THREAD_CONFIGS} thread configurations, "
          f"{REPEATS} repeats each (median reported)...\n")
 
    # --- Sequential baseline ---
    seq_times = [run_sequential_once(original_data) for _ in range(REPEATS)]
    seq_median = sorted(seq_times)[len(seq_times) // 2]
    print(f"Sequential baseline | median runtime: {seq_median:.4f}s "
          f"(runs: {[f'{t:.3f}' for t in seq_times]})")
 
    # --- Parallel configurations ---
    runtimes = []
    speedups = []
    for nt in THREAD_CONFIGS:
        times = [run_parallel_once(original_data, nt) for _ in range(REPEATS)]
        med = sorted(times)[len(times) // 2]
        sp = seq_median / med
        runtimes.append(med)
        speedups.append(sp)
        print(f"Threads: {nt:<3} | median runtime: {med:.4f}s | speedup: {sp:.2f}x "
              f"(runs: {[f'{t:.3f}' for t in times]})")
 
    efficiencies = [s / t for s, t in zip(speedups, THREAD_CONFIGS)]
 
    print("\n" + "=" * 80)
    print(f"{'Threads':>8} {'Time(s)':>10} {'Speedup':>10} {'Efficiency':>12}")
    print("-" * 44)
    for nt, rt, sp, ef in zip(THREAD_CONFIGS, runtimes, speedups, efficiencies):
        print(f"{nt:>8} {rt:>10.4f} {sp:>9.2f}x {ef*100:>10.1f}%")
 
    # =====================================================================
    # 4. PLOT: Speedup vs Thread Count
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.suptitle("Task 5: Parallel Merge Sort — Scalability & Efficiency (Python threading)",
                 fontsize=14, fontweight='bold', y=0.98)
 
    ax1.plot(THREAD_CONFIGS, runtimes, color="#4A90E2", linewidth=2.5,
              marker='o', markersize=8, label="Measured Runtime (s)")
    ax1.set_title("Wall-Clock Runtime vs Thread Count", fontsize=11, fontweight='semibold')
    ax1.set_xlabel("Thread Count (p)")
    ax1.set_ylabel("Execution Time (seconds)")
    ax1.set_xticks(THREAD_CONFIGS)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    for t, r in zip(THREAD_CONFIGS, runtimes):
        ax1.annotate(f'{r:.3f}s', xy=(t, r), xytext=(8, 5),
                     textcoords="offset points", fontsize=9, fontweight='semibold')
 
    color_sp = '#D9534F'
    l1 = ax2.plot(THREAD_CONFIGS, speedups, color=color_sp, marker='o',
                   linewidth=2.5, label="Empirical Speedup ($S_p$)", zorder=3)
    ax2.plot(THREAD_CONFIGS, THREAD_CONFIGS, color='#7f7f7f', linestyle='--',
              linewidth=1.2, label="Ideal Linear Speedup", alpha=0.7)
    ax2.set_title("Speedup & Efficiency vs Thread Count", fontsize=11, fontweight='semibold')
    ax2.set_xlabel("Thread Count (p)")
    ax2.set_ylabel("Speedup ($S_p$)", color=color_sp)
    ax2.tick_params(axis='y', labelcolor=color_sp)
    ax2.set_xticks(THREAD_CONFIGS)
    ax2.grid(True, linestyle='--', alpha=0.5)
 
    ax3 = ax2.twinx()
    color_ef = '#2CA02C'
    l2 = ax3.plot(THREAD_CONFIGS, efficiencies, color=color_ef, marker='s',
                   linestyle='-.', linewidth=2, label="Efficiency ($E_p$)", zorder=3)
    ax3.set_ylabel("Efficiency ($E_p = S_p/p$)", color=color_ef)
    ax3.tick_params(axis='y', labelcolor=color_ef)
    ax3.set_ylim(0, 1.1)
 
    lines = l1 + l2
    ax2.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=9)
 
    plt.tight_layout()
    plt.savefig("task5_speedup_dashboard.png", bbox_inches='tight')
    print("\nSaved plot to task5_speedup_dashboard.png")
    plt.show()
 