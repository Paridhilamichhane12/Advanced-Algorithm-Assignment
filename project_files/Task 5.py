import time
import random
import threading
import matplotlib.pyplot as plt

# Set random seed for consistent performance benchmarks
random.seed(42)

# =====================================================================
# 1. PARALLEL MERGE SORT WITH MUTEX SYNCHRONIZATION
# =====================================================================
class ParallelMergeSort:
    def __init__(self, max_threads=4):
        self.max_threads = max_threads
        self.thread_count = 1
        self.lock = threading.Lock()  # Mutex protecting the active thread counter

    def _merge(self, arr, low, mid, high):
        """Standard merge step combining two sorted sub-arrays."""
        left = arr[low:mid + 1]
        right = arr[mid + 1:high + 1]
        
        i = j = 0
        k = low
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
            
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
            
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1

    def _sequential_sort(self, arr, low, high):
        """Fallback to high-performance sequential merge sort."""
        if low < high:
            mid = (low + high) // 2
            self._sequential_sort(arr, low, mid)
            self._sequential_sort(arr, mid + 1, high)
            self._merge(arr, low, mid, high)

    def _parallel_sort(self, arr, low, high):
        """Recursive sorting using thread-spawning with global limits."""
        if low >= high:
            return

        mid = (low + high) // 2
        spawn_thread = False

        # --- CRITICAL SECTION ---
        # Safely query and update the active thread count using the mutex lock
        with self.lock:
            if self.thread_count < self.max_threads:
                self.thread_count += 1
                spawn_thread = True

        if spawn_thread:
            # Spawn a concurrent thread to sort the left half
            left_thread = threading.Thread(
                target=self._parallel_sort, 
                args=(arr, low, mid)
            )
            left_thread.start()

            # The parent thread processes the right half concurrently
            self._parallel_sort(arr, mid + 1, high)

            # Synchronization: Wait for the child thread to finish
            left_thread.join()

            # Safely release the thread slot
            with self.lock:
                self.thread_count -= 1
        else:
            # Thread limit reached: Fallback to sequential execution
            self._sequential_sort(arr, low, mid)
            self._sequential_sort(arr, mid + 1, high)

        # Merge the two halves
        self._merge(arr, low, mid, high)

    def sort(self, arr):
        self.thread_count = 1
        self._parallel_sort(arr, 0, len(arr) - 1)


# =====================================================================
# 2. EXPERIMENT RUNNER & PERFORMANCE PROFILER
# =====================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("                TASK 5: CONCURRENT PROGRAMMING BENCHMARK")
    print("=" * 80)

    # Dataset Size (Adjust according to local CPU speeds)
    dataset_size = 150000
    original_data = [random.random() for _ in range(dataset_size)]
    
    print(f"Dataset Initialized: Sorting {dataset_size:,} elements.")
    print("Running benchmarks across 1, 2, 4, and 8 thread configurations...\n")

    thread_configs = [1, 2, 4, 8]
    runtimes = []
    speedups = []

    # 1. Measure 1 Thread (Baseline)
    baseline_data = list(original_data)
    t0 = time.perf_counter()
    sorter_1 = ParallelMergeSort(max_threads=1)
    sorter_1.sort(baseline_data)
    t_baseline = time.perf_counter() - t0
    
    runtimes.append(t_baseline)
    speedups.append(1.0)
    print(f"Threads: 1 (Baseline) | Runtime: {t_baseline:.4f}s | Speedup: 1.00x")

    # 2. Run Benchmarks for Multi-Threaded Configurations
    for t in thread_configs[1:]:
        test_data = list(original_data)
        t0 = time.perf_counter()
        sorter = ParallelMergeSort(max_threads=t)
        sorter.sort(test_data)
        t_elapsed = time.perf_counter() - t0
        
        runtimes.append(t_elapsed)
        speedup = t_baseline / t_elapsed
        speedups.append(speedup)
        print(f"Threads: {t:<12} | Runtime: {t_elapsed:.4f}s | Speedup: {speedup:.2f}x")

    # Calculate Parallel Efficiency (E_p = S_p / p)
    efficiencies = [s / t for s, t in zip(speedups, thread_configs)]

    print("\n" + "=" * 80)
    print("Generating pure line-graph scalability dashboard...")

    # =====================================================================
    # 3. DUAL-PANEL LINE GRAPH DASHBOARD
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.suptitle("Task 5: Multi-Threaded Scalability & Efficiency Analysis", 
                 fontsize=14, fontweight='bold', y=0.98)

    # -----------------------------------------------------------------
    # PANEL 1: Absolute Wall-Clock Runtime (Line Graph)
    # -----------------------------------------------------------------
    ax1.plot(thread_configs, runtimes, 
             color="#4A90E2", linewidth=2.5, marker='o', markersize=8, 
             label="Measured Runtime (seconds)")
    ax1.set_title("Absolute Wall-Clock Runtime Decay", fontsize=11, fontweight='semibold', pad=10)
    ax1.set_xlabel("Thread Count (p)", fontsize=10)
    ax1.set_ylabel("Execution Time (Seconds)", fontsize=10)
    ax1.set_xticks(thread_configs)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=9, loc="upper right")

    # Add precise runtime text annotations next to each data point
    for t, r in zip(thread_configs, runtimes):
        ax1.annotate(f'{r:.3f}s',
                     xy=(t, r),
                     xytext=(8, 5),  # offset labels to the right and slightly up
                     textcoords="offset points",
                     ha='left', va='center', fontsize=9, fontweight='semibold')

    # -----------------------------------------------------------------
    # PANEL 2: Speedup vs. Parallel Efficiency (Dual-Axis Line Graph)
    # -----------------------------------------------------------------
    # Primary Y-Axis: Speedup Factor
    color_speedup = '#D9534F'
    line1 = ax2.plot(thread_configs, speedups, color=color_speedup, marker='o', 
                     linewidth=2.5, label="Empirical Speedup ($S_p$)", zorder=3)
    ax2.plot(thread_configs, thread_configs, color='#7f7f7f', linestyle='--', 
             linewidth=1.2, label="Ideal Linear Speedup ($S_p = p$)", alpha=0.7)

    ax2.set_title("Speedup Factor & Multi-Core Efficiency", fontsize=11, fontweight='semibold', pad=10)
    ax2.set_xlabel("Thread Count (p)", fontsize=10)
    ax2.set_ylabel("Measured Speedup Factor ($S_p$)", color=color_speedup, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=color_speedup)
    ax2.set_xticks(thread_configs)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Secondary Y-Axis: System Parallel Efficiency
    ax3 = ax2.twinx()
    color_eff = '#2CA02C'
    line2 = ax3.plot(thread_configs, efficiencies, color=color_eff, marker='s', linestyle='-.',
                     linewidth=2, label="Parallel Efficiency ($E_p$)", zorder=3)
    ax3.set_ylabel("Parallel Efficiency ($E_p = S_p / p$)", color=color_eff, fontsize=10)
    ax3.tick_params(axis='y', labelcolor=color_eff)
    ax3.set_ylim(0, 1.1)

    # Combined Legend for the Dual-Axis Panel
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper left", fontsize=9)

    # Adjust spacing and save
    plt.tight_layout()
    output_filename = "task5_scalability_line_dashboard.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Success! Pure line-graph scalability dashboard saved as '{output_filename}'.")
    plt.show()