import time
import math
import random
import copy
import matplotlib.pyplot as plt

random.seed(101)

# =====================================================================
# 1. CORE DATA STRUCTURES
# =====================================================================
class MultidimensionalBin:
    def __init__(self, bin_id, capacities):
        self.id = bin_id
        self.max_capacities = capacities
        self.current_load = [0.0] * len(capacities)
        self.items = []

    def can_fit(self, item_reqs):
        return all(self.current_load[d] + item_reqs[d] <= self.max_capacities[d] 
                   for d in range(len(self.max_capacities)))

    def add_item(self, item):
        self.items.append(item)
        for d in range(len(self.max_capacities)):
            self.current_load[d] += item['requirements'][d]

    def get_average_utilization(self):
        return sum(self.current_load[d] / self.max_capacities[d] 
                   for d in range(len(self.max_capacities))) / len(self.max_capacities)


def generate_workload_items(num_items, dimensions=3):
    items = []
    for i in range(num_items):
        reqs = [random.uniform(10.0, 50.0) for _ in range(dimensions)]
        items.append({'id': i, 'requirements': reqs})
    return items

# =====================================================================
# 2. HEURISTIC 1: Greedy Multidimensional FFD (MFFD)
# =====================================================================
def greedy_mffd(items, bin_capacities):
    sorted_items = sorted(items, key=lambda x: sum(x['requirements']), reverse=True)
    bins = []
    for item in sorted_items:
        placed = False
        for b in bins:
            if b.can_fit(item['requirements']):
                b.add_item(item)
                placed = True
                break
        if not placed:
            new_bin = MultidimensionalBin(len(bins), bin_capacities)
            new_bin.add_item(item)
            bins.append(new_bin)
    return bins

# =====================================================================
# 3. HEURISTIC 2: Local Search (First-Improvement Hill Climbing)
# =====================================================================
def local_search_optimize(bin_capacities, initial_solution, max_iter=300):
    
    current_state = copy.deepcopy(initial_solution)

    for _ in range(max_iter):
        active_bins = [b for b in current_state if b.items]
        if len(active_bins) < 2:
            break

        # Target the least-utilized bin as the source for relocation
        source_bin = min(active_bins, key=lambda b: b.get_average_utilization())

        improved = False
        # Try relocating each item out of the source bin, least improvement first
        for target_item in list(source_bin.items):
            for dest in active_bins:
                if dest.id == source_bin.id:
                    continue
                if dest.can_fit(target_item['requirements']):
                    # Perform the move
                    source_bin.items.remove(target_item)
                    for d in range(len(bin_capacities)):
                        source_bin.current_load[d] -= target_item['requirements'][d]
                    dest.add_item(target_item)
                    improved = True
                    break
            if improved:
                break

        # Remove any bin that has been fully emptied
        current_state = [b for b in current_state if b.items]
        for idx, b in enumerate(current_state):
            b.id = idx

        if not improved:
            # No beneficial move found this pass; search has converged
            break

    return current_state

# =====================================================================
# 4. HEURISTIC 3: Simulated Annealing (SA) Metaheuristic
# =====================================================================
def evaluate_solution_quality(bins):
    num_bins = len(bins)
    if num_bins == 0:
        return 0
    avg_util = sum(b.get_average_utilization() ** 2 for b in bins) / num_bins
    return num_bins - (0.15 * avg_util)


def simulated_annealing_optimize(items, bin_capacities, initial_solution, max_iter=400):
    current_state = copy.deepcopy(initial_solution)
    current_cost = evaluate_solution_quality(current_state)
    
    best_state = copy.deepcopy(current_state)
    best_cost = current_cost

    T = 2.0
    cooling_rate = 0.94
    
    for _ in range(max_iter):
        if T < 0.005:
            break
            
        candidate_state = copy.deepcopy(current_state)
        active_bins = [b for b in candidate_state if b.items]
        if not active_bins:
            continue
            
        source_bin = random.choice(active_bins)
        target_item = random.choice(source_bin.items)
        
        dest_choices = [b for b in candidate_state if b.id != source_bin.id]
        random.shuffle(dest_choices)
        
        reallocated = False
        for dest in dest_choices:
            if dest.can_fit(target_item['requirements']):
                source_bin.items.remove(target_item)
                for d in range(len(bin_capacities)):
                    source_bin.current_load[d] -= target_item['requirements'][d]
                dest.add_item(target_item)
                reallocated = True
                break
                
        if not reallocated:
            source_bin.items.remove(target_item)
            for d in range(len(bin_capacities)):
                source_bin.current_load[d] -= target_item['requirements'][d]
            new_bin = MultidimensionalBin(len(candidate_state), bin_capacities)
            new_bin.add_item(target_item)
            candidate_state.append(new_bin)

        candidate_state = [b for b in candidate_state if b.items]
        for idx, b in enumerate(candidate_state):
            b.id = idx
            
        candidate_cost = evaluate_solution_quality(candidate_state)
        delta_energy = candidate_cost - current_cost
        
        if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
            current_state = candidate_state
            current_cost = candidate_cost
            
            if current_cost < best_cost:
                best_state = copy.deepcopy(current_state)
                best_cost = current_cost
                
        T *= cooling_rate
        
    return best_state

# =====================================================================
# 5. BENCHMARKING ENGINE & TRI-PANEL GRAPH GENERATOR
# =====================================================================
if __name__ == "__main__":
    print("=" * 100)
    print("      SCALABILITY EVALUATION: MULTI-DIMENSIONAL BIN PACKING (CPU, RAM, BANDWIDTH)")
    print("=" * 100)
    print(f"{'Scale':<8}{'Greedy Bins':<13}{'Greedy(ms)':<13}{'LS Bins':<10}{'LS(ms)':<12}{'SA Bins':<10}{'SA(ms)':<12}")
    print("-" * 100)

    input_scales = [20, 40, 60, 80, 100, 120, 140]
    resource_limits = [100.0, 100.0, 100.0]

    mffd_times, ls_times, sa_times = [], [], []
    mffd_bin_usage, ls_bin_usage, sa_bin_usage = [], [], []

    for n in input_scales:
        workload = generate_workload_items(n, dimensions=3)
        
        # Greedy MFFD
        t0 = time.perf_counter()
        mffd_result = greedy_mffd(workload, resource_limits)
        t_mffd = (time.perf_counter() - t0) * 1000
        mffd_times.append(t_mffd)
        mffd_bin_usage.append(len(mffd_result))
        
        # Local Search (refines the Greedy result)
        t0 = time.perf_counter()
        ls_result = local_search_optimize(resource_limits, mffd_result, max_iter=300)
        t_ls = (time.perf_counter() - t0) * 1000
        ls_times.append(t_ls)
        ls_bin_usage.append(len(ls_result))

        # Simulated Annealing (refines the Greedy result)
        t0 = time.perf_counter()
        sa_result = simulated_annealing_optimize(workload, resource_limits, mffd_result, max_iter=300)
        t_sa = (time.perf_counter() - t0) * 1000
        sa_times.append(t_sa)
        sa_bin_usage.append(len(sa_result))

        print(f"{n:<8}{len(mffd_result):<13}{t_mffd:<13.4f}{len(ls_result):<10}{t_ls:<12.4f}{len(sa_result):<10}{t_sa:<12.4f}")

    print("=" * 100)
    print("Generating tri-algorithm comparative graphs...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Left Plot: Solution Quality
    ax1.plot(input_scales, mffd_bin_usage, 
             label="Greedy MFFD (Baseline)", 
             color="#1f77b4", linewidth=2.5, marker='o', markersize=6)
    ax1.plot(input_scales, ls_bin_usage, 
             label="Local Search (Refined)", 
             color="#ff7f0e", linewidth=2.5, marker='d', markersize=6)
    ax1.plot(input_scales, sa_bin_usage, 
             label="Simulated Annealing (Optimized)", 
             color="#2ca02c", linewidth=2.5, marker='^', markersize=6)
    
    ax1.set_title("Solution Quality Comparison\n(Lower Bin Count = Better Packing)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Problem Scale (Total Items to Pack)", fontsize=10)
    ax1.set_ylabel("Total Bins Required", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=9, loc="upper left")

    # Right Plot: Runtime Scaling
    ax2.plot(input_scales, mffd_times, 
             label="Greedy MFFD", 
             color="#1f77b4", linewidth=2.5, marker='o', markersize=6)
    ax2.plot(input_scales, ls_times, 
             label="Local Search", 
             color="#ff7f0e", linewidth=2.5, marker='d', markersize=6)
    ax2.plot(input_scales, sa_times, 
             label="Simulated Annealing", 
             color="#d62728", linewidth=2.5, marker='s', markersize=6)
    
    ax2.set_yscale('log')
    ax2.set_title("Computational Cost Scaling Profile\n(Logarithmic Scale Runtime Analysis)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Problem Scale (Total Items to Pack)", fontsize=10)
    ax2.set_ylabel("Measured Execution Runtime (ms, Log Scale)", fontsize=10)
    ax2.grid(True, which="both", linestyle='--', alpha=0.5)
    ax2.legend(fontsize=9, loc="upper left")

    plt.suptitle("Task 4: MDBPP Performance and Scalability Trade-offs (Greedy vs Local Search vs SA)", fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_filename = "task4_mdbpp_tripanel_evaluation.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f" Success! Saved as: '{output_filename}'")