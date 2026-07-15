import time
import math
import random
import copy
import matplotlib.pyplot as plt

# Set random seed for consistent, reproducible results
random.seed(101)

# =====================================================================
# 1. CORE DATA STRUCTURES
# =====================================================================
class MultidimensionalBin:
    def __init__(self, bin_id, capacities):
        self.id = bin_id
        self.max_capacities = capacities  # [CPU, RAM, Bandwidth]
        self.current_load = [0.0] * len(capacities)
        self.items = []

    def can_fit(self, item_reqs):
        """Checks structural viability across all three dimensions simultaneously."""
        return all(self.current_load[d] + item_reqs[d] <= self.max_capacities[d] 
                   for d in range(len(self.max_capacities)))

    def add_item(self, item):
        self.items.append(item)
        for d in range(len(self.max_capacities)):
            self.current_load[d] += item['requirements'][d]

    def get_average_utilization(self):
        """Returns normalized space utilization across all three dimensions."""
        return sum(self.current_load[d] / self.max_capacities[d] 
                   for d in range(len(self.max_capacities))) / len(self.max_capacities)


def generate_workload_items(num_items, dimensions=3):
    """Generates server items with resource demands between 10% and 50% of capacity."""
    items = []
    for i in range(num_items):
        # Generates demand requirements for CPU, RAM, and Bandwidth
        reqs = [random.uniform(10.0, 50.0) for _ in range(dimensions)]
        items.append({'id': i, 'requirements': reqs})
    return items

# =====================================================================
# 2. HEURISTIC 1: Greedy Multidimensional FFD (MFFD)
# =====================================================================
def greedy_mffd(items, bin_capacities):
    """Sorts items by aggregate normalized resource footprint and packs them greedily."""
    # Sort items descending by total resource load across all three dimensions
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
# 3. HEURISTIC 2: Simulated Annealing (SA) Metaheuristic
# =====================================================================
def evaluate_solution_quality(bins):
    """
    Evaluates state cost. Minimizes total bins used while penalizing 
    loosely packed active bins to guide the optimization search space.
    """
    num_bins = len(bins)
    if num_bins == 0:
        return 0
    avg_util = sum(b.get_average_utilization() ** 2 for b in bins) / num_bins
    return num_bins - (0.15 * avg_util)


def simulated_annealing_optimize(items, bin_capacities, initial_solution, max_iter=400):
    """Refines an existing allocation layout using structured cooling search."""
    current_state = copy.deepcopy(initial_solution)
    current_cost = evaluate_solution_quality(current_state)
    
    best_state = copy.deepcopy(current_state)
    best_cost = current_cost

    # Annealing schedule configurations
    T = 2.0
    cooling_rate = 0.94
    
    for _ in range(max_iter):
        if T < 0.005:
            break
            
        candidate_state = copy.deepcopy(current_state)
        active_bins = [b for b in candidate_state if b.items]
        if not active_bins:
            continue
            
        # Select a random item to move
        source_bin = random.choice(active_bins)
        target_item = random.choice(source_bin.items)
        
        # Identify alternative placement choices
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
                
        # If no existing bin fits the item, provision a new container
        if not reallocated:
            source_bin.items.remove(target_item)
            for d in range(len(bin_capacities)):
                source_bin.current_load[d] -= target_item['requirements'][d]
            new_bin = MultidimensionalBin(len(candidate_state), bin_capacities)
            new_bin.add_item(target_item)
            candidate_state.append(new_bin)

        # Cleanup resulting empty bins from tracking array
        candidate_state = [b for b in candidate_state if b.items]
        for idx, b in enumerate(candidate_state):
            b.id = idx
            
        candidate_cost = evaluate_solution_quality(candidate_state)
        delta_energy = candidate_cost - current_cost
        
        # Metropolis-Hastings acceptance condition
        if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
            current_state = candidate_state
            current_cost = candidate_cost
            
            if current_cost < best_cost:
                best_state = copy.deepcopy(current_state)
                best_cost = current_cost
                
        T *= cooling_rate
        
    return best_state

# =====================================================================
# 4. BENCHMARKING ENGINE & DUAL-PANEL GRAPH GENERATOR
# =====================================================================
if __name__ == "__main__":
    print("=" * 85)
    print("      SCALABILITY EVALUATION: MULTI-DIMENSIONAL BIN PACKING (CPU, RAM, BANDWIDTH)")
    print("=" * 85)
    print(f"{'Scale (Tasks)':<15}{'Greedy Bins':<15}{'Greedy Time (ms)':<20}{'SA Bins':<12}{'SA Time (ms)':<15}")
    print("-" * 85)

    input_scales = [20, 40, 60, 80, 100, 120, 140]
    resource_limits = [100.0, 100.0, 100.0]  # Bin limit parameters [CPU, RAM, Bandwidth]

    mffd_times, sa_times = [], []
    mffd_bin_usage, sa_bin_usage = [], []

    for n in input_scales:
        workload = generate_workload_items(n, dimensions=3)
        
        # Benchmark MFFD
        t0 = time.perf_counter()
        mffd_result = greedy_mffd(workload, resource_limits)
        t_mffd = (time.perf_counter() - t0) * 1000
        mffd_times.append(t_mffd)
        mffd_bin_usage.append(len(mffd_result))
        
        # Benchmark Simulated Annealing
        t0 = time.perf_counter()
        sa_result = simulated_annealing_optimize(workload, resource_limits, mffd_result, max_iter=300)
        t_sa = (time.perf_counter() - t0) * 1000
        sa_times.append(t_sa)
        sa_bin_usage.append(len(sa_result))

        print(f"{n:<15}{len(mffd_result):<15}{t_mffd:<20.4f}{len(sa_result):<12}{t_sa:<15.4f}")

    print("=" * 85)
    print("Generating high-fidelity dual-panel comparative graphs...")

    # Set up Side-by-Side Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Left Plot: Solution Quality (Objective value metric)
    ax1.plot(input_scales, mffd_bin_usage, 
             label="Greedy MFFD (Baseline)", 
             color="#1f77b4", linewidth=2.5, marker='o', markersize=6)
    ax1.plot(input_scales, sa_bin_usage, 
             label="Simulated Annealing (Optimized)", 
             color="#2ca02c", linewidth=2.5, marker='^', markersize=6)
    
    ax1.set_title("Solution Quality Comparison\n(Lower Bin Count = Better Packing)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Problem Scale (Total Items to Pack)", fontsize=10)
    ax1.set_ylabel("Total Bins Required", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=9, loc="upper left")

    # Right Plot: Logarithmic Scaling Runtime Graph
    ax2.plot(input_scales, mffd_times, 
             label="Greedy MFFD", 
             color="#1f77b4", linewidth=2.5, marker='o', markersize=6)
    ax2.plot(input_scales, sa_times, 
             label="Simulated Annealing", 
             color="#d62728", linewidth=2.5, marker='s', markersize=6)
    
    ax2.set_yscale('log')  # Uses a Log Scale to show both curves in perspective
    ax2.set_title("Computational Cost Scaling Profile\n(Logarithmic Scale Runtime Analysis)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Problem Scale (Total Items to Pack)", fontsize=10)
    ax2.set_ylabel("Measured Execution Runtime (ms, Log Scale)", fontsize=10)
    ax2.grid(True, which="both", linestyle='--', alpha=0.5)
    ax2.legend(fontsize=9, loc="upper left")

    # Layout adjustment
    plt.suptitle("Task 4: MDBPP Performance and Scalability Trade-offs", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_filename = "task4_mdbpp_dual_panel_evaluation.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"🎉 Success! High-fidelity dual-panel line graph saved as: '{output_filename}'")
    plt.show()