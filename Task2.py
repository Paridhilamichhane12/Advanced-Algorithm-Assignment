import time
import heapq
import random
import matplotlib.pyplot as plt

# =====================================================================
# 1. GRAPH STRUCTURE IMPLEMENTATION (Adjacency List)
# =====================================================================
class TransportationNetwork:
    """Represents a city traffic network layout using Adjacency Lists."""
    def __init__(self):
        self.adj_list = {}

    def add_vertex(self, vertex):
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []

    def add_edge(self, source, destination, weight):
        self.add_vertex(source)
        self.add_vertex(destination)
        self.adj_list[source].append((destination, weight))

    def get_vertices(self):
        return list(self.adj_list.keys())

    def get_edges(self):
        edges = []
        for u in self.adj_list:
            for v, w in self.adj_list[u]:
                edges.append((u, v, w))
        return edges

# =====================================================================
# 2. THE THREE ALGORITHMS (With Distinct Return Types)
# =====================================================================

def dijkstra(graph, start):
    """
    Dijkstra's Shortest Path Algorithm: O((V + E) log V) using Min-Heap
    RETURNS: Dictionary of shortest cumulative costs from the start node.
    """
    distances = {v: float('inf') for v in graph.get_vertices()}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    
    while pq:
        curr_dist, u = heapq.heappop(pq)
        if u in visited: 
            continue
        visited.add(u)
        
        for v, weight in graph.adj_list.get(u, []):
            if v in visited: 
                continue
            new_dist = curr_dist + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    return distances

def bellman_ford(graph, start):
    """
    Bellman-Ford Shortest Path Algorithm: O(V * E)
    RETURNS: Dictionary of shortest cumulative costs, OR a string error flag if a cycle exists.
    """
    vertices = graph.get_vertices()
    distances = {v: float('inf') for v in vertices}
    distances[start] = 0
    edges = graph.get_edges()
    
    # Relax edges V - 1 times
    for _ in range(len(vertices) - 1):
        for u, v, w in edges:
            if distances[u] != float('inf') and distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
                
    # Negative cycle verification pass
    for u, v, w in edges:
        if distances[u] != float('inf') and distances[u] + w < distances[v]:
            return "NEGATIVE_CYCLE_DETECTED"
            
    return distances

def prim(graph, start):
    """
    Prim's Minimum Spanning Tree Algorithm: O((V + E) log V)
    RETURNS: A list of edge tuples outlining the structure of the spanning backbone tree.
    """
    visited = set([start])
    mst_edges = []
    pq = []
    
    for v, w in graph.adj_list.get(start, []):
        heapq.heappush(pq, (w, start, v))
        
    while pq:
        w, u, v = heapq.heappop(pq)
        if v in visited: 
            continue
        visited.add(v)
        mst_edges.append((u, v, w))
        
        for next_v, next_w in graph.adj_list.get(v, []):
            if next_v not in visited:
                heapq.heappush(pq, (next_w, v, next_v))
    return mst_edges

# =====================================================================
# 3. BENCHMARKING PIPELINE & DUAL METRIC CHART PLOTTER
# =====================================================================
def run_benchmark_pipeline():
    # Structural ranges targeting nodes 10 up to 1000
    node_scales = [10, 50, 100, 250, 500, 750, 1000]
    
    dijkstra_results = []
    prim_results = []
    bellman_results = []

    print("=" * 75)
    print(" RUNNING SYSTEM BENCHMARKS FOR TASK 2")
    print("=" * 75)
    print(f"{'Nodes (V)':<12}{'Edges (E)':<12}{'Dijkstra (ms)':<16}{'Prim (ms)':<16}{'Bellman-Ford (ms)':<16}")
    print("-" * 75)

    for v_size in node_scales:
        test_net = TransportationNetwork()
        for i in range(v_size):
            test_net.add_vertex(i)
        
        # Sparse layout graph setup (Avg 4 paths per node)
        for i in range(v_size):
            for _ in range(4):
                target = random.randint(0, v_size - 1)
                if i != target:
                    test_net.add_edge(i, target, random.randint(1, 25))
                    
        actual_e = len(test_net.get_edges())

        # Time Dijkstra Execution
        t0 = time.perf_counter()
        dijkstra(test_net, 0)
        t_dijkstra = (time.perf_counter() - t0) * 1000
        dijkstra_results.append(t_dijkstra)

        # Time Prim Execution
        t0 = time.perf_counter()
        prim(test_net, 0)
        t_prim = (time.perf_counter() - t0) * 1000
        prim_results.append(t_prim)

        # Time Bellman-Ford Execution
        t0 = time.perf_counter()
        bellman_ford(test_net, 0)
        t_bellman = (time.perf_counter() - t0) * 1000
        bellman_results.append(t_bellman)

        # Print layout updates live to terminal console
        print(f"{v_size:<12}{actual_e:<12}{t_dijkstra:<16.3f}{t_prim:<16.3f}{t_bellman:<16.3f}")

    print("=" * 75)
    print("Generating dual-metric evaluation plot window layout...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- GRAPH 1: Standard Linear Display View ---
    ax1.plot(node_scales, dijkstra_results, label="Dijkstra's Algorithm - O((V+E) log V)", color="#1f77b4", linewidth=2.5, marker='o')
    ax1.plot(node_scales, bellman_results, label="Bellman-Ford Algorithm - O(V*E)", color="#ff7f0e", linewidth=2.5, marker='s')
    ax1.plot(node_scales, prim_results, label="Prim's Algorithm - O((V+E) log V)", color="#2ca02c", linewidth=2.5, marker='^')
    ax1.set_title("Standard Linear Scaling Profile View\n(Shows Absolute Execution Gaps)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Graph Structural Scale (Total Nodes V)")
    ax1.set_ylabel("Measured Operational Latency (ms)")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc="upper left")

    # --- GRAPH 2: Logarithmic Normalized Display View ---
    ax2.plot(node_scales, dijkstra_results, label="Dijkstra's Algorithm", color="#1f77b4", linewidth=2.5, marker='o')
    ax2.plot(node_scales, bellman_results, label="Bellman-Ford Algorithm", color="#ff7f0e", linewidth=2.5, marker='s')
    ax2.plot(node_scales, prim_results, label="Prim's Algorithm", color="#2ca02c", linewidth=2.5, marker='^')
    
    ax2.set_yscale('log') # Split out compressed baseline measurements clearly
    ax2.set_title("Logarithmic Scaling Profile View\n(Reveals Lower Heap Scaling Curves)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Graph Structural Scale (Total Nodes V)")
    ax2.set_ylabel("Measured Operational Latency (ms, Log Scale)")
    ax2.grid(True, which="both", linestyle='--', alpha=0.5)
    ax2.legend(loc="upper left")

    plt.suptitle("Task 2: Pathfinding and Spanning Tree Algorithmic Performance Profile", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save chart layout artifact file straight to current directory
    output_filename = "task2_comprehensive_benchmarks.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"🎉 Success! High-resolution output graph saved locally as: '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    # 1. Verification structural check to display variations in return formatting
    sample_net = TransportationNetwork()
    for edge in [('A','B',4), ('A','C',2), ('B','C',1), ('B','D',5), ('C','D',8)]:
        sample_net.add_edge(edge[0], edge[1], edge[2])
        
    print("VERIFICATION DATA DEMONSTRATING STRUCTURAL RETURNING SCHEMES:")
    print(f" -> Dijkstra Path Result (Map Data Format):    {dijkstra(sample_net, 'A')}")
    print(f" -> Bellman-Ford Result (Map Data Format):     {bellman_ford(sample_net, 'A')}")
    print(f" -> Prim Minimal Tree Result (List of Edges):   {prim(sample_net, 'A')}\n")
    
    # 2. Fire up scalability benchmarking routines
    run_benchmark_pipeline()