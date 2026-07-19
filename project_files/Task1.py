import sys
import time
import random
import math
import matplotlib.pyplot as plt

# Increase recursion depth for deep un-balanced BSTs during worst-case testing
sys.setrecursionlimit(50000)

# =====================================================================
# 0. Core Data Structure: City Object
# =====================================================================
class City:
    def __init__(self, name: str, lat: float, lon: float, population: int, distance: float):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.population = population
        self.distance = distance  

    def __repr__(self):
        return f"{self.name}(Dist: {self.distance})"


# =====================================================================
# 1. Binary Search Tree (BST)
# =====================================================================
class BSTNode:
    def __init__(self, city: City):
        self.city = city
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, city: City):
        self.root = self._insert_recursive(self.root, city)

    def _insert_recursive(self, node: BSTNode, city: City) -> BSTNode:
        if not node:
            return BSTNode(city)
        if city.name < node.city.name:
            node.left = self._insert_recursive(node.left, city)
        else:
            node.right = self._insert_recursive(node.right, city)
        return node

    def search(self, name: str) -> City:
        return self._search_recursive(self.root, name)

    def _search_recursive(self, node: BSTNode, name: str) -> City:
        if not node or node.city.name == name:
            return node.city if node else None
        if name < node.city.name:
            return self._search_recursive(node.left, name)
        return self._search_recursive(node.right, name)

    def delete(self, name: str):
        self.root = self._delete_recursive(self.root, name)

    def _delete_recursive(self, node: BSTNode, name: str) -> BSTNode:
        if not node:
            return None
        if name < node.city.name:
            node.left = self._delete_recursive(node.left, name)
        elif name > node.city.name:
            node.right = self._delete_recursive(node.right, name)
        else:
            if not node.left:
                return node.right
            elif not node.right:
                return node.left

            successor = self._min_value_node(node.right)
            node.city = successor.city
            node.right = self._delete_recursive(node.right, successor.city.name)
        return node

    def _min_value_node(self, node: BSTNode) -> BSTNode:
        current = node
        while current.left:
            current = current.left
        return current


# =====================================================================
# 2. AVL Tree (Self-Balancing)
# =====================================================================
class AVLNode:
    def __init__(self, city: City):
        self.city = city
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def __init__(self):
        self.root = None

    def get_height(self, node: AVLNode) -> int:
        return node.height if node else 0

    def get_balance(self, node: AVLNode) -> int:
        return self.get_height(node.left) - self.get_height(node.right) if node else 0

    def right_rotate(self, z: AVLNode) -> AVLNode:
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y

    def left_rotate(self, z: AVLNode) -> AVLNode:
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y

    def insert(self, city: City):
        self.root = self._insert_recursive(self.root, city)

    def _insert_recursive(self, node: AVLNode, city: City) -> AVLNode:
        if not node:
            return AVLNode(city)

        if city.name < node.city.name:
            node.left = self._insert_recursive(node.left, city)
        else:
            node.right = self._insert_recursive(node.right, city)

        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))
        balance = self.get_balance(node)

        # Left Left Case
        if balance > 1 and city.name < node.left.city.name:
            return self.right_rotate(node)
        # Right Right Case
        if balance < -1 and city.name > node.right.city.name:
            return self.left_rotate(node)
        # Left Right Case
        if balance > 1 and city.name > node.left.city.name:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)
        # Right Left Case
        if balance < -1 and city.name < node.right.city.name:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        return node

    def search(self, name: str) -> City:
        current = self.root
        while current:
            if name == current.city.name:
                return current.city
            current = current.left if name < current.city.name else current.right
        return None

    def delete(self, name: str):
        self.root = self._delete_recursive(self.root, name)

    def _delete_recursive(self, node: AVLNode, name: str) -> AVLNode:
        if not node:
            return node

        if name < node.city.name:
            node.left = self._delete_recursive(node.left, name)
        elif name > node.city.name:
            node.right = self._delete_recursive(node.right, name)
        else:
            if not node.left:
                return node.right
            elif not node.right:
                return node.left

            successor = self._min_value_node(node.right)
            node.city = successor.city
            node.right = self._delete_recursive(node.right, successor.city.name)

        if not node:
            return node

        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))
        balance = self.get_balance(node)

        if balance > 1 and self.get_balance(node.left) >= 0:
            return self.right_rotate(node)
        if balance > 1 and self.get_balance(node.left) < 0:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)
        if balance < -1 and self.get_balance(node.right) <= 0:
            return self.left_rotate(node)
        if balance < -1 and self.get_balance(node.right) > 0:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        return node

    def _min_value_node(self, node: AVLNode) -> AVLNode:
        current = node
        while current.left:
            current = current.left
        return current


# =====================================================================
# 3. Min-Heap (Array-Based Priority Queue)
# =====================================================================
class MinHeap:
    def __init__(self):
        self.heap = []

    def insert(self, city: City):
        self.heap.append(city)
        self._heapify_up(len(self.heap) - 1)

    def extract_min(self) -> City:
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()

        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        return root

    def _heapify_up(self, index: int):
        parent = (index - 1) // 2
        while index > 0 and self.heap[index].distance < self.heap[parent].distance:
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            index = parent
            parent = (index - 1) // 2

    def _heapify_down(self, index: int):
        smallest = index
        left = 2 * index + 1
        right = 2 * index + 2
        size = len(self.heap)

        if left < size and self.heap[left].distance < self.heap[smallest].distance:
            smallest = left
        if right < size and self.heap[right].distance < self.heap[smallest].distance:
            smallest = right

        if smallest != index:
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            self._heapify_down(smallest)


# =====================================================================
# 4. Hash Table (Chaining Collision Handling)
# =====================================================================
class HashTable:
    def __init__(self, capacity: int = 10007):
        self.capacity = capacity
        self.table = [[] for _ in range(capacity)]

    def _hash(self, key: str) -> int:
        hash_val = 0
        for char in key:
            hash_val = (hash_val * 31 + ord(char)) % self.capacity
        return hash_val

    def insert(self, city: City):
        index = self._hash(city.name)
        for pair in self.table[index]:
            if pair[0] == city.name:
                pair[1] = city
                return
        self.table[index].append([city.name, city])

    def search(self, name: str) -> City:
        index = self._hash(name)
        for pair in self.table[index]:
            if pair[0] == name:
                return pair[1]
        return None

    def delete(self, name: str):
        index = self._hash(name)
        for i, pair in enumerate(self.table[index]):
            if pair[0] == name:
                del self.table[index][i]
                return True
        return False


# =====================================================================
# 5. Benchmarking & Data Plotting Suite
# =====================================================================
def run_benchmarks_and_plot():
    sizes = [100, 1000, 10000]

    # Target vectors to collect tracking data
    bst_times = []
    avl_times = []
    heap_times = []
    hash_times = []

    print("=" * 70)
    print("RUNNING EMPIRICAL PERFORMANCE TESTS (Times in Milliseconds)")
    print("=" * 70)

    for size in sizes:
        # Generate random mock datasets
        cities = [
            City(f"City_{i}", random.uniform(-90, 90), random.uniform(-180, 180),
                 random.randint(1000, 5000000), random.uniform(1.0, 5000.0))
            for i in range(size)
        ]

        # Shuffle input data to simulate typical insertion variability
        random.shuffle(cities)

        # 1. Benchmark BST
        bst = BST()
        start = time.perf_counter()
        for c in cities: bst.insert(c)
        bst_times.append((time.perf_counter() - start) * 1000)

        # 2. Benchmark AVL Tree
        avl = AVLTree()
        start = time.perf_counter()
        for c in cities: avl.insert(c)
        avl_times.append((time.perf_counter() - start) * 1000)

        # 3. Benchmark Min-Heap
        heap = MinHeap()
        start = time.perf_counter()
        for c in cities: heap.insert(c)
        heap_times.append((time.perf_counter() - start) * 1000)

        # 4. Benchmark Hash Table
        ht = HashTable(capacity=size * 2)  # Balance alpha load factor around ~0.5
        start = time.perf_counter()
        for c in cities: ht.insert(c)
        hash_times.append((time.perf_counter() - start) * 1000)

        print(f"\u2713 Benchmarked dataset size N = {size} successfully.")

    # --- TEXT METRIC DISPLAY FOR ASSIGNMENT DATA TABLES ---
    print("\n" + "=" * 65)
    print("EMPIRICAL TIMING RESULTS (Copy and paste into your report)")
    print("=" * 65)
    for idx, size in enumerate(sizes):
        print(f"\n--- DATASET SIZE: {size} NODES ---")
        print(f"{'Structure':<15} | {'Insertion Run-Time (ms)':<25}")
        print("-" * 45)
        print(f"BST             | {bst_times[idx]:<25.5f}")
        print(f"AVL Tree        | {avl_times[idx]:<25.5f}")
        print(f"Min-Heap        | {heap_times[idx]:<25.5f}")
        print(f"Hash Table      | {hash_times[idx]:<25.5f}")

    # --- GRAPH GENERATION USING MATPLOTLIB: 1 bar graph + 2 line graphs (linear + log scale) ---
    structures = ['BST', 'AVL Tree', 'Min-Heap', 'Hash Table']
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(19, 6))

    # Panel 1: Bar graph comparing all structures at the largest dataset size
    largest_idx = len(sizes) - 1
    values = [bst_times[largest_idx], avl_times[largest_idx], heap_times[largest_idx], hash_times[largest_idx]]
    bars = ax0.bar(structures, values, color=bar_colors)
    ax0.set_title(f'Task 1: Insertion Time at N = {sizes[largest_idx]}', fontsize=13, fontweight='bold', pad=12)
    ax0.set_ylabel('Total Insertion Execution Time (ms)', fontsize=11)
    ax0.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax0.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, values):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # Panels 2 & 3: Line graphs across dataset sizes, linear scale and log scale
    for ax, yscale, subtitle in [(ax1, 'linear', 'Linear Scale'), (ax2, 'log', 'Log Scale')]:
        ax.plot(sizes, bst_times, label='BST (Empirical)', marker='o', color='#1f77b4', linewidth=2)
        ax.plot(sizes, avl_times, label='AVL Tree (Empirical Balanced)', marker='s', color='#ff7f0e', linewidth=2)
        ax.plot(sizes, heap_times, label='Min-Heap (Empirical Priority)', marker='^', color='#2ca02c', linewidth=2)
        ax.plot(sizes, hash_times, label='Hash Table (Empirical Chained)', marker='d', color='#d62728', linewidth=2)
        ax.set_title(f'Task 1: Insertion Performance ({subtitle})', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Dataset Size (Number of Cities, N)', fontsize=11)
        ax.set_ylabel('Total Insertion Execution Time (ms)', fontsize=11)
        ax.set_xticks(sizes)
        ax.set_yscale(yscale)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()

    # Save chart layout image to local execution workspace folder
    graph_filename = 'task1_performance.png'
    plt.savefig(graph_filename, dpi=300)
    print(f"\n[SUCCESS] Chart visual artifact exported as '{graph_filename}'!")

    print("Chart saved (interactive display window skipped in this environment).")

if __name__ == "__main__":
    run_benchmarks_and_plot()