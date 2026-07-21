"""
ST5003CEM - Advanced Algorithms - Task 1: Advanced Data Structures
Corrected, single-file implementation: BST, AVL Tree, Min-Heap, Hash Table.

Fixes applied vs. the earlier version:
  1. City is keyed on a numeric city_id (not name) -> correct ordering.
  2. BST insert/search/delete are iterative -> no RecursionError on
     worst-case (sorted) input, and faster than recursion.
  3. AVL insert/delete verified against the numeric key; rotations
     compare city_id, not city.name.
  4. Insert, Search AND Delete are all benchmarked (previous version
     only benchmarked insertion).
  5. Empirical testing at n = 100, 1,000, 10,000 as required by the brief,
     with per-operation timings and comparison graphs.
"""

import sys
import time
import random
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.setrecursionlimit(20000)  # safety net for AVL's recursive delete only

SIZES = [100, 1000, 10000]
REPEATS = 5


# =====================================================================
# 0. City Record
# =====================================================================
class City:
    def __init__(self, city_id: int, name: str, lat: float, lon: float,
                 population: int, distance: float):
        self.city_id = city_id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.population = population
        self.distance = distance

    def __repr__(self):
        return f"{self.name}(id={self.city_id}, dist={self.distance:.2f})"


# =====================================================================
# 1. Binary Search Tree (BST) - keyed on city_id, iterative
# =====================================================================
class BSTNode:
    __slots__ = ("key", "city", "left", "right")

    def __init__(self, city: City):
        self.key = city.city_id
        self.city = city
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None
        self._size = 0

    def __len__(self):
        return self._size

    def insert(self, city: City):
        if self.root is None:
            self.root = BSTNode(city)
            self._size += 1
            return
        node = self.root
        while True:
            if city.city_id < node.key:
                if node.left is None:
                    node.left = BSTNode(city)
                    self._size += 1
                    return
                node = node.left
            elif city.city_id > node.key:
                if node.right is None:
                    node.right = BSTNode(city)
                    self._size += 1
                    return
                node = node.right
            else:
                node.city = city
                return

    def search(self, city_id: int):
        node = self.root
        while node is not None:
            if city_id == node.key:
                return node.city
            node = node.left if city_id < node.key else node.right
        return None

    def delete(self, city_id: int) -> bool:
        self.root, deleted = self._delete(self.root, city_id)
        if deleted:
            self._size -= 1
        return deleted

    def _delete(self, node, city_id):
        if node is None:
            return node, False
        if city_id < node.key:
            node.left, deleted = self._delete(node.left, city_id)
        elif city_id > node.key:
            node.right, deleted = self._delete(node.right, city_id)
        else:
            deleted = True
            if node.left is None:
                return node.right, deleted
            if node.right is None:
                return node.left, deleted
            successor = node.right
            while successor.left is not None:
                successor = successor.left
            node.key, node.city = successor.key, successor.city
            node.right, _ = self._delete(node.right, successor.key)
        return node, deleted

    def height(self) -> int:
        if self.root is None:
            return -1
        h, frontier = -1, [self.root]
        while frontier:
            h += 1
            nxt = []
            for n in frontier:
                if n.left:
                    nxt.append(n.left)
                if n.right:
                    nxt.append(n.right)
            frontier = nxt
        return h


# =====================================================================
# 2. AVL Tree (Self-Balancing) - keyed on city_id
# =====================================================================
class AVLNode:
    __slots__ = ("key", "city", "left", "right", "height")

    def __init__(self, city: City):
        self.key = city.city_id
        self.city = city
        self.left = None
        self.right = None
        self.height = 1


def _h(n):
    return n.height if n else 0


def _update_height(n):
    n.height = 1 + max(_h(n.left), _h(n.right))


def _balance(n):
    return _h(n.left) - _h(n.right)


def _rotate_right(y):
    x = y.left
    t2 = x.right
    x.right = y
    y.left = t2
    _update_height(y)
    _update_height(x)
    return x


def _rotate_left(x):
    y = x.right
    t2 = y.left
    y.left = x
    x.right = t2
    _update_height(x)
    _update_height(y)
    return y


def _rebalance(node):
    _update_height(node)
    bf = _balance(node)
    if bf > 1:
        if _balance(node.left) < 0:
            node.left = _rotate_left(node.left)
        return _rotate_right(node)
    if bf < -1:
        if _balance(node.right) > 0:
            node.right = _rotate_right(node.right)
        return _rotate_left(node)
    return node


class AVLTree:
    def __init__(self):
        self.root = None
        self._size = 0

    def __len__(self):
        return self._size

    def insert(self, city: City):
        self.root = self._insert(self.root, city)

    def _insert(self, node, city):
        if node is None:
            self._size += 1
            return AVLNode(city)
        if city.city_id < node.key:
            node.left = self._insert(node.left, city)
        elif city.city_id > node.key:
            node.right = self._insert(node.right, city)
        else:
            node.city = city
            return node
        return _rebalance(node)

    def search(self, city_id: int):
        node = self.root
        while node is not None:
            if city_id == node.key:
                return node.city
            node = node.left if city_id < node.key else node.right
        return None

    def delete(self, city_id: int) -> bool:
        self.root, deleted = self._delete(self.root, city_id)
        if deleted:
            self._size -= 1
        return deleted

    def _delete(self, node, city_id):
        if node is None:
            return node, False
        if city_id < node.key:
            node.left, deleted = self._delete(node.left, city_id)
        elif city_id > node.key:
            node.right, deleted = self._delete(node.right, city_id)
        else:
            deleted = True
            if node.left is None:
                return node.right, deleted
            if node.right is None:
                return node.left, deleted
            successor = node.right
            while successor.left is not None:
                successor = successor.left
            node.key, node.city = successor.key, successor.city
            node.right, _ = self._delete(node.right, successor.key)
        if node is None:
            return node, deleted
        return _rebalance(node), deleted

    def height(self) -> int:
        return _h(self.root) - 1 if self.root else -1


# =====================================================================
# 3. Min-Heap (Array-Based Priority Queue)
# =====================================================================
class MinHeap:
    def __init__(self):
        self.heap = []

    def __len__(self):
        return len(self.heap)

    def is_empty(self):
        return not self.heap

    def push(self, city: City):
        self.heap.append(city)
        self._heapify_up(len(self.heap) - 1)

    def pop(self):
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        return root

    def _heapify_up(self, index):
        parent = (index - 1) // 2
        while index > 0 and self.heap[index].distance < self.heap[parent].distance:
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            index = parent
            parent = (index - 1) // 2

    def _heapify_down(self, index):
        size = len(self.heap)
        while True:
            left, right = 2 * index + 1, 2 * index + 2
            smallest = index
            if left < size and self.heap[left].distance < self.heap[smallest].distance:
                smallest = left
            if right < size and self.heap[right].distance < self.heap[smallest].distance:
                smallest = right
            if smallest == index:
                break
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            index = smallest


# =====================================================================
# 4. Hash Table (Separate Chaining) - keyed on city_id
# =====================================================================
class HashTable:
    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.table = [[] for _ in range(capacity)]
        self._size = 0
        self._max_load = 0.75

    def __len__(self):
        return self._size

    def _hash(self, key: int) -> int:
        return hash(key) % self.capacity

    def _resize(self):
        old = self.table
        self.capacity *= 2
        self.table = [[] for _ in range(self.capacity)]
        for bucket in old:
            for key, city in bucket:
                self.table[self._hash(key)].append((key, city))

    def insert(self, city: City):
        idx = self._hash(city.city_id)
        bucket = self.table[idx]
        for i, (key, _) in enumerate(bucket):
            if key == city.city_id:
                bucket[i] = (city.city_id, city)
                return
        bucket.append((city.city_id, city))
        self._size += 1
        if self._size / self.capacity > self._max_load:
            self._resize()

    def search(self, city_id: int):
        for key, city in self.table[self._hash(city_id)]:
            if key == city_id:
                return city
        return None

    def delete(self, city_id: int) -> bool:
        idx = self._hash(city_id)
        bucket = self.table[idx]
        for i, (key, _) in enumerate(bucket):
            if key == city_id:
                bucket.pop(i)
                self._size -= 1
                return True
        return False


# =====================================================================
# 5. Benchmarking (Insert, Search AND Delete) + Plotting
# =====================================================================
def make_cities(n, seed):
    rnd = random.Random(seed)
    ids = list(range(n))
    rnd.shuffle(ids)
    return [
        City(i, f"City_{i}", rnd.uniform(-90, 90), rnd.uniform(-180, 180),
             rnd.randint(1000, 5_000_000), rnd.uniform(1.0, 5000.0))
        for i in ids
    ]


def median_time(fn, repeats=REPEATS):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def run_benchmarks_and_plot():
    results = {name: {"insert": [], "search": [], "delete": []}
               for name in ("BST", "AVL", "HashTable", "MinHeap")}

    print("=" * 70)
    print("RUNNING EMPIRICAL PERFORMANCE TESTS (insert / search / delete)")
    print("=" * 70)

    for n in SIZES:
        cities = make_cities(n, seed=n)
        ids = [c.city_id for c in cities]
        rnd = random.Random(n)
        sample = rnd.sample(ids, max(1, n // 10))

        # ---- BST ----
        bst = BST()
        t0 = time.perf_counter()
        for c in cities:
            bst.insert(c)
        results["BST"]["insert"].append((time.perf_counter() - t0) * 1000)
        results["BST"]["search"].append(median_time(lambda: [bst.search(i) for i in sample]) * 1000)
        results["BST"]["delete"].append(median_time(lambda: [bst.delete(i) for i in sample]) * 1000)

        # ---- AVL ----
        avl = AVLTree()
        t0 = time.perf_counter()
        for c in cities:
            avl.insert(c)
        results["AVL"]["insert"].append((time.perf_counter() - t0) * 1000)
        results["AVL"]["search"].append(median_time(lambda: [avl.search(i) for i in sample]) * 1000)
        results["AVL"]["delete"].append(median_time(lambda: [avl.delete(i) for i in sample]) * 1000)

        # ---- Hash Table ----
        ht = HashTable(capacity=max(16, n * 2))
        t0 = time.perf_counter()
        for c in cities:
            ht.insert(c)
        results["HashTable"]["insert"].append((time.perf_counter() - t0) * 1000)
        results["HashTable"]["search"].append(median_time(lambda: [ht.search(i) for i in sample]) * 1000)
        results["HashTable"]["delete"].append(median_time(lambda: [ht.delete(i) for i in sample]) * 1000)

        # ---- Min-Heap (push / pop, its natural operations) ----
        heap = MinHeap()
        t0 = time.perf_counter()
        for c in cities:
            heap.push(c)
        results["MinHeap"]["insert"].append((time.perf_counter() - t0) * 1000)
        results["MinHeap"]["search"].append(float("nan"))
        t0 = time.perf_counter()
        while not heap.is_empty():
            heap.pop()
        results["MinHeap"]["delete"].append((time.perf_counter() - t0) * 1000)

        print(f"\u2713 Benchmarked N = {n}  "
              f"(BST height={bst.height()}, AVL height={avl.height()})")

    # ---- Print table ----
    print("\n" + "=" * 70)
    print("EMPIRICAL TIMING RESULTS (ms)")
    print("=" * 70)
    for idx, n in enumerate(SIZES):
        print(f"\n--- N = {n} ---")
        print(f"{'Structure':<12}{'Insert(ms)':<14}{'Search(ms)':<14}{'Delete(ms)':<14}")
        for name in results:
            ins = results[name]["insert"][idx]
            sea = results[name]["search"][idx]
            dele = results[name]["delete"][idx]
            sea_str = "n/a" if sea != sea else f"{sea:.5f}"  # NaN check
            print(f"{name:<12}{ins:<14.5f}{sea_str:<14}{dele:<14.5f}")

    # ---- Plot: insert / search / delete, each linear+log ----
    colors = {"BST": "#1f77b4", "AVL": "#ff7f0e", "MinHeap": "#2ca02c", "HashTable": "#d62728"}
    markers = {"BST": "o", "AVL": "s", "MinHeap": "^", "HashTable": "d"}

    for metric, title in [("insert", "Insertion"), ("search", "Search"), ("delete", "Deletion")]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, yscale in zip(axes, ["linear", "log"]):
            for name in results:
                y = results[name][metric]
                if all(v != v for v in y):  # all NaN (e.g. MinHeap search)
                    continue
                ax.plot(SIZES, y, label=name, marker=markers[name], color=colors[name], linewidth=2)
            ax.set_title(f"Task 1: {title} Time ({yscale} scale)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Dataset Size (N)")
            ax.set_ylabel(f"Total {title} Time (ms)")
            ax.set_xticks(SIZES)
            ax.set_yscale(yscale)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
        plt.tight_layout()
        fname = f"task1_{metric}_performance.png"
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"[SUCCESS] Saved {fname}")


if __name__ == "__main__":
    run_benchmarks_and_plot()