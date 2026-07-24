# Advanced Algorithms Coursework (ST5003CEM)

This repository contains the full Python implementation and benchmark results for five tasks covering efficient data structures, graph algorithms, algorithmic design strategies, NP-Hard heuristics, and concurrent programming.

## Repository Structure

```
.
└── project_files/
    ├── Task1.py                              # Efficient Data Structures
    ├── Task2.py                              # Graph Algorithms and Pathfinding
    ├── Task3.py                              # Algorithmic Strategies (DP, Greedy, Backtracking)
    ├── Task4.py                              # NP-Hard Problem and Heuristics
    ├── Task5.py                              # Concurrent Programming
    │
    ├── task1_delete_performance.png          # Task1 – deletion benchmark
    ├── task1_insert_performance.png          # Task1 – insertion benchmark
    ├── task1_search_performance.png          # Task1 – search benchmark
    ├── task2_comprehensive_benchmark.png     # Task2 – graph algorithm comparison
    ├── task4_mdbpp_tripanel_evaluation.png   # Task4 – multi-dimensional bin packing evaluation
    └── task5_speedup_dashboard.png           # Task5 – threading speedup dashboard
```


## Requirements

- Python 3.10+
- Optional, for benchmarking/plotting utilities:
  - `matplotlib`
  - `numpy`

Install optional dependencies:

```bash
pip install matplotlib numpy
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Paridhilamichhane12/Advanced-Algorithm-Assignment.git
cd Advanced-Algorithm-Assignment/project_files
```

### Run a task

Each task file can be run directly:

```bash
python Task1.py
python Task2.py
python Task3.py
python Task4.py
python Task5.py
```

Running each script regenerates its benchmark plots (the PNGs in this folder).

