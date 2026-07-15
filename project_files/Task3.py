
import time
import random
import matplotlib.pyplot as plt

# =========================
# PART 1 - Dynamic Programming
# Matrix Chain Multiplication
# =========================
def matrix_chain_order(dimensions):
    n = len(dimensions)
    dp = [[0] * n for _ in range(n)]
    for L in range(2, n):
        for i in range(1, n - L + 1):
            j = i + L - 1
            dp[i][j] = float("inf")
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dimensions[i-1]*dimensions[k]*dimensions[j]
                if cost < dp[i][j]:
                    dp[i][j] = cost
    return dp[1][n-1]

# =========================
# PART 2 - Greedy
# Minimum Number of Platforms
# =========================
def minimum_platforms(arrival, departure):
    arrival = sorted(arrival)
    departure = sorted(departure)
    i = 1
    j = 0
    plat = 1
    ans = 1
    while i < len(arrival) and j < len(departure):
        if arrival[i] <= departure[j]:
            plat += 1
            ans = max(ans, plat)
            i += 1
        else:
            plat -= 1
            j += 1
    return ans

# =========================
# PART 3 - Backtracking
# Hamiltonian Cycle
# =========================
def hamiltonian_cycle(graph):
    V = len(graph)
    path = [-1] * V
    path[0] = 0

    def safe(v, pos):
        return graph[path[pos-1]][v] and v not in path

    def solve(pos):
        if pos == V:
            return graph[path[-1]][path[0]] == 1
        for v in range(1, V):
            if safe(v, pos):
                path[pos] = v
                if solve(pos + 1):
                    return True
                path[pos] = -1
        return False

    return path + [0] if solve(1) else None

print("=== PART 1 ===")
dims = [10,20,30,40,30]
print("Minimum multiplications:", matrix_chain_order(dims))

print("\n=== PART 2 ===")
arr = [900,940,950,1100,1500,1800]
dep = [910,1200,1120,1130,1900,2000]
print("Minimum platforms:", minimum_platforms(arr, dep))

print("\n=== PART 3 ===")
g = [
[0,1,0,1,0],
[1,0,1,1,1],
[0,1,0,0,1],
[1,1,0,0,1],
[0,1,1,1,0]
]
print("Hamiltonian Cycle:", hamiltonian_cycle(g))

# Graph 1
sizes=[5,10,15,20,25]
times=[]
for s in sizes:
    d=[random.randint(5,50) for _ in range(s)]
    t=time.perf_counter()
    matrix_chain_order(d)
    times.append((time.perf_counter()-t)*1000)
plt.figure()
plt.plot(sizes,times,marker='o')
plt.title("Dynamic Programming")
plt.xlabel("Matrices")
plt.ylabel("Time (ms)")
plt.grid(True)

# Graph 2
sizes2=[100,200,300,400,500]
times2=[]
for s in sizes2:
    a=sorted(random.sample(range(900,900+s*5),s))
    d=[x+random.randint(1,50) for x in a]
    t=time.perf_counter()
    minimum_platforms(a,d)
    times2.append((time.perf_counter()-t)*1000)
plt.figure()
plt.plot(sizes2,times2,marker='o')
plt.title("Greedy Algorithm")
plt.xlabel("Number of Trains")
plt.ylabel("Time (ms)")
plt.grid(True)

# Graph 3
sizes3=[4,5,6,7]
times3=[]
for n in sizes3:
    cg=[[1 if i!=j else 0 for j in range(n)] for i in range(n)]
    t=time.perf_counter()
    hamiltonian_cycle(cg)
    times3.append((time.perf_counter()-t)*1000)
plt.figure()
plt.plot(sizes3,times3,marker='o')
plt.title("Backtracking")
plt.xlabel("Vertices")
plt.ylabel("Time (ms)")
plt.grid(True)

plt.show()
