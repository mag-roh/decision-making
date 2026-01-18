import numpy as np
import math

ROUTES = [
    [0, 1, 11, 22, 7, 14, 25, 0],
    [0, 8, 10, 6, 17, 4, 23, 0],
    [0, 9, 0],
    [0, 15, 18, 19, 12, 16, 20, 0],
    [0, 21, 5, 3, 24, 2, 13, 0],
]

def read_instance(path="instance.txt"):
    lines = [ln.strip() for ln in open(path) if ln.strip()]
    Q = int(lines[0])
    dem = list(map(int, lines[1].split()))  
    q = np.array([0] + dem, dtype=int)      
    return Q, q

def simulate_12b(routes, Q, q_nominal, k=1000, seed=0):
    """
    Implements 1.2(b):
    - sample integer demands q_tilde_i in [0.9 q_i, 1.1 q_i]
    - compute per-route violation max(0, load - Q)
    - total violation is sum of route violations
    """
    rng = np.random.default_rng(seed)
    n = len(q_nominal) - 1  

    totalV = np.zeros(k, dtype=int)

    for t in range(k):
        # sample demands
        q_tilde = np.zeros(n + 1, dtype=int)
        for i in range(1, n + 1):
            lo = int(math.floor(0.9 * q_nominal[i]))
            hi = int(math.ceil(1.1 * q_nominal[i]))
            q_tilde[i] = rng.integers(lo, hi + 1)  

        V = 0
        for r in routes:
            cust = [i for i in r if i != 0]
            load = int(q_tilde[cust].sum())
            V += max(0, load - Q)
        totalV[t] = V

    num_violations = int(np.sum(totalV > 0))
    avg_violation = float(np.mean(totalV))
    max_violation = int(np.max(totalV))

    return num_violations, avg_violation, max_violation

if __name__ == "__main__":
    Q, q = read_instance("instance.txt")
    k = 1000
    numV, avgV, maxV = simulate_12b(ROUTES, Q, q, k=k, seed=0)

    print("===== 1.2(b) Robustness Simulation =====")
    print(f"Capacity Q = {Q}")
    print(f"k = {k}")
    print(f"Number of samples with violation (V>0) : {numV}/{k}")
    print(f"Average total violation      : {avgV:.3f}")
    print(f"Maximum total violation      : {maxV}")

