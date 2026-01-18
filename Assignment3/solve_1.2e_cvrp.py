import numpy as np
import math

def read_instance(path="instance.txt"):
    lines = [ln.strip() for ln in open(path) if ln.strip()]
    Q = int(lines[0])
    dem = list(map(int, lines[1].split()))
    n = len(dem)

    C = []
    for r in range(n + 1):
        C.append(list(map(int, lines[2 + r].split())))
    C = np.array(C, dtype=int)

    q = np.array([0] + dem, dtype=int)
    return Q, q, C

def sample_demands(q_nominal, rng):
    n = len(q_nominal) - 1
    q_tilde = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        lo = int(math.floor(0.9 * q_nominal[i]))
        hi = int(math.ceil(1.1 * q_nominal[i]))
        q_tilde[i] = rng.integers(lo, hi + 1)
    return q_tilde

def apply_refill_recourse(route, q_tilde, Q, C):
    base_cost = 0
    for a, b in zip(route[:-1], route[1:]):
        base_cost += int(C[a, b])

    extra = 0
    cap = Q

    for node in route[1:]:
        if node == 0:
            break
        demand = int(q_tilde[node])

        if demand <= cap:
            cap -= demand
            continue

        rem = demand - cap
        cap = 0

        while rem > 0:
            extra += int(C[node, 0]) + int(C[0, node])
            cap = Q
            take = min(cap, rem)
            rem -= take
            cap -= take

    return base_cost, base_cost + extra, extra

def simulate_solution(routes, Q, q_nom, C, k=1000, seed=0):
    rng = np.random.default_rng(seed)
    total_base = []
    total_recourse = []
    total_extra = []
    any_violation = 0

    for _ in range(k):
        q_tilde = sample_demands(q_nom, rng)

        base_sum = 0
        rec_sum = 0
        extra_sum = 0

        for r in routes:
            base, rec, extra = apply_refill_recourse(r, q_tilde, Q, C)
            base_sum += base
            rec_sum += rec
            extra_sum += extra

        if extra_sum > 0:
            any_violation += 1

        total_base.append(base_sum)
        total_recourse.append(rec_sum)
        total_extra.append(extra_sum)

    return {
            "viol_samples": any_violation,
            "avg_base": float(np.mean(total_base)),
            "avg_recourse": float(np.mean(total_recourse)),
            "avg_extra": float(np.mean(total_extra)),
            "max_extra": int(np.max(total_extra)),
            "max_recourse": float(np.max(total_recourse)),
            }

if __name__ == "__main__":
    Q, q_nom, C = read_instance("instance.txt")

    SOLUTIONS = {
            1: [
                [0, 1, 11, 22, 7, 14, 25, 0],
                [0, 8, 10, 6, 17, 4, 23, 0],
                [0, 9, 0],
                [0, 15, 18, 19, 12, 16, 20, 0],
                [0, 21, 5, 3, 24, 2, 13, 0],
                ],
            2: [ [0, 11, 22, 7, 14, 1, 0],
                [0, 13, 18, 19, 12, 16, 15, 0],
                [0, 21, 2, 24, 3, 5, 0], 
                [0, 23, 8, 9, 0], 
                [0, 25, 10, 20, 17, 4, 6, 0]],
            3: [  [0, 1, 14, 7, 22, 11, 0],
                [0, 4, 17, 20, 15, 10, 25, 0], 
                [0, 9, 8, 23, 0],
                [0, 13, 18, 19, 12, 16, 6, 0],
                [0, 21, 2, 24, 3, 5, 0]],
            4: [  [0, 1, 14, 7, 22, 11, 0],
                [0, 5, 3, 24, 2, 21, 0],
                [0, 9, 25, 10, 8, 0],
                [0, 18, 19, 12, 16, 15, 0],
                [0, 23, 6, 4, 17, 20, 13, 0]],
            5: [  [0, 1, 14, 7, 22, 11, 0],
                [0, 5, 3, 24, 2, 21, 0],
                [0, 9, 25, 10, 8, 0],
                [0, 18, 19, 12, 16, 15, 0],
                [0, 23, 6, 4, 17, 20, 13, 0]],
    }

    k = 1000
    print("===== 1.2(e) Recourse simulation (k=1000) =====")
    print(f"Capacity Q = {Q}")

    for it, routes in SOLUTIONS.items():
        res = simulate_solution(routes, Q, q_nom, C, k=k, seed=100 + it)
        print(f"\n--- Iteration {it} solution ---")
        print(f"Scenarios needing recourse : {res['viol_samples']}/{k}")
        print(f"Avg base cost             : {res['avg_base']:.2f}")
        print(f"Avg recourse cost         : {res['avg_recourse']:.2f}")
        print(f"Avg extra cost            : {res['avg_extra']:.3f}")
        print(f"Max extra cost            : {res['max_extra']}")
        print(f"Max recourse total cost   : {res['max_recourse']:.2f}")

