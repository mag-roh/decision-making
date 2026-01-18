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

def route_violation(route, q_tilde, Q):
    cust = [i for i in route if i != 0]
    load = int(q_tilde[cust].sum())
    return max(0, load - Q), load

def apply_refill_recourse(route, q_tilde, Q, C, verbose=False):
    """
    Order-preserving refill recourse.
    Returns: (base_cost, recourse_cost, extra_cost)
    """
    base_cost = 0
    for a, b in zip(route[:-1], route[1:]):
        base_cost += int(C[a, b])

    extra = 0
    cap = Q

    if verbose:
        print("Route:", route)
        print(f"Start capacity = {Q}")

    for node in route[1:]:  # skip starting depot
        if node == 0:
            if verbose:
                print("Return to depot. Route finished.")
            break

        demand = int(q_tilde[node])
        if verbose:
            print(f"\nArrive at customer {node}: demand={demand}, cap_left={cap}")

        if demand <= cap:
            cap -= demand
            if verbose:
                print(f"Serve fully. New cap_left={cap}")
            continue

        rem = demand
        served = cap
        rem -= served
        if verbose:
            print(f"Serve {served} (partial). Remaining={rem}. Need refill trips.")
        cap = 0

        # refill until remaining is served
        while rem > 0:
            # detour i -> depot -> i
            extra += int(C[node, 0]) + int(C[0, node])
            cap = Q
            if verbose:
                print(f"Refill trip: +({C[node,0]} + {C[0,node]}) distance. cap_left reset to {Q}.")

            take = min(cap, rem)
            rem -= take
            cap -= take
            if verbose:
                print(f"Serve {take} after refill. Remaining={rem}. cap_left={cap}")

    recourse_cost = base_cost + extra
    return base_cost, recourse_cost, extra

if __name__ == "__main__":
    Q, q_nom, C = read_instance("instance.txt")
    rng = np.random.default_rng(0)

    # find one violating scenario + route
    for attempt in range(1, 100000):
        q_tilde = sample_demands(q_nom, rng)

        viols = []
        for r_idx, r in enumerate(ROUTES):
            v, load = route_violation(r, q_tilde, Q)
            if v > 0:
                viols.append((v, load, r_idx, r))

        if viols:
            viols.sort(reverse=True, key=lambda x: x[0])
            v, load, r_idx, r = viols[0]
            print("Found a violating scenario!")
            print("q_tilde (customers 1..n):", q_tilde[1:].tolist())
            print(f"Most violating route index={r_idx+1}, load={load}, violation={v}")
            print("\n--- Recourse trace (order-preserving refill) ---")
            base, rec, extra = apply_refill_recourse(r, q_tilde, Q, C, verbose=True)
            print("\nBase route cost     :", base)
            print("Extra recourse cost :", extra)
            print("Recourse total cost :", rec)
            break

