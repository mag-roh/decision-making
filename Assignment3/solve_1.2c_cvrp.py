import time
import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def read_instance(path="instance.txt"):
    lines = [ln.strip() for ln in open(path) if ln.strip()]
    Q = int(lines[0])
    dem = list(map(int, lines[1].split()))  # customers 1..n
    n = len(dem)

    C = []
    for r in range(n + 1):
        C.append(list(map(int, lines[2 + r].split())))
    C = np.array(C, dtype=int)

    q = np.array([0] + dem, dtype=int)  # depot demand = 0
    return Q, q, C


def extract_routes_from_x(x, n_customers):
    N = range(n_customers + 1)
    succ = {i: [] for i in N}

    for i in N:
        for j in N:
            if i != j and x[i, j].X > 0.5:
                succ[i].append(j)

    routes = []
    # depot has K outgoing arcs
    for first in succ[0]:
        rt = [0, first]
        cur = first
        while cur != 0:
            nxts = succ[cur]
            if not nxts:
                break
            cur = nxts[0]
            rt.append(cur)
        routes.append(rt)
    return routes


# -----------------------------
# Simulation as in 1.2(b), plus "most violating scenario"
# -----------------------------
def simulate_k_and_get_worst(routes, Q, q_nominal, k=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(q_nominal) - 1

    totalV = np.zeros(k, dtype=int)
    worst_q = None
    worst_V = -1

    # optional per-route violation counts
    route_viol_counts = np.zeros(len(routes), dtype=int)

    for t in range(k):
        q_tilde = np.zeros(n + 1, dtype=int)
        for i in range(1, n + 1):
            lo = int(math.floor(0.9 * q_nominal[i]))
            hi = int(math.ceil(1.1 * q_nominal[i]))
            q_tilde[i] = rng.integers(lo, hi + 1)

        V = 0
        for r_idx, r in enumerate(routes):
            cust = [i for i in r if i != 0]
            load = int(q_tilde[cust].sum())
            viol = max(0, load - Q)
            if viol > 0:
                route_viol_counts[r_idx] += 1
            V += viol

        totalV[t] = V
        if V > worst_V:
            worst_V = V
            worst_q = q_tilde.copy()

    numV = int(np.sum(totalV > 0))
    avgV = float(np.mean(totalV))
    maxV = int(np.max(totalV))

    return {
        "num_violating_samples": numV,
        "avg_total_violation": avgV,
        "max_total_violation": maxV,
        "route_violation_counts": route_viol_counts.tolist(),
        "worst_q": worst_q,          # full vector with depot at index 0
        "worst_V": int(worst_V),
    }


# -----------------------------
# Scenario-based two-index CVRP (shared x, scenario-specific u^s)
# -----------------------------
def solve_scenario_based_two_index(Q, C, scenarios, K=5, timelimit=600, output=1):
    """
    scenarios: list of demand vectors q^s, each length n+1 (q^s[0]=0)
    """
    n = len(scenarios[0]) - 1
    N = range(n + 1)
    CUST = range(1, n + 1)

    m = gp.Model("cvrp_scenario_based")
    m.Params.TimeLimit = timelimit
    m.Params.OutputFlag = output

    # shared routing decisions
    x = m.addVars(N, N, vtype=GRB.BINARY, name="x")

    # no self arcs
    for i in N:
        m.addConstr(x[i, i] == 0)

    # each customer: exactly one in and one out
    for i in CUST:
        m.addConstr(gp.quicksum(x[i, j] for j in N if j != i) == 1)
        m.addConstr(gp.quicksum(x[j, i] for j in N if j != i) == 1)

    # depot: K vehicles
    m.addConstr(gp.quicksum(x[0, j] for j in CUST) == K)
    m.addConstr(gp.quicksum(x[j, 0] for j in CUST) == K)

    # scenario-specific load variables + constraints
    # u[s, i] for each scenario s and customer i
    u = {}
    for s_idx, q_s in enumerate(scenarios):
        for i in CUST:
            u[s_idx, i] = m.addVar(lb=0.0, ub=float(Q), vtype=GRB.CONTINUOUS, name=f"u[{s_idx},{i}]")

        # bounds: u_i^s in [q_i^s, Q]
        for i in CUST:
            m.addConstr(u[s_idx, i] >= int(q_s[i]))
            m.addConstr(u[s_idx, i] <= Q)

        # MTZ-load constraints (capacity + subtour elimination) for scenario s
        for i in CUST:
            for j in CUST:
                if i != j:
                    # u_i^s - u_j^s + Q*x_ij <= Q - q_j^s
                    m.addConstr(u[s_idx, i] - u[s_idx, j] + Q * x[i, j] <= Q - int(q_s[j]))

    # objective: travel cost (same for all scenarios)
    m.setObjective(gp.quicksum(C[i, j] * x[i, j] for i in N for j in N), GRB.MINIMIZE)

    start = time.time()
    m.optimize()
    end = time.time()

    info = {
        "status": m.Status,
        "obj": m.ObjVal if m.SolCount > 0 else None,
        "runtime": end - start,
        "gap": m.MIPGap if m.SolCount > 0 and m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else None,
    }
    return m, x, info


# -----------------------------
# 1.2(c) Cutting-plane loop (5 iterations)
# -----------------------------
def run_cutting_plane(iterations=5, k=1000, seed=0, timelimit=600, K=5, output=1):
    Q, q_nom, C = read_instance("instance.txt")
    n = len(q_nom) - 1

    # scenario set S starts with only nominal q  :contentReference[oaicite:2]{index=2}
    S = [q_nom.copy()]

    # store results for reporting
    history = []

    for it in range(1, iterations + 1):
        print(f"\n================ 1.2(c) Iteration {it} ================")
        print(f"Scenario set size |S| = {len(S)}")

        # (1) solve scenario-based model with scenario set S  :contentReference[oaicite:3]{index=3}
        m, x, solve_info = solve_scenario_based_two_index(
            Q=Q, C=C, scenarios=S, K=K, timelimit=timelimit, output=output
        )

        print("Solve info:", solve_info)
        if solve_info["obj"] is None:
            print("No feasible solution found. Stopping.")
            break

        # extract routes from incumbent solution
        routes = extract_routes_from_x(x, n_customers=n)
        print("Routes:")
        for r in routes:
            print(r)

        # (2) simulate k=1000 realisations as in (b)  :contentReference[oaicite:4]{index=4}
        sim = simulate_k_and_get_worst(routes, Q, q_nom, k=k, seed=seed + it)

        print(f"Robustness (k={k}):")
        print(f"  #samples with violation (V>0): {sim['num_violating_samples']}/{k}")
        print(f"  avg total violation: {sim['avg_total_violation']:.3f}")
        print(f"  max total violation: {sim['max_total_violation']}")
        print(f"  route-wise violation counts: {sim['route_violation_counts']}")

        # (3) add the most violating scenario to S, if any  :contentReference[oaicite:5]{index=5}
        worst_V = sim["worst_V"]
        worst_q = sim["worst_q"]

        if worst_V > 0:
            # avoid adding duplicates
            duplicate = any(np.array_equal(worst_q, q_s) for q_s in S)
            print("Most violating scenario total violation:", worst_V)
            print("Most violating scenario demands (customers 1..n):")
            print(worst_q[1:].tolist())

            if not duplicate:
                S.append(worst_q)
                print("-> Added this scenario to S.")
            else:
                print("-> Worst scenario already in S; not adding.")
        else:
            print("No violations found in simulation; no scenario added.")

        # store per-iteration report values requested by assignment :contentReference[oaicite:6]{index=6}
        history.append({
            "iteration": it,
            "S_size": len(S),
            "solve_time": solve_info["runtime"],
            "gap": solve_info["gap"],
            "routing_cost": solve_info["obj"],
            "num_viol_samples": sim["num_violating_samples"],
            "avg_violation": sim["avg_total_violation"],
            "max_violation": sim["max_total_violation"],
            "worst_demands_customers": worst_q[1:].tolist() if worst_V > 0 else None,
        })


if __name__ == "__main__":
    # As allowed: 10 minute time limit per optimization model :contentReference[oaicite:7]{index=7}
    run_cutting_plane(
        iterations=5,
        k=1000,
        seed=0,
        timelimit=600,
        K=5,
        output=1   # set to 0 if you want it quiet
    )

