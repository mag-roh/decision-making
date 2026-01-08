import math
import gurobipy as gp
from gurobipy import GRB


def read_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        capacity = int(f.readline().split()[0])
        demands = [int(x) for x in f.readline().split()]
    return capacity, demands


def read_routes(path, n_customers):
    distances = []
    customers_in_route = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = [int(x) for x in line.split()]
            dist = parts[0]
            nodes = parts[2:]
            cust = set(v for v in nodes if v != 0 and 1 <= v <= n_customers)

            distances.append(dist)
            customers_in_route.append(cust)

    return distances, customers_in_route


def fmt_num(x, digits=4):
    if x is None:
        return "nan"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


def main():
    vehicles = 5
    eps = 0.05
    z_star = 8831

    capacity, demands = read_instance("instance.txt")
    n = len(demands)

    c, route_customers = read_routes("routes.txt", n)
    p = c
    R = range(len(c))

    a = [[] for _ in range(n + 1)]
    for r in R:
        for i in route_customers[r]:
            a[i].append(r)

    for i in range(1, n + 1):
        if not a[i]:
            raise ValueError(f"Customer {i} is not covered by any route")

    model = gp.Model("cvrp_range_vehicle_index")
    model.Params.OutputFlag = 0

    K = range(vehicles)
    R = range(len(c))
    x = model.addVars(vehicles, len(c), vtype=GRB.BINARY, name="x")
    gamma = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="gamma")
    eta = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="eta")

    for i in range(1, n + 1):
        model.addConstr(gp.quicksum(x[k, r] for k in K for r in a[i]) == 1)

    for k in K:
        model.addConstr(gp.quicksum(x[k, r] for r in R) == 1)

    model.addConstr(
        gp.quicksum(c[r] * x[k, r] for k in K for r in R) <= (1.0 + eps) * z_star
    )

    for k in K:
        payoff_k = gp.quicksum(p[r] * x[k, r] for r in R)
        model.addConstr(gamma <= payoff_k)
        model.addConstr(payoff_k <= eta)

    model.setObjective(eta - gamma, GRB.MINIMIZE)
    model.optimize()

    runtime = float(getattr(model, "Runtime", float("nan")))
    gap = float(getattr(model, "MIPGap", float("nan")))
    cost = float("nan")
    route_range = float("nan")

    if model.SolCount > 0:
        cost = 0.0
        chosen_dist = []
        for k in K:
            for r in R:
                if x[k, r].X > 0.5:
                    cost += c[r]
                    chosen_dist.append(p[r])
                    break
        if chosen_dist:
            route_range = float(max(chosen_dist) - min(chosen_dist))

    # One final line with the required metrics
    print(
        "runtime_s="
        + fmt_num(runtime, 4)
        + " gap_pct="
        + ("nan" if math.isnan(gap) else f"{100.0 * gap:.4f}")
        + " cost="
        + ("nan" if math.isnan(cost) else f"{cost:.0f}")
        + " range="
        + ("nan" if math.isnan(route_range) else f"{route_range:.0f}")
    )


if __name__ == "__main__":
    main()
