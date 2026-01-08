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

    capacity, demands = read_instance("instance.txt")
    n = len(demands)

    distances, custs = read_routes("routes.txt", n)
    R = range(len(distances))

    cover = [[] for _ in range(n + 1)]
    for r in R:
        for i in custs[r]:
            cover[i].append(r)

    for i in range(1, n + 1):
        if not cover[i]:
            raise ValueError(f"Customer {i} is not covered by any route")

    model = gp.Model("cvrp_route_based")
    model.Params.OutputFlag = 0

    x = model.addVars(len(distances), vtype=GRB.BINARY, name="x")

    for i in range(1, n + 1):
        model.addConstr(gp.quicksum(x[r] for r in cover[i]) == 1)

    model.addConstr(gp.quicksum(x[r] for r in R) <= vehicles)
    model.setObjective(gp.quicksum(distances[r] * x[r] for r in R), GRB.MINIMIZE)

    model.optimize()

    runtime = float(getattr(model, "Runtime", float("nan")))
    gap = float(getattr(model, "MIPGap", float("nan")))
    cost = float("nan")
    route_range = float("nan")

    if model.SolCount > 0:
        cost = float(model.ObjVal)
        chosen = [r for r in R if x[r].X > 0.5]
        chosen_dist = [distances[r] for r in chosen]
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
