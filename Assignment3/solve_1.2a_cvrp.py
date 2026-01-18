import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# -----------------------------
# Read instance.txt (Assignment 3)
# -----------------------------
def read_instance(path="instance.txt"):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    Q = int(lines[0])  # capacity
    q = list(map(int, lines[1].split()))  # demands for customers 1..n
    n = len(q)

    # Next n+1 lines: distance matrix rows for depot(0), customer1..customern
    C = []
    for r in range(n + 1):
        row = list(map(int, lines[2 + r].split()))
        C.append(row)
    C = np.array(C, dtype=int)

    return Q, np.array([0] + q, dtype=int), C  # q[0]=0 for depot


# -----------------------------
# Two-index CVRP (MTZ-load style)
# -----------------------------
def solve_cvrp_two_index(Q, q, C, K=5, timelimit=600, output=1):
    """
    Nodes: 0..n (0 is depot)
    q[i] demand, q[0]=0
    C[i,j] distance
    K vehicles
    """
    n = len(q) - 1  # number of customers
    N = range(n + 1)      # 0..n
    CUST = range(1, n + 1)

    m = gp.Model("cvrp_two_index")
    m.Params.TimeLimit = timelimit
    m.Params.OutputFlag = output

    x = m.addVars(N, N, vtype=GRB.BINARY, name="x")

    # no self arcs
    for i in N:
        m.addConstr(x[i, i] == 0)

    # each customer has exactly one in and one out
    for i in CUST:
        m.addConstr(gp.quicksum(x[i, j] for j in N if j != i) == 1)
        m.addConstr(gp.quicksum(x[j, i] for j in N if j != i) == 1)

    # depot degree equals number of vehicles (K routes)
    m.addConstr(gp.quicksum(x[0, j] for j in CUST) == K)
    m.addConstr(gp.quicksum(x[j, 0] for j in CUST) == K)

    # MTZ-load variables (only meaningful for customers)
    u = m.addVars(CUST, lb=0.0, ub=float(Q), vtype=GRB.CONTINUOUS, name="u")

    # bounds: if customer visited (always), load at i at least demand and at most Q
    for i in CUST:
        m.addConstr(u[i] >= q[i])
        m.addConstr(u[i] <= Q)

    # capacity + subtour elimination (MTZ-load)
    # For i != j, both customers:
    # u[i] - u[j] + Q * x[i,j] <= Q - q[j]
    for i in CUST:
        for j in CUST:
            if i != j:
                m.addConstr(u[i] - u[j] + Q * x[i, j] <= Q - q[j])

    # objective
    m.setObjective(gp.quicksum(C[i, j] * x[i, j] for i in N for j in N), GRB.MINIMIZE)

    start = time.time()
    m.optimize()
    end = time.time()

    return m, x, {
        "status": m.Status,
        "obj": m.ObjVal if m.SolCount > 0 else None,
        "runtime": end - start,
        "gap": m.MIPGap if m.SolCount > 0 and m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else None
    }


# -----------------------------
# Extract routes from x (for part b/c/d/e)
# -----------------------------
def extract_routes(x, n_customers):
    """
    x is Gurobi tupledict on nodes 0..n
    returns list of routes, each like [0, ..., 0]
    """
    N = range(n_customers + 1)
    succ = {}
    for i in N:
        for j in N:
            if i != j and x[i, j].X > 0.5:
                succ[i] = succ.get(i, []) + [j]

    routes = []
    # depot has K outgoing arcs
    for first in succ.get(0, []):
        rt = [0, first]
        cur = first
        while cur != 0:
            nxts = succ.get(cur, [])
            if not nxts:
                break
            cur = nxts[0]
            rt.append(cur)
        routes.append(rt)
    return routes


if __name__ == "__main__":
    Q, q, C = read_instance("instance.txt")
    m, x, info = solve_cvrp_two_index(Q, q, C, K=5, timelimit=600, output=1)

    print("\n=== 1.2(a) Two-index CVRP ===")
    print(info)

    if info["obj"] is not None:
        routes = extract_routes(x, n_customers=len(q) - 1)
        print("\nRoutes:")
        for r in routes:
            print(r)

