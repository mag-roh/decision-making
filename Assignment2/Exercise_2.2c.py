import pandas as pd
import time
from gurobipy import Model, GRB, quicksum

U = ["PL3", "PL4"]

F = {"PL3": 315000, "PL4": 385000}   # €/year
C = {"PL3": 400, "PL4": 600}         # seats
L = {"PL3": 80,  "PL4": 110}         # meters

# Seat demand per (Line, Direction)
D_DEMAND = {
    (800,  "north"): 1235,
    (800,  "south"): 810,
    (3000, "north"): 1055,
    (3000, "south"): 890,
    (3100, "north"): 850,
    (3100, "south"): 780,
    (3500, "north"): 900,
    (3500, "south"): 670,
    (3900, "north"): 750,
}

P = {
    "PL3":      {"PL3": 1, "PL4": 0},
    "PL4":      {"PL3": 0, "PL4": 1},
    "PL3-PL3":  {"PL3": 2, "PL4": 0},
    "PL3-PL4":  {"PL3": 1, "PL4": 1},
    "PL4-PL4":  {"PL3": 0, "PL4": 2},
    "PL3-PL3-PL3": {"PL3": 3, "PL4": 0},
    "PL3-PL3-PL4": {"PL3": 2, "PL4": 1},
    "PL3-PL4-PL4": {"PL3": 1, "PL4": 2},
}

def seats(p):  return sum(C[u] * P[p][u] for u in U)
def length(p): return sum(L[u] * P[p][u] for u in U)

T = [
    ("800",  "north", 300),
    ("800",  "south", 300),
    ("3000", "south", 300),
    ("3100", "north", 300),
    ("3500", "north", 300),
    ("3500", "south", 300),
    ("3900", "north", 200),
]

def solve_composition_model():
    start = time.time()

    m = Model("composition_model")

    X = m.addVars(range(len(T)), P.keys(), vtype=GRB.BINARY, name="X")
    n = m.addVars(U, vtype=GRB.INTEGER, lb=0, name="n")

    # Objective
    m.setObjective(quicksum(F[u] * n[u] for u in U), GRB.MINIMIZE)

    # One composition per train
    for t in range(len(T)):
        m.addConstr(quicksum(X[t, p] for p in P) == 1)

    # Seat demand
    for t, (line, direction, _) in enumerate(T):
        D_t = D_DEMAND[(int(line), direction)]
        m.addConstr(quicksum(seats(p) * X[t, p] for p in P) >= D_t)

    # Length limits
    for t, (_, _, Lmax) in enumerate(T):
        m.addConstr(quicksum(length(p) * X[t, p] for p in P) <= Lmax)

    # Fleet size
    for u in U:
        m.addConstr(
            quicksum(P[p][u] * X[t, p] for t in range(len(T)) for p in P)
            <= n[u]
        )

    # Manufacturer balance
    m.addConstr(n["PL3"] <= 1.25 * n["PL4"])
    m.addConstr(n["PL4"] <= 1.25 * n["PL3"])

    m.optimize()
    runtime = time.time() - start

    return m, X, n, runtime

model, X, n, runtime = solve_composition_model()

print("\n================= COMPOSITION FORMULATION =================")
print(f"Runtime: {runtime:.4f} seconds")
print(f"Optimal cost: {model.objVal:,.0f} €")
print(f"Fleet: PL3 = {int(n['PL3'].X)}, PL4 = {int(n['PL4'].X)}")

print("\nAssigned compositions per cross-section train:")
for t, (line, direction, _) in enumerate(T):
    for p in P:
        if X[t, p].X > 0.5:
            print(f"Train {line} {direction}: {p}")

