import pandas as pd
from gurobipy import Model, GRB, quicksum
import time

XLSX_FILE = "a2_part2.xlsx"

U = ["PL3", "PL4"]

F = {"PL3": 315000, "PL4": 385000}   # €/year
C = {"PL3": 400, "PL4": 600}         # seats
L = {"PL3": 80,  "PL4": 110}         # meters

D_DEMAND = {
    (800,  "south"): 810,
    (800,  "north"): 1235,
    (3000, "south"): 890,
    (3000, "north"): 1055,
    (3100, "south"): 780,
    (3100, "north"): 850,
    (3500, "south"): 670,
    (3500, "north"): 900,
    (3900, "south"): 540,
    (3900, "north"): 750,
}

PERIOD = 30
START_TIME = 300
END_TIME = 1440
TARGET_TIME = 8 * 60

# Length limits
L_MAX_DEFAULT = 300
L_MAX_3900 = 200


def normalize_direction(d):
    return str(d).strip().lower()


def minutes_to_hhmm(m):
    return f"{m//60:02d}:{m%60:02d}"


def build_cross_section():
    df = pd.read_excel(XLSX_FILE, sheet_name="Timetable")
    df["Type_norm"] = df["Type"].astype(str).str.strip().str.lower()
    df["Dir_norm"] = df["Direction"].apply(normalize_direction)

    patterns = []

    for (line, direction), g in df.groupby(["Line", "Dir_norm"]):
        dep = g[g["Type_norm"] == "dep"]["Time"].min()
        arr_times = g[g["Type_norm"] == "arr"]["Time"]

        duration = max((t - dep) % PERIOD for t in arr_times)
        if duration == 0:
            duration = PERIOD

        patterns.append({
            "Line": int(line),
            "Direction": direction,
            "Dep0": dep,
            "Arr0": dep + duration
        })

    patt = pd.DataFrame(patterns)

    cross = []
    for _, r in patt.iterrows():
        k = 0
        while True:
            dep = r.Dep0 + k * PERIOD
            arr = r.Arr0 + k * PERIOD
            if dep >= END_TIME:
                break
            if dep >= START_TIME and dep <= TARGET_TIME <= arr:
                cross.append({
                    "Line": r.Line,
                    "Direction": r.Direction,
                    "Departure": dep,
                    "Arrival": arr
                })
            k += 1

    Tdf = pd.DataFrame(cross).drop_duplicates().reset_index(drop=True)
    Tdf["t"] = range(len(Tdf))
    return Tdf

def main():
    Tdf = build_cross_section()
    T = Tdf.t.tolist()

    # Demand and length limits
    D = {}
    Lmax = {}

    for r in Tdf.itertuples():
        key = (r.Line, r.Direction)
        D[r.t] = D_DEMAND[key]
        Lmax[r.t] = L_MAX_3900 if r.Line == 3900 else L_MAX_DEFAULT
    m = Model("Exercise_2_1c")

    N = m.addVars(U, T, vtype=GRB.INTEGER, lb=0, name="N")
    n = m.addVars(U, vtype=GRB.INTEGER, lb=0, name="n")

    # Objective
    m.setObjective(quicksum(F[u] * n[u] for u in U), GRB.MINIMIZE)

    # Fleet-size constraints
    for u in U:
        m.addConstr(quicksum(N[u, t] for t in T) <= n[u])

    # Capacity constraints
    for t in T:
        m.addConstr(quicksum(C[u] * N[u, t] for u in U) >= D[t])

    # Length constraints
    for t in T:
        m.addConstr(quicksum(L[u] * N[u, t] for u in U) <= Lmax[t])

    # Manufacturer balance
    m.addConstr(n["PL3"] <= 1.25 * n["PL4"])
    m.addConstr(n["PL4"] <= 1.25 * n["PL3"])

    # Solve
    start_time = time.time()
    m.optimize()
    end_time = time.time()
    runtime = end_time - start_time

    print("\n================ OPTIMAL SOLUTION =================")
    print(f"Optimal yearly cost: {m.objVal:,.0f} €")
    print(f"Buy PL3 units: {int(n['PL3'].X)}")
    print(f"Buy PL4 units: {int(n['PL4'].X)}")
    print(f"Execution time (basic formulation): {runtime:.4f} seconds")


    print("\n=========== ALLOCATION PER CROSS-SECTION TRAIN ==========")
    rows = []
    for r in Tdf.itertuples():
        t = r.t
        n3 = int(N["PL3", t].X)
        n4 = int(N["PL4", t].X)
        rows.append({
            "Line": r.Line,
            "Direction": r.Direction,
            "Dep": minutes_to_hhmm(r.Departure),
            "Arr": minutes_to_hhmm(r.Arrival),
            "Demand": D[t],
            "PL3": n3,
            "PL4": n4,
            "Seats": n3*C["PL3"] + n4*C["PL4"],
            "Length": n3*L["PL3"] + n4*L["PL4"]
        })

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()



