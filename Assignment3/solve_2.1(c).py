import gurobipy as gp
from gurobipy import GRB
import time
import os
import sys

def read_network(filename):
    """
    Reads the implicit network arcs from the file.
    Returns a list of unique nodes and a list of arcs (u, v).
    """
    arcs = []
    nodes = set()
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.split()
                # Skip empty lines or headers
                if len(parts) < 2:
                    continue
                try:
                    u, v = int(parts[0]), int(parts[1])
                    arcs.append((u, v))
                    nodes.add(u)
                    nodes.add(v)
                except ValueError:
                    continue # Skip lines that don't start with integers
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return [], []
    return list(nodes), arcs

def read_pairs(filename):
    """
    Reads the OD pairs (commodities) for a company.
    Expected format: volume origin destination
    Skips header lines automatically.
    """
    commodities = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        # Try to convert to numbers. If this fails (e.g. header), skip line.
                        vol = float(parts[0])
                        orig = int(parts[1])
                        dest = int(parts[2])
                        commodities.append({'vol': vol, 'orig': orig, 'dest': dest})
                    except ValueError:
                        continue # Skip header or malformed lines
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []
    return commodities

def solve_charging_location(nodes, arcs, commodities, company_name):
    print(f"\n--- Solving for Company {company_name} ---")
    
    Q = 10.0  # Capacity per station
    
    m = gp.Model(f"ChargingLocation_{company_name}")
    m.setParam('OutputFlag', 0)
    
    y = m.addVars(nodes, vtype=GRB.BINARY, name="y")
    arc_tuples = gp.tuplelist(arcs)
    x = {}
    
    # Create variables only for necessary arcs to save memory
    for k_idx, comm in enumerate(commodities):
        for i, j in arcs:
            x[k_idx, i, j] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)

    m.setObjective(gp.quicksum(y[i] for i in nodes), GRB.MINIMIZE)
    
    # Constraints
    for k_idx, comm in enumerate(commodities):
        s_k = comm['orig']
        t_k = comm['dest']
        
        for i in nodes:
            flow_in = gp.quicksum(x[k_idx, j, i] for j, i_node in arc_tuples.select('*', i))
            flow_out = gp.quicksum(x[k_idx, i, j] for i_node, j in arc_tuples.select(i, '*'))
            
            rhs = 0
            if i == s_k: rhs = 1
            elif i == t_k: rhs = -1
            
            m.addConstr(flow_out - flow_in == rhs)

    for i in nodes:
        leaving_volume = gp.quicksum(
            comm['vol'] * x[k_idx, i, j]
            for k_idx, comm in enumerate(commodities) 
            if comm['orig'] != i
            for i_node, j in arc_tuples.select(i, '*')
        )
        m.addConstr(leaving_volume <= Q * y[i])

    start_time = time.time()
    m.optimize()
    end_time = time.time()
    
    if m.status == GRB.OPTIMAL:
        obj_val = m.objVal
        open_locs = [i for i in nodes if y[i].X > 0.5]
        print(f"Status: Optimal")
        print(f"Time: {end_time - start_time:.4f}s")
        print(f"Optimal Number of Facilities: {obj_val}")
        print(f"Locations: {sorted(open_locs)}")
        return obj_val, open_locs
    else:
        print(f"Status: {m.status}")
        return None, None

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    def get_path(filename): return os.path.join(script_dir, filename)

    print(f"Working Directory: {script_dir}")
    
    nodes, arcs = read_network(get_path("network.txt"))
    if not nodes:
        print("CRITICAL: Network not loaded.")
        sys.exit(1)
        
    print(f"Network loaded: {len(nodes)} nodes, {len(arcs)} arcs")
    
    companies = ['A', 'B', 'C']
    results = {}
    
    for company in companies:
        filename = get_path(f"pairs{company}.txt")
        if not os.path.exists(filename):
            print(f"Skipping {company}: File not found")
            continue
            
        commodities = read_pairs(filename)
        print(f"Loaded {len(commodities)} commodities for Company {company}")
        
        if len(commodities) > 0:
            obj, locs = solve_charging_location(nodes, arcs, commodities, company)
            results[company] = (obj, locs)
        else:
            print(f"Warning: No valid commodities found in pairs{company}.txt")

    print("\n" + "="*40)
    print("FINAL RESULTS SUMMARY")
    print("="*40)
    for comp in companies:
        if comp in results and results[comp][0] is not None:
            print(f"{comp:<5} | Count: {int(results[comp][0])} | Locs: {sorted(results[comp][1])}")
        else:
            print(f"{comp:<5} | N/A")