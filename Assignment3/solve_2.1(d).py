import gurobipy as gp
from gurobipy import GRB
import time
import os
import sys
from itertools import combinations
import math

def read_network(filename):
    arcs = []
    nodes = set()
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2: continue
                try:
                    u, v = int(parts[0]), int(parts[1])
                    arcs.append((u, v))
                    nodes.add(u); nodes.add(v)
                except ValueError: continue
    except Exception: return [], []
    return list(nodes), arcs

def read_pairs(filename):
    commodities = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        vol = float(parts[0])
                        orig = int(parts[1])
                        dest = int(parts[2])
                        commodities.append({'vol': vol, 'orig': orig, 'dest': dest})
                    except ValueError: continue
    except Exception: return []
    return commodities

def solve_model(nodes, arcs, commodities, label):
    m = gp.Model(f"Coalition_{label}")
    m.setParam('OutputFlag', 0)
    
    Q = 10.0
    y = m.addVars(nodes, vtype=GRB.BINARY, name="y")
    x = {}
    arc_tuples = gp.tuplelist(arcs)
    
    for k_idx, comm in enumerate(commodities):
        for i, j in arcs:
            x[k_idx, i, j] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)

    m.setObjective(gp.quicksum(y[i] for i in nodes), GRB.MINIMIZE)
    
    for k_idx, comm in enumerate(commodities):
        s_k = comm['orig']; t_k = comm['dest']
        for i in nodes:
            flow_in = gp.quicksum(x[k_idx, j, i] for j, i_node in arc_tuples.select('*', i))
            flow_out = gp.quicksum(x[k_idx, i, j] for i_node, j in arc_tuples.select(i, '*'))
            rhs = 1 if i == s_k else (-1 if i == t_k else 0)
            m.addConstr(flow_out - flow_in == rhs)

    for i in nodes:
        leaving_volume = gp.quicksum(
            comm['vol'] * x[k_idx, i, j]
            for k_idx, comm in enumerate(commodities) 
            if comm['orig'] != i
            for i_node, j in arc_tuples.select(i, '*')
        )
        m.addConstr(leaving_volume <= Q * y[i])

    m.optimize()
    if m.status == GRB.OPTIMAL:
        return m.objVal
    return None

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    def get_path(filename): return os.path.join(script_dir, filename)
    
    print("Loading network...")
    nodes, arcs = read_network(get_path("network.txt"))
    
    all_companies = ['A', 'B', 'C']
    company_data = {}
    for c in all_companies:
        company_data[c] = read_pairs(get_path(f"pairs{c}.txt"))

    v = {} 
    
    subsets = []
    for r in range(1, 4):
        subsets.extend(combinations(all_companies, r))
    
    print(f"\n--- Computing Costs for {len(subsets)} Coalitions ---")
    
    for coalition in subsets:
        coalition_name = "".join(coalition)
        
        combined_commodities = []
        for c in coalition:
            combined_commodities.extend(company_data[c])
            
        print(f"Solving for {{{coalition_name}}} ... ", end="")
        cost = solve_model(nodes, arcs, combined_commodities, coalition_name)
        
        if cost is not None:
            v[coalition] = cost
            print(f"Cost: {cost}")
        else:
            print("Infeasible")
            v[coalition] = 0

    print("\n--- Shapley Value Calculation ---")
    shapley_values = {i: 0.0 for i in all_companies}
    N = len(all_companies)
    
    for i in all_companies:
        print(f"\nCalculating for {i}:")
        remaining_companies = [c for c in all_companies if c != i]
        
        S_list = []
        for r in range(0, len(remaining_companies) + 1):
            S_list.extend(combinations(remaining_companies, r))
            
        for S in S_list:
            S_union_i = tuple(sorted(list(S) + [i]))
            
            val_S = v[S] if S in v else 0.0
            val_S_union_i = v[S_union_i]
            
            marginal_contribution = val_S_union_i - val_S
            
            s_len = len(S)
            weight = (math.factorial(s_len) * math.factorial(N - s_len - 1)) / math.factorial(N)
            
            term = weight * marginal_contribution
            shapley_values[i] += term
            
            S_str = "".join(S) if S else "{}"
            print(f"  S={{{S_str:<2}}} | Marg: {val_S_union_i:.1f} - {val_S:.1f} = {marginal_contribution:.1f} | W: {weight:.3f} | Term: {term:.3f}")

    print("\n" + "="*50)
    print(f"{'Coalition':<15} | {'Cost v(S)':<10}")
    print("-" * 50)
    for coalition, cost in v.items():
        name = "{" + ",".join(coalition) + "}"
        print(f"{name:<15} | {cost}")
    print("="*50)
    
    print(f"{'Company':<10} | {'Stand-alone':<12} | {'Shapley Cost':<12} | {'Savings':<10}")
    print("-" * 50)
    for c in all_companies:
        standalone = v[(c,)]
        shapley = shapley_values[c]
        saving = standalone - shapley
        print(f"{c:<10} | {standalone:<12.1f} | {shapley:<12.2f} | {saving:<10.2f}")
    print("="*50)