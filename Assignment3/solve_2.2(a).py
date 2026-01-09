import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import os
import sys

def read_network_with_dist(filename):
    dist_map = {}
    nodes = set()
    arcs = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3: continue
                try:
                    u, v = int(parts[0]), int(parts[1])
                    d = float(parts[2])
                    dist_map[(u, v)] = d
                    arcs.append((u, v))
                    nodes.add(u); nodes.add(v)
                except ValueError: continue
    except Exception as e:
        print(f"Error: {e}")
        return [], [], {}
    return list(nodes), arcs, dist_map

def read_all_pairs(script_dir):
    all_commodities = []
    for company in ['A', 'B', 'C']:
        fname = os.path.join(script_dir, f"pairs{company}.txt")
        if not os.path.exists(fname): continue
        with open(fname, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        vol = float(parts[0])
                        orig = int(parts[1])
                        dest = int(parts[2])
                        all_commodities.append({'vol': vol, 'orig': orig, 'dest': dest, 'company': company})
                    except ValueError: continue
    return all_commodities

def solve_system_optimum(nodes, arcs, dist_map, commodities):
    print("\n--- Step 1: Solving System Optimal (Grand Coalition) ---")
    m = gp.Model("GrandCoalition")
    m.setParam('OutputFlag', 0)
    
    Q = 10.0
    y = m.addVars(nodes, vtype=GRB.BINARY, name="y")
    x = {}
    arc_tuples = gp.tuplelist(arcs)
    
    for k_idx, comm in enumerate(commodities):
        for i, j in arcs:
            x[k_idx, i, j] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)

    obj_facilities = gp.quicksum(y[i] for i in nodes)
    m.setObjectiveN(obj_facilities, index=0, priority=2, name="MinFacilities")

    obj_distance = gp.quicksum(
        dist_map[(i,j)] * x[k_idx, i, j] 
        for k_idx in range(len(commodities)) 
        for i, j in arcs
    )
    m.setObjectiveN(obj_distance, index=1, priority=1, name="MinDistance")

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
        open_stations = {i for i in nodes if y[i].X > 0.5}
        
        total_dist = 0.0
        for k_idx in range(len(commodities)):
            for i, j in arcs:
                if x[k_idx, i, j].X > 0.001:
                    total_dist += dist_map[(i, j)]
        
        return open_stations, total_dist
    return None, None

def simulate_selfish_routing(nodes, arcs, dist_map, commodities, open_stations):
    print("\n--- Step 2: Simulating Selfish Routing ---")
    
    G = nx.DiGraph()
    for u in nodes:
        G.add_node(u)
    
    total_selfish_distance = 0.0
    station_usage = {i: 0.0 for i in nodes}
    
    FullG = nx.DiGraph()
    for (u, v), d in dist_map.items():
        FullG.add_edge(u, v, weight=d)
        
    print(f"Routing {len(commodities)} travelers...")
    
    for comm in commodities:
        s = comm['orig']
        t = comm['dest']
        vol = comm['vol']
        
        valid_nodes = open_stations.union({s, t})
        subG = FullG.subgraph(valid_nodes)
        
        try:
            path = nx.shortest_path(subG, source=s, target=t, weight='weight')
            
            path_dist = 0
            for k in range(len(path)-1):
                u, v = path[k], path[k+1]
                path_dist += dist_map[(u, v)]
            total_selfish_distance += path_dist
            
            for node in path[1:-1]: 
                station_usage[node] += vol
                
        except nx.NetworkXNoPath:
            print(f"Warning: No valid path for traveler {s}->{t}!")

    return total_selfish_distance, station_usage

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    def get_path(filename): return os.path.join(script_dir, filename)

    nodes, arcs, dist_map = read_network_with_dist(get_path("network.txt"))
    commodities = read_all_pairs(script_dir)
    print(f"Loaded {len(nodes)} nodes, {len(arcs)} arcs, {len(commodities)} commodities.")

    open_stations, system_dist = solve_system_optimum(nodes, arcs, dist_map, commodities)
    
    if open_stations:
        print(f"System Optimal Locations ({len(open_stations)}): {sorted(list(open_stations))}")
        print(f"System Optimal Total Distance: {system_dist:.2f}")
        
        selfish_dist, usage = simulate_selfish_routing(nodes, arcs, dist_map, commodities, open_stations)
        
        print(f"Selfish Routing Total Distance: {selfish_dist:.2f}")
        
        print("\n--- Capacity Check ---")
        print(f"{'Station':<10} | {'Usage':<10} | {'Status'}")
        print("-" * 40)
        
        violations = 0
        for st in sorted(list(open_stations)):
            load = usage[st]
            status = "OK"
            if load > 10.0 + 1e-5:
                status = f"VIOLATION (+{load-10:.1f})"
                violations += 1
            print(f"{st:<10} | {load:<10.1f} | {status}")
            
        print("-" * 40)
        print(f"Total Capacity Violations: {violations}")
        
    else:
        print("Failed to solve system optimal model.")