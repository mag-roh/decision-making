import math

import gurobipy as gp
from gurobipy import GRB


def read_instance(path):
	with open(path, "r", encoding="utf-8") as f:
		capacity = int(f.readline().split()[0])
		demands = [int(x) for x in f.readline().split()]
	return capacity, demands


def read_routes(path, n_customers):
	c = []
	route_customers = []
	last_customer = []

	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue

			parts = [int(x) for x in line.split()]
			dist = parts[0]
			nodes = parts[2:]
			cust = [v for v in nodes if v != 0 and 1 <= v <= n_customers]
			if not cust:
				continue

			c.append(dist)
			route_customers.append(set(cust))
			last_customer.append(cust[-1])

	return c, route_customers, last_customer


def fmt_num(x, digits=4):
	if x is None:
		return "nan"
	if isinstance(x, float) and math.isnan(x):
		return "nan"
	return f"{x:.{digits}f}"


def solve_for_eps(eps, z_star, vehicles, c, a, last_routes, n):
	p = c
	M = max(p)

	model = gp.Model("cvrp_last_customer")
	model.Params.OutputFlag = 0

	R = range(len(c))
	x = model.addVars(len(c), vtype=GRB.BINARY, name="x")
	gamma = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="gamma")
	eta = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="eta")

	for i in range(1, n + 1):
		model.addConstr(gp.quicksum(x[r] for r in a[i]) == 1)

	model.addConstr(gp.quicksum(c[r] * x[r] for r in R) <= (1.0 + eps) * z_star)
	model.addConstr(gp.quicksum(x[r] for r in R) == vehicles)

	for i in range(1, n + 1):
		model.addConstr(gp.quicksum(p[r] * x[r] for r in last_routes[i]) <= eta)
		model.addConstr(
			M * (1 - gp.quicksum(x[r] for r in last_routes[i]))
			+ gp.quicksum(p[r] * x[r] for r in last_routes[i])
			>= gamma
		)

	model.setObjective(eta - gamma, GRB.MINIMIZE)
	model.optimize()

	runtime = float(getattr(model, "Runtime", float("nan")))
	gap = float(getattr(model, "MIPGap", float("nan")))
	total_cost = float("nan")
	route_range = float("nan")

	if model.SolCount > 0:
		chosen = [r for r in R if x[r].X > 0.5]
		chosen_dist = [c[r] for r in chosen]
		total_cost = float(sum(chosen_dist))
		if chosen_dist:
			route_range = float(max(chosen_dist) - min(chosen_dist))

	return runtime, gap, total_cost, route_range


def main():
	vehicles = 5
	z_star = 8831
	eps_list = [0.01, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15]

	capacity, demands = read_instance("instance.txt")
	n = len(demands)

	c, route_customers, last_customer = read_routes("routes.txt", n)
	R = range(len(c))

	a = [[] for _ in range(n + 1)]
	for r in R:
		for i in route_customers[r]:
			a[i].append(r)

	for i in range(1, n + 1):
		if not a[i]:
			raise ValueError(f"Customer {i} is not covered by any route")

	last_routes = [[] for _ in range(n + 1)]
	for r in R:
		i_last = last_customer[r]
		last_routes[i_last].append(r)

	for eps in eps_list:
		runtime, gap, total_cost, route_range = solve_for_eps(
			eps, z_star, vehicles, c, a, last_routes, n
		)

		print(
			"eps="
			+ fmt_num(eps, 3)
			+ " runtime_s="
			+ fmt_num(runtime, 4)
			+ " gap_pct="
			+ ("nan" if math.isnan(gap) else f"{100.0 * gap:.4f}")
			+ " cost="
			+ ("nan" if math.isnan(total_cost) else f"{total_cost:.0f}")
			+ " range="
			+ ("nan" if math.isnan(route_range) else f"{route_range:.0f}")
		)


if __name__ == "__main__":
	main()
