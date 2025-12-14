import os
import gurobipy as gp
import pandas as pd
from gurobipy import  Model, GRB, quicksum
from collections import defaultdict



def load_data_from_excel(filepath='a2_part1.xlsx'):
    
    # Load Travel Times sheet
    df_travel = pd.read_excel(filepath, sheet_name='Travel Times')
    travel_times = {}

    for _, row in df_travel.iterrows():
        from_station = row['From']
        to_station = row['To']
        travel_time = int(row['Travel Time'])
        travel_times[(from_station, to_station)] = travel_time

    # Lines
    df_lines = pd.read_excel(filepath, sheet_name='Lines')
    lines = {}
    for _, row in df_lines.iterrows():
        line = int(row['Name'])
        stops = []
        for col in df_lines.columns:
            if col not in ['Name', 'Frequency']:
                val = row[col]
                if pd.notna(val):
                    s = str(val).strip()
                    if s and s.lower() != 'nan':
                        stops.append(s)
        lines[line] = stops
    return travel_times, lines


class PESPModel:
    def __init__(self, lines, travel_times, T=30):
        self.T = T  # 30 minutes
        self.events = {}  
        self.event_list = []  
        self.activities = {}  
        self.event_counter = 0
        self.activity_counter = 0
        self.lines = lines  # Store lines data
        self.travel_times = travel_times  # Store travel times data
        
    def create_event(self, line, direction, station, event_type):
        """Create event with numeric ID"""
        event_id = self.event_counter  
        self.events[event_id] = (line, direction, station, event_type)
        self.event_list.append(event_id)
        self.event_counter += 1
        return event_id
    
    def create_activity(self, from_event, to_event, activity_type, lower_bound, upper_bound, weight=0):
        """Create an activity linking two events."""
        activity_id = self.activity_counter
        self.activities[activity_id] = {
            'from': from_event,
            'to': to_event,
            'type': activity_type,
            'lower': lower_bound,
            'upper': upper_bound,
            'weight': weight,
        }
        self.activity_counter += 1
        return activity_id
    
    def build_network(self):
        """Build the complete event-activity network."""
        event_dict = {}  
        
        # Events
        for line, station_sequence in self.lines.items():  
            for direction in ['F', 'B']:
                path = list(reversed(station_sequence)) if direction == 'B' else station_sequence
                for station in path:
                    event_d = self.create_event(line, direction, station, 'D')
                    event_a = self.create_event(line, direction, station, 'A')
                    event_dict[(line, direction, station, 'D')] = event_d
                    event_dict[(line, direction, station, 'A')] = event_a
        
        # Create activities
        self._create_running_activities(event_dict)
        self._create_dwell_activities(event_dict)
        self._create_headway_activities(event_dict)
        self._create_synchronization_activities(event_dict)
        self._create_transfer_activities(event_dict)

        return event_dict
        
        
    def _create_running_activities(self, event_dict):
        """Running activities: departure at station i and arrival at station i+1."""
        for line, station_sequence in self.lines.items():  
            for direction in ['F', 'B']:
                path = list(reversed(station_sequence)) if direction == 'B' else station_sequence
                for i in range(len(path) - 1):
                    station_from, station_to = path[i] , path[i + 1]

                    travel_time = self.travel_times.get((station_from, station_to), self.travel_times.get((station_to, station_from)))
                    if travel_time is None:
                        print(f"Warning: No travel time for {station_from} to {station_to}")
                        continue
                    
                    event_dep = event_dict[(line, direction, station_from, 'D')]
                    event_arr = event_dict[(line, direction, station_to, 'A')]
                    self.create_activity(event_dep, event_arr,'running',travel_time,travel_time,weight=100) # Used ChatGPT for Weights
    
    def _create_dwell_activities(self, event_dict):
        """Dwell activities: arrival at station and departure at same station."""
        for line, seq in self.lines.items():
            for direction in ['F', 'B']:
                path = list(reversed(seq)) if direction == 'B' else seq
                for i, st in enumerate(path):
                    if i == 0 or i == len(path) - 1:
                        continue
                    arr = event_dict[(line, direction, st, 'A')]
                    dep = event_dict[(line, direction, st, 'D')]
                    self.create_activity(arr, dep, 'dwell', 2, 8, 50)
    
    def _create_headway_activities(self, event_dict):
        """Headway activities on shared track sections."""
        shared_departures = [
            # 800 & 3000: Amr–Asd–Ut
            (800, 3000, 'Amr', ['F', 'B']),
            (800, 3000, 'Asd', ['F', 'B']),
            (800, 3000, 'Ut', ['F', 'B']),    
            
            # 800 & 3500: Ut–Ehv
            (800, 3500, 'Ut', ['F', 'B']),
            (800, 3500, 'Ehv', ['F', 'B']),    
            
            # 800 & 3900: Ehv–Std
            (800, 3900, 'Ehv', ['F', 'B']),
            (800, 3900, 'Std', ['F', 'B']),    
            
            # 3000 & 3100: Ut–Nm
            (3000, 3100, 'Ut', ['F', 'B']),
            (3000, 3100, 'Nm', ['F', 'B']),    
            
            # 3100 & 3500: Shl–Ut
            (3100, 3500, 'Shl', ['F', 'B']),
            (3100, 3500, 'Ut', ['F', 'B']),    
        ]
        
        for l1, l2, station, directions in shared_departures:
            for direction in directions:
                e1 = event_dict.get((l1, direction, station, 'D'))
                e2 = event_dict.get((l2, direction, station, 'D'))
                
                if e1 and e2:
                    # Bidirectional headway
                    self.create_activity(e1, e2, 'headway', 3, self.T, 0)
                    self.create_activity(e2, e1, 'headway', 3, self.T, 0)
        
        
    def _create_synchronization_activities(self, event_dict):
        """Synchronization: exactly 15 minutes on shared sections."""
        sync_pairs = [
            (800, 3000, 'Asd', ['F', 'B']),
            (800, 3900, 'Std', ['F', 'B']),
            (3000, 3100, 'Ut', ['F', 'B']),
            (3100, 3500, 'Shl', ['F', 'B']),
            # (800, 3500, 'Ut', ['F', 'B']), # not-included Ut–Ehv shared section. Used ChatGPT Help
        ]
        
        for l1, l2, station, directions in sync_pairs:
            for direction in directions:
                e1 = event_dict.get((l1, direction, station, 'D'))
                e2 = event_dict.get((l2, direction, station, 'D'))
                if e1 and e2:
                    self.create_activity(e1, e2, 'synchronization', 15, 15, 0)

    
    def _create_transfer_activities(self, event_dict):
        """Transfer activities: Heerlen & Utrecht via Eindhoven (BOTH DIRECTIONS)."""
        # Heerlen to Utrecht
        ev_3900_arr = event_dict.get((3900, 'B', 'Ehv', 'A'))
        lines_ehv_to_ut = [ln for ln, stops in self.lines.items() if 'Ehv' in stops and 'Ut' in stops]
        for ln in lines_ehv_to_ut:
            ev_dep = event_dict.get((ln, 'F', 'Ehv', 'D'))
            if ev_3900_arr and ev_dep:
                self.create_activity(ev_3900_arr, ev_dep, 'transfer_Hrl_to_Ut', 2, 5, 30)

        # Utrecht to Heerlen
        ev_3900_dep = event_dict.get((3900, 'F', 'Ehv', 'D'))
        for ln in lines_ehv_to_ut:
            ev_arr = event_dict.get((ln, 'B', 'Ehv', 'A'))
            if ev_arr and ev_3900_dep:
                self.create_activity(ev_arr, ev_3900_dep, 'transfer_Ut_to_Hrl', 2, 5, 30)


def solve_pesp(pesp_model, event_dict):
    """Solve the PESP using Gurobi."""
    
    model = gp.Model("A2_Corridor_PESP")
    model.Params.OutputFlag = 1
    
    # Decision variables
    pi = {}  
    x = {}   
    p = {}   
    
    # Event times
    for event_id in pesp_model.events:
        pi[event_id] = model.addVar(lb=0, ub=pesp_model.T, name=f"pi_{event_id}")
    
    # Create activity duration and period variables
    for activity_id, activity in pesp_model.activities.items():
        lower = activity['lower']
        upper = activity['upper']
        x[activity_id] = model.addVar(lb=lower, ub=upper, name=f"x_{activity_id}")
        p[activity_id] = model.addVar(lb=0, vtype=GRB.INTEGER, name=f"p_{activity_id}")
    
    model.update()
    
    # Activity duration constraints
    for activity_id, activity in pesp_model.activities.items():
        event_i = activity['from']
        event_j = activity['to']
        model.addConstr(
            x[activity_id] == pi[event_j] - pi[event_i] + pesp_model.T * p[activity_id],
            name=f"duration_{activity_id}"
        )
    
    # For headway activities, enforce: |π[e2] - π[e1]| >= 3 OR |π[e2] - π[e1]| <= T-3
    # This prevents both from being in [0, 3) or (T-3, T]
    # Used GitHub Copilot for this part
    
    headway_pairs = set()
    for activity_id, activity in pesp_model.activities.items():
        if activity['type'] == 'headway':
            e1, e2 = activity['from'], activity['to']
            pair = tuple(sorted([e1, e2]))
            if pair not in headway_pairs:
                headway_pairs.add(pair)
                # Add constraint: π[e2] - π[e1] must be outside the "forbidden zone"
                # Use binary variable to enforce disjunction
                y = model.addVar(vtype=GRB.BINARY, name=f"y_headway_{e1}_{e2}")
                
                # If y=0: π[e2] >= π[e1] + 3
                # If y=1: π[e2] <= π[e1] - 3 (or equivalently: π[e1] >= π[e2] + 3)
                model.addConstr(pi[e2] >= pi[e1] + 3 - pesp_model.T * y, name=f"hw_sep1_{e1}_{e2}")
                model.addConstr(pi[e1] >= pi[e2] + 3 - pesp_model.T * (1 - y), name=f"hw_sep2_{e1}_{e2}")
    
    # Line 3500 departs Schiphol at :09
    for event_id in pesp_model.events:
        line, direction, station, event_type = pesp_model.events[event_id]
        if line == 3500 and station == 'Shl' and event_type == 'D' and direction == 'F':
            model.addConstr(pi[event_id] == 9, name=f"fixed_3500_Shl")
    
    model.update()
    
    # Objective
    model.setObjective(
        quicksum(
            activity['weight'] * x[activity_id]
            for activity_id, activity in pesp_model.activities.items()
            if activity['weight'] > 0
        ),
        GRB.MINIMIZE
    )
    
    model.optimize()
    
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
    return model, pi, x, p

def print_timetable(model, pesp_model, pi):
    """Print the timetable in a readable format. Used ChatGPT Help to generate the output format."""
    if model.status != GRB.OPTIMAL:
        print("Model did not find an optimal solution.")
        return
    
    print("="*100)
    print("TIMETABLE SOLUTION".center(100))
    print("="*100)
    
    # Group events by (line, direction, station) and sort
    events_by_station = defaultdict(list)
    
    for event_id in pesp_model.event_list:
        line, direction, station, event_type = pesp_model.events[event_id]
        time = pi[event_id].X
        events_by_station[(line, direction, station)].append((event_id, event_type, time))
    
    # Print by line and direction
    for line in sorted(lines.keys()):
        print(f"{'LINE ' + str(line):^100}")
        print("-" * 100)
        for direction in ['F', 'B']:
            name = "Forward" if direction == 'F' else "Backward"
            seq = list(reversed(lines[line])) if direction == 'B' else lines[line]
            print(f"{name}:")
            print(f"{'Station':<25} {'Departure':<15} {'Arrival':<15}")
            print("-"*85)
            
            for station in seq:
                key = (line, direction, station)
                if key in events_by_station:
                    events = {et: time for _, et, time in events_by_station[key]}
                    dep_time = events.get('D', None)
                    arr_time = events.get('A', None)
                    
                    dep_str = f"00:{int(round(dep_time % pesp_model.T)):02d}" if dep_time is not None else "---"
                    arr_str = f"00:{int(round(arr_time % pesp_model.T)):02d}" if arr_time is not None else "---"
                    
                    print(f"{station:<25} {dep_str:<15} {arr_str:<15}")
    
    print("="*100)
    print(f"Objective Value (Total Passenger-Minutes): {model.ObjVal:.2f}")
    print("="*100 )

if __name__ == "__main__":

    travel_times, lines = load_data_from_excel()

    pesp = PESPModel(lines, travel_times, T=30)  
    event_dict = pesp.build_network()
    
    print("Solving PESP model...")

    model, pi, x, p = solve_pesp(pesp,event_dict)
    print_timetable(model, pesp, pi)
