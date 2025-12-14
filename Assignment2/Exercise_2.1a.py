import pandas as pd

def solve_2_1_a():
    # =====================================================
    # 1. Load the Timetable Data
    # =====================================================
    df = pd.read_excel("a2_part2.xlsx", sheet_name="Timetable")

    print("Columns:", df.columns.tolist())
    print("Unique Type values:", df["Type"].unique())

    # =====================================================
    # 2. Normalize Type column
    # =====================================================
    df["Type_norm"] = df["Type"].astype(str).str.strip().str.lower()

    # =====================================================
    # 3. Extract periodic trip patterns (per line & direction)
    # =====================================================
    PERIOD = 30  # minutes

    trips_pattern = []

    for (line, direction), group in df.groupby(["Line", "Direction"]):

        dep_times = group[group["Type_norm"] == "dep"]["Time"]
        arr_times = group[group["Type_norm"] == "arr"]["Time"]

        if dep_times.empty or arr_times.empty:
            continue

        # Earliest departure in the 30-min period
        dep0 = dep_times.min()

        # Correct duration: maximum forward difference modulo 30
        duration = max((t - dep0) % PERIOD for t in arr_times)
        if duration == 0:
            duration = PERIOD

        trips_pattern.append({
            "Line": line,
            "Direction": direction,
            "Dep0": dep0,
            "Arr0": dep0 + duration
        })

    trips_pattern_df = pd.DataFrame(trips_pattern)

    print("\nPeriodic trip patterns:")
    print(trips_pattern_df)

    # =====================================================
    # 4. Expand periodic timetable & check 08:00 cross-section
    # =====================================================
    START_TIME = 300      # 05:00
    END_TIME = 1440       # 24:00
    TARGET_TIME = 8 * 60  # 08:00

    cross_section_trips = []

    for _, trip in trips_pattern_df.iterrows():
        dep0 = trip["Dep0"]
        arr0 = trip["Arr0"]

        k = 0
        while True:
            dep = dep0 + k * PERIOD
            arr = arr0 + k * PERIOD

            if dep >= END_TIME:
                break

            if dep >= START_TIME and dep <= TARGET_TIME <= arr:
                cross_section_trips.append({
                    "Line": trip["Line"],
                    "Direction": trip["Direction"],
                    "Departure_abs": dep,
                    "Arrival_abs": arr
                })

            k += 1

    cross_section_df = pd.DataFrame(cross_section_trips)

    # =====================================================
    # 5. Output
    # =====================================================
    print("\n" + "-" * 50)
    print(f"Total Cross-Section Trains at 08:00: {len(cross_section_df)}")
    print("-" * 50)

    if not cross_section_df.empty:
        print(cross_section_df.sort_values(
            ["Line", "Direction", "Departure_abs"]
        ))
    else:
        print("❌ No cross-section trains found (this should NOT happen).")

    return cross_section_df


if __name__ == "__main__":
    solve_2_1_a()

