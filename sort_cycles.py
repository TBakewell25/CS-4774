import pandas as pd
import sys


# Time gap (in minutes) between consecutive pings that indicates a new cycle.
CYCLE_GAP_MINUTES = 30


def main(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    df["RecordedAtTime"] = pd.to_datetime(df["RecordedAtTime"])
    df["Date"] = df["RecordedAtTime"].dt.date

    group_keys = ["PublishedLineName", "DirectionRef", "VehicleRef"]
    df = df.sort_values(group_keys + ["RecordedAtTime"])

    gap = df.groupby(group_keys)["RecordedAtTime"].diff()
    new_cycle = gap.isna() | (gap > pd.Timedelta(minutes=CYCLE_GAP_MINUTES))
    df["CycleNumber"] = new_cycle.groupby([df[k] for k in group_keys]).cumsum()

    sort_cols = group_keys + ["CycleNumber", "RecordedAtTime"]
    df = df.sort_values(sort_cols)

    df.to_csv(output_path, index=False)
    print(f"Written {len(df)} rows to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.csv> <output.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
