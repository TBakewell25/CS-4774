import os
import pandas as pd
import numpy as np
import sys


# Number of within-cycle lagged stops to include for delay and speed.
N_LAGS = 3


def load_weather(weather_path: str) -> pd.DataFrame:
    """Load hourly weather CSV, return (Date str, hour int) keyed DataFrame."""
    w = pd.read_csv(weather_path, parse_dates=["time"])
    w["Date"] = w["time"].dt.strftime("%Y-%m-%d")
    w["hour"] = w["time"].dt.hour
    w = w.rename(columns={
        "temperature_2m (°C)":  "temperature_c",
        "precipitation (mm)":   "precipitation_mm",
        "rain (mm)":            "rain_mm",
        "windspeed_10m (km/h)": "windspeed_kmh",
    })
    return w[["Date", "hour", "temperature_c", "precipitation_mm",
              "rain_mm", "windspeed_kmh"]]


def parse_scheduled_times(dates, time_strs):
    """Vectorized parse of ScheduledArrivalTime, which has no date component
    and may have hours >= 24 (MTA overnight notation, e.g. '24:06:14')."""
    parts = time_strs.str.split(":", expand=True).astype(float)
    delta = (
        pd.to_timedelta(parts[0], unit="h")
        + pd.to_timedelta(parts[1], unit="m")
        + pd.to_timedelta(parts[2], unit="s")
    )
    return pd.to_datetime(dates) + delta


def main(input_path: str, output_path: str,
         weather_path: str | None = None) -> None:
    df = pd.read_csv(input_path)

    df["RecordedAtTime"] = pd.to_datetime(df["RecordedAtTime"])
    df["ExpectedArrivalTime"] = pd.to_datetime(df["ExpectedArrivalTime"])

    # sort_cycles.py adds 'Date'; derive it here when running on unsorted input.
    if "Date" not in df.columns:
        df["Date"] = df["RecordedAtTime"].dt.strftime("%Y-%m-%d")

    # Reconstruct full ScheduledArrival datetime (time-only col + Date col).
    valid_sched = df["ScheduledArrivalTime"].notna()
    df.loc[valid_sched, "ScheduledArrival"] = parse_scheduled_times(
        df.loc[valid_sched, "Date"],
        df.loc[valid_sched, "ScheduledArrivalTime"],
    )

    # --- Core derived features ---

    # Delay in seconds: positive = late, negative = early.
    df["delay_s"] = (
        df["ExpectedArrivalTime"] - df["ScheduledArrival"]
    ).dt.total_seconds()

    # Fix day-boundary errors (~±24h): happens when a bus running just past
    # midnight has a ScheduledArrivalTime from the prior service evening, or
    # vice versa. Shift by ±86400s to minimize |delay|.
    off = df["delay_s"].abs() > 12 * 3600
    df.loc[off, "delay_s"] += np.where(df.loc[off, "delay_s"] < 0, 86400, -86400)

    # Speed: distance to next stop (m) / seconds until expected arrival.
    time_to_arrival = (
        df["ExpectedArrivalTime"] - df["RecordedAtTime"]
    ).dt.total_seconds()
    # Zero or negative time_to_arrival (bus already at/past stop) -> NaN speed.
    df["speed"] = df["DistanceFromStop"] / time_to_arrival.where(time_to_arrival > 0)

    # --- Cyclical time encodings ---
    hour = df["RecordedAtTime"].dt.hour + df["RecordedAtTime"].dt.minute / 60
    dow = df["RecordedAtTime"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"] = np.sin(2 * np.pi * dow / 7)
    df["day_cos"] = np.cos(2 * np.pi * dow / 7)

    # Rush hour binary flag: 08:00-09:00 and 17:00-18:00.
    h = df["RecordedAtTime"].dt.hour
    df["rush_hour"] = (((h >= 8) & (h < 9)) | ((h >= 17) & (h < 18))).astype(int)

    # --- Lagged features (within each cycle) ---
    # CycleNumber from sort_cycles.py is unique per (line, direction, vehicle)
    # across the month, so this group identifies a single trip.
    cycle_group = ["PublishedLineName", "DirectionRef", "VehicleRef", "CycleNumber"]
    grouped_delay = df.groupby(cycle_group)["delay_s"]
    grouped_speed = df.groupby(cycle_group)["speed"]

    for n in range(1, N_LAGS + 1):
        df[f"delay_lag{n}"] = grouped_delay.shift(n)
        df[f"speed_lag{n}"] = grouped_speed.shift(n)

    # --- Hourly weather join (optional) ---
    if weather_path is not None:
        weather = load_weather(weather_path)
        df["hour"] = df["RecordedAtTime"].dt.hour
        df = df.merge(weather, on=["Date", "hour"], how="left")
        df.drop(columns=["hour"], inplace=True)
        n_nan = df[["temperature_c", "precipitation_mm",
                    "rain_mm", "windspeed_kmh"]].isna().any(axis=1).sum()
        if n_nan:
            print(f"Warning: {n_nan:,} rows missing weather data (filling 0)")
            df[["temperature_c", "precipitation_mm",
                "rain_mm", "windspeed_kmh"]] = \
                df[["temperature_c", "precipitation_mm",
                    "rain_mm", "windspeed_kmh"]].fillna(0)

    df.to_csv(output_path, index=False)
    print(f"Written {len(df)} rows to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(f"Usage: {sys.argv[0]} <sorted_input.csv> <output.csv> [weather.csv]")
        sys.exit(1)
    weather_path = sys.argv[3] if len(sys.argv) == 4 else None
    main(sys.argv[1], sys.argv[2], weather_path)
