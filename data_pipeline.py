"""
Data columns:
    RecordedAtTime, DirectionRef, PublishedLineName, OriginName,
    OriginLat, OriginLong, DestinationName, DestinationLat, DestinationLong,
    VehicleRef, VehicleLocation.Latitude, VehicleLocation.Longitude,
    NextStopPointName, ArrivalProximityText, DistanceFromStop,
    ExpectedArrivalTime, ScheduledArrivalTime, Date, CycleNumber,
    ScheduledArrival, delay_s, speed, hour_sin, hour_cos, day_sin, day_cos,
    rush_hour, delay_lag1, speed_lag1, delay_lag2, speed_lag2,
    delay_lag3, speed_lag3
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
import calendar

# Column names
COL_TIMESTAMP   = "RecordedAtTime"
COL_LINE        = "PublishedLineName"
COL_DIRECTION   = "DirectionRef"
COL_VEHICLE     = "VehicleRef"
COL_DATE        = "Date"
COL_CYCLE       = "CycleNumber"
COL_TARGET      = "delay_s"

# Features already calculated in data
PRECOMPUTED_FEATURES = [
    "speed",
    "hour_sin", "hour_cos",
    "day_sin",  "day_cos",
    "rush_hour",
    "delay_lag1", "speed_lag1",
    "delay_lag2", "speed_lag2",
    "delay_lag3", "speed_lag3",
]

# Subset used for RNN
RNN_FEATURE_COLS = [
    "delay_s",
    "speed",
    "hour_sin", "hour_cos",
    "day_sin",  "day_cos",
    "rush_hour",
    "DistanceFromStop",
    "stop_idx_norm",
    "line_mean_delay",
]

# Subset used for Random Forest
RF_FEATURE_COLS = [
    "speed",
    "hour_sin", "hour_cos",
    "day_sin",  "day_cos",
    "rush_hour",
    "delay_lag1", "speed_lag1",
    "delay_lag2", "speed_lag2",
    "delay_lag3", "speed_lag3",
]

# Only these columns are needed from the raw CSVs; everything else (stop names,
# GPS coords, arrival time strings) is dropped at read time to save memory.
KEEP_COLS: frozenset = frozenset(
    [COL_TIMESTAMP, COL_LINE, COL_DIRECTION, COL_VEHICLE, COL_DATE, COL_CYCLE, COL_TARGET,
     "DistanceFromStop"]
    + PRECOMPUTED_FEATURES
)

# Main pipeline
class MTADataPipeline:

    def __init__(self, csv_paths: List[str], chunksize: int = 300_000):
        self.csv_paths = [Path(p) for p in csv_paths]
        self.chunksize = chunksize
        self.trips: List[pd.DataFrame] = []

    # Loading

    def load(self) -> "MTADataPipeline":
        #Read all CSVs in and concatenate
        frames = []
        for i, path in enumerate(self.csv_paths):
            print(f"  Loading {path.name} ...")
            for chunk in pd.read_csv(
                path, chunksize=self.chunksize, low_memory=False,
                usecols=lambda c: c in KEEP_COLS,
            ):
                chunk = self._parse_chunk(chunk)
                if chunk is not None:
                    chunk["_source"] = np.int8(i)
                    frames.append(chunk)
        self.raw = pd.concat(frames, ignore_index=True)
        print(f"  Loaded {len(self.raw):,} rows total.")
        return self

    def _parse_chunk(self, df: pd.DataFrame) -> "pd.DataFrame | None":
        required = {COL_TIMESTAMP, COL_LINE, COL_DIRECTION,
                    COL_VEHICLE, COL_DATE, COL_CYCLE, COL_TARGET}
        missing = required - set(df.columns)
        if missing:
            print(f"    WARNING: missing columns {missing} - skipping chunk.")
            return None

        df = df.copy()
        df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], errors="coerce")
        df = df.dropna(subset=[COL_TIMESTAMP, COL_TARGET]).copy()

        # Ensure numeric target and feature columns
        for col in [COL_TARGET] + PRECOMPUTED_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # Trip Reconstruction

    def reconstruct_trips(self, min_len: int = 3) -> "MTADataPipeline":
        #Group rows into individual route cycles.

        print("Reconstructing trips ...")
        trips = []
        # Group by source-file index instead of Date: CycleNumber resets to 1
        # per file in sort_cycles.py, so _source disambiguates same-numbered
        # cycles from different months without fragmenting within-month cycles
        # by calendar date (which would create ~7x more trips and OOM at 128 GB).
        groups = self.raw.groupby(
            [COL_LINE, COL_DIRECTION, COL_VEHICLE, "_source", COL_CYCLE],
            sort=False
        )
        for _, grp in groups:
            grp = grp.sort_values(COL_TIMESTAMP).reset_index(drop=True)
            if len(grp) >= min_len:
                grp["stop_idx_norm"] = np.arange(len(grp)) / max(len(grp) - 1, 1.0)
                trips.append(grp)

        self.trips = trips
        # Free the raw DataFrame — it's no longer needed after trip reconstruction
        del self.raw
        print(f"  Reconstructed {len(trips):,} route cycles.")
        return self

    # Temporal Train/Val/Test Split

    def temporal_split(self) -> Tuple[List, List, List]:
        #2/3 for training
        by_month: dict = defaultdict(list)
        for trip in self.trips:
            date_val = trip[COL_DATE].iloc[0]
            # Handle both string and datetime
            if isinstance(date_val, str):
                date_val = pd.to_datetime(date_val)
            by_month[(date_val.year, date_val.month)].append(trip)

        train, val, test = [], [], []
        for (y, m), month_trips in by_month.items():
            days_in_month = calendar.monthrange(y, m)[1]
            cutoff_train  = int(days_in_month * 2 / 3)
            cutoff_val    = cutoff_train + (days_in_month - cutoff_train) // 2

            for trip in month_trips:
                date_val = trip[COL_DATE].iloc[0]
                if isinstance(date_val, str):
                    date_val = pd.to_datetime(date_val)
                day = date_val.day
                if day <= cutoff_train:
                    train.append(trip)
                elif day <= cutoff_val:
                    val.append(trip)
                else:
                    test.append(trip)

        print(f"  Split -> train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
        return train, val, test

    # Encoded features

    @staticmethod
    def encode_line_stats(
        train_trips: List[pd.DataFrame],
        val_trips:   List[pd.DataFrame],
        test_trips:  List[pd.DataFrame],
    ) -> Tuple[List, List, List]:
        """Compute mean delay_s per bus line from training data only, then
        join it back to all splits as `line_mean_delay`. Unknown lines in
        val/test fall back to the global training mean."""
        line_sum:   dict = defaultdict(float)
        line_count: dict = defaultdict(int)
        for trip in train_trips:
            line = trip[COL_LINE].iloc[0]
            vals = trip[COL_TARGET].dropna()
            line_sum[line]   += float(vals.sum())
            line_count[line] += len(vals)

        global_mean = sum(line_sum.values()) / max(sum(line_count.values()), 1)
        line_mean   = {ln: line_sum[ln] / line_count[ln] for ln in line_sum}
        print(f"  Line stats: {len(line_mean)} lines, global mean delay {global_mean:.1f}s")

        for split in (train_trips, val_trips, test_trips):
            for trip in split:
                trip["line_mean_delay"] = line_mean.get(trip[COL_LINE].iloc[0], global_mean)

        return train_trips, val_trips, test_trips

    # RNN Tensor Builder

    @staticmethod
    def trips_to_rnn_tensors(
        trips: List[pd.DataFrame],
        feature_cols: List[str] = None,
        max_len: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        #Pad trip sequences into 3-D tensors for RNN.
        if feature_cols is None:
            feature_cols = RNN_FEATURE_COLS

        lengths = np.array([len(t) for t in trips])
        if max_len is None:
            # Cap at 99th percentile to avoid a single outlier blowing up the tensor
            max_len = int(np.percentile(lengths, 99))
        N, k = len(trips), len(feature_cols)
        print(f"  RNN tensor: N={N:,}  max_len={max_len}  k={k}"
              f"  (trip len p50={int(np.median(lengths))} p99={max_len} abs_max={lengths.max()})")

        X     = np.zeros((N, max_len, k), dtype=np.float32)
        y     = np.zeros((N, max_len),    dtype=np.float32)
        masks = np.zeros((N, max_len),    dtype=bool)

        for i, trip in enumerate(trips):
            L    = min(len(trip), max_len)
            feat = trip[feature_cols].iloc[:L].fillna(0.0).values.astype(np.float32)
            tgt      = trip[COL_TARGET].iloc[:L].fillna(0.0).values.astype(np.float32)
            next_tgt = np.empty_like(tgt)
            next_tgt[:-1] = tgt[1:]
            next_tgt[-1]  = tgt[-1]
            X[i, :L, :] = feat
            y[i, :L]    = next_tgt
            masks[i, :L] = True

        return X, y, masks

    # Random Forest Feature Builder

    @staticmethod
    def trips_to_rf_features(
        trips: List[pd.DataFrame],
        feature_cols: List[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        if feature_cols is None:
            feature_cols = RF_FEATURE_COLS

        rows_X, rows_y = [], []
        for trip in trips:
            feat = trip[feature_cols].fillna(0.0).values.astype(np.float32)
            tgt  = trip[COL_TARGET].fillna(0.0).values.astype(np.float32)
            # Target = delay at next stop; last stop repeats its own delay
            next_tgt = np.roll(tgt, -1)
            next_tgt[-1] = tgt[-1]
            rows_X.append(feat)
            rows_y.append(next_tgt)

        return (np.vstack(rows_X).astype(np.float32),
                np.concatenate(rows_y).astype(np.float32))
