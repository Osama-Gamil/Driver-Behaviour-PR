"""
Loader for the labelled Sagbas (2026) driving dataset.

Source  : https://github.com/arifsagbas/2025_obd_dataset  (Zenodo 10.5281/zenodo.17286874)
Paper   : Sagbas, E. A. (2026). Temporal and modal contributions to smartphone-based
          multimodal driving behavior classification ... PeerJ Computer Science 12:e3493.

Facts checked on the downloaded files (Nov 2026):
  * 1,601 CSV files, one labelled window each, ~250 rows at 50 Hz (~5 s).
    One file (5_20250127_235711.csv) is truncated inside the RAR archive.
  * No header row, ';' separated, 55 columns in the order of the README;
    gps_speed uses a decimal comma ("107,64"); battery_voltage has a "V" suffix.
  * OBD values (speed, rpm, engine_load, ...) refresh about once per second,
    so they are resampled to 1 Hz here; phone IMU stays at 50 Hz.
  * File name: <label>_<YYYYMMDD>_<HHMMSS>.csv; recorded on 5 days.
  * Labels: 1 very calm, 2 normal, 3 aggressive, 4 eco constant speed,
    5 non-eco (high RPM) constant speed, 6 calm eco highway.
  * The README does not say how many drivers took part. Treat it as a
    single-driver / single-vehicle dataset unless the paper says otherwise:
    evaluation is then "unseen DRIVES", not "unseen drivers".

Sessions: consecutive windows of the same label recorded less than
SESSION_GAP_S apart belong to one drive. Cross-validation is grouped by
session, so neighbouring seconds of the same drive never sit in both the
training and the test fold (that leak would inflate accuracy).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .. import physics as P

COLUMNS = [
    "accel_x", "accel_y", "accel_z", "gravity_x", "gravity_y", "gravity_z",
    "linear_x", "linear_y", "linear_z", "gyro_x", "gyro_y", "gyro_z",
    "magnet_x", "magnet_y", "magnet_z", "light", "proximity",
    "latitude", "longitude", "altitude", "gps_speed", "rpm", "engine_load", "speed",
    "coolant_temp", "intake_air_temp", "maf", "map", "ambient_temp", "battery_voltage",
    "timestamp", "maxSpeed", "highway", "name", "oneWay", "ref", "int_ref", "lanes",
    "minSpeed", "surface", "toll",
    "rms", "zcr", "decibel", "avg_amplitude", "signal_energy", "dominant_freq",
    "spectral_centroid", "spectral_spread", "spectral_flatness", "spectral_entropy",
    "skewness", "kurtosis", "bandwidth", "rolloff",
]

LABEL_NAMES = {
    1: "very_calm", 2: "normal", 3: "aggressive",
    4: "eco_constant", 5: "non_eco_constant", 6: "calm_eco_highway",
}
# Three-class task used by the paper and by our safety question.
SAFETY_3CLASS = {1: "calm", 2: "normal", 3: "aggressive"}

SESSION_GAP_S = 60.0
OBD_COLUMNS = ["speed", "rpm", "engine_load", "maf", "map", "maxSpeed"]


@dataclass
class Window:
    file: str
    label: int
    session: str
    start: datetime
    obd: pd.DataFrame       # 1 Hz: t, speed_kmh, rpm, engine_load, maf, map, speed_limit_kmh, surface
    motion: pd.DataFrame    # 50 Hz: t, a_h_g, a_lat_g, yaw_deg_s, mu


def read_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", header=None, names=COLUMNS, dtype=str,
                     keep_default_na=False)
    df["gps_speed"] = df["gps_speed"].str.replace(",", ".", regex=False)
    df["battery_voltage"] = df["battery_voltage"].str.rstrip("Vv")
    for c in COLUMNS:
        if c not in ("timestamp", "highway", "name", "oneWay", "ref", "int_ref", "surface", "toll"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"]).reset_index(drop=True)


def to_canonical(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = (raw["timestamp"] - raw["timestamp"].iloc[0]).dt.total_seconds().to_numpy()

    # ---- 1 Hz OBD frame: last value in each second, like a 1 Hz ELM327 poll ----
    sec = np.floor(t).astype(int)
    obd = raw[OBD_COLUMNS].assign(_sec=sec).groupby("_sec").last().reset_index()
    obd = obd.rename(columns={"_sec": "t", "speed": "speed_kmh", "maxSpeed": "speed_limit_kmh"})
    obd["t"] = obd["t"].astype(float)
    obd["surface"] = raw["surface"].iloc[0] if raw["surface"].iloc[0] not in ("", "NoData") else None

    # ---- 50 Hz motion frame (orientation-free quantities only) ----
    lin = raw[["linear_x", "linear_y", "linear_z"]].to_numpy(float)
    grav = raw[["gravity_x", "gravity_y", "gravity_z"]].to_numpy(float)
    gyro = raw[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(float)
    g_norm = np.linalg.norm(grav, axis=1, keepdims=True)
    g_hat = np.divide(grav, g_norm, out=np.zeros_like(grav), where=g_norm > 0)
    # Yaw rate = rotation about the vertical (gravity) axis, whatever the phone mounting.
    yaw = np.degrees(np.abs(np.sum(gyro * g_hat, axis=1)))
    v_interp = np.interp(t, obd["t"].to_numpy(float), obd["speed_kmh"].to_numpy(float))
    mu = P.friction_mu(obd["surface"].iloc[0])
    motion = pd.DataFrame({
        "t": t,
        "a_h_g": P.horizontal_accel_g(lin, grav),
        "a_lat_g": P.lateral_accel_g(v_interp, yaw),   # a_lat = v * omega
        "yaw_deg_s": yaw,
        "mu": mu,
    })
    # The phone logs some rows with identical millisecond timestamps; average
    # them so derivatives (jerk) never divide by a zero time step.
    motion = motion.groupby("t", as_index=False).mean()
    return obd, motion


def _parse_name(path: str):
    base = os.path.basename(path)[:-4]
    label, date, hms = base.split("_")[:3]
    return int(label), datetime.strptime(date + hms, "%Y%m%d%H%M%S")


def load(data_dir: str, labels=None) -> list[Window]:
    paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    items = []
    for p in paths:
        try:
            label, start = _parse_name(p)
        except ValueError:
            continue
        if labels is not None and label not in labels:
            continue
        items.append((start, label, p))
    items.sort()

    windows: list[Window] = []
    session_id, prev = 0, None
    for start, label, p in items:
        try:
            raw = read_raw(p)
        except Exception:
            continue
        if len(raw) < 50:            # truncated / empty file
            continue
        if prev is None or label != prev[0] or (start - prev[1]).total_seconds() > SESSION_GAP_S:
            session_id += 1
        prev = (label, start)
        obd, motion = to_canonical(raw)
        windows.append(Window(os.path.basename(p), label,
                              f"{start:%Y%m%d}-s{session_id:03d}", start, obd, motion))
    return windows
