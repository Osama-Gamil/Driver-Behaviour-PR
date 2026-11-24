"""
Harsh-event detection on a canonical trip frame.

Canonical frame (pandas DataFrame, one row per sample, any sample rate):
    t                 seconds (float, increasing)               required
    speed_kmh         OBD PID 0x0D                              required
    a_long_g          longitudinal accel (accelerometer)        optional
    a_lat_g           lateral accel (accelerometer)             optional
    lat, lon          GPS, used for lateral accel if no a_lat_g optional
    speed_limit_kmh   road limit (map / simulator)              optional
    surface           e.g. 'asphalt', 'gravel'                  optional
    wet               bool, rain                                optional
    is_night          bool                                      optional

What each part of the theory contributes:
    * Simons-Morton (2013) thresholds decide WHETHER a sample is harsh.
    * The friction circle decides HOW SEVERE it is (share of grip used).
    * The Power Model decides how severe an overspeed episode is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import physics as P


@dataclass
class Event:
    type: str                 # harsh_braking | rapid_acceleration | harsh_cornering | overspeed
    t_start: float
    t_end: float
    severity: str             # low | medium | high | extreme
    points: int               # Castignani deduction
    peak_g: float | None = None
    friction_utilisation: float | None = None
    relative_risk: float | None = None
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _col(df: pd.DataFrame, name: str):
    if name in df.columns and df[name].notna().any():
        return df[name].to_numpy(dtype=float)
    return None


def kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Add a_long_g, a_lat_g, a_h_g to a copy of the frame.

    Accelerometer values win when present; otherwise longitudinal comes from
    OBD speed and lateral from GPS heading.  `lateral_source` records which.
    """
    out = df.copy()
    t = out["t"].to_numpy(dtype=float)
    v = out["speed_kmh"].to_numpy(dtype=float)

    a_long = _col(out, "a_long_g")
    if a_long is None:
        a_long = P.accel_from_speed(t, v)
    a_lat = _col(out, "a_lat_g")
    lateral_source = "accelerometer"
    if a_lat is None:
        if _col(out, "lat") is not None and _col(out, "lon") is not None and len(out) > 2:
            heading = P.bearing_deg(out["lat"].to_numpy(float), out["lon"].to_numpy(float))
            a_lat = P.lateral_accel_g(v, P.yaw_rate_deg_s(t, heading))
            lateral_source = "gps_heading"
        else:
            a_lat = np.zeros_like(v)
            lateral_source = "none"

    out["a_long_g"] = a_long
    out["a_lat_g"] = a_lat
    out["a_h_g"] = P.combined_accel_g(a_long, a_lat)
    out.attrs["lateral_source"] = lateral_source
    return out


def _mu_for(df: pd.DataFrame) -> np.ndarray:
    surface = df["surface"] if "surface" in df.columns else pd.Series([None] * len(df))
    wet = df["wet"] if "wet" in df.columns else pd.Series([False] * len(df))
    return np.array([P.friction_mu(s if isinstance(s, str) else None, bool(w))
                     for s, w in zip(surface, wet.fillna(False))])


def _runs(mask: np.ndarray, t: np.ndarray, merge_gap_s: float):
    """Yield (start_idx, end_idx) of runs where mask is True, merging runs
    separated by less than merge_gap_s."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return
    start = prev = idx[0]
    for i in idx[1:]:
        if t[i] - t[prev] > merge_gap_s:
            yield start, prev
            start = i
        prev = i
    yield start, prev


def detect_events(df: pd.DataFrame) -> list[Event]:
    k = kinematics(df)
    t = k["t"].to_numpy(float)
    a_long = k["a_long_g"].to_numpy(float)
    a_lat = k["a_lat_g"].to_numpy(float)
    a_h = k["a_h_g"].to_numpy(float)
    mu = _mu_for(k)
    u = P.friction_utilisation(a_h, 1.0) / mu      # a_h is already in g
    night = k["is_night"].fillna(False).to_numpy(bool) if "is_night" in k.columns \
        else np.zeros(len(k), bool)

    events: list[Event] = []
    rules = [
        ("harsh_braking", a_long <= C.HARSH_BRAKE_G, a_long),
        ("rapid_acceleration", a_long >= C.RAPID_ACCEL_G, a_long),
    ]
    if k.attrs.get("lateral_source") != "none":
        rules.append(("harsh_cornering", np.abs(a_lat) >= C.HARSH_CORNER_G, a_lat))

    for etype, mask, signal in rules:
        for s, e in _runs(mask, t, C.EVENT_MERGE_GAP_S):
            seg = slice(s, e + 1)
            peak_i = s + int(np.argmax(np.abs(signal[seg])))
            u_max = float(np.max(u[seg]))
            level = P.friction_severity(u_max)
            if C.NIGHT_RAISES_SEVERITY and night[peak_i]:
                level = P.raise_severity(level)
            events.append(Event(
                type=etype, t_start=float(t[s]), t_end=float(t[e]),
                severity=level, points=C.SEVERITY_POINTS[level],
                peak_g=round(float(signal[peak_i]), 3),
                friction_utilisation=round(u_max, 3),
                detail={"mu": float(mu[peak_i]),
                        "lateral_source": k.attrs.get("lateral_source")},
            ))

    events.extend(detect_overspeed(k))
    events.sort(key=lambda ev: ev.t_start)
    return events


def detect_overspeed(df: pd.DataFrame) -> list[Event]:
    """Overspeed episodes weighted by the Power Model.

    The relative risk uses the MEAN speed of the episode, because the Power
    Model is a model of mean speed, not of momentary peaks.
    """
    limit = _col(df, "speed_limit_kmh")
    if limit is None:
        return []
    t = df["t"].to_numpy(float)
    v = df["speed_kmh"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        over = P.plausible_limit(v, limit) & (v > limit * C.OVERSPEED_RATIO)
    night = df["is_night"].fillna(False).to_numpy(bool) if "is_night" in df.columns \
        else np.zeros(len(df), bool)

    events = []
    for s, e in _runs(over, t, C.EVENT_MERGE_GAP_S):
        if t[e] - t[s] < C.OVERSPEED_MIN_S:
            continue
        seg = slice(s, e + 1)
        v_mean = float(np.mean(v[seg]))
        lim = float(np.median(limit[seg]))
        rr = float(P.power_model_relative_risk(v_mean, lim))
        level = P.power_severity(rr)
        if C.NIGHT_RAISES_SEVERITY and night[s]:
            level = P.raise_severity(level)
        events.append(Event(
            type="overspeed", t_start=float(t[s]), t_end=float(t[e]),
            severity=level, points=C.SEVERITY_POINTS[level],
            relative_risk=round(rr, 3),
            detail={"mean_speed_kmh": round(v_mean, 1), "speed_limit_kmh": lim,
                    "peak_speed_kmh": round(float(np.max(v[seg])), 1),
                    "duration_s": round(float(t[e] - t[s]), 1),
                    "power_outcome": C.POWER_OUTCOME},
        ))
    return events


def distance_km(df: pd.DataFrame) -> np.ndarray:
    """Cumulative distance (km) at each sample, by trapezoidal integration."""
    t = df["t"].to_numpy(float)
    v = df["speed_kmh"].to_numpy(float) / 3600.0
    if len(t) < 2:
        return np.zeros(len(t))
    steps = np.diff(t) * (v[1:] + v[:-1]) / 2.0
    return np.concatenate([[0.0], np.cumsum(steps)])
