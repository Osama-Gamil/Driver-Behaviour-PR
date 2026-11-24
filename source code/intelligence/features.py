"""
Window features.

Feature groups follow the research question RQ1 ("does a model using all OBD
signals beat the speed-only reference?"), so the groups can be switched on
one at a time in the experiment (an ablation study):

    SPEED   what the published reference score sees: OBD speed and the
            acceleration derived from it, plus driving-volatility measures
            (Mohammadnazar, Arvin & Khattak 2021, after Kamrani et al.)
    ENGINE  RPM, engine load, MAF, MAP  (OBD PIDs 0x0C, 0x04, 0x10, 0x0B)
    MOTION  friction-circle demand, lateral acceleration, yaw rate
            (accelerometer/gyro or GPS heading)
    CONTEXT speed relative to the road limit and Power-Model relative risk

Volatility measures for a signal x in one window (definitions as used in the
driving-volatility literature; check them against Mohammadnazar et al. 2021
Table 1 before the thesis is submitted):
    cv     = sd / |mean|                coefficient of variation
    dmean  = mean |x - mean|            mean absolute deviation
    qcv    = (Q3 - Q1) / (Q3 + Q1)      quartile coefficient of variation
    vf     = share of x outside mean +/- 2 sd
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import physics as P

SPEED, ENGINE, MOTION, CONTEXT = "speed", "engine", "motion", "context"


def volatility(x: np.ndarray, prefix: str) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {f"{prefix}_cv": 0.0, f"{prefix}_dmean": 0.0,
                f"{prefix}_qcv": 0.0, f"{prefix}_vf": 0.0}
    m, sd = x.mean(), x.std()
    q1, q3 = np.percentile(x, [25, 75])
    return {
        f"{prefix}_cv": float(sd / abs(m)) if abs(m) > 1e-6 else 0.0,
        f"{prefix}_dmean": float(np.mean(np.abs(x - m))),
        f"{prefix}_qcv": float((q3 - q1) / (q3 + q1)) if abs(q3 + q1) > 1e-6 else 0.0,
        f"{prefix}_vf": float(np.mean(np.abs(x - m) > 2 * sd)) if sd > 0 else 0.0,
    }


def _stats(x, prefix, which=("mean", "max", "sd")) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out = {}
    for w in which:
        if len(x) == 0:
            out[f"{prefix}_{w}"] = 0.0
        elif w == "mean":
            out[f"{prefix}_{w}"] = float(x.mean())
        elif w == "max":
            out[f"{prefix}_{w}"] = float(x.max())
        elif w == "min":
            out[f"{prefix}_{w}"] = float(x.min())
        elif w == "sd":
            out[f"{prefix}_{w}"] = float(x.std())
    return out


def speed_features(obd: pd.DataFrame) -> dict:
    """From OBD speed only (the reference score's view of the world)."""
    t = obd["t"].to_numpy(float)
    v = obd["speed_kmh"].to_numpy(float)
    a_ms2 = P.accel_from_speed(t, v) * C.G
    f = _stats(v, "speed")
    f.update({
        "obd_accel_max_ms2": float(a_ms2.max()) if len(a_ms2) else 0.0,
        "obd_decel_max_ms2": float(-a_ms2.min()) if len(a_ms2) else 0.0,
        "obd_a_abs_max_ms2": float(np.abs(a_ms2).max()) if len(a_ms2) else 0.0,
        "obd_a_sd_ms2": float(a_ms2.std()) if len(a_ms2) else 0.0,
    })
    f.update(volatility(v, "vol_speed"))
    f.update(volatility(a_ms2, "vol_accel"))
    return f


def engine_features(obd: pd.DataFrame) -> dict:
    f = {}
    for col, name in (("rpm", "rpm"), ("engine_load", "load"), ("maf", "maf"), ("map", "map")):
        if col in obd.columns:
            f.update(_stats(obd[col], name))
    if "rpm" in obd.columns:
        v = obd["speed_kmh"].to_numpy(float)
        rpm = obd["rpm"].to_numpy(float)
        moving = v > 5
        # RPM per km/h is a proxy for the gear chosen: high value = low gear /
        # high revs for the speed (the eco vs non-eco difference).
        f["rpm_per_kmh"] = float(np.median(rpm[moving] / v[moving])) if moving.any() else 0.0
    return f


def motion_features(motion: pd.DataFrame) -> dict:
    """From an accelerometer/gyro frame with a_h_g, a_lat_g, yaw_deg_s and
    optionally mu (friction coefficient) columns."""
    f = {}
    if "a_h_g" in motion.columns:
        a_h = motion["a_h_g"].to_numpy(float)
        mu = motion["mu"].to_numpy(float) if "mu" in motion.columns \
            else np.full(len(a_h), P.friction_mu(None))
        u = a_h / mu
        f.update(_stats(a_h, "a_h_g"))
        f["friction_u_max"] = float(np.nanmax(u)) if len(u) else 0.0
        f["friction_u_p90"] = float(np.nanpercentile(u, 90)) if len(u) else 0.0
        # Volatility of the accelerometer signal: closest to the 10 Hz
        # acceleration volatility Mohammadnazar et al. actually used.
        f.update(volatility(a_h, "vol_ah"))
        if len(a_h) > 2 and "t" in motion.columns:
            jerk = np.gradient(a_h, motion["t"].to_numpy(float))
            f["jerk_sd"] = float(np.nanstd(jerk))
    if "a_lat_g" in motion.columns:
        f.update(_stats(np.abs(motion["a_lat_g"]), "a_lat_abs_g", ("mean", "max")))
    if "yaw_deg_s" in motion.columns:
        f.update(_stats(np.abs(motion["yaw_deg_s"]), "yaw_abs", ("mean", "max")))
    return f


def context_features(obd: pd.DataFrame) -> dict:
    if "speed_limit_kmh" not in obd.columns:
        return {}
    v = obd["speed_kmh"].to_numpy(float)
    lim = obd["speed_limit_kmh"].to_numpy(float)
    ok = P.plausible_limit(v, lim)
    if not ok.any():
        return {"speed_ratio_max": 0.0, "speed_ratio_mean": 0.0,
                "power_rr": 0.0, "power_log_rr": 0.0, "limit_known": 0.0}
    ratio = v[ok] / lim[ok]
    # Power Model on the window's MEAN speed (mean speed is what the model is
    # about). The log form p*ln(v/v0) is used as the ML feature because the
    # risk itself is very skewed.
    rr = float(P.power_model_relative_risk(v[ok].mean(), np.median(lim[ok])))
    return {
        "speed_ratio_max": float(ratio.max()),
        "speed_ratio_mean": float(ratio.mean()),
        "power_rr": rr,
        "power_log_rr": float(np.log(rr)) if rr > 0 else -10.0,
        "limit_known": 1.0,
    }


def window_features(obd: pd.DataFrame, motion: pd.DataFrame | None = None,
                    groups=(SPEED, ENGINE, MOTION, CONTEXT)) -> dict:
    f = {}
    if SPEED in groups:
        f.update(speed_features(obd))
    if ENGINE in groups:
        f.update(engine_features(obd))
    if MOTION in groups and motion is not None:
        f.update(motion_features(motion))
    if CONTEXT in groups:
        f.update(context_features(obd))
    return f


def sliding_windows(df: pd.DataFrame, length_s: float = C.WINDOW_S,
                    step_s: float = C.WINDOW_STEP_S, min_samples: int = 3):
    """Yield (t_start, sub-frame) windows over a trip frame with a 't' column."""
    if df.empty:
        return
    t = df["t"].to_numpy(float)
    start = t[0]
    while start + length_s <= t[-1] + 1e-9:
        sel = (t >= start) & (t < start + length_s)
        if sel.sum() >= min_samples:
            yield start, df.loc[sel]
        start += step_s


# Names of the volatility features used by the unsupervised classifier.
# OBD-only version (what the live pipeline has at 1 Hz):
VOLATILITY_COLUMNS = [
    "vol_speed_cv", "vol_speed_dmean", "vol_speed_qcv", "vol_speed_vf",
    "vol_accel_dmean", "obd_a_sd_ms2",
]
# Accelerometer version (closer to the 10 Hz data of the original paper):
VOLATILITY_COLUMNS_IMU = [
    "vol_speed_cv", "vol_speed_dmean", "vol_ah_cv", "vol_ah_dmean", "vol_ah_qcv", "vol_ah_vf",
]
