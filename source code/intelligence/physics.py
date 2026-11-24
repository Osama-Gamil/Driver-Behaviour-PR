"""
Kinetic models: acceleration from OBD speed, lateral acceleration from GPS,
the friction circle, and the Power Model of speed and crash risk.

All functions are pure (numpy in, numpy/float out) so they can be unit-tested
and reused by the batch experiments and the live scoring service alike.
"""

from __future__ import annotations

import numpy as np

from . import config as C


# ---------------------------------------------------------------------------
# Longitudinal acceleration from OBD speed (PID 0x0D)
# ---------------------------------------------------------------------------

def accel_from_speed(t_s: np.ndarray, speed_kmh: np.ndarray) -> np.ndarray:
    """a = dv/dt in g, from speed in km/h sampled at times t (seconds).

    Uses central differences (np.gradient), which handles irregular sampling.
    This is the method of Ameen et al. (2021) and slide 10 of the deck:
    km/h / 3.6 -> m/s, change per second, / 9.81 -> g.
    """
    t_s = np.asarray(t_s, dtype=float)
    v = np.asarray(speed_kmh, dtype=float) / 3.6
    if len(v) < 2:
        return np.zeros_like(v)
    return np.gradient(v, t_s) / C.G


# ---------------------------------------------------------------------------
# Lateral acceleration from GPS heading (when no accelerometer is present)
# ---------------------------------------------------------------------------

def bearing_deg(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Initial bearing between consecutive GPS fixes, degrees [0, 360).
    The last value repeats the previous one so the output has len(lat)."""
    lat1, lat2 = np.radians(lat[:-1]), np.radians(lat[1:])
    dlon = np.radians(np.asarray(lon[1:]) - np.asarray(lon[:-1]))
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    b = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return np.append(b, b[-1] if len(b) else 0.0)


def yaw_rate_deg_s(t_s: np.ndarray, heading_deg: np.ndarray) -> np.ndarray:
    """Rate of heading change, handling the 359 -> 0 wrap."""
    h = np.unwrap(np.radians(np.asarray(heading_deg, dtype=float)))
    if len(h) < 2:
        return np.zeros_like(h)
    return np.degrees(np.gradient(h, np.asarray(t_s, dtype=float)))


def lateral_accel_g(speed_kmh: np.ndarray, yaw_deg_s: np.ndarray,
                    min_speed_kmh: float = 10.0) -> np.ndarray:
    """Centripetal acceleration a_lat = v * omega (equivalent to v^2 / r).

    Below min_speed_kmh GPS heading is mostly noise, so the result is 0 there.
    """
    v = np.asarray(speed_kmh, dtype=float) / 3.6
    omega = np.radians(np.asarray(yaw_deg_s, dtype=float))
    a = v * omega / C.G
    return np.where(np.asarray(speed_kmh) >= min_speed_kmh, a, 0.0)


def horizontal_accel_g(linear_xyz: np.ndarray, gravity_xyz: np.ndarray) -> np.ndarray:
    """Magnitude of a phone's acceleration in the road plane, in g.

    Projects the gravity-free acceleration onto the plane perpendicular to
    gravity, so the result does not depend on how the phone is mounted.
    This is exactly the quantity the friction circle limits.
    """
    a = np.asarray(linear_xyz, dtype=float)
    g = np.asarray(gravity_xyz, dtype=float)
    g_norm = np.linalg.norm(g, axis=1, keepdims=True)
    g_hat = np.divide(g, g_norm, out=np.zeros_like(g), where=g_norm > 0)
    vertical = np.sum(a * g_hat, axis=1, keepdims=True) * g_hat
    return np.linalg.norm(a - vertical, axis=1) / C.G


# ---------------------------------------------------------------------------
# Friction circle
# ---------------------------------------------------------------------------

def friction_mu(surface: str | None = None, wet: bool = False) -> float:
    """Peak friction coefficient for a surface; wet asphalt if wet=True."""
    if wet and (surface in (None, "asphalt", "dry_asphalt")):
        return C.FRICTION_MU["wet_asphalt"]
    key = {"asphalt": "dry_asphalt", None: C.DEFAULT_SURFACE}.get(surface, surface)
    return C.FRICTION_MU.get(key, C.FRICTION_MU[C.DEFAULT_SURFACE])


def combined_accel_g(a_long_g, a_lat_g) -> np.ndarray:
    """Total horizontal demand on the tyres: sqrt(a_long^2 + a_lat^2)."""
    return np.hypot(np.asarray(a_long_g, dtype=float), np.asarray(a_lat_g, dtype=float))


def friction_utilisation(a_horizontal_g, mu: float) -> np.ndarray:
    """Share of the available grip in use: u = a_h / (mu * g) (in g units: a_h / mu).
    u >= 1 means the manoeuvre exceeds what the road can deliver."""
    return np.asarray(a_horizontal_g, dtype=float) / mu


def _band(value: float, bands) -> str:
    for upper, level in bands:
        if value < upper:
            return level
    return bands[-1][1]


def friction_severity(u: float) -> str:
    return _band(float(u), C.FRICTION_SEVERITY_BANDS)


# ---------------------------------------------------------------------------
# Power Model (Nilsson)
# ---------------------------------------------------------------------------

def power_model_relative_risk(v_kmh, v_ref_kmh, outcome: str = C.POWER_OUTCOME,
                              exponent: float | None = None):
    """Relative crash risk when travelling at v instead of v_ref:
        R = (v / v_ref) ^ p,  p = 4 (fatal), 3 (serious), 2 (injury).
    Example: 10% above the reference -> 1.1^4 = 1.46x fatal-crash risk."""
    p = C.POWER_EXPONENTS[outcome] if exponent is None else exponent
    return (np.asarray(v_kmh, dtype=float) / np.asarray(v_ref_kmh, dtype=float)) ** p


def plausible_limit(v_kmh, limit_kmh) -> np.ndarray:
    """True where a speed limit can be trusted (data-quality rule, see config)."""
    v = np.asarray(v_kmh, dtype=float)
    lim = np.asarray(limit_kmh, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.isfinite(lim) & (lim > 0) & (
            (lim >= C.MIN_PLAUSIBLE_LIMIT_KMH) | (v <= lim * C.MAX_PLAUSIBLE_RATIO))


def power_severity(relative_risk: float) -> str:
    return _band(float(relative_risk), C.POWER_SEVERITY_BANDS)


def raise_severity(level: str, steps: int = 1) -> str:
    i = min(C.SEVERITY_ORDER.index(level) + steps, len(C.SEVERITY_ORDER) - 1)
    return C.SEVERITY_ORDER[i]
