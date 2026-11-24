"""
Reference trip score (Castignani et al., 2015) with physics-based severity.

    S = clamp(100 - sum(P_i) + B, 0, 100)
    P_i : points for event i, from its severity (friction circle / Power Model)
    B   : +1 for every 0.5 km segment of the trip that contains no event
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .events import Event, detect_events, distance_km


@dataclass
class TripScore:
    score: float
    score_class: str
    distance_km: float
    points_lost: int
    bonus: int
    n_events: int
    events_per_100km: float
    events_by_type: dict
    top_reasons: list        # most costly event types, for explainability / the Coach


def score_class(score: float) -> str:
    if score >= C.SCORE_SAFE_MIN:
        return "safe"
    if score >= C.SCORE_MODERATE_MIN:
        return "moderate"
    return "risky"


def bonus_points(event_km: list[float], total_km: float) -> int:
    """Count 0.5 km segments of the trip that contain no event."""
    n_seg = int(total_km // C.BONUS_SEGMENT_KM)
    if n_seg == 0:
        return 0
    hit = {int(km // C.BONUS_SEGMENT_KM) for km in event_km}
    return C.BONUS_POINTS * sum(1 for s in range(n_seg) if s not in hit)


def score_trip(df: pd.DataFrame, events: list[Event] | None = None) -> tuple[TripScore, list[Event]]:
    if events is None:
        events = detect_events(df)
    cum_km = distance_km(df)
    total_km = float(cum_km[-1]) if len(cum_km) else 0.0
    t = df["t"].to_numpy(float)
    event_km = [float(np.interp(ev.t_start, t, cum_km)) for ev in events]

    lost = sum(ev.points for ev in events)
    bonus = bonus_points(event_km, total_km)
    score = float(np.clip(100 - lost + bonus, 0, 100))

    by_type: dict[str, int] = {}
    cost: dict[str, int] = {}
    for ev in events:
        by_type[ev.type] = by_type.get(ev.type, 0) + 1
        cost[ev.type] = cost.get(ev.type, 0) + ev.points
    top = sorted(cost, key=cost.get, reverse=True)[:3]

    return TripScore(
        score=round(score, 1),
        score_class=score_class(score),
        distance_km=round(total_km, 3),
        points_lost=lost,
        bonus=bonus,
        n_events=len(events),
        events_per_100km=round(len(events) / total_km * 100, 2) if total_km > 0 else 0.0,
        events_by_type=by_type,
        top_reasons=top,
    ), events


def driver_score(trip_scores: list[TripScore]) -> float | None:
    """Distance-weighted mean of a driver's trip scores (a long safe trip
    counts more than a 300 m drive to the shop)."""
    km = np.array([ts.distance_km for ts in trip_scores], dtype=float)
    s = np.array([ts.score for ts in trip_scores], dtype=float)
    if len(s) == 0:
        return None
    if km.sum() <= 0:
        return float(s.mean())
    return float(np.average(s, weights=km))
