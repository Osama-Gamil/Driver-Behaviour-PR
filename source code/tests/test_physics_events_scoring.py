import math

import numpy as np
import pandas as pd
import pytest

from intelligence import config as C
from intelligence import physics as P
from intelligence.events import detect_events, distance_km
from intelligence.scoring import bonus_points, score_class, score_trip


# ---------------- physics ----------------

def test_accel_from_speed_matches_deck_example():
    # Slide 10: 60 -> 43 km/h in 1 s is -0.48 g
    a = P.accel_from_speed(np.array([0.0, 1.0]), np.array([60.0, 43.0]))
    assert a[0] == pytest.approx(-0.481, abs=1e-3)


def test_lateral_accel_matches_v2_over_r():
    # 60 km/h around r = 50 m -> v^2/r = 5.56 m/s^2 = 0.57 g (slide 11)
    v = 60 / 3.6
    omega_deg = math.degrees(v / 50.0)
    a = P.lateral_accel_g(np.array([60.0]), np.array([omega_deg]))
    assert a[0] == pytest.approx(0.566, abs=2e-3)


def test_horizontal_accel_ignores_phone_orientation():
    # Gravity along z, braking 4.41 m/s^2 along x, plus vertical bump on z
    lin = np.array([[-4.41, 0.0, 1.0]])
    grav = np.array([[0.0, 0.0, 9.81]])
    assert P.horizontal_accel_g(lin, grav)[0] == pytest.approx(0.45, abs=1e-3)
    # Same physical event with the phone rotated 90 degrees about y
    lin_r = np.array([[1.0, 0.0, 4.41]])
    grav_r = np.array([[9.81, 0.0, 0.0]])
    assert P.horizontal_accel_g(lin_r, grav_r)[0] == pytest.approx(0.45, abs=1e-3)


def test_friction_severity_rises_on_wet_road():
    dry = P.friction_utilisation(0.45, P.friction_mu("asphalt"))
    wet = P.friction_utilisation(0.45, P.friction_mu("asphalt", wet=True))
    assert P.friction_severity(dry) == "medium"
    assert P.friction_severity(wet) == "extreme"


def test_power_model_examples():
    assert P.power_model_relative_risk(110, 100) == pytest.approx(1.4641)
    assert P.power_model_relative_risk(110, 100, outcome="injury") == pytest.approx(1.21)
    assert P.power_severity(1.4641) == "medium"
    assert P.power_severity(2.07) == "extreme"


# ---------------- events ----------------

def _trip(speeds, limit=None, **extra):
    df = pd.DataFrame({"t": np.arange(len(speeds), dtype=float), "speed_kmh": speeds})
    if limit is not None:
        df["speed_limit_kmh"] = limit
    for k, v in extra.items():
        df[k] = v
    return df


def test_harsh_braking_detected_once_with_friction_severity():
    speeds = [60, 60, 60, 43, 26, 20, 20, 20]   # two seconds of ~-0.48 g
    ev = [e for e in detect_events(_trip(speeds)) if e.type == "harsh_braking"]
    assert len(ev) == 1
    assert ev[0].severity == "medium"           # u = 0.48/0.8 ~ 0.6
    assert ev[0].points == 4


def test_gentle_driving_has_no_events():
    speeds = list(np.linspace(0, 50, 30)) + [50] * 30
    assert detect_events(_trip(speeds)) == []


def test_overspeed_uses_power_model_and_min_duration():
    short = _trip([50] * 5 + [60] * 2 + [50] * 5, limit=50)
    assert [e for e in detect_events(short) if e.type == "overspeed"] == []
    long = _trip([50] * 5 + [60] * 10 + [50] * 5, limit=50)
    ov = [e for e in detect_events(long) if e.type == "overspeed"]
    assert len(ov) == 1
    assert ov[0].relative_risk == pytest.approx((60 / 50) ** 4, rel=1e-3)
    assert ov[0].severity == "extreme"          # 20% over -> 2.07x fatal risk


def test_night_raises_severity():
    speeds = [60, 60, 60, 43, 26, 20, 20, 20]
    ev = [e for e in detect_events(_trip(speeds, is_night=True)) if e.type == "harsh_braking"]
    assert ev[0].severity == "high"


# ---------------- scoring ----------------

def test_distance_integration():
    df = _trip([36.0] * 101)                    # 10 m/s for 100 s = 1 km
    assert distance_km(df)[-1] == pytest.approx(1.0)


def test_bonus_counts_event_free_half_km_segments():
    assert bonus_points([], 5.0) == 10
    assert bonus_points([0.1, 0.2, 2.7], 5.0) == 8


def test_score_classes_from_deck_slide_20():
    assert score_class(78) == "safe"
    assert score_class(77.9) == "moderate"
    assert score_class(47) == "risky"


def test_clean_trip_scores_100():
    ts, events = score_trip(_trip([50.0] * 600))
    assert events == [] and ts.score == 100 and ts.score_class == "safe"
