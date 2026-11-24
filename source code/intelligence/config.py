"""
Single source of truth for every number the intelligence layer uses.

Every constant names where it comes from. Three kinds of origin are used and
must never be mixed up in the thesis:

  [PAPER]   taken from a published paper (citation given)
  [PHYSICS] standard vehicle-dynamics / physics value
  [DESIGN]  a design choice of this project, with the reasoning written next
            to it, so it can be defended or changed in one place

The simulator (python-client/single-client.py) imports the harsh-event
thresholds from here, so the edge rule engine and the cloud analysis can no
longer disagree.
"""

G = 9.81  # m/s^2  [PHYSICS]

# ---------------------------------------------------------------------------
# 1. Kinetic event thresholds  [PAPER]
#    Simons-Morton, Cheon, Guo & Albert (2013), Accident Analysis & Prevention
#    51, 27-32, p.3: "longitudinal deceleration/hard braking (<= -0.45 g),
#    longitudinal acceleration/rapid starts (>= 0.35 g), lateral ... (<= -0.50 g)
#    and (>= 0.50 g) accelerations, and yaw rate (+/- 6 degrees per second)."
# ---------------------------------------------------------------------------
HARSH_BRAKE_G = -0.45
RAPID_ACCEL_G = 0.35
HARSH_CORNER_G = 0.50
YAW_RATE_DEG_S = 6.0

# Minimum gap between two events of the same type, so one long manoeuvre is
# one event, not one event per sample.  [DESIGN]
EVENT_MERGE_GAP_S = 2.0

# ---------------------------------------------------------------------------
# 2. Reference classes on longitudinal acceleration  [PAPER]
#    Ameen, Mahamad, Saon, Ahmadon & Yamaguchi (2021), ASTESJ 6(1) 550-569:
#    safe up to +/-2 m/s^2, normal 2-4, risk of collision from 4, dangerous
#    from 7 m/s^2. Derived from OBD speed at 1 Hz - the same signal we use.
# ---------------------------------------------------------------------------
AMEEN_NORMAL_MS2 = 2.0
AMEEN_AGGRESSIVE_MS2 = 4.0
AMEEN_DANGEROUS_MS2 = 7.0

# ---------------------------------------------------------------------------
# 3. Friction circle  [PHYSICS]
#    A tyre can deliver at most mu*g of horizontal acceleration in total:
#        sqrt(a_long^2 + a_lat^2) <= mu * g
#    Peak tyre-road friction coefficients below are typical textbook values
#    (see e.g. Gillespie, Fundamentals of Vehicle Dynamics, 1992; Wong,
#    Theory of Ground Vehicles). They vary with tyre, speed and texture, so
#    they are configurable.  Cite the textbook you actually use.
# ---------------------------------------------------------------------------
FRICTION_MU = {
    "dry_asphalt": 0.80,
    "wet_asphalt": 0.50,
    "gravel": 0.60,
    "snow": 0.20,
    "ice": 0.10,
}
DEFAULT_SURFACE = "dry_asphalt"

# Severity from friction utilisation u = a_horizontal / (mu * g).
# [DESIGN, anchored in physics]: u -> 1 means the tyres are at the limit of
# grip (skid / loss of control).  The bands split the range between a harsh
# event (u ~ 0.45-0.55 on dry road) and the grip limit.
# Worked check: harsh braking at -0.45 g on dry asphalt -> u = 0.56 (medium);
# the same braking on wet asphalt -> u = 0.90 (extreme).  Bad weather raises
# severity for a physical reason, replacing the arbitrary weather mapping.
FRICTION_SEVERITY_BANDS = [  # (upper bound of u, level)
    (0.50, "low"),
    (0.70, "medium"),
    (0.90, "high"),
    (float("inf"), "extreme"),
]

# ---------------------------------------------------------------------------
# 4. Power Model of speed and crashes  [PAPER]
#    Nilsson (2004, Lund University thesis; first formulated 1981-82):
#      fatal crashes   ~ (v1/v0)^4
#      serious injury  ~ (v1/v0)^3
#      injury crashes  ~ (v1/v0)^2
#    Elvik (2009, TOI report 1034/2009) re-estimated the exponents by road
#    type and found smaller effects in urban areas.  Fill in the urban/rural
#    values from that report before using road-type specific exponents; until
#    then the original Nilsson exponents are used everywhere.
#    CAVEAT for the thesis: the model describes changes of MEAN speed of a
#    traffic population. Applying it to one driver's episode (mean speed of
#    the episode vs. the limit) is an adaptation of this project.
# ---------------------------------------------------------------------------
POWER_EXPONENTS = {"fatal": 4.0, "serious": 3.0, "injury": 2.0}
POWER_OUTCOME = "fatal"          # [DESIGN] which outcome drives severity

# An overspeed episode starts when speed exceeds the limit by this ratio and
# lasts at least OVERSPEED_MIN_S.  [DESIGN]  5% ignores speedometer tolerance.
OVERSPEED_RATIO = 1.05
OVERSPEED_MIN_S = 3.0

# Data quality: map-matched limits are sometimes wrong (the Sagbas data has
# 10-20 km/h limits on highways, giving speed/limit up to x11).  A limit is
# treated as implausible when it is below MIN_PLAUSIBLE_LIMIT_KMH and the car
# drives at more than MAX_PLAUSIBLE_RATIO times it.  [DESIGN]
MIN_PLAUSIBLE_LIMIT_KMH = 30.0
MAX_PLAUSIBLE_RATIO = 2.0

# Relative risk -> severity.  [DESIGN]  With exponent 4 the bands correspond to
# speed/limit of about 1.06, 1.11 and 1.19 (e.g. 106 / 111 / 119 on a 100 road).
POWER_SEVERITY_BANDS = [  # (upper bound of relative risk, level)
    (1.25, "low"),
    (1.50, "medium"),
    (2.00, "high"),
    (float("inf"), "extreme"),
]

# ---------------------------------------------------------------------------
# 5. Trip score  [PAPER] Castignani, Derrmann, Frank & Engel (2015),
#    IEEE ITS Magazine 7(1) 91-102: start at 100, deduct points per event by
#    severity, +1 bonus per 0.5 km without events, clamp to [0, 100].
# ---------------------------------------------------------------------------
SEVERITY_POINTS = {"low": 2, "medium": 4, "high": 6, "extreme": 8}
SEVERITY_ORDER = ["low", "medium", "high", "extreme"]
BONUS_SEGMENT_KM = 0.5
BONUS_POINTS = 1
NIGHT_RAISES_SEVERITY = True   # Castignani raise severity at night [PAPER, mapping DESIGN]

# Score classes.  Slide 20 of the deck: boundaries from Castignani's own
# cluster results (midpoint of 67.8 and 88.75 -> 78; calm/aggressive split at
# 47.75 -> 48).  Re-check on our data with k-means.
SCORE_SAFE_MIN = 78
SCORE_MODERATE_MIN = 48

# ---------------------------------------------------------------------------
# 6. Windows for features / volatility  [DESIGN, after Mohammadnazar et al. 2021]
#    They used 3 s windows on 10 Hz data. OBD gives ~1 Hz, so windows are longer.
#    5 s matches the labelled Sagbas (2026) windows exactly.
# ---------------------------------------------------------------------------
WINDOW_S = 5.0
WINDOW_STEP_S = 5.0
