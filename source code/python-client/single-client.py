#!/usr/bin/env python3
"""
Fleet Simulator — Driver Behavior & Vehicle Health AI Platform
==============================================================

Spawns N independent simulated vehicles. Each vehicle:
  * owns its own MQTT connection to EMQX (separate client_id = separate node)
  * runs its own driving state machine (stop -> accelerate -> cruise -> decelerate)
  * derives correlated OBD-II PID values from that physical state
  * publishes telemetry, locally-detected harsh events, and DTCs
  * publishes an LWT so the broker marks it offline if it dies

Topics follow the architecture document:
    telemetry/{fleetId}/{vin}/raw       device -> cloud   (telemetry batches)
    telemetry/{fleetId}/{vin}/events    device -> cloud   (harsh events / DTCs)
    device/{vin}/status                 device -> cloud   (LWT presence, retained)
    device/{vin}/config                 cloud  -> device  (subscribed)

Usage:
    python fleet_simulator.py                        # will prompt for vehicle count
    python fleet_simulator.py -n 50
    python fleet_simulator.py -n 200 --host 10.0.0.5 --interval 0.5 --batch 5
    python fleet_simulator.py -n 3 --dry-run         # print payloads, no broker

Requires: pip install paho-mqtt
"""

import argparse
import json
import math
import random
import signal
import string
import sys
import threading
import time
import warnings
from dataclasses import dataclass

# We deliberately use the v1 callback signatures so this script runs unchanged
# on both paho-mqtt 1.x and 2.x; silence the resulting deprecation notice.
warnings.filterwarnings("ignore", message=".*Callback API version 1 is deprecated.*")

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULTS = {
    "host": "localhost",
    "port": 1883,
    "keepalive": 60,
    "qos": 1,
    "fleet_id": "fleet-001",
    "interval": 1.0,      # seconds between simulation ticks (sample rate)
    "batch": 1,           # telemetry samples per MQTT publish
}

# Map base — vehicles start scattered around this point (Cairo).
BASE_LAT, BASE_LON = 30.0444, 31.2357
SPAWN_RADIUS_DEG = 0.08

G = 9.81  # m/s^2

# Harsh-event thresholds (Section 9.2 of the architecture doc)
HARSH_BRAKE_G = -0.35
HARSH_ACCEL_G = 0.30
HARSH_TURN_G = 0.30
OVERSPEED_MARGIN = 10.0   # km/h above the road limit

# A few realistic diagnostic trouble codes we can inject
DTC_CATALOG = [
    ("P0128", "Coolant thermostat below regulating temperature", "medium"),
    ("P0300", "Random/multiple cylinder misfire detected", "high"),
    ("P0171", "System too lean (Bank 1)", "medium"),
    ("P0420", "Catalyst system efficiency below threshold", "low"),
    ("P0562", "System voltage low", "high"),
    ("P0217", "Engine over temperature condition", "critical"),
]

VEHICLE_MODELS = [
    ("Toyota", "Corolla", "gasoline"),
    ("Hyundai", "Elantra", "gasoline"),
    ("Ford", "Transit", "diesel"),
    ("Mercedes-Benz", "Sprinter", "diesel"),
    ("Nissan", "Sunny", "gasoline"),
    ("Isuzu", "NPR", "diesel"),
]


# --------------------------------------------------------------------------
# Driver profiles — this is what makes the driver-scoring model interesting
# --------------------------------------------------------------------------

@dataclass
class DriverProfile:
    name: str
    accel_g: float          # typical acceleration
    brake_g: float          # typical deceleration
    harsh_chance: float     # probability a manoeuvre becomes a harsh event
    overspeed_bias: float   # km/h habitually over the limit
    idle_chance: float      # tendency to sit idling

    @staticmethod
    def random():
        return random.choices(
            [
                DriverProfile("calm",       0.10, -0.13, 0.01,  -2.0, 0.10),
                DriverProfile("normal",     0.16, -0.20, 0.05,   4.0, 0.20),
                DriverProfile("aggressive", 0.24, -0.30, 0.22,  18.0, 0.35),
            ],
            weights=[0.35, 0.45, 0.20],
        )[0]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def generate_vin(index: int) -> str:
    """17-character VIN-shaped identifier (I, O, Q excluded as per the standard)."""
    alphabet = "".join(c for c in string.ascii_uppercase if c not in "IOQ") + string.digits
    wmi = "5SM"  # fake World Manufacturer Identifier for the simulator
    body = "".join(random.choice(alphabet) for _ in range(8))
    return f"{wmi}{body}{index:06d}"


def gear_for_speed(kmh: float) -> int:
    for gear, upper in enumerate([0, 20, 40, 65, 95, 130], start=0):
        if kmh <= upper:
            return max(gear, 1)
    return 6


def clamp(value, low, high):
    return max(low, min(high, value))


# --------------------------------------------------------------------------
# The vehicle simulator
# --------------------------------------------------------------------------

class VehicleSimulator(threading.Thread):
    """One simulated vehicle = one thread + one independent MQTT connection."""

    def __init__(self, index, cfg, stop_event):
        super().__init__(daemon=True, name=f"vehicle-{index:04d}")
        self.index = index
        self.cfg = cfg
        self.stop_event = stop_event

        # --- identity ---
        self.vin = generate_vin(index)
        self.device_id = f"obd-{self.vin[-8:]}"
        self.make, self.model, self.fuel_type = random.choice(VEHICLE_MODELS)
        self.driver_id = f"drv-{index:05d}"
        self.profile = DriverProfile.random()

        # --- physical state ---
        self.speed = 0.0                       # km/h
        self.accel = 0.0                       # m/s^2
        self.rpm = 0.0
        self.throttle = 0.0                    # %
        self.engine_load = 0.0                 # %
        self.coolant_temp = random.uniform(15, 30)   # deg C, cold start
        self.intake_temp = self.coolant_temp
        self.fuel_level = random.uniform(25, 100)    # %
        self.voltage = 12.5                    # V
        self.odometer_km = random.uniform(5_000, 240_000)
        self.trip_km = 0.0
        self.engine_on = False

        # --- driving state machine ---
        # Stagger the fleet: roughly half the vehicles are already on the road.
        self.speed_limit = random.choice([50, 60, 80, 90, 100, 120])
        self.state = random.choices(
            ["PARKED", "IDLING", "ACCEL", "CRUISE"], weights=[0.30, 0.10, 0.15, 0.45]
        )[0]
        self.state_timer = random.uniform(3, 40)
        self.target_speed = self.speed_limit + self.profile.overspeed_bias

        if self.state in ("IDLING", "ACCEL", "CRUISE"):
            self.engine_on = True
            self.trip_id = f"trip_{self.vin[-6:]}_{int(time.time())}"
            self.coolant_temp = random.uniform(70, 92)   # already warmed up
        if self.state == "CRUISE":
            self.speed = clamp(self.target_speed + random.uniform(-15, 5), 5, 160)

        # --- position ---
        self.lat = BASE_LAT + random.uniform(-SPAWN_RADIUS_DEG, SPAWN_RADIUS_DEG)
        self.lon = BASE_LON + random.uniform(-SPAWN_RADIUS_DEG, SPAWN_RADIUS_DEG)
        self.heading = random.uniform(0, 360)
        self.turn_rate = 0.0                   # deg/s

        # --- faults ---
        self.active_dtcs = []
        self.fault_mode = None                 # 'overheat' | 'charging' | None

        # --- bookkeeping ---
        self.idle_seconds = 0.0
        self.last_idle_alert = 0.0
        self.event_cooldown = {}          # event type -> last emitted timestamp
        self.overspeed_since = None       # start ts of the current overspeed episode
        self.overspeed_peak = 0.0
        self.seq = 0
        self.trip_id = None
        self.batch = []
        self.connected = False
        self.client = None

    # ---------------- MQTT ----------------

    def topic_raw(self):
        return f"telemetry/{self.cfg['fleet_id']}/{self.vin}/raw"

    def topic_events(self):
        return f"telemetry/{self.cfg['fleet_id']}/{self.vin}/events"

    def topic_status(self):
        return f"device/{self.vin}/status"

    def _build_client(self):
        """Create a paho client that works on both paho-mqtt 1.x and 2.x."""
        client_id = f"{self.device_id}"
        try:
            # paho-mqtt >= 2.0
            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION1,
                client_id=client_id,
                clean_session=False,
            )
        except AttributeError:
            # paho-mqtt 1.x
            client = mqtt.Client(client_id=client_id, clean_session=False)

        if self.cfg.get("username"):
            client.username_pw_set(self.cfg["username"], self.cfg.get("password"))

        # Last Will and Testament: broker announces us offline if we drop.
        client.will_set(
            self.topic_status(),
            json.dumps({"vin": self.vin, "status": "offline", "ts": self._now()}),
            qos=1,
            retain=True,
        )
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        # Retry/backoff, mirroring the edge gateway's Retry Manager.
        client.reconnect_delay_set(min_delay=1, max_delay=60)
        return client

    def _on_connect(self, client, userdata, flags, rc, *args):
        if rc == 0:
            self.connected = True
            client.subscribe(f"device/{self.vin}/config", qos=1)
            client.subscribe(f"device/{self.vin}/command", qos=1)
            client.publish(
                self.topic_status(),
                json.dumps({"vin": self.vin, "status": "online", "ts": self._now()}),
                qos=1,
                retain=True,
            )
            if self.cfg["verbose"]:
                print(f"[{self.vin}] connected ({self.profile.name} driver, "
                      f"{self.make} {self.model})")
        else:
            print(f"[{self.vin}] connect failed rc={rc}", file=sys.stderr)

    def _on_disconnect(self, client, userdata, rc, *args):
        self.connected = False
        if rc != 0 and self.cfg["verbose"]:
            print(f"[{self.vin}] unexpected disconnect rc={rc}, reconnecting...")

    def _on_message(self, client, userdata, msg):
        """Cloud -> device config/commands (Section 4.2)."""
        try:
            payload = json.loads(msg.payload.decode())
        except (ValueError, UnicodeDecodeError):
            return
        if msg.topic.endswith("/config"):
            if "interval" in payload:
                self.cfg["interval"] = float(payload["interval"])
            if self.cfg["verbose"]:
                print(f"[{self.vin}] config update: {payload}")

    # ---------------- simulation ----------------

    @staticmethod
    def _now():
        return time.time()

    def _pick_next_state(self):
        """Driving state machine. Each state runs for a randomised duration."""
        if self.state == "PARKED":
            self.state = "IDLING" if random.random() < self.profile.idle_chance else "ACCEL"
            self.state_timer = random.uniform(5, 30)
            self.engine_on = True
            self.trip_id = f"trip_{self.vin[-6:]}_{int(time.time())}"
            self.trip_km = 0.0
            self.speed_limit = random.choice([50, 60, 80, 90, 100, 120])
            self.target_speed = self.speed_limit + self.profile.overspeed_bias + random.uniform(-8, 8)

        elif self.state == "IDLING":
            self.state = "ACCEL"
            self.state_timer = random.uniform(8, 25)
            self.target_speed = self.speed_limit + self.profile.overspeed_bias + random.uniform(-8, 8)

        elif self.state == "ACCEL":
            self.state = "CRUISE"
            self.state_timer = random.uniform(20, 120)

        elif self.state == "CRUISE":
            roll = random.random()
            if roll < 0.15:
                # new road segment -> new limit and target
                self.speed_limit = random.choice([50, 60, 80, 90, 100, 120])
                self.target_speed = self.speed_limit + self.profile.overspeed_bias + random.uniform(-8, 8)
                self.state = "ACCEL" if self.target_speed > self.speed else "DECEL"
                self.state_timer = random.uniform(6, 20)
            elif roll < 0.55:
                self.state = "DECEL"          # traffic light / junction
                self.target_speed = random.choice([0, 0, 15, 30])
                self.state_timer = random.uniform(4, 15)
            else:
                self.state_timer = random.uniform(20, 90)

        elif self.state == "DECEL":
            if self.speed < 2:
                self.state = "IDLING" if random.random() < 0.6 else "PARKED"
                self.state_timer = random.uniform(5, 45)
                if self.state == "PARKED":
                    self.engine_on = False
            else:
                self.state = "CRUISE"
                self.state_timer = random.uniform(15, 60)

    def _step(self, dt):
        """Advance the physical model by dt seconds and return any events."""
        events = []

        self.state_timer -= dt
        if self.state_timer <= 0:
            self._pick_next_state()

        # ---- longitudinal dynamics ----
        harsh = random.random() < self.profile.harsh_chance

        if self.state in ("PARKED", "IDLING"):
            target_a = -1.5 if self.speed > 0 else 0.0
        elif self.state == "ACCEL":
            target_a = self.profile.accel_g * G * (1.6 if harsh else 1.0)
            if self.speed >= self.target_speed:
                target_a = 0.0
        elif self.state == "DECEL":
            target_a = self.profile.brake_g * G * (1.7 if harsh else 1.0)
            if self.speed <= self.target_speed:
                target_a = 0.0
        else:  # CRUISE — small corrections around target
            error = self.target_speed - self.speed
            target_a = clamp(error * 0.15, -1.2, 1.2)

        # Smooth toward the target acceleration so the jerk stays plausible.
        self.accel += (target_a - self.accel) * clamp(dt * 2.5, 0, 1)
        self.accel += random.gauss(0, 0.05)

        prev_speed = self.speed
        self.speed = clamp(self.speed + self.accel * dt * 3.6, 0.0, 180.0)
        if self.speed == 0.0:
            self.accel = 0.0

        actual_g = ((self.speed - prev_speed) / 3.6) / dt / G

        # ---- lateral dynamics / heading ----
        if self.speed > 3:
            if random.random() < 0.06:
                self.turn_rate = random.gauss(0, 12 if harsh else 5)
            self.turn_rate *= 0.85
            self.heading = (self.heading + self.turn_rate * dt) % 360
            lateral_g = abs((self.speed / 3.6) * math.radians(self.turn_rate)) / G
        else:
            self.turn_rate = 0.0
            lateral_g = 0.0

        # ---- position ----
        if self.speed > 0:
            dist_km = self.speed * dt / 3600.0
            self.trip_km += dist_km
            self.odometer_km += dist_km
            rad = math.radians(self.heading)
            self.lat += (dist_km / 111.32) * math.cos(rad)
            self.lon += (dist_km / (111.32 * math.cos(math.radians(self.lat)))) * math.sin(rad)

        # ---- engine derived signals ----
        if self.engine_on:
            gear = gear_for_speed(self.speed)
            base_rpm = 780 + (self.speed / gear) * 62
            self.rpm = clamp(base_rpm + max(self.accel, 0) * 260 + random.gauss(0, 40), 600, 6500)
            self.throttle = clamp(
                8 + max(self.accel, 0) * 22 + self.speed * 0.16 + random.gauss(0, 2), 0, 100
            )
            self.engine_load = clamp(
                12 + self.throttle * 0.7 + max(self.accel, 0) * 9 + random.gauss(0, 3), 0, 100
            )
            self.voltage = clamp(14.2 + random.gauss(0, 0.08), 13.4, 14.8)

            # Coolant warm-up toward operating temperature.
            target_temp = 88 + self.engine_load * 0.06
            if self.fault_mode == "overheat":
                target_temp = 118
            self.coolant_temp += (target_temp - self.coolant_temp) * clamp(dt / 90.0, 0, 1)
            self.intake_temp = clamp(28 + self.coolant_temp * 0.09 + random.gauss(0, 1.5), 10, 70)

            burn = (self.rpm / 6500.0) * (self.engine_load / 100.0) * dt * 0.0022
            self.fuel_level = clamp(self.fuel_level - burn, 0.0, 100.0)
            if self.fuel_level <= 0.5:
                self.fuel_level = random.uniform(85, 100)   # refuelled
        else:
            self.rpm = 0.0
            self.throttle = 0.0
            self.engine_load = 0.0
            self.voltage = clamp(12.5 + random.gauss(0, 0.05), 11.8, 12.8)
            self.coolant_temp += (25 - self.coolant_temp) * clamp(dt / 600.0, 0, 1)

        if self.fault_mode == "charging":
            self.voltage = clamp(self.voltage - 2.1, 10.8, 13.0)

        # ---- local event detection (edge Rule Engine, Section 3.2) ----
        if actual_g <= HARSH_BRAKE_G and self._cooled_down("harsh_braking", 4):
            events.append(self._event("harsh_braking", "high", {
                "gForce": round(actual_g, 3),
                "speedBefore": round(prev_speed, 1),
                "speedAfter": round(self.speed, 1),
            }))
        elif actual_g >= HARSH_ACCEL_G and self._cooled_down("rapid_acceleration", 4):
            events.append(self._event("rapid_acceleration", "medium", {
                "gForce": round(actual_g, 3),
                "speedBefore": round(prev_speed, 1),
                "speedAfter": round(self.speed, 1),
            }))

        if lateral_g >= HARSH_TURN_G and self._cooled_down("sharp_turn", 4):
            events.append(self._event("sharp_turn", "medium", {
                "lateralG": round(lateral_g, 3),
                "turnRateDegPerSec": round(self.turn_rate, 2),
                "speed": round(self.speed, 1),
            }))

        # Overspeed is reported as an episode (start -> end), not per sample,
        # otherwise one long motorway stretch floods the alerts topic.
        if self.speed > self.speed_limit + OVERSPEED_MARGIN:
            if self.overspeed_since is None:
                self.overspeed_since = self._now()
                self.overspeed_peak = self.speed
            else:
                self.overspeed_peak = max(self.overspeed_peak, self.speed)
        elif self.overspeed_since is not None:
            duration = self._now() - self.overspeed_since
            if duration >= 3.0:       # ignore momentary blips
                events.append(self._event("overspeed", "medium", {
                    "durationSeconds": round(duration, 1),
                    "peakSpeed": round(self.overspeed_peak, 1),
                    "speedLimit": self.speed_limit,
                    "excess": round(self.overspeed_peak - self.speed_limit, 1),
                }))
            self.overspeed_since = None
            self.overspeed_peak = 0.0

        # Idle = engine running but stationary (Section 9.2)
        if self.engine_on and self.speed < 1.0:
            self.idle_seconds += dt
        else:
            self.idle_seconds = 0.0
            self.last_idle_alert = 0.0

        if self.idle_seconds >= 180 and (self.idle_seconds - self.last_idle_alert) >= 60:
            self.last_idle_alert = self.idle_seconds
            events.append(self._event("excessive_idling", "low", {
                "idleSeconds": round(self.idle_seconds),
                "rpm": round(self.rpm, 0),
            }))

        # ---- fault injection ----
        # Roughly one injected fault per vehicle every few hours of driving.
        if self.fault_mode is None and self.engine_on and random.random() < 0.00008 * dt:
            code, desc, severity = random.choice(DTC_CATALOG)
            if code not in [d["code"] for d in self.active_dtcs]:
                self.active_dtcs.append({"code": code, "description": desc})
                if code in ("P0217", "P0128"):
                    self.fault_mode = "overheat"
                elif code == "P0562":
                    self.fault_mode = "charging"
                events.append(self._event("dtc_set", severity, {
                    "code": code, "description": desc
                }))

        if self.coolant_temp > 112:
            events.append(self._event("overheating", "critical", {
                "coolantTempC": round(self.coolant_temp, 1)
            }))

        return events

    def _cooled_down(self, event_type, min_gap_seconds):
        """Suppress repeat events of the same type inside a short window."""
        now = self._now()
        if now - self.event_cooldown.get(event_type, 0.0) < min_gap_seconds:
            return False
        self.event_cooldown[event_type] = now
        return True

    def _event(self, event_type, severity, detail):
        return {
            "eventId": f"{self.vin}-{int(time.time() * 1000)}-{random.randint(100, 999)}",
            "vin": self.vin,
            "driverId": self.driver_id,
            "tripId": self.trip_id,
            "type": event_type,
            "severity": severity,
            "ts": round(self._now(), 3),
            "location": {"lat": round(self.lat, 6), "lon": round(self.lon, 6)},
            "detail": detail,
            "source": "edge_rule_engine",
        }

    def _sample(self):
        """One telemetry sample, keyed by OBD-II Mode 01 PID (Section 2.3)."""
        self.seq += 1
        return {
            "seq": self.seq,
            "ts": round(self._now(), 3),
            "pids": {
                "0x0C": round(self.rpm, 1),            # Engine RPM
                "0x0D": round(self.speed, 1),          # Vehicle speed (km/h)
                "0x05": round(self.coolant_temp, 1),   # Coolant temperature (C)
                "0x11": round(self.throttle, 1),       # Throttle position (%)
                "0x2F": round(self.fuel_level, 1),     # Fuel tank level (%)
                "0x04": round(self.engine_load, 1),    # Calculated engine load (%)
                "0x42": round(self.voltage, 2),        # Control module voltage (V)
                "0x0F": round(self.intake_temp, 1),    # Intake air temperature (C)
            },
            "derived": {
                "accelMs2": round(self.accel, 3),
                "gForce": round(self.accel / G, 3),
                "gear": gear_for_speed(self.speed) if self.engine_on else 0,
                "engineOn": self.engine_on,
                "state": self.state,
            },
            "gps": {
                "lat": round(self.lat, 6),
                "lon": round(self.lon, 6),
                "heading": round(self.heading, 1),
                "speedKmh": round(self.speed, 1),
            },
            "trip": {
                "tripId": self.trip_id,
                "distanceKm": round(self.trip_km, 3),
                "odometerKm": round(self.odometer_km, 1),
            },
            "dtcs": [d["code"] for d in self.active_dtcs],
        }

    def _flush(self):
        """Publish the buffered batch. If offline, the batch is kept (store-and-forward)."""
        if not self.batch:
            return
        payload = {
            "vin": self.vin,
            "deviceId": self.device_id,
            "fleetId": self.cfg["fleet_id"],
            "driverId": self.driver_id,
            "vehicle": {"make": self.make, "model": self.model, "fuelType": self.fuel_type},
            "batchTs": round(self._now(), 3),
            "samples": self.batch,
        }
        body = json.dumps(payload, separators=(",", ":"))

        if self.cfg["dry_run"]:
            print(f"[{self.vin}] {self.topic_raw()} -> {body}")
            self.batch = []
            return

        if self.connected:
            self.client.publish(self.topic_raw(), body, qos=self.cfg["qos"])
            self.batch = []
            if self.cfg["verbose"]:
                last = payload["samples"][-1]["pids"]
                print(f"[{self.vin}] speed={last['0x0D']:6.1f} rpm={last['0x0C']:6.0f} "
                      f"thr={last['0x11']:5.1f} temp={last['0x05']:5.1f} "
                      f"state={self.state}")
        else:
            # Offline: cap the local buffer like the edge Local Cache ring buffer.
            if len(self.batch) > 5000:
                self.batch = self.batch[-5000:]

    def _publish_events(self, events):
        for event in events:
            body = json.dumps(event, separators=(",", ":"))
            if self.cfg["dry_run"]:
                print(f"[{self.vin}] EVENT {event['type']} ({event['severity']}) -> {body}")
            elif self.connected:
                self.client.publish(self.topic_events(), body, qos=1)
            if self.cfg["verbose"] and not self.cfg["dry_run"]:
                print(f"  !! [{self.vin}] {event['type']} ({event['severity']})")

    # ---------------- thread entry point ----------------

    def run(self):
        if not self.cfg["dry_run"]:
            self.client = self._build_client()
            try:
                self.client.connect_async(self.cfg["host"], self.cfg["port"], self.cfg["keepalive"])
                self.client.loop_start()
            except Exception as exc:
                print(f"[{self.vin}] connection error: {exc}", file=sys.stderr)
                return

        dt = self.cfg["interval"]
        next_tick = time.time()

        try:
            while not self.stop_event.is_set():
                events = self._step(dt)
                self.batch.append(self._sample())
                if events:
                    self._publish_events(events)
                if len(self.batch) >= self.cfg["batch"]:
                    self._flush()

                next_tick += dt
                sleep_for = next_tick - time.time()
                if sleep_for > 0:
                    self.stop_event.wait(sleep_for)
                else:
                    next_tick = time.time()   # we fell behind; resync
        finally:
            self._shutdown()

    def _shutdown(self):
        if self.cfg["dry_run"] or self.client is None:
            return
        try:
            self._flush()
            self.client.publish(
                self.topic_status(),
                json.dumps({"vin": self.vin, "status": "offline", "ts": self._now()}),
                qos=1,
                retain=True,
            )
            time.sleep(0.05)
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Simulate a fleet of OBD-II vehicles against EMQX.")
    p.add_argument("-n", "--vehicles", type=int, help="number of vehicles to simulate")
    p.add_argument("--host", default=DEFAULTS["host"], help="EMQX broker host")
    p.add_argument("--port", type=int, default=DEFAULTS["port"], help="EMQX broker port")
    p.add_argument("--username", help="MQTT username (if auth is enabled)")
    p.add_argument("--password", help="MQTT password")
    p.add_argument("--fleet-id", default=DEFAULTS["fleet_id"], help="fleet identifier used in topics")
    p.add_argument("--interval", type=float, default=DEFAULTS["interval"],
                   help="seconds between telemetry samples")
    p.add_argument("--batch", type=int, default=DEFAULTS["batch"],
                   help="samples buffered per MQTT publish")
    p.add_argument("--qos", type=int, choices=[0, 1, 2], default=DEFAULTS["qos"])
    p.add_argument("--stagger", type=float, default=0.02,
                   help="seconds between vehicle starts (avoids connection thundering herd)")
    p.add_argument("--duration", type=float, help="stop automatically after N seconds")
    p.add_argument("--dry-run", action="store_true", help="print payloads instead of publishing")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress per-sample logging")
    return p.parse_args()


def ask_vehicle_count():
    while True:
        try:
            raw = input("How many vehicles do you want to simulate? ").strip()
            count = int(raw)
            if count > 0:
                return count
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a whole number.")
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)


def main():
    args = parse_args()

    if mqtt is None and not args.dry_run:
        sys.exit("paho-mqtt is not installed. Run: pip install paho-mqtt")

    count = args.vehicles if args.vehicles else ask_vehicle_count()

    cfg = {
        "host": args.host,
        "port": args.port,
        "keepalive": DEFAULTS["keepalive"],
        "username": args.username,
        "password": args.password,
        "fleet_id": args.fleet_id,
        "interval": args.interval,
        "batch": max(1, args.batch),
        "qos": args.qos,
        "dry_run": args.dry_run,
        "verbose": not args.quiet,
    }

    stop_event = threading.Event()

    def handle_signal(signum, frame):
        print("\nShutting down fleet...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    target = "dry-run (no broker)" if args.dry_run else f"{args.host}:{args.port}"
    print(f"Starting {count} simulated vehicle(s) -> {target}")
    print(f"Sample interval {cfg['interval']}s, batch size {cfg['batch']}, QoS {cfg['qos']}")
    print("Press Ctrl+C to stop.\n")

    vehicles = []
    for i in range(1, count + 1):
        vehicle = VehicleSimulator(i, dict(cfg), stop_event)
        vehicle.start()
        vehicles.append(vehicle)
        if args.stagger:
            time.sleep(args.stagger)

    deadline = time.time() + args.duration if args.duration else None
    try:
        while not stop_event.is_set():
            if deadline and time.time() >= deadline:
                stop_event.set()
                break
            time.sleep(0.25)
    except KeyboardInterrupt:
        stop_event.set()

    for vehicle in vehicles:
        vehicle.join(timeout=5)

    total_samples = sum(v.seq for v in vehicles)
    total_km = sum(v.trip_km for v in vehicles)
    print(f"\nStopped. {len(vehicles)} vehicles, {total_samples} samples generated, "
          f"{total_km:.1f} km simulated.")


if __name__ == "__main__":
    main()