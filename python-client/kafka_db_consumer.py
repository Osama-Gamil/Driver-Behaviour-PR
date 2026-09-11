#!/usr/bin/env python3
"""
Kafka -> TimescaleDB consumer.

mqtt_kafka_bridge.py forwards everything published under "telemetry/#" on
EMQX into the single Kafka topic "vehicle-telemetry" - that includes both
telemetry batches (telemetry/{fleetId}/{vin}/raw, shape: {..., "samples": [...]})
and discrete events (telemetry/{fleetId}/{vin}/events, shape: {..., "type": ...}).
This consumer tells the two apart by shape and writes each to its own table:

    batch w/ "samples" -> vehicles (upsert) + telemetry_samples (append)
    single event        -> vehicle_events (append)

Offsets are committed only after the DB transaction that wrote them commits,
so a crash between "read from Kafka" and "write to Postgres" replays the
batch instead of silently losing it (at-least-once delivery).

Requires: pip install -r requirements.txt
"""

import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from kafka import KafkaConsumer

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:29092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "vehicle-telemetry")
KAFKA_GROUP_ID = os.environ.get("KAFKA_GROUP_ID", "db-writer")

DB_DSN = os.environ.get(
    "DB_DSN",
    # Port 5433, not 5432: a native Postgres install on this machine already
    # owns 5432, so docker-compose.yml maps the container to 5433 instead.
    "host=localhost port=5433 dbname=telemetry user=iot password=password",
)

# Flush whichever comes first: this many buffered rows, or this much time.
BATCH_MAX_ROWS = 200
BATCH_MAX_SECONDS = 2.0

running = True


def _handle_signal(signum, frame):
    global running
    running = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def connect_db(retries=30, delay=2):
    """Postgres' port can accept TCP connections slightly before it's
    actually ready to serve queries, so retry on OperationalError too."""
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(DB_DSN)
            conn.autocommit = False
            return conn
        except psycopg2.OperationalError as exc:
            print(f"[db] not ready ({attempt}/{retries}): {exc}")
            time.sleep(delay)
    sys.exit("[db] could not connect, giving up.")


def connect_kafka(retries=30, delay=2):
    for attempt in range(1, retries + 1):
        try:
            return KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id=KAFKA_GROUP_ID,
                enable_auto_commit=False,
                auto_offset_reset="earliest",
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            )
        except Exception as exc:
            print(f"[kafka] not ready ({attempt}/{retries}): {exc}")
            time.sleep(delay)
    sys.exit("[kafka] could not connect, giving up.")


def to_ts(epoch_seconds):
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)


def flatten_telemetry(msg):
    """One raw-telemetry message -> (vehicle_row, [sample_row, ...])."""
    vehicle_row = (
        msg["vin"],
        msg["deviceId"],
        msg["fleetId"],
        msg["driverId"],
        msg["vehicle"]["make"],
        msg["vehicle"]["model"],
        msg["vehicle"]["fuelType"],
    )

    sample_rows = []
    for s in msg["samples"]:
        pids = s["pids"]
        derived = s["derived"]
        gps = s["gps"]
        trip = s["trip"]
        sample_rows.append((
            to_ts(s["ts"]), msg["vin"], trip["tripId"], s["seq"],
            pids["0x0C"], pids["0x0D"], pids["0x05"], pids["0x11"],
            pids["0x2F"], pids["0x04"], pids["0x42"], pids["0x0F"],
            derived["accelMs2"], derived["gForce"], derived["gear"],
            derived["engineOn"], derived["state"],
            gps["lat"], gps["lon"], gps["heading"],
            trip["distanceKm"], trip["odometerKm"],
            psycopg2.extras.Json(s["dtcs"]),
        ))
    return vehicle_row, sample_rows


def flatten_event(msg):
    """One event message -> event_row."""
    loc = msg.get("location", {})
    return (
        to_ts(msg["ts"]), msg["eventId"], msg["vin"], msg.get("driverId"),
        msg.get("tripId"), msg["type"], msg.get("severity"),
        loc.get("lat"), loc.get("lon"),
        psycopg2.extras.Json(msg.get("detail", {})), msg.get("source"),
    )


VEHICLE_UPSERT = """
    INSERT INTO vehicles (vin, device_id, fleet_id, driver_id, make, model, fuel_type)
    VALUES %s
    ON CONFLICT (vin) DO UPDATE SET
        driver_id = EXCLUDED.driver_id,
        last_seen = now()
"""

SAMPLE_INSERT = """
    INSERT INTO telemetry_samples (
        time, vin, trip_id, seq, rpm, speed_kmh, coolant_temp_c, throttle_pct,
        fuel_level_pct, engine_load_pct, voltage, intake_temp_c, accel_ms2,
        g_force, gear, engine_on, state, lat, lon, heading, distance_km,
        odometer_km, dtcs
    ) VALUES %s
"""

EVENT_INSERT = """
    INSERT INTO vehicle_events (
        time, event_id, vin, driver_id, trip_id, type, severity, lat, lon, detail, source
    ) VALUES %s
    ON CONFLICT (event_id, time) DO NOTHING
"""


def flush(conn, vehicles, samples, events):
    if not (vehicles or samples or events):
        return
    with conn.cursor() as cur:
        if vehicles:
            psycopg2.extras.execute_values(cur, VEHICLE_UPSERT, list(vehicles.values()))
        if samples:
            psycopg2.extras.execute_values(cur, SAMPLE_INSERT, samples)
        if events:
            psycopg2.extras.execute_values(cur, EVENT_INSERT, events)
    conn.commit()
    print(f"[flush] {len(vehicles)} vehicles, {len(samples)} samples, {len(events)} events")


def main():
    conn = connect_db()
    consumer = connect_kafka()
    print(f"Consuming '{KAFKA_TOPIC}' from {KAFKA_BOOTSTRAP} -> {DB_DSN}")

    vehicles, samples, events = {}, [], []
    last_flush = time.time()

    try:
        while running:
            records = consumer.poll(timeout_ms=1000, max_records=500)
            for batch in records.values():
                for record in batch:
                    msg = record.value
                    try:
                        if "samples" in msg:
                            v_row, s_rows = flatten_telemetry(msg)
                            vehicles[v_row[0]] = v_row
                            samples.extend(s_rows)
                        elif "type" in msg:
                            events.append(flatten_event(msg))
                        else:
                            print(f"[warn] unrecognized message shape, skipping: {msg}")
                    except (KeyError, TypeError) as exc:
                        print(f"[warn] malformed message, skipping: {exc}")

            due = (
                len(samples) + len(events) >= BATCH_MAX_ROWS
                or time.time() - last_flush >= BATCH_MAX_SECONDS
            )
            if due and (vehicles or samples or events):
                try:
                    flush(conn, vehicles, samples, events)
                    consumer.commit()
                except psycopg2.Error as exc:
                    conn.rollback()
                    print(f"[error] DB write failed, will retry next tick: {exc}")
                else:
                    vehicles, samples, events = {}, [], []
                last_flush = time.time()
    finally:
        # Flush whatever's still buffered so Ctrl+C doesn't drop a partial batch.
        try:
            flush(conn, vehicles, samples, events)
            consumer.commit()
        except Exception as exc:
            print(f"[warn] final flush failed: {exc}")
        consumer.close()
        conn.close()
        print("Consumer stopped.")


if __name__ == "__main__":
    main()
