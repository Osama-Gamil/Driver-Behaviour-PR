-- Runs once, automatically, the first time the timescaledb container
-- initializes an empty data volume (see docker-compose.yml volumes mount
-- of ./db-init -> /docker-entrypoint-initdb.d).

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ---------------------------------------------------------------------
-- Dimension table: one row per vehicle. Small, mostly static, upserted
-- by the consumer every time it sees a batch (driver/last_seen can change).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vehicles (
    vin         TEXT PRIMARY KEY,
    device_id   TEXT NOT NULL,
    fleet_id    TEXT NOT NULL,
    driver_id   TEXT NOT NULL,
    make        TEXT,
    model       TEXT,
    fuel_type   TEXT,
    first_seen  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- Fact table: one row per OBD-II sample. High volume, append-only ->
-- hypertable so Timescale auto-partitions it into time-based chunks.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS telemetry_samples (
    time            TIMESTAMPTZ       NOT NULL,
    vin             TEXT              NOT NULL REFERENCES vehicles(vin),
    trip_id         TEXT,
    seq             INTEGER,
    rpm             DOUBLE PRECISION,
    speed_kmh       DOUBLE PRECISION,
    coolant_temp_c  DOUBLE PRECISION,
    throttle_pct    DOUBLE PRECISION,
    fuel_level_pct  DOUBLE PRECISION,
    engine_load_pct DOUBLE PRECISION,
    voltage         DOUBLE PRECISION,
    intake_temp_c   DOUBLE PRECISION,
    accel_ms2       DOUBLE PRECISION,
    g_force         DOUBLE PRECISION,
    gear            SMALLINT,
    engine_on       BOOLEAN,
    state           TEXT,
    lat             DOUBLE PRECISION,
    lon             DOUBLE PRECISION,
    heading         DOUBLE PRECISION,
    distance_km     DOUBLE PRECISION,
    odometer_km     DOUBLE PRECISION,
    dtcs            JSONB
);

SELECT create_hypertable('telemetry_samples', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_telemetry_vin_time ON telemetry_samples (vin, time DESC);

-- ---------------------------------------------------------------------
-- Fact table: one row per harsh-event / DTC / overspeed episode.
-- Also a hypertable: lower volume than telemetry, but still time-series.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vehicle_events (
    time        TIMESTAMPTZ NOT NULL,
    event_id    TEXT        NOT NULL,
    vin         TEXT        NOT NULL REFERENCES vehicles(vin),
    driver_id   TEXT,
    trip_id     TEXT,
    type        TEXT        NOT NULL,
    severity    TEXT,
    lat         DOUBLE PRECISION,
    lon         DOUBLE PRECISION,
    detail      JSONB,
    source      TEXT,
    PRIMARY KEY (event_id, time)
);

SELECT create_hypertable('vehicle_events', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_events_vin_time ON vehicle_events (vin, time DESC);
