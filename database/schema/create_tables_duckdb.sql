-- DuckDB schema for Oracle Red Bull Racing F1 Performance Database

CREATE TABLE IF NOT EXISTS circuits (
    circuit_id INTEGER PRIMARY KEY,
    circuit_ref TEXT UNIQUE,
    circuit_name TEXT,
    location TEXT,
    country TEXT,
    lat DOUBLE,
    lng DOUBLE,
    altitude INTEGER,
    url TEXT
);

CREATE TABLE IF NOT EXISTS seasons (
    year INTEGER PRIMARY KEY,
    url TEXT
);

CREATE TABLE IF NOT EXISTS constructors (
    constructor_id INTEGER PRIMARY KEY,
    constructor_ref TEXT UNIQUE,
    constructor_name TEXT,
    nationality TEXT,
    url TEXT
);

CREATE TABLE IF NOT EXISTS drivers (
    driver_id INTEGER PRIMARY KEY,
    driver_ref TEXT UNIQUE,
    driver_number INTEGER,
    code TEXT,
    forename TEXT,
    surname TEXT,
    dob TEXT,
    nationality TEXT,
    url TEXT
);

CREATE TABLE IF NOT EXISTS races (
    race_id INTEGER PRIMARY KEY,
    year INTEGER,
    round INTEGER,
    circuit_id INTEGER NOT NULL,
    race_name TEXT,
    race_date TEXT,
    race_time TEXT,
    url TEXT,
    FOREIGN KEY (year) REFERENCES seasons(year),
    FOREIGN KEY (circuit_id) REFERENCES circuits(circuit_id)
);

CREATE TABLE IF NOT EXISTS qualifying (
    race_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    constructor_id INTEGER NOT NULL,
    number INTEGER,
    position INTEGER,
    q1 TEXT,
    q2 TEXT,
    q3 TEXT,
    PRIMARY KEY (race_id, driver_id),
    FOREIGN KEY (race_id) REFERENCES races(race_id),
    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id),
    FOREIGN KEY (constructor_id) REFERENCES constructors(constructor_id)
);

CREATE TABLE IF NOT EXISTS results (
    race_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    constructor_id INTEGER NOT NULL,
    number INTEGER,
    grid INTEGER,
    position INTEGER,
    position_text TEXT,
    position_order INTEGER CHECK (position_order >= 0),
    points DOUBLE CHECK (points >= 0),
    laps INTEGER CHECK (laps >= 0),
    time_result TEXT,
    milliseconds INTEGER,
    fastest_lap INTEGER,
    fastest_lap_rank INTEGER,
    fastest_lap_time TEXT,
    fastest_lap_speed TEXT,
    status TEXT,
    PRIMARY KEY (race_id, driver_id, constructor_id),
    FOREIGN KEY (race_id) REFERENCES races(race_id),
    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id),
    FOREIGN KEY (constructor_id) REFERENCES constructors(constructor_id)
);

CREATE TABLE IF NOT EXISTS pit_stops (
    race_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    stop INTEGER,
    lap INTEGER,
    time_of_day TEXT,
    duration TEXT,
    milliseconds INTEGER,
    PRIMARY KEY (race_id, driver_id, stop),
    FOREIGN KEY (race_id) REFERENCES races(race_id),
    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
);

CREATE TABLE IF NOT EXISTS constructor_standings (
    race_id INTEGER NOT NULL,
    constructor_id INTEGER NOT NULL,
    points DOUBLE CHECK (points >= 0),
    position INTEGER,
    position_text TEXT,
    wins INTEGER,
    PRIMARY KEY (race_id, constructor_id),
    FOREIGN KEY (race_id) REFERENCES races(race_id),
    FOREIGN KEY (constructor_id) REFERENCES constructors(constructor_id)
);

CREATE TABLE IF NOT EXISTS driver_standings (
    race_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    points DOUBLE CHECK (points >= 0),
    position INTEGER,
    position_text TEXT,
    wins INTEGER,
    PRIMARY KEY (race_id, driver_id),
    FOREIGN KEY (race_id) REFERENCES races(race_id),
    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
);

CREATE TABLE IF NOT EXISTS status (
    status_id INTEGER PRIMARY KEY,
    status TEXT
);

-- FastF1 lap-by-lap telemetry: sector times, tyre compound, stint data.
CREATE TABLE IF NOT EXISTS laps (
    race_id          INTEGER NOT NULL,
    driver_id        INTEGER NOT NULL,
    lap_number       INTEGER NOT NULL,
    lap_time_s       DOUBLE,
    sector1_s        DOUBLE,
    sector2_s        DOUBLE,
    sector3_s        DOUBLE,
    compound         TEXT,
    tyre_life        INTEGER,
    stint            INTEGER,
    is_personal_best INTEGER DEFAULT 0,
    pit_in           INTEGER DEFAULT 0,
    pit_out          INTEGER DEFAULT 0,
    track_status     TEXT,
    PRIMARY KEY (race_id, driver_id, lap_number),
    FOREIGN KEY (race_id)   REFERENCES races(race_id),
    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
);

CREATE INDEX IF NOT EXISTS idx_races_year ON races(year);
CREATE INDEX IF NOT EXISTS idx_races_circuit ON races(circuit_id);
CREATE INDEX IF NOT EXISTS idx_results_race ON results(race_id);
CREATE INDEX IF NOT EXISTS idx_results_driver ON results(driver_id);
CREATE INDEX IF NOT EXISTS idx_results_constructor ON results(constructor_id);
CREATE INDEX IF NOT EXISTS idx_qualifying_race ON qualifying(race_id);
CREATE INDEX IF NOT EXISTS idx_qualifying_driver ON qualifying(driver_id);
CREATE INDEX IF NOT EXISTS idx_pit_stops_race   ON pit_stops(race_id);
CREATE INDEX IF NOT EXISTS idx_pit_stops_driver ON pit_stops(driver_id);
CREATE INDEX IF NOT EXISTS idx_constructor_standings_race ON constructor_standings(race_id);
CREATE INDEX IF NOT EXISTS idx_driver_standings_race ON driver_standings(race_id);
CREATE INDEX IF NOT EXISTS idx_laps_race   ON laps(race_id);
CREATE INDEX IF NOT EXISTS idx_laps_driver ON laps(driver_id);
CREATE INDEX IF NOT EXISTS idx_laps_compound ON laps(compound);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT,
    ended_at TEXT,
    status TEXT,
    source_url TEXT,
    mode TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_run_tables (
    run_id TEXT,
    table_name TEXT,
    rows_loaded INTEGER,
    PRIMARY KEY (run_id, table_name)
);
