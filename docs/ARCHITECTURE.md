# Architecture Overview

This document summarizes the pipeline and data model at a glance.

## Pipeline Flow

```text
Ergast-compatible API
        |
        v
scripts/extract_data.py
        |
        v
data/raw/*.csv + data/cache/*.json (resume state)
        |
        v
scripts/transform_data.py
        |
        v
data/processed/*_clean.csv
        |
        v
scripts/load_data.py                     FastF1 (lap-by-lap telemetry)
        |                                            |
        v                                            v
        +------------->  DuckDB / SQLite / MySQL  <--+  scripts/extract_telemetry.py (laps table)
                                    |
                                    v
                          scripts/data_quality.py
                                    |
                +-------------------+-------------------+
                v                                        v
scripts/run_queries.py / scripts/analytics.py    dbt/ (staging views + mart models)
        (notebooks, Power BI, dashboard)
```

## Data Model (Core Tables)

```text
dimensions
  circuits
  seasons
  constructors
  drivers

facts
  races
  results
  qualifying
  pit_stops
  laps                  (FastF1 telemetry: sector times, tyre compound, stint)
  constructor_standings
  driver_standings
```

Key relationships:
- `races.circuit_id -> circuits.circuit_id`
- `results.race_id -> races.race_id`
- `results.driver_id -> drivers.driver_id`
- `results.constructor_id -> constructors.constructor_id`
- `qualifying`, `pit_stops`, and `laps` share `race_id` and `driver_id` with `results`
