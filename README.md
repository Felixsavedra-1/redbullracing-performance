# Oracle Red Bull Racing · F1 Analytics

![Dashboard](docs/dashboard.gif)

ETL pipeline extracting 6 seasons of Formula 1 race data (2020–2025) from the Ergast API into a DuckDB star schema, with resumable extraction, schema-validated transforms, and 15+ quality gates enforced in GitHub Actions CI. Analytical layer built with dbt, exposing 14 parameterized models consumed by Power BI and a Three.js interactive dashboard.

## Stack

| Layer | Technology |
|---|---|
| Extraction | Python `requests` — resumable, adaptive backoff |
| Transformation | `pandas` — ref resolution, schema contracts |
| Storage | DuckDB (default), SQLite, or MySQL |
| Analytical layer | dbt — staging views + materialized mart models |
| Analysis | `scipy` OLS/Poisson MLE, SQL, Jupyter, Power BI |
| Visualization | Plotly, Three.js PBR |
| Source | [api.jolpi.ca/ergast/f1](https://api.jolpi.ca/ergast/f1) |

## Quickstart

```bash
pip install -r requirements.txt
cp scripts/config.example.py scripts/config.py
python scripts/run_pipeline.py
```

Runs extract → transform → load → 15+ quality checks → dbt.

## Pipeline

1. **Extract** `scripts/extract_data.py` — Ergast API → `data/raw/*.csv`; resume state in `data/cache/`
2. **Transform** `scripts/transform_data.py` — cleans CSVs, resolves `*_ref` string keys to integer `*_id` surrogates
3. **Load** `scripts/load_data.py` — validates against schema contracts, inserts into DuckDB
4. **Quality** `scripts/data_quality.py` — 15+ checks; raises `RuntimeError` in CI, warns locally
5. **dbt** `dbt/` — 6 staging views + 4 materialized mart tables (`driver_summary`, `pit_stop_efficiency`, `championship_progression`, `qualifying_vs_race`)
6. **Analysis** `scripts/run_analysis.py` — 5 statistical models → PNG charts + interactive 3D dashboard

## Analysis

```bash
python scripts/run_analysis.py --export
open data/exports/dashboard.html
```

| Model | Method | Output |
|---|---|---|
| Championship trajectory | Cumulative points per driver per round | Line chart |
| Teammate comparison | Mean position delta, 95% t-interval, p-value | Bar chart |
| Grid → finish | OLS regression, R², slope, significance | Scatter |
| Pit stop efficiency | Z-score vs season field distribution | Bar chart |
| DNF rate | Poisson MLE, exact 95% CI | Bar chart |

Dashboard: self-contained HTML — Three.js PBR F1 car model + 5 Plotly charts (championship trajectory, bump chart, points gap, performance heatmap, grid vs finish). No server required.

## Queries

14 parameterized SQL queries by constructor:

```bash
python scripts/run_queries.py --list
python scripts/run_queries.py --query driver_summary
python scripts/run_queries.py --query all --export
```

`driver_summary` · `championship_progression` · `pit_stop_efficiency` · `qualifying_vs_race_performance` · `reliability_analysis` · `failure_modes` · `race_start_analysis` · `fastest_laps_analysis` · and 6 more.

## Data Model

Star schema — `f1_analytics.duckdb`

**Dimensions:** `circuits` `seasons` `constructors` `drivers`  
**Facts:** `races` `results` `qualifying` `pit_stops` `constructor_standings` `driver_standings`

Transform resolves `*_ref` string keys to integer `*_id` surrogates before load. Schema DDL: `database/schema/`. Contracts: `scripts/schema_contracts.py`.

## Quality

15+ checks after each load (`scripts/data_quality.py`):

- **Non-empty** — `results`, `drivers`, `races` must have rows
- **Year bounds** — no out-of-range races; all expected years present
- **Uniqueness** — no duplicate PKs in dimension tables
- **FK integrity** — no orphaned keys in `results`, `qualifying`, `pit_stops`
- **Non-negative** — `points`, `laps`, `grid`, `position_order`

Failures raise `RuntimeError` in CI (`GITHUB_ACTIONS=true`). CI also runs `dbt compile` to validate all model SQL.

## Tests

```bash
python -m unittest discover -s tests
python -m unittest tests.test_smoke           # end-to-end pipeline
python -m unittest tests.test_quality_checks  # quality gate integration
python -m unittest tests.test_etl_unit        # schema, DNF sentinel, FK refs
```
