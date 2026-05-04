# Red Bull F1 Analytics

![Dashboard](docs/dashboard.png)

ETL pipeline for Formula 1 performance analysis — Oracle Red Bull Racing, 2020–2025. Extracts from the Ergast-compatible API into a star-schema SQLite database, runs 15 quality gates, and exports five statistical models with an interactive 3D dashboard.

## Stack

| Layer | Technology |
|---|---|
| Extraction | Python `requests` — resumable, adaptive backoff |
| Transformation | `pandas` — ref resolution, schema validation |
| Storage | SQLite (default) or MySQL |
| Analysis | SQL, `scipy` OLS/MLE, Jupyter, Power BI |
| Visualization | Plotly 3D, Three.js PBR |
| Source | [api.jolpi.ca/ergast/f1](https://api.jolpi.ca/ergast/f1) |

## Quickstart

```bash
pip install -r requirements.txt
cp scripts/config.example.py scripts/config.py
python scripts/run_pipeline.py
```

## Flags

```bash
python scripts/run_pipeline.py --fast                        # 2021–2025, reduced retries
python scripts/run_pipeline.py --start-year 2022 --end-year 2024
python scripts/run_pipeline.py --skip-extract                # reuse cached data
python scripts/run_pipeline.py --skip-pit-stops
python scripts/run_pipeline.py --incremental                 # upsert instead of full refresh
python scripts/run_pipeline.py --base-delay 2.0 --max-retries 8
```

## Analysis

```bash
python scripts/run_analysis.py --export
open data/exports/dashboard.html
```

| Model | Method |
|---|---|
| Championship trajectory | Cumulative points per driver per round |
| Teammate comparison | Mean position delta, 95% t-interval, p-value |
| Grid → finish | OLS, R², slope, significance |
| Pit stop efficiency | Z-score vs season field |
| DNF rate | Poisson MLE, exact 95% CI |

Self-contained HTML — no server. Three.js PBR car model above three Plotly charts (trajectory, finish positions, grid vs finish). Neon glow traces, orbital rotation, black background.

## Queries

14 parameterized SQL queries by constructor.

```bash
python scripts/run_queries.py --list
python scripts/run_queries.py --query driver_summary
python scripts/run_queries.py --query all --export
```

`driver_summary` · `championship_progression` · `pit_stop_efficiency` · `qualifying_vs_race_performance` · `reliability_analysis` · `failure_modes`

## Data Model

Star schema — `f1_analytics.db`.

**Dimensions:** `circuits` `seasons` `constructors` `drivers`  
**Facts:** `races` `results` `qualifying` `pit_stops` `constructor_standings` `driver_standings`

Transform resolves `*_ref` string keys to integer `*_id` surrogates. DDL: `database/schema/`. Contracts: `scripts/schema_contracts.py`.

## Quality

15+ checks after each load (`--skip-quality` to bypass):

- **Non-empty** — `results`, `drivers`, `races`
- **Year bounds** — no out-of-range races; all expected years present
- **Uniqueness** — no duplicate PKs in dimensions
- **FK integrity** — no orphaned keys in `results`, `qualifying`, `pit_stops`
- **Non-negative** — `points`, `laps`, `grid`, `position_order`

In CI (`GITHUB_ACTIONS=true`), any failure raises `RuntimeError`.

## Tests

```bash
python -m unittest discover -s tests
python -m unittest tests.test_smoke           # end-to-end pipeline
python -m unittest tests.test_quality_checks  # quality gate integration
python -m unittest tests.test_etl_unit        # schema, DNF sentinel, FK refs
```

## MySQL

SQLite needs no configuration. For MySQL:

```bash
cp scripts/config.example.py scripts/config.py
mysql -u root -p < database/schema/create_tables.sql
```

## Structure

```
├── data/
│   ├── raw/               # extracted CSVs
│   ├── processed/         # transformed CSVs
│   ├── cache/             # extraction resume state
│   └── exports/           # dashboard.html + charts/
├── database/
│   ├── queries/           # analytical_queries.yaml + .sql
│   └── schema/            # DDL for SQLite and MySQL
├── scripts/
│   ├── run_pipeline.py    # main entry point
│   ├── run_analysis.py    # models + export
│   ├── run_queries.py     # query runner
│   ├── extract_data.py
│   ├── transform_data.py
│   ├── load_data.py
│   ├── analytics.py
│   ├── dashboard.py
│   ├── data_quality.py
│   └── schema_contracts.py
└── powerbi/
```

## Troubleshooting

**Rate limiting** — increase `--base-delay` (default 1.5 s) or narrow the year range. Extraction is resumable.

**Incomplete season** — unpublished rounds are skipped automatically.

**MySQL errors** — verify `scripts/config.py` and check server reachability (`mysql.server status`).
