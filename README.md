# Oracle Red Bull Racing · F1 Analytics

<p align="center">
  <img src="docs/dashboard.gif" alt="Interactive dashboard — Three.js 3D visualization + playable F1 game" width="100%">
</p>

Production-style ETL pipeline covering 6 seasons of F1 race data (2020–2025). Resumable extraction from the Ergast API, schema-validated transforms, 15+ CI-enforced quality gates, 5 statistical models, and a self-contained Three.js dashboard with an embedded playable F1 game — no server required.

---

## Stack

| Layer | Technology |
|---|---|
| Extraction | Python `requests` — resumable, adaptive backoff |
| Transformation | `pandas` — ref resolution, schema contracts |
| Storage | SQLite (default) · MySQL · DuckDB |
| Analytical layer | dbt — 6 staging views + 4 mart models |
| Analysis | `scipy` OLS, Poisson MLE · 14 parameterized SQL queries |
| Visualization | Plotly 3D · Three.js · Power BI |
| CI | GitHub Actions — quality gates + dbt compile |

---

## Pipeline

```bash
pip install -r requirements.txt
cp scripts/config.example.py scripts/config.py
python scripts/run_pipeline.py
```

| Step | Script | Output |
|---|---|---|
| Extract | `extract_data.py` | `data/raw/*.csv` + resume state in `data/cache/` |
| Transform | `transform_data.py` | `*_ref` → `*_id` surrogate key resolution |
| Load | `load_data.py` | Schema-validated insert into database |
| Quality | `data_quality.py` | 15+ checks; raises in CI, warns locally |
| dbt | `dbt/` | Staging views + `driver_summary`, `pit_stop_efficiency`, `championship_progression`, `qualifying_vs_race` |

---

## Analysis

```bash
python scripts/run_analysis.py --export
open data/exports/dashboard.html
```

| Model | Method | Output |
|---|---|---|
| Championship trajectory | Cumulative points per driver per round | Line chart |
| Teammate comparison | Mean Δ position, 95% t-interval, p-value | Bar chart |
| Grid → finish | OLS regression — R², slope, significance | Scatter |
| Pit stop efficiency | Z-score vs season field | Bar chart |
| DNF rate | Poisson MLE, exact 95% CI | Bar chart |

Dashboard: self-contained HTML — Three.js PBR F1 car, 5 Plotly charts, playable F1 racing game with physics-based AI (Pacejka tire model, PID steering), sector timing, and live race positions.

---

## Data Model

Star schema — `f1_analytics.db`

**Dimensions:** `circuits` · `seasons` · `constructors` · `drivers`  
**Facts:** `races` · `results` · `qualifying` · `pit_stops` · `constructor_standings` · `driver_standings`

Schema DDL: `database/schema/` · Validation contracts: `scripts/schema_contracts.py`

---

## Quality Gates

15+ checks run after every load:

- **Non-empty** — `results`, `drivers`, `races` must have rows
- **Year bounds** — no out-of-range races; all expected seasons present
- **Uniqueness** — no duplicate PKs across dimension tables
- **FK integrity** — no orphaned keys in `results`, `qualifying`, `pit_stops`
- **Non-negative** — `points`, `laps`, `grid`, `position_order`

Failures raise `RuntimeError` in CI (`GITHUB_ACTIONS=true`). CI also runs `dbt compile` to validate all model SQL.

---

## Tests

```bash
python -m unittest discover -s tests
```

| Suite | Coverage |
|---|---|
| `test_smoke` | End-to-end pipeline |
| `test_quality_checks` | Quality gate integration |
| `test_etl_unit` | Schema contracts, DNF sentinel, FK ref resolution |

---

<p align="center">
  <img src="company.JPG" alt="Company Logo" />
</p>
