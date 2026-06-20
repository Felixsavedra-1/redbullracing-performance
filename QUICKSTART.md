# Quick Start Guide

This guide summarises the minimal steps required to reproduce the analysis.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Recommended: Python 3.11+

Optional (for notebooks/visuals):
```bash
pip install -r requirements-notebooks.txt
```

## Step 2: Run the Pipeline
SQLite is used by default; no separate database service needs to be configured.

```bash
python scripts/run_pipeline.py
```

For a faster demo run:
```bash
python scripts/run_pipeline.py --fast
```
Fast mode skips pit stop extraction to avoid long, rate-limited requests.

This will:
- Download F1 data from the Ergast API
- Clean and transform the data
- Load the processed data into your database

Note: The project is scoped to the 2020–2025 seasons. Year ranges outside that window are clamped.

## Step 3: Run Your First Query

```bash
# See what queries are available
python scripts/run_queries.py --list

# Run a query
python scripts/run_queries.py --query kpi_summary

# Export results
python scripts/run_queries.py --query kpi_summary --export
```

## Summary

You now have a complete F1 Red Bull analytics database running locally.

### Next Steps
- Explore queries in `database/queries/analytical_queries.sql`
- Connect Power BI to SQLite (`f1_analytics.db`) or MySQL for visualizations
- Run custom queries: `sqlite3 f1_analytics.db` or `mysql -u root -p F1_RedBull_Analytics`

### Optional Flags
- Incremental load: `python scripts/run_pipeline.py --incremental`
- Skip data quality checks: `python scripts/run_pipeline.py --skip-quality`
- Relax schema enforcement: `python scripts/run_pipeline.py --no-strict-schema`

### Need Help?
- Consult the full [README.md](README.md) for detailed documentation.
- For connection issues, confirm that your database service is running and `config.py` contains valid credentials.
