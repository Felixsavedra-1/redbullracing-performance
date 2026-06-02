# Copy this file to config.py. Do not commit config.py to version control.

# family_refs must match the constructors table: SELECT constructor_ref FROM constructors
TEAM_CONFIG = {
    "family_refs": ["red_bull"],
    "name": "Oracle Red Bull Racing",
    "colors": {
        "primary": "#C9A96E",
        "accent":  "#8B5E3C",
        "neutral": "#D4C5A9",
    },
}

# Option 1: DuckDB (recommended — OLAP-optimized, no server required)
DB_CONFIG = {
    "type": "duckdb",
    "filename": "f1_analytics.duckdb",
}

# Option 2: SQLite (lightweight, no dependencies)
# DB_CONFIG = {
#     "type": "sqlite",
#     "filename": "f1_analytics.db",
# }

# Option 3: MySQL
# DB_CONFIG = {
#     "type": "mysql",
#     "host": "localhost",
#     "port": 3306,
#     "user": "root",
#     "password": "your_password_here",
#     "database": "f1_analytics",
# }

DATA_PATHS = {
    "raw_data": "data/raw/",
    "processed_data": "data/processed/",
}
