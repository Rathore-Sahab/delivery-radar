"""Run every .sql file in this folder with DuckDB and print the results.

    pip install duckdb
    python sql/run_queries.py        (run from the repo root)
"""
from pathlib import Path

import duckdb

for f in sorted(Path(__file__).parent.glob("*.sql")):
    print(f"\n=== {f.name} ===")
    print(duckdb.sql(f.read_text()).df().to_string(index=False))
