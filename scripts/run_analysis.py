import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive; must precede pyplot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from load_data import _build_connection_string
from logging_utils import setup_logging, format_table
from constants import TEAM_NAME, TEAM_COLORS, TEAM_REFS
from analytics import (
    teammate_delta, qualifying_race_ols,
    pit_stop_efficiency, championship_trajectory, dnf_rate_model,
)
from charts import (
    championship as chart_championship,
    teammate_delta_chart, qualifying_regression,
    pit_stops_chart, reliability_chart,
)
from dashboard import generate_dashboard

try:
    from config import DB_CONFIG
except ImportError:
    DB_CONFIG = {"type": "sqlite", "filename": "f1_analytics.db"}

logger = setup_logging()
_EXPORTS = os.path.join("data", "exports", "charts")


def _engine():
    return create_engine(_build_connection_string(DB_CONFIG))


def _out(name: str, export: bool) -> str | None:
    return os.path.join(_EXPORTS, f"{name}.png") if export else None


def _close(fig: plt.Figure | None, name: str, export: bool) -> None:
    if fig:
        plt.close(fig)
        if export:
            logger.info("  → %s", _out(name, True))
    elif export:
        logger.warning("%s chart: no data to plot", name)


def run(export: bool = False) -> None:
    if export:
        os.makedirs(_EXPORTS, exist_ok=True)

    engine = _engine()

    logger.info("Championship progression...")
    traj = championship_trajectory(engine)
    fig = chart_championship(traj, _out("championship", export), TEAM_NAME, TEAM_COLORS)
    _close(fig, "championship", export)

    logger.info("Teammate head-to-head...")
    delta = teammate_delta(engine)
    fig = teammate_delta_chart(delta, _out("teammate_delta", export), TEAM_COLORS)
    _close(fig, "teammate_delta", export)
    if not delta.empty:
        headers = ["Pair", "Δ mean", "CI 95%", "n", "p"]
        rows = [
            [
                f"{r.driver_a.split()[-1]} / {r.driver_b.split()[-1]}",
                f"{r.mean_delta:+.2f}",
                f"[{r.ci_lower:+.2f}, {r.ci_upper:+.2f}]",
                str(r.n),
                f"{r.p_value:.4f}",
            ]
            for r in delta.itertuples(index=False)
        ]
        logger.info("\n%s", format_table(headers, rows, {1, 2, 3}))

    logger.info("Qualifying → race regression...")
    ols_stats, scatter = qualifying_race_ols(engine)
    if ols_stats["slope"] is not None:
        fig = qualifying_regression(scatter, ols_stats, _out("qualifying_ols", export), TEAM_COLORS)
        if fig:
            plt.close(fig)
        p = ols_stats["p_value"]
        logger.info(
            "  R² = %.3f  slope = %.3f  p = %s  n = %d",
            ols_stats["r2"], ols_stats["slope"],
            "< 0.001" if p < 0.001 else f"{p:.4f}",
            ols_stats["n"],
        )
    else:
        logger.info("  qualifying_race_ols: insufficient data (n=%d)", ols_stats["n"])

    logger.info("Pit stop efficiency...")
    pit = pit_stop_efficiency(engine)
    fig = pit_stops_chart(pit, _out("pit_stop_efficiency", export), TEAM_COLORS)
    _close(fig, "pit_stop_efficiency", export)

    logger.info("DNF rate model...")
    dnf = dnf_rate_model(engine)
    fig = reliability_chart(dnf, _out("reliability", export), TEAM_COLORS)
    _close(fig, "reliability", export)
    if not dnf.empty:
        headers = ["Driver", "Races", "DNFs", "Rate", "CI 95%"]
        rows = [
            [
                r.driver.split()[-1],
                str(int(r.races)),
                str(int(r.dnfs)),
                f"{r.rate:.3f}",
                f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}]",
            ]
            for r in dnf.itertuples(index=False)
        ]
        logger.info("\n%s", format_table(headers, rows, {1, 2, 3}))

    if export:
        logger.info("Charts saved to %s/", _EXPORTS)
        logger.info("Generating dashboard...")
        dash_path = os.path.join("data", "exports", "dashboard.html")
        try:
            generate_dashboard(engine, TEAM_REFS, TEAM_NAME, dash_path)
            logger.info("  → %s", dash_path)
        except Exception:
            logger.exception("Dashboard generation failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{TEAM_NAME} F1 performance analytics")
    parser.add_argument(
        "--export", action="store_true",
        help=f"Save 300 DPI PNG charts to {_EXPORTS}/",
    )
    run(export=parser.parse_args().export)


if __name__ == "__main__":
    main()
