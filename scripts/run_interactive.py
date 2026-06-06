"""
Blueprint-style interactive F1 performance dashboard.
Styled as a technical CAD drawing sheet — monospace, cyan-on-navy, title block.

Usage:
    python scripts/run_interactive.py           # writes data/exports/f1_dashboard.html
    python scripts/run_interactive.py --open    # also opens in browser
"""
import argparse
import logging
import os
import re
import sys
import webbrowser
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats
from sqlalchemy import create_engine, text

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from constants import CONSTRUCTOR_ID, TEAM_REFS, TEAM_NAME
from load_data import _build_connection_string
from logging_utils import setup_logging

try:
    from config import DB_CONFIG
except ImportError:
    DB_CONFIG = {"type": "sqlite", "filename": "f1_analytics.db"}

_log = logging.getLogger("f1_analytics")
_OUT = os.path.join("data", "exports", "f1_dashboard.html")

_BG_SHEET = "#0b1929"   # drawing sheet
_BG_PLOT  = "#071220"   # plot area (slightly deeper)
_FG       = "#c8e6ff"   # primary text / lines
_GRID     = "#142d4a"   # subtle grid
_ACCENT   = "#00b4d8"   # cyan highlight
_DIM      = "#4a9ec4"   # dimension annotations / borders
_CALLOUT  = "#ff9f43"   # OLS line / orange callout

_PALETTE  = ["#00b4d8", "#48cae4", "#90e0ef",
             "#0096c7", "#0077b6", "#ade8f4",
             "#023e8a", "#caf0f8"]

_FONT = "'Courier New', Courier, monospace"

_REF_RE = re.compile(r"^[a-z0-9_]+$")


def _refs_sql(refs: list[str]) -> str:
    for r in refs:
        if not _REF_RE.match(r):
            raise ValueError(f"Invalid team ref: {r!r}")
    return ", ".join(f"'{r}'" for r in refs)


def _color_map(drivers: list[str]) -> dict[str, str]:
    return {d: _PALETTE[i % len(_PALETTE)] for i, d in enumerate(drivers)}


def _layout(**overrides) -> dict:
    return {
        "paper_bgcolor": _BG_SHEET,
        "plot_bgcolor":  _BG_PLOT,
        "font":          dict(color=_FG, size=10, family=_FONT),
        "xaxis":         dict(gridcolor=_GRID, linecolor=_DIM, zerolinecolor=_DIM, tickfont_family=_FONT),
        "yaxis":         dict(gridcolor=_GRID, linecolor=_DIM, zerolinecolor=_DIM, tickfont_family=_FONT),
        "legend":        dict(bgcolor=_BG_SHEET, bordercolor=_DIM, borderwidth=1,
                              font=dict(family=_FONT, size=9)),
        "title":         dict(font=dict(family=_FONT, size=11, color=_ACCENT)),
        "margin":        dict(t=52, b=36, l=60, r=20),
        **overrides,
    }


def _championship(engine) -> go.Figure | None:
    sql = f"""
    SELECT ra.year, ra.round, ra.race_name,
           COALESCE(d.forename,'') || ' ' || COALESCE(d.surname,'') AS driver,
           ds.points
    FROM driver_standings ds
    JOIN races        ra  ON ds.race_id         = ra.race_id
    JOIN drivers      d   ON ds.driver_id        = d.driver_id
    JOIN results      res ON res.race_id         = ra.race_id
                         AND res.driver_id       = ds.driver_id
    JOIN constructors c   ON res.constructor_id  = c.constructor_id
    WHERE c.constructor_ref IN ({_refs_sql(TEAM_REFS)})
    ORDER BY d.driver_id, ra.year, ra.round
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    if df.empty:
        return None

    years   = sorted(df["year"].unique())
    drivers = df.groupby("driver")["points"].max().sort_values(ascending=False).index.tolist()
    ncols   = min(3, len(years))
    nrows   = (len(years) + ncols - 1) // ncols

    fig = px.line(
        df, x="round", y="points", color="driver",
        facet_col="year", facet_col_wrap=ncols,
        color_discrete_map=_color_map(drivers),
        markers=True,
        hover_data={"race_name": True, "year": False},
        labels={"round": "ROUND", "points": "PTS", "driver": ""},
        title="[01]  CHAMPIONSHIP POINTS PROGRESSION",
    )
    fig.update_traces(marker_symbol="circle-open", marker_size=6, line_width=1.8)
    fig.for_each_annotation(lambda a: a.update(text=(a.text or "").split("=")[-1], font_family=_FONT))
    fig.update_layout(**_layout(height=300 * nrows))
    return fig


def _qualifying_scatter(engine) -> go.Figure | None:
    sql = """
    SELECT COALESCE(d.forename,'') || ' ' || COALESCE(d.surname,'') AS driver,
           ra.year, ra.race_name, r.grid,
           r.position_order AS finish
    FROM results r
    JOIN drivers d  ON r.driver_id = d.driver_id
    JOIN races   ra ON r.race_id   = ra.race_id
    WHERE r.constructor_id = :cid
      AND r.grid > 0 AND r.position_order < 999
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"cid": CONSTRUCTOR_ID})
    if df.empty:
        return None

    drivers                    = df["driver"].unique().tolist()
    slope, intercept, r_val, p, _ = stats.linregress(df["grid"], df["finish"])
    x_fit                      = np.linspace(df["grid"].min(), df["grid"].max(), 200)
    p_str                      = "p<0.001" if p < 0.001 else ("n/a" if np.isnan(p) else f"p={p:.4f}")

    fig = px.scatter(
        df, x="grid", y="finish", color="driver",
        color_discrete_map=_color_map(drivers),
        hover_data={"race_name": True, "year": True, "driver": False},
        opacity=0.65,
        labels={"grid": "GRID POS", "finish": "FINISH POS", "driver": ""},
        title=f"[02]  GRID → FINISH  |  R²={r_val**2:.3f}  ·  {p_str}  ·  n={len(df)}",
    )
    fig.update_traces(marker_symbol="cross-thin", marker_size=10,
                      marker_line_width=1.5, marker_line_color=None)
    fig.add_trace(go.Scatter(
        x=x_fit, y=intercept + slope * x_fit,
        mode="lines", name="OLS",
        line=dict(color=_CALLOUT, width=2, dash="dot"),
        hoverinfo="skip",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(**_layout(height=520))
    return fig


def _pit_stops(engine) -> go.Figure | None:
    sql = f"""
    WITH season_stats AS (
        SELECT ra.year,
               AVG(p.milliseconds) AS mu,
               SQRT(AVG(p.milliseconds * p.milliseconds)
                    - AVG(p.milliseconds) * AVG(p.milliseconds)) AS sigma
        FROM pit_stops p
        JOIN races ra ON p.race_id = ra.race_id
        WHERE p.milliseconds BETWEEN 15000 AND 60000
        GROUP BY ra.year
    )
    SELECT ra.year, ra.race_name,
           COALESCE(d.forename,'') || ' ' || COALESCE(d.surname,'') AS driver,
           ROUND(p.milliseconds / 1000.0, 2)                        AS duration_s,
           ROUND((p.milliseconds - ss.mu) / NULLIF(ss.sigma, 0), 3) AS z
    FROM pit_stops p
    JOIN races        ra  ON p.race_id         = ra.race_id
    JOIN season_stats ss  ON ss.year            = ra.year
    JOIN results      res ON res.race_id        = p.race_id
                         AND res.driver_id      = p.driver_id
    JOIN constructors c   ON res.constructor_id = c.constructor_id
    JOIN drivers      d   ON p.driver_id        = d.driver_id
    WHERE c.constructor_ref IN ({_refs_sql(TEAM_REFS)})
      AND p.milliseconds BETWEEN 15000 AND 60000
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn).dropna(subset=["z"])
    if df.empty:
        return None

    order = df.groupby("driver")["z"].mean().sort_values(ascending=False).index.tolist()

    fig = px.strip(
        df, x="z", y="driver", color="driver",
        color_discrete_map=_color_map(order),
        hover_data={"race_name": True, "duration_s": True, "year": True, "driver": False},
        category_orders={"driver": order},
        labels={"z": "Z-SCORE VS FIELD AVG", "driver": ""},
        title="[03]  PIT STOP EFFICIENCY  |  z < 0 = FASTER THAN FIELD",
    )
    fig.update_traces(marker_symbol="line-ns-open", marker_size=9, marker_line_width=1.5)
    fig.add_vline(x=0, line_dash="dot", line_color=_CALLOUT, line_width=1.5, opacity=0.7)
    fig.update_layout(**_layout(height=max(280, len(order) * 52)))
    return fig


_COMPOUND_COLOR = {
    "SOFT":         "#E8002D",
    "MEDIUM":       "#FFF200",
    "HARD":         "#FFFFFF",
    "INTERMEDIATE": "#39B54A",
    "WET":          "#0067FF",
    "UNKNOWN":      "#888888",
}


def _tyre_strategy(engine) -> go.Figure | None:
    sql = """
    SELECT ra.race_name, ra.round, l.lap_number,
           COALESCE(d.forename,'') || ' ' || COALESCE(d.surname,'') AS driver,
           COALESCE(l.compound,'UNKNOWN') AS compound
    FROM laps l
    JOIN races   ra ON l.race_id   = ra.race_id
    JOIN drivers d  ON l.driver_id = d.driver_id
    WHERE ra.year = (
        SELECT MAX(ra2.year) FROM laps l2 JOIN races ra2 ON l2.race_id = ra2.race_id
    )
    ORDER BY ra.round, driver, l.lap_number
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    if df.empty:
        return None

    with engine.connect() as conn:
        season_year = conn.execute(text(
            "SELECT MAX(ra.year) FROM laps l JOIN races ra ON l.race_id = ra.race_id"
        )).scalar()

    ncols = 4
    nrows = (df["race_name"].nunique() + ncols - 1) // ncols
    n_drivers = df["driver"].nunique()

    fig = px.scatter(
        df, x="lap_number", y="driver", color="compound",
        facet_col="race_name", facet_col_wrap=ncols,
        color_discrete_map=_COMPOUND_COLOR,
        labels={"lap_number": "LAP", "driver": "", "compound": "COMPOUND"},
        title=f"[04]  TYRE STRATEGY  |  {season_year} SEASON",
    )
    fig.update_traces(marker_symbol="square", marker_size=5)
    fig.for_each_annotation(lambda a: a.update(
        text=(a.text or "").split("=")[-1][:20], font_family=_FONT, font_size=8,
    ))
    fig.update_layout(**_layout(height=max(260, n_drivers * 30) * nrows))
    return fig


def _sector_comparison(engine) -> go.Figure | None:
    sql = """
    SELECT COALESCE(d.forename,'') || ' ' || COALESCE(d.surname,'') AS driver,
           ROUND(AVG(l.sector1_s), 3) AS s1_mean,
           ROUND(AVG(l.sector2_s), 3) AS s2_mean,
           ROUND(AVG(l.sector3_s), 3) AS s3_mean,
           COUNT(*) AS n
    FROM laps l
    JOIN results res ON l.race_id  = res.race_id AND l.driver_id = res.driver_id
    JOIN drivers d   ON l.driver_id = d.driver_id
    WHERE res.constructor_id = :cid
      AND l.track_status = '1'
      AND l.sector1_s IS NOT NULL
      AND l.sector2_s IS NOT NULL
      AND l.sector3_s IS NOT NULL
    GROUP BY l.driver_id, d.forename, d.surname
    HAVING COUNT(*) >= 10
    ORDER BY s1_mean + s2_mean + s3_mean
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"cid": CONSTRUCTOR_ID})
    if df.empty:
        return None

    melted = df.melt(
        id_vars="driver", value_vars=["s1_mean", "s2_mean", "s3_mean"],
        var_name="sector", value_name="time_s",
    )
    melted["sector"] = melted["sector"].map({"s1_mean": "S1", "s2_mean": "S2", "s3_mean": "S3"})

    fig = px.bar(
        melted, x="driver", y="time_s", color="sector",
        barmode="group",
        color_discrete_map={"S1": _ACCENT, "S2": _CALLOUT, "S3": _DIM},
        labels={"driver": "", "time_s": "MEAN TIME (s)", "sector": "SECTOR"},
        title="[05]  SECTOR TIME BREAKDOWN  |  GREEN-FLAG LAPS",
    )
    fig.update_layout(**_layout(height=420))
    return fig


def _build_html(figs: list[go.Figure]) -> str:
    divs = [
        pio.to_html(fig, full_html=False,
                    include_plotlyjs=("cdn" if i == 0 else False),
                    config={"displayModeBar": True, "scrollZoom": True})
        for i, fig in enumerate(figs)
    ]
    today     = date.today().isoformat()
    zones_h   = "".join(f"<span>{n}</span>" for n in range(1, 9))
    zones_v   = "".join(f"<span>{c}</span>" for c in "ABCDEFGH")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{TEAM_NAME} — Performance Drawing</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0 }}

    body {{
      background: #040d18;
      font-family: {_FONT};
      padding: 24px;
      min-height: 100vh;
    }}

    /* ── Drawing sheet ─────────────────────────────── */
    .sheet {{
      background: {_BG_SHEET};
      border: 2px solid {_DIM};
      outline: 6px solid {_BG_SHEET};
      outline-offset: -8px;
      max-width: 1440px;
      margin: 0 auto;
      display: grid;
      grid-template-rows: 28px 1fr 28px;
      grid-template-columns: 24px 1fr 24px;
      min-height: calc(100vh - 48px);
    }}

    /* ── Zone reference strips ─────────────────────── */
    .zone {{
      display: flex;
      align-items: center;
      justify-content: space-around;
      color: {_DIM};
      font-size: 9px;
      letter-spacing: 2px;
      border: 1px solid {_DIM};
      opacity: .7;
    }}
    .zone-top    {{ grid-column: 2; grid-row: 1; flex-direction: row; }}
    .zone-bottom {{ grid-column: 2; grid-row: 3; flex-direction: row; }}
    .zone-left   {{ grid-column: 1; grid-row: 2; flex-direction: column; border-right: none; }}
    .zone-right  {{ grid-column: 3; grid-row: 2; flex-direction: column; border-left: none; }}
    .corner      {{ border: 1px solid {_DIM}; opacity: .7; }}
    .corner-tl   {{ grid-column: 1; grid-row: 1; }}
    .corner-tr   {{ grid-column: 3; grid-row: 1; }}
    .corner-bl   {{ grid-column: 1; grid-row: 3; }}
    .corner-br   {{ grid-column: 3; grid-row: 3; }}

    /* ── Main drawing area ─────────────────────────── */
    .drawing {{
      grid-column: 2;
      grid-row: 2;
      padding: 28px 32px 20px;
      display: flex;
      flex-direction: column;
      gap: 36px;
    }}

    /* ── Page header ────────────────────────────────── */
    .page-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 1px solid {_DIM};
      padding-bottom: 12px;
    }}
    .page-title {{
      color: {_ACCENT};
      font-size: 1.05rem;
      font-weight: bold;
      letter-spacing: .12em;
      text-transform: uppercase;
    }}
    .page-sub {{
      color: {_DIM};
      font-size: .7rem;
      letter-spacing: .08em;
      margin-top: 4px;
    }}

    /* ── Title block (bottom-right, CAD standard) ───── */
    .title-block {{
      align-self: flex-end;
      border: 1px solid {_DIM};
      font-size: .68rem;
      letter-spacing: .05em;
      text-transform: uppercase;
      min-width: 340px;
    }}
    .title-block table {{
      border-collapse: collapse;
      width: 100%;
    }}
    .title-block td {{
      padding: 3px 8px;
      border: 1px solid {_DIM};
      vertical-align: middle;
    }}
    .title-block .lbl {{ color: {_DIM}; width: 90px; }}
    .title-block .val {{ color: {_FG}; }}
    .title-block .val.accent {{ color: {_ACCENT}; font-weight: bold; }}

    /* ── Section label above each chart ─────────────── */
    .section-label {{
      font-size: .65rem;
      color: {_DIM};
      letter-spacing: .18em;
      text-transform: uppercase;
      margin-bottom: -28px;
    }}

    /* ── Chart wrapper ───────────────────────────────── */
    .chart {{ width: 100%; }}
  </style>
</head>
<body>
<div class="sheet">

  <!-- Corners -->
  <div class="corner corner-tl"></div>
  <div class="corner corner-tr"></div>
  <div class="corner corner-bl"></div>
  <div class="corner corner-br"></div>

  <!-- Zone strips -->
  <div class="zone zone-top">{zones_h}</div>
  <div class="zone zone-bottom">{zones_h}</div>
  <div class="zone zone-left">{zones_v}</div>
  <div class="zone zone-right">{zones_v}</div>

  <!-- Main content -->
  <div class="drawing">

    <div class="page-header">
      <div>
        <div class="page-title">{TEAM_NAME} — F1 Performance Analysis</div>
        <div class="page-sub">INTERACTIVE TECHNICAL DRAWING  ·  DATA SOURCE: ERGAST API  ·  REV A</div>
      </div>
    </div>

    {"".join(f'<div class="chart">{div}</div>' for div in divs)}

    <div class="title-block">
      <table>
        <tr>
          <td class="lbl">PROJECT</td>
          <td class="val accent">{TEAM_NAME} F1 ANALYTICS</td>
        </tr>
        <tr>
          <td class="lbl">DRAWING</td>
          <td class="val">PERFORMANCE DASHBOARD</td>
        </tr>
        <tr>
          <td class="lbl">VIEWS</td>
          <td class="val">CHAMPIONSHIP  ·  GRID/FINISH  ·  PIT  ·  TYRES  ·  SECTORS</td>
        </tr>
        <tr>
          <td class="lbl">DATE</td>
          <td class="val">{today}</td>
        </tr>
        <tr>
          <td class="lbl">REVISION</td>
          <td class="val">A</td>
        </tr>
        <tr>
          <td class="lbl">SCALE</td>
          <td class="val">1 : 1</td>
        </tr>
      </table>
    </div>

  </div><!-- .drawing -->
</div><!-- .sheet -->
</body>
</html>"""


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Export blueprint-style F1 performance dashboard")
    parser.add_argument("--open", action="store_true", help="Open in browser after export")
    args = parser.parse_args()

    engine  = create_engine(_build_connection_string(DB_CONFIG))
    builders = [_championship, _qualifying_scatter, _pit_stops,
                _tyre_strategy, _sector_comparison]
    figs    = [f for fn in builders if (f := fn(engine)) is not None]

    if not figs:
        _log.warning("No data to render — run the pipeline first.")
        return

    html = _build_html(figs)
    os.makedirs(os.path.dirname(os.path.abspath(_OUT)), exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as fh:
        fh.write(html)

    _log.info("Dashboard written → %s", os.path.abspath(_OUT))
    if args.open:
        webbrowser.open(f"file://{os.path.abspath(_OUT)}")


if __name__ == "__main__":
    main()
