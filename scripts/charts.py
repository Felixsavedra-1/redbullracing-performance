from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_PALETTE = [
    "#1E41FF", "#D6203F", "#9DA3A8", "#ECE5D5",
    "#00B4FF", "#FF6B35", "#C8FF00", "#7E858B",
]

_RC: dict = {
    "figure.facecolor":     "#0B0A08",
    "axes.facecolor":       "#000000",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.edgecolor":       "#9DA3A8",
    "axes.grid":            True,
    "axes.grid.axis":       "y",
    "grid.alpha":           0.22,
    "grid.linewidth":       0.6,
    "grid.color":           "#26231C",
    "text.color":           "#ECE5D5",
    "axes.labelcolor":      "#ECE5D5",
    "xtick.color":          "#6A7176",
    "ytick.color":          "#6A7176",
    "font.family":          "sans-serif",
    "font.size":            10,
    "axes.titlesize":       12,
    "axes.titleweight":     "semibold",
    "axes.labelsize":       10,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      9,
    "legend.frameon":       False,
    "legend.facecolor":     "#0B0A08",
    "legend.edgecolor":     "#26231C",
    "lines.linewidth":      2.2,
    "patch.linewidth":      0.5,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
}

matplotlib.rcParams.update(_RC)


def _primary(colors: dict | None) -> str:
    return (colors or {}).get("primary", "#1E41FF")


def _accent(colors: dict | None) -> str:
    return (colors or {}).get("accent", "#D6203F")


def _driver_colors(names: list[str], primary: str) -> dict[str, str]:
    ordered = sorted(names)
    palette = [primary] + [c for c in _PALETTE if c != primary]
    return {name: palette[i % len(palette)] for i, name in enumerate(ordered)}


def _sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _save(fig: plt.Figure, path: str | None) -> plt.Figure:
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def championship(
    df: pd.DataFrame,
    path: str | None = None,
    team_name: str = "",
    colors: dict | None = None,
) -> plt.Figure | None:
    if df.empty:
        return None
    primary = _primary(colors)

    years = sorted(df["year"].unique())
    ncols = min(3, len(years))
    nrows = (len(years) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    all_drivers: set[str] = set()
    for year in years:
        yd = df[df["year"] == year]
        all_drivers.update(
            yd.groupby("driver")["points"].max()
            .sort_values(ascending=False).index.tolist()
        )
    cmap = _driver_colors(sorted(all_drivers), primary)

    for i, year in enumerate(years):
        ax = axes[i]
        yd = df[df["year"] == year]
        drivers = (
            yd.groupby("driver")["points"].max()
            .sort_values(ascending=False).index.tolist()
        )
        for driver in drivers:
            d = yd[yd["driver"] == driver].sort_values("round")
            ax.plot(d["round"], d["points"],
                    color=cmap[driver], label=driver.split()[-1])
        ax.set_title(str(year))
        ax.set_xlabel("Round", fontsize=8)
        ax.set_ylabel("Points", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=8)

    for ax in axes[len(years):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8,
               frameon=False, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"{team_name}  —  Championship Progression",
                 fontsize=13, fontweight="semibold", y=1.01)
    fig.tight_layout()
    return _save(fig, path)


def teammate_delta_chart(
    df: pd.DataFrame,
    path: str | None = None,
    colors: dict | None = None,
) -> plt.Figure | None:
    if df.empty:
        return None
    primary = _primary(colors)
    accent  = _accent(colors)

    n_pairs = len(df)
    fig, ax = plt.subplots(figsize=(9, max(3.5, n_pairs * 0.9)))

    y_pos = list(range(n_pairs))
    labels = (df["driver_a"].str.split().str[-1] + " / " + df["driver_b"].str.split().str[-1]).tolist()
    bar_colors = np.where(df["mean_delta"] < 0, primary, accent).tolist()
    xerr = np.column_stack([df["mean_delta"] - df["ci_lower"], df["ci_upper"] - df["mean_delta"]]).T

    ax.barh(y_pos, df["mean_delta"].tolist(), xerr=xerr,
            color=bar_colors, alpha=0.82,
            error_kw={"linewidth": 1.4, "capsize": 4, "color": "#9DA3A8"})
    ax.axvline(0, color="#888888", linewidth=1, linestyle="--", alpha=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Position Delta  (driver A relative to driver B)")
    ax.set_title("Teammate Head-to-Head")
    ax.grid(axis="x", alpha=0.22)
    ax.grid(axis="y", alpha=0)

    for pos, (_, row) in enumerate(df.iterrows()):
        sig = _sig_label(row.p_value)
        ax.text(
            row.ci_upper + 0.05, pos,
            f"  {sig}  n={row.n}",
            va="center", fontsize=8, color="#9DA3A8",
        )

    fig.tight_layout()
    return _save(fig, path)


def qualifying_regression(
    df: pd.DataFrame,
    ols_stats: dict,
    path: str | None = None,
    colors: dict | None = None,
) -> plt.Figure | None:
    if df.empty:
        return None
    primary = _primary(colors)

    fig, ax = plt.subplots(figsize=(7, 7))

    drivers = df["driver"].unique().tolist()
    cmap = _driver_colors(drivers, primary)
    for driver in drivers:
        d = df[df["driver"] == driver]
        ax.scatter(d["grid"], d["finish"],
                   color=cmap[driver], alpha=0.45, s=22,
                   label=driver.split()[-1])

    x = np.linspace(df["grid"].min(), df["grid"].max(), 120)
    ax.plot(x, ols_stats["intercept"] + ols_stats["slope"] * x,
            color="#9DA3A8", linewidth=2, zorder=5, label="_fit")

    p = ols_stats["p_value"]
    p_str = "p < 0.001" if p < 0.001 else ("n/a" if pd.isna(p) else f"p = {p:.4f}")
    ax.text(
        0.05, 0.97,
        f"R² = {ols_stats['r2']:.3f}\n{p_str}\nn = {ols_stats['n']}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#2A2A2A",
              "alpha": 0.9, "edgecolor": "#555555"},
    )

    ax.invert_yaxis()
    ax.set_xlabel("Grid Position")
    ax.set_ylabel("Finish Position")
    ax.set_title("Grid → Finish  (DNFs excluded)")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    ax.grid(axis="both", alpha=0.22)
    fig.tight_layout()
    return _save(fig, path)


def pit_stops_chart(
    df: pd.DataFrame,
    path: str | None = None,
    colors: dict | None = None,
) -> plt.Figure | None:
    if df.empty:
        return None
    primary = _primary(colors)

    fig, ax = plt.subplots(figsize=(max(7, len(df) * 0.9), 5))

    x = list(range(len(df)))
    ax.bar(x, df["mean_z"].tolist(), color=primary, alpha=0.75, zorder=3)
    ax.errorbar(
        x, df["mean_z"].tolist(),
        yerr=df["std_z"].fillna(0).tolist(),
        fmt="none", color="#9DA3A8", linewidth=1.4, capsize=4, zorder=4,
    )
    ax.axhline(0, color="#888888", linewidth=1, linestyle="--", alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(df["driver"].str.split().str[-1].tolist(), rotation=30, ha="right")
    ax.set_ylabel("Z-score vs Season Average")
    ax.set_title("Pit Stop Efficiency  (z < 0 = faster than field)")

    for idx, (_, row) in enumerate(df.iterrows()):
        offset = 0.08 if row["mean_z"] >= 0 else -0.15
        ax.text(idx, row["mean_z"] + offset, f"n={int(row['n_stops'])}",
                ha="center", fontsize=7.5, color="#AA9988")

    fig.tight_layout()
    return _save(fig, path)


def tyre_degradation_chart(
    df: pd.DataFrame,
    path: str | None = None,
    colors: dict | None = None,
) -> plt.Figure | None:
    if df.empty:
        return None

    _COMPOUND_COLORS = {"SOFT": "#FF1800", "MEDIUM": "#FFD700", "HARD": "#AAAAAA"}
    compounds = [c for c in ("SOFT", "MEDIUM", "HARD") if c in df["compound"].unique()]
    drivers_sorted = (
        df.groupby("driver")["deg_rate_s"].mean().sort_values().index.tolist()
    )
    n = len(drivers_sorted)
    width = 0.65 / max(len(compounds), 1)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(7, n * 1.8), 5))

    for i, compound in enumerate(compounds):
        cdf = df[df["compound"] == compound].set_index("driver")
        heights = [
            float(cdf.loc[d, "deg_rate_s"]) if d in cdf.index else 0.0
            for d in drivers_sorted
        ]
        offset = (i - len(compounds) / 2 + 0.5) * width
        ax.bar(
            x + offset, heights, width=width * 0.9,
            color=_COMPOUND_COLORS.get(compound, "#888888"),
            alpha=0.82, label=compound, zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([d.split()[-1] for d in drivers_sorted])
    ax.set_ylabel("Degradation Rate (s / lap)")
    ax.set_title("Tyre Degradation · Rate by Compound")
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def reliability_chart(
    df: pd.DataFrame,
    path: str | None = None,
    colors: dict | None = None,
) -> plt.Figure | None:
    if df.empty:
        return None
    primary = _primary(colors)

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(df) * 0.65)))

    y_pos = list(range(len(df)))
    xerr = np.column_stack([df["rate"] - df["ci_lower"], df["ci_upper"] - df["rate"]]).T

    ax.barh(y_pos, df["rate"].tolist(), xerr=xerr,
            color=primary, alpha=0.78,
            error_kw={"linewidth": 1.4, "capsize": 4, "color": "#9DA3A8"})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        (df["driver"].str.rsplit(" ", n=1).str[-1] + "  ("
         + df["dnfs"].astype(int).astype(str) + "/"
         + df["races"].astype(int).astype(str) + ")").tolist()
    )
    ax.set_xlabel("DNF Rate per Race  (95% Poisson CI)")
    ax.set_title("Reliability — DNF Rate by Driver")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="x", alpha=0.22)
    ax.grid(axis="y", alpha=0)

    fig.tight_layout()
    return _save(fig, path)
