#!/usr/bin/env python3
"""Generate plots from the SimAI-NS3 sweep's summary.csv.

Produces in `output/<sweep>/plots/`:

  Per-collective:
    <coll>_time.png    log-log time vs payload size, NCCL meas + SimAI-NS3 sim
    <coll>_busbw.png   busbw vs payload size, same legend

  Overviews:
    overview_time.png   all three modelled collectives on one log-log plot
    overview_busbw.png  same for busbw
    error_pct.png       (sim/meas - 1) % across sizes per collective

  Cross-simulator comparison (if the AstraSim and SimAI-Analytical sweeps exist):
    compare_<coll>_busbw.png  NCCL vs AstraSim vs SimAI-Analytical vs SimAI-NS3
    compare_<coll>_time.png   same for time
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

SIM_COLLECTIVES = ["all_reduce", "all_gather", "reduce_scatter"]

# Distinct colors per "source" so the cross-simulator plot stays readable.
COL_NCCL    = "#1f77b4"   # measured
COL_NS3     = "#d62728"   # SimAI-NS3 (this sweep)
COL_SIMA    = "#9467bd"   # SimAI-Analytical
COL_ASTRA   = "#2ca02c"   # AstraSim

COLLECTIVE_PALETTE = {
    "all_reduce":     "#1f77b4",
    "all_gather":     "#2ca02c",
    "reduce_scatter": "#9467bd",
}


def _set_style() -> None:
    plt.rcParams.update({
        "figure.figsize":    (10.0, 6.0),
        "figure.dpi":        120,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "axes.grid":         True,
        "grid.alpha":        0.30,
        "grid.linestyle":    "--",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.titleweight":  "bold",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "legend.frameon":    False,
        "legend.fontsize":   10,
        "lines.linewidth":   1.8,
        "lines.markersize":  6,
        "font.family":       "DejaVu Sans",
    })


def _human_size(b: int) -> str:
    for unit, denom in [("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10), ("B", 1)]:
        if b >= denom:
            v = b / denom
            return f"{v:.0f}\u202F{unit}" if v >= 10 or unit == "B" else f"{v:.1f}\u202F{unit}"
    return str(b)


def _format_size_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ticks = [1 << k for k in (10, 14, 18, 22, 26, 30, 33)]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _human_size(int(v))))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Payload size")


# --- per-collective plots ------------------------------------------------

def plot_time(df: pd.DataFrame, collective: str, out_path: Path) -> None:
    sub = df[df.collective == collective].sort_values("size_bytes")
    fig, ax = plt.subplots()
    ax.plot(sub.size_bytes, sub.nccl_time_us, "o-", color=COL_NCCL,
            label="NCCL measured")
    if sub.sim_time_us.notna().any():
        ax.plot(sub.size_bytes, sub.sim_time_us, "s-", color=COL_NS3,
                label="SimAI-NS3 simulated")
    ax.set_yscale("log")
    _format_size_axis(ax)
    ax.set_ylabel("Time (\u00b5s)  \u2014 log scale")
    ax.set_title(f"{collective}: time vs payload size  (4\u00d7H100, intra-node, NV6)")
    ax.legend(loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)


def plot_busbw(df: pd.DataFrame, collective: str, out_path: Path) -> None:
    sub = df[df.collective == collective].sort_values("size_bytes")
    fig, ax = plt.subplots()
    ax.plot(sub.size_bytes, sub.nccl_busbw_GBs, "o-", color=COL_NCCL,
            label="NCCL bus bandwidth")
    if sub.sim_busbw_GBs.notna().any():
        ax.plot(sub.size_bytes, sub.sim_busbw_GBs, "s-", color=COL_NS3,
                label="SimAI-NS3 bus bandwidth")
    _format_size_axis(ax)
    ax.set_ylabel("Bus bandwidth (GB/s)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{collective}: bus bandwidth vs payload size  (4\u00d7H100, intra-node, NV6)")
    ax.legend(loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)


# --- overview plots ------------------------------------------------------

def plot_overview_time(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots()
    for coll in SIM_COLLECTIVES:
        sub = df[df.collective == coll].sort_values("size_bytes")
        c = COLLECTIVE_PALETTE[coll]
        ax.plot(sub.size_bytes, sub.nccl_time_us, "o-",  color=c,
                label=f"{coll} \u2014 measured")
        ax.plot(sub.size_bytes, sub.sim_time_us, "s--", color=c, alpha=0.7,
                label=f"{coll} \u2014 SimAI-NS3")
    ax.set_yscale("log")
    _format_size_axis(ax)
    ax.set_ylabel("Time (\u00b5s) \u2014 log scale")
    ax.set_title("SimAI-NS3 vs measured time across modelled collectives")
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(out_path)
    plt.close(fig)


def plot_overview_busbw(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots()
    for coll in SIM_COLLECTIVES:
        sub = df[df.collective == coll].sort_values("size_bytes")
        c = COLLECTIVE_PALETTE[coll]
        ax.plot(sub.size_bytes, sub.nccl_busbw_GBs, "o-",  color=c,
                label=f"{coll} \u2014 measured")
        ax.plot(sub.size_bytes, sub.sim_busbw_GBs, "s--", color=c, alpha=0.7,
                label=f"{coll} \u2014 SimAI-NS3")
    _format_size_axis(ax)
    ax.set_ylabel("Bus bandwidth (GB/s)")
    ax.set_ylim(bottom=0)
    ax.set_title("SimAI-NS3 vs measured bus bandwidth across modelled collectives")
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(out_path)
    plt.close(fig)


def plot_error_pct(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots()
    for coll in SIM_COLLECTIVES:
        sub = df[df.collective == coll].sort_values("size_bytes")
        sub = sub[sub.err_pct_time.notna()]
        ax.plot(sub.size_bytes, sub.err_pct_time, "o-",
                color=COLLECTIVE_PALETTE[coll], label=coll)
    ax.axhline(0.0, color="black", linewidth=0.8)
    for thresh in (-30, -10, 10, 30):
        ax.axhline(thresh, color="lightgrey", linewidth=0.6, linestyle=":")
    _format_size_axis(ax)
    ax.set_ylabel("(sim - measured) / measured  [%]")
    ax.set_title("SimAI-NS3 error % vs measured (time) across the sweep")
    ax.legend(loc="lower right")
    fig.savefig(out_path)
    plt.close(fig)


# --- cross-simulator comparison ------------------------------------------

def _load_optional(p: Path, cols: dict[str, str]) -> Optional[pd.DataFrame]:
    """Load a sister sweep's summary.csv if present.

    `cols` maps the column name we want to its name in that source CSV.
    """
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    out = pd.DataFrame()
    out["collective"] = df["collective"]
    out["size_bytes"] = df["size_bytes"]
    for our, theirs in cols.items():
        out[our] = df[theirs] if theirs in df.columns else None
    return out


def plot_compare(
    ns3: pd.DataFrame,
    sima: Optional[pd.DataFrame],
    astra: Optional[pd.DataFrame],
    collective: str,
    metric: str,
    out_path: Path,
) -> None:
    """Overlay measured + up to 3 simulators on one collective/metric plot."""
    fig, ax = plt.subplots()
    sub_n = ns3[ns3.collective == collective].sort_values("size_bytes")

    nccl_col = "nccl_time_us"   if metric == "time" else "nccl_busbw_GBs"
    sim_col  = "sim_time_us"    if metric == "time" else "sim_busbw_GBs"

    ax.plot(sub_n.size_bytes, sub_n[nccl_col], "o-", color=COL_NCCL, label="NCCL measured")
    ax.plot(sub_n.size_bytes, sub_n[sim_col],  "s-", color=COL_NS3,  label="SimAI-NS3")

    if astra is not None:
        sub_a = astra[astra.collective == collective].sort_values("size_bytes")
        if not sub_a.empty:
            ax.plot(sub_a.size_bytes, sub_a[sim_col], "^--", color=COL_ASTRA,
                    label="AstraSim (calibrated)")
    if sima is not None:
        sub_s = sima[sima.collective == collective].sort_values("size_bytes")
        if not sub_s.empty:
            ax.plot(sub_s.size_bytes, sub_s[sim_col], "v--", color=COL_SIMA,
                    label="SimAI-Analytical (-nv 370)")

    if metric == "time":
        ax.set_yscale("log")
        ax.set_ylabel("Time (\u00b5s) \u2014 log scale")
    else:
        ax.set_ylabel("Bus bandwidth (GB/s)")
        ax.set_ylim(bottom=0)
    _format_size_axis(ax)
    ax.set_title(f"{collective}: NCCL vs three simulators  (4\u00d7H100, intra-node, NV6)")
    ax.legend(loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)


# --- driver --------------------------------------------------------------

def generate_all(csv_path: Path, plots_dir: Path) -> list[Path]:
    df = pd.read_csv(csv_path)
    plots_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    for coll in SIM_COLLECTIVES:
        t = plots_dir / f"{coll}_time.png"
        b = plots_dir / f"{coll}_busbw.png"
        plot_time(df, coll, t)
        plot_busbw(df, coll, b)
        produced += [t, b]

    for name, fn in [
        ("overview_time.png", plot_overview_time),
        ("overview_busbw.png", plot_overview_busbw),
        ("error_pct.png", plot_error_pct),
    ]:
        p = plots_dir / name
        fn(df, p)
        produced.append(p)

    # Cross-simulator overlay if the sister sweeps are present.
    here = Path(__file__).resolve().parent.parent
    sima_csv  = here.parent / "h100-simai-analytical-replay" / "output" / "sweep_latest" / "summary.csv"
    astra_csv = here.parent / "h100-nccl-analytical-1gib-replay" / "output" / "sweep_latest" / "all_data.csv"

    sima = _load_optional(sima_csv, {
        "sim_time_us":   "sim_time_us",
        "sim_busbw_GBs": "sim_busbw_GBs",
    })
    astra = _load_optional(astra_csv, {
        "sim_time_us":   "sim_time_us",
        "sim_busbw_GBs": "sim_busbw_GBs",
    })

    if sima is not None or astra is not None:
        for coll in SIM_COLLECTIVES:
            for metric in ("time", "busbw"):
                out = plots_dir / f"compare_{coll}_{metric}.png"
                plot_compare(df, sima, astra, coll, metric, out)
                produced.append(out)

    return produced


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--csv", type=Path,
        default=here / "output" / "sweep_latest" / "summary.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}\nRun ./scripts/run_sweep.py first.")
    out_dir = args.out_dir if args.out_dir else args.csv.parent / "plots"
    _set_style()
    produced = generate_all(args.csv, out_dir)
    print(f"Wrote {len(produced)} plots into {out_dir}/")
    for p in produced:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
