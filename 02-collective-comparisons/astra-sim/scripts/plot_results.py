#!/usr/bin/env python3
"""Generate plots from a sweep's all_data.csv.

Reads `output/<sweep>/all_data.csv` (default: output/sweep_latest/) and writes
PNGs into `output/<sweep>/plots/`.

What gets produced (all in the sweep's plots/ subfolder):

  Per-collective (for all_reduce, all_gather, reduce_scatter):
    <coll>_time.png    log-log time vs payload size, NCCL meas + AstraSim sim
    <coll>_busbw.png   busbw vs payload size (lin y, log x), same legend

  Cross-collective overviews (the 3 modelled collectives):
    overview_time.png         all three on one log-log time plot
    overview_busbw.png        all three on one busbw plot
    error_pct.png             (sim/meas - 1) % across sizes for each collective

  Bonus (uses the full nccl-tests data, including collectives we don't model):
    nccl_all_busbw.png        measured busbw curves for all 7 NCCL collectives
                              (only NCCL data; no sim component)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# 3 collectives we actually simulate, in display order.
SIM_COLLECTIVES = ["all_reduce", "all_gather", "reduce_scatter"]
ALL_NCCL_COLLECTIVES = [
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "reduce",
    "all_to_all",
    "sendrecv",
]

# Color palette: keep "measured" cool and "sim" warm, consistent across plots.
COL_NCCL = "#1f77b4"   # measured (NCCL busbw / time, out-of-place column)
COL_SIM  = "#d62728"   # AstraSim sim (warm red so it pops)

# Colors for the cross-collective overview / nccl-all plots.
COLLECTIVE_PALETTE = {
    "all_reduce":     "#1f77b4",
    "all_gather":     "#2ca02c",
    "reduce_scatter": "#9467bd",
    "broadcast":      "#ff7f0e",
    "reduce":         "#8c564b",
    "all_to_all":     "#e377c2",
    "sendrecv":       "#7f7f7f",
}


def _set_style() -> None:
    plt.rcParams.update({
        "figure.figsize":   (10.0, 6.0),
        "figure.dpi":       120,
        "savefig.dpi":      150,
        "savefig.bbox":     "tight",
        "axes.grid":        True,
        "grid.alpha":       0.30,
        "grid.linestyle":   "--",
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
    """Log-scale x-axis with byte labels at the nice power-of-2 ticks."""
    ax.set_xscale("log", base=2)
    ticks = [1 << k for k in (10, 14, 18, 22, 26, 30, 33)]  # 1KiB, 16KiB, ..., 8GiB
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _human_size(int(v))))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Payload size")


def plot_time(df: pd.DataFrame, collective: str, out_path: Path) -> None:
    sub = df[df.collective == collective].sort_values("size_bytes")
    fig, ax = plt.subplots()

    ax.plot(sub.size_bytes, sub.nccl_time_us_oop, "o-", color=COL_NCCL,
            label="NCCL measured")
    if sub.sim_time_us.notna().any():
        ax.plot(sub.size_bytes, sub.sim_time_us, "s-", color=COL_SIM,
                label="AstraSim simulated")

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

    ax.plot(sub.size_bytes, sub.nccl_busbw_GBs_oop, "o-", color=COL_NCCL,
            label="NCCL bus bandwidth")
    if sub.sim_busbw_GBs.notna().any():
        ax.plot(sub.size_bytes, sub.sim_busbw_GBs, "s-", color=COL_SIM,
                label="AstraSim bus bandwidth")

    _format_size_axis(ax)
    ax.set_ylabel("Bus bandwidth (GB/s)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{collective}: bus bandwidth vs payload size  (4\u00d7H100, intra-node, NV6)")
    ax.legend(loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)


def plot_overview_time(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots()
    for coll in SIM_COLLECTIVES:
        sub = df[df.collective == coll].sort_values("size_bytes")
        c = COLLECTIVE_PALETTE[coll]
        ax.plot(sub.size_bytes, sub.nccl_time_us_oop, "o-",  color=c,
                label=f"{coll} \u2014 measured")
        ax.plot(sub.size_bytes, sub.sim_time_us,      "s--", color=c, alpha=0.7,
                label=f"{coll} \u2014 simulated")
    ax.set_yscale("log")
    _format_size_axis(ax)
    ax.set_ylabel("Time (\u00b5s) \u2014 log scale")
    ax.set_title("Sim vs measured time across modelled collectives")
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(out_path)
    plt.close(fig)


def plot_overview_busbw(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots()
    for coll in SIM_COLLECTIVES:
        sub = df[df.collective == coll].sort_values("size_bytes")
        c = COLLECTIVE_PALETTE[coll]
        ax.plot(sub.size_bytes, sub.nccl_busbw_GBs_oop, "o-",  color=c,
                label=f"{coll} \u2014 measured")
        ax.plot(sub.size_bytes, sub.sim_busbw_GBs,      "s--", color=c, alpha=0.7,
                label=f"{coll} \u2014 simulated")
    _format_size_axis(ax)
    ax.set_ylabel("Bus bandwidth (GB/s)")
    ax.set_ylim(bottom=0)
    ax.set_title("Sim vs measured bus bandwidth across modelled collectives")
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(out_path)
    plt.close(fig)


def plot_error_pct(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots()
    for coll in SIM_COLLECTIVES:
        sub = df[df.collective == coll].sort_values("size_bytes")
        sub = sub[sub.err_pct_vs_oop.notna()]
        ax.plot(sub.size_bytes, sub.err_pct_vs_oop, "o-",
                color=COLLECTIVE_PALETTE[coll], label=coll)
    ax.axhline(0.0, color="black", linewidth=0.8)
    for thresh in (-30, -10, 10, 30):
        ax.axhline(thresh, color="lightgrey", linewidth=0.6, linestyle=":")
    _format_size_axis(ax)
    ax.set_ylabel("(sim - measured) / measured  [%]")
    ax.set_title("AstraSim error % vs measured (out-of-place) across the sweep")
    ax.legend(loc="lower right")
    fig.savefig(out_path)
    plt.close(fig)


def plot_nccl_all_busbw(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    for coll in ALL_NCCL_COLLECTIVES:
        sub = df[df.collective == coll].sort_values("size_bytes")
        if sub.empty:
            continue
        ax.plot(sub.size_bytes, sub.nccl_busbw_GBs_oop, "o-",
                color=COLLECTIVE_PALETTE[coll], label=coll)
    _format_size_axis(ax)
    ax.set_ylabel("Bus bandwidth (GB/s)")
    ax.set_ylim(bottom=0)
    ax.set_title("NCCL measured bus bandwidth for all 7 collectives  (4\u00d7H100, intra-node)")
    ax.legend(loc="lower right", ncol=2)
    fig.savefig(out_path)
    plt.close(fig)


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

    over_t = plots_dir / "overview_time.png"
    over_b = plots_dir / "overview_busbw.png"
    err    = plots_dir / "error_pct.png"
    nccl_all = plots_dir / "nccl_all_busbw.png"

    plot_overview_time(df, over_t)
    plot_overview_busbw(df, over_b)
    plot_error_pct(df, err)
    plot_nccl_all_busbw(df, nccl_all)
    produced += [over_t, over_b, err, nccl_all]
    return produced


def parse_args(argv: list[str]) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    results_dir = here.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv",
        type=Path,
        default=results_dir / "output" / "sweep_latest" / "all_data.csv",
        help="Path to a sweep's all_data.csv (default: output/sweep_latest/all_data.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write plots into (default: <csv-dir>/plots/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}\nRun ./scripts/run_sweep.sh first.")
    out_dir = args.out_dir if args.out_dir else args.csv.parent / "plots"
    _set_style()
    produced = generate_all(args.csv, out_dir)
    print(f"Wrote {len(produced)} plots into {out_dir}/")
    for p in produced:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
