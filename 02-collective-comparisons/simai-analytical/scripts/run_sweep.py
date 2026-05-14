#!/usr/bin/env python3
"""
SimAI-Analytical sweep for the H100 NCCL-tests replay.

Generates 24-size workloads (one per collective), runs SimAI-Analytical,
parses the per-layer CSV, joins with the measured NCCL data, and writes
a summary.csv that is directly comparable to the AstraSim sibling project.

This script is intentionally minimal: no calibration loop, no plotting,
no in-place/out-of-place split. We feed SimAI a single `-nv` value
(default 370 GiB/s = H100 SM90 18-NVLink datasheet) and report what it
predicts vs reality.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

# --- constants -----------------------------------------------------------

REPLAY_DIR = Path(__file__).resolve().parents[1]
SIMAI_DIR = REPLAY_DIR.parent.parent / "SimAI"
SIMAI_BIN = SIMAI_DIR / "bin" / "SimAI_analytical"
RATIO_SRC = SIMAI_DIR / "astra-sim-alibabacloud" / "inputs" / "ratio"

NCCL_ALL_DATA = (
    REPLAY_DIR.parent
    / "h100-nccl-analytical-1gib-replay"
    / "output"
    / "sweep_latest"
    / "all_data.csv"
)

# 1 KiB .. 8 GiB doubling, matches the NCCL/AstraSim sweep exactly.
SIZES_BYTES = [1 << k for k in range(10, 34)]

# SimAI's collective tokens (case-insensitive in cal_busbw, but we keep it
# stable to make grep-friendly logs).
COLLECTIVES = {
    "all_reduce":     "ALLREDUCE",
    "all_gather":     "ALLGATHER",
    "reduce_scatter": "REDUCESCATTER",
}

# NCCL bus-bandwidth factors for n ranks. SimAI uses these internally;
# we recompute here only to derive algbw -> busbw on the SimAI side
# (SimAI reports both, so this is just sanity).
def bus_factor(coll: str, n: int) -> float:
    if coll == "all_reduce":
        return 2 * (n - 1) / n
    if coll in ("all_gather", "reduce_scatter"):
        return (n - 1) / n
    raise ValueError(coll)


# --- workload generation ------------------------------------------------

def write_workload(path: Path, simai_coll: str, sizes: list[int], n_gpus: int) -> None:
    """One workload.txt with one layer per payload size.

    NB: SimAI's `comm_size` for ALLGATHER is the TOTAL output buffer
    (opposite of AstraSim, where it's the per-rank chunk). Verified
    empirically: feeding 1 GiB to ALLGATHER produces a time matching the
    NCCL AllGather 1 GiB row to within ~20%, whereas feeding 1 GiB / n
    produces a time ~n times too small. We therefore feed the raw size
    for all three collectives.
    """
    lines: list[str] = []
    lines.append(
        f"HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: {n_gpus} "
        f"ep: 1 pp: 1 vpp: 1 ga: 1 all_gpus: {n_gpus} "
        f"checkpoints: 0 checkpoint_initiates: 0"
    )
    lines.append(str(len(sizes)))
    for s in sizes:
        # name    dep  fwd_comp  fwd_comm   fwd_size  ig_comp ig_comm ig_size  wg_comp wg_comm wg_size  loops
        lines.append(
            f"layer_{s}\t-1\t1\t{simai_coll}\t{s}\t1\tNONE\t0\t1\tNONE\t0\t1"
        )
    path.write_text("\n".join(lines) + "\n")


# --- run + parse --------------------------------------------------------

def run_simai(
    workload: Path,
    out_prefix: str,
    *,
    nv: float,
    n_gpus: int,
    cwd: Path,
) -> Path:
    """Invoke SimAI_analytical. Returns the path to the produced EndToEnd.csv.

    SimAI auto-prepends './results/' to -r. We run from `cwd` so the ratio
    CSV symlink resolves and outputs land where expected.
    """
    (cwd / "results").mkdir(exist_ok=True)
    cmd = [
        str(SIMAI_BIN),
        "-w", str(workload),
        "-g", str(n_gpus),
        "-g_p_s", str(n_gpus),
        "-g_type", "H100",
        "-nv", str(nv),
        "-nic", "48.5",         # dummy: single-node, but parser requires >0
        "-n_p_s", "1",
        "-nic_t", "cx7",
        "-r", out_prefix,
    ]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(
            f"SimAI_analytical failed (rc={proc.returncode}) for {workload.name}"
        )
    return cwd / "results" / f"{out_prefix}EndToEnd.csv"


def parse_endtoend(csv_path: Path, sizes: list[int]) -> list[dict]:
    """SimAI EndToEnd.csv layout:

        line 1: summary header
        line 2: summary row
        line 3: per-layer header
        line 4+: per-layer rows (one per workload layer, in order)

    We pull `fwd total comm` (microseconds), `algbw`, `busbw` for each layer.
    """
    rows = csv_path.read_text().splitlines()
    if len(rows) < 4:
        raise RuntimeError(f"unexpected SimAI CSV layout: {csv_path}")
    per_layer = rows[3:]
    if len(per_layer) != len(sizes):
        raise RuntimeError(
            f"layer count mismatch in {csv_path}: got {len(per_layer)}, expected {len(sizes)}"
        )
    out: list[dict] = []
    for size, line in zip(sizes, per_layer):
        f = [c.strip() for c in line.split(",")]
        # columns: layer_name, run_name, fwd_compute, wg_compute, ig_compute,
        #          fwd_exposed, wg_exposed, ig_exposed,
        #          fwd_total_comm, fwd_algbw, fwd_busbw, ...
        out.append({
            "size_bytes": size,
            "sim_time_us": float(f[8]),
            "sim_algbw_GiBs": float(f[9]),
            "sim_busbw_GiBs": float(f[10]),
        })
    return out


# --- NCCL ground truth --------------------------------------------------

def load_nccl_measured() -> dict[tuple[str, int], dict]:
    """Map (collective, size) -> measured numbers, out-of-place columns only."""
    out: dict[tuple[str, int], dict] = {}
    with NCCL_ALL_DATA.open() as fh:
        for r in csv.DictReader(fh):
            coll = r["collective"]
            if coll not in COLLECTIVES:
                continue
            out[(coll, int(r["size_bytes"]))] = {
                "nccl_time_us":   float(r["nccl_time_us_oop"]),
                "nccl_algbw_GBs": float(r["nccl_algbw_GBs_oop"]),
                "nccl_busbw_GBs": float(r["nccl_busbw_GBs_oop"]),
            }
    return out


# --- main ---------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nv", type=float, default=370.0,
                    help="per-GPU NVLink aggregate BW (GiB/s). "
                         "Default 370 = H100 SM90 datasheet (18 lanes x 20.6).")
    ap.add_argument("--gpus", type=int, default=4,
                    help="number of GPUs (single node). Default 4.")
    ap.add_argument("--out", type=Path,
                    default=REPLAY_DIR / "output" / "sweep_latest",
                    help="output directory")
    args = ap.parse_args()

    if not SIMAI_BIN.exists():
        print(f"SimAI binary not found at {SIMAI_BIN}", file=sys.stderr)
        return 1

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Workspace for SimAI runs: needs a `results/` subdir for the prefix to
    # land in, and a sibling `astra-sim-alibabacloud/inputs/ratio` for the
    # ratio CSV reads (cal_ratio uses a hardcoded relative path).
    run_root = out_dir / "simai_runs"
    run_root.mkdir(exist_ok=True)
    ratio_link = run_root / "astra-sim-alibabacloud" / "inputs" / "ratio"
    if not ratio_link.exists():
        ratio_link.parent.mkdir(parents=True, exist_ok=True)
        ratio_link.symlink_to(RATIO_SRC)

    workloads_dir = REPLAY_DIR / "inputs" / "workloads"
    workloads_dir.mkdir(parents=True, exist_ok=True)

    nccl = load_nccl_measured()

    summary_rows: list[dict] = []

    for nccl_coll, simai_coll in COLLECTIVES.items():
        wl = workloads_dir / f"{nccl_coll}.txt"
        write_workload(wl, simai_coll, SIZES_BYTES, args.gpus)

        csv_path = run_simai(
            workload=wl,
            out_prefix=f"{nccl_coll}-",
            nv=args.nv,
            n_gpus=args.gpus,
            cwd=run_root,
        )
        sim_rows = parse_endtoend(csv_path, SIZES_BYTES)

        for r in sim_rows:
            size = r["size_bytes"]
            m = nccl.get((nccl_coll, size), {})
            sim_busbw_GBs = r["sim_busbw_GiBs"] * (1024 ** 3) / 1e9
            sim_algbw_GBs = r["sim_algbw_GiBs"] * (1024 ** 3) / 1e9
            row = {
                "collective": nccl_coll,
                "size_bytes": size,
                "nccl_time_us":   m.get("nccl_time_us"),
                "nccl_algbw_GBs": m.get("nccl_algbw_GBs"),
                "nccl_busbw_GBs": m.get("nccl_busbw_GBs"),
                "sim_time_us":   r["sim_time_us"],
                "sim_algbw_GBs": round(sim_algbw_GBs, 4),
                "sim_busbw_GBs": round(sim_busbw_GBs, 4),
            }
            if m:
                row["err_pct_time"] = round(
                    100 * (r["sim_time_us"] - m["nccl_time_us"]) / m["nccl_time_us"], 2
                )
                row["err_pct_busbw"] = round(
                    100 * (sim_busbw_GBs - m["nccl_busbw_GBs"]) / m["nccl_busbw_GBs"], 2
                )
            summary_rows.append(row)

    summary_csv = out_dir / "summary.csv"
    fieldnames = [
        "collective", "size_bytes",
        "nccl_time_us", "nccl_algbw_GBs", "nccl_busbw_GBs",
        "sim_time_us",  "sim_algbw_GBs",  "sim_busbw_GBs",
        "err_pct_time", "err_pct_busbw",
    ]
    with summary_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    # Console summary: one row per (collective, size) with the headline diff.
    print(f"\nSimAI-Analytical sweep, -nv {args.nv} GiB/s, {args.gpus} GPUs single-node")
    print(f"NCCL ground truth: {NCCL_ALL_DATA}")
    print(f"Output: {summary_csv}\n")
    print(f"{'collective':<16} {'size':>11}  "
          f"{'nccl_us':>10} {'sim_us':>10}  {'err%':>7}   "
          f"{'nccl_busbw':>11} {'sim_busbw':>11}  {'err%':>7}")
    for r in summary_rows:
        print(
            f"{r['collective']:<16} {r['size_bytes']:>11}  "
            f"{r['nccl_time_us']:>10.2f} {r['sim_time_us']:>10.2f}  "
            f"{r.get('err_pct_time', float('nan')):>+7.1f}   "
            f"{r['nccl_busbw_GBs']:>11.3f} {r['sim_busbw_GBs']:>11.3f}  "
            f"{r.get('err_pct_busbw', float('nan')):>+7.1f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
