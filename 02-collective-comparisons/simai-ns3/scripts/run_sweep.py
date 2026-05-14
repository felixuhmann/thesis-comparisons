#!/usr/bin/env python3
"""
SimAI-Simulation (NS3 backend) sweep for the H100 NCCL-tests replay.

Mirrors the AstraSim and SimAI-Analytical replays:
- 24 payload sizes (1 KiB .. 8 GiB doubling)
- 3 collectives (all_reduce, all_gather, reduce_scatter)
- 4-GPU single-node H100 topology, NV6, NVLS enabled

NS3 mode has no `-r` output prefix flag; the binary always writes to
./ncclFlowModel_EndToEnd.csv in the CWD. We work around that by running
each collective in its own subdirectory, then collecting outputs.

Per-run cost: ~30-60 seconds (compared to ~10 ms for SimAI-Analytical).
Full sweep: ~25-45 minutes total for 3 collectives x 24 sizes.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --- constants -----------------------------------------------------------

REPLAY_DIR = Path(__file__).resolve().parents[1]
SIMAI_DIR = REPLAY_DIR.parent.parent / "SimAI"
SIMAI_BIN = SIMAI_DIR / "bin" / "SimAI_simulator"
TOPO_FILE = REPLAY_DIR / "inputs" / "topo" / "No_Rail_Opti_4g_4gps_SingleToR_400Gbps_H100"
CONF_FILE = REPLAY_DIR / "inputs" / "SimAI.conf"
NCCL_ALL_DATA = (
    REPLAY_DIR.parent
    / "h100-nccl-analytical-1gib-replay"
    / "output"
    / "sweep_latest"
    / "all_data.csv"
)

SIZES_BYTES = [1 << k for k in range(10, 34)]

COLLECTIVES = {
    "all_reduce":     "ALLREDUCE",
    "all_gather":     "ALLGATHER",
    "reduce_scatter": "REDUCESCATTER",
}


# --- workload generation ------------------------------------------------

def write_workload(path: Path, simai_coll: str, sizes: list[int], n_gpus: int) -> None:
    """One workload with one layer per size. Same semantics as the
    SimAI-Analytical replay: feed the raw size for all collectives
    (SimAI's ALLGATHER comm_size is the total buffer, not per-rank chunk).
    """
    lines = [
        f"HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: {n_gpus} "
        f"ep: 1 pp: 1 vpp: 1 ga: 1 all_gpus: {n_gpus} "
        f"checkpoints: 0 checkpoint_initiates: 0",
        str(len(sizes)),
    ]
    for s in sizes:
        lines.append(
            f"layer_{s}\t-1\t1\t{simai_coll}\t{s}\t1\tNONE\t0\t1\tNONE\t0\t1"
        )
    path.write_text("\n".join(lines) + "\n")


# --- run a single NS3 simulation ----------------------------------------

def run_ns3(
    workload: Path,
    *,
    threads: int,
    send_lat: int,
    nvls_enable: bool,
    workdir: Path,
) -> Path:
    """Invoke SimAI_simulator in `workdir`. Returns path to ncclFlowModel_EndToEnd.csv.

    The binary writes outputs to the CWD; we run in `workdir` to isolate them.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["AS_SEND_LAT"] = str(send_lat)
    env["AS_NVLS_ENABLE"] = "1" if nvls_enable else "0"
    cmd = [
        str(SIMAI_BIN),
        "-t", str(threads),
        "-w", str(workload),
        "-n", str(TOPO_FILE),
        "-c", str(CONF_FILE),
    ]
    start = time.time()
    proc = subprocess.run(
        cmd, cwd=workdir, env=env, capture_output=True, text=True
    )
    elapsed = time.time() - start
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(
            f"SimAI_simulator failed (rc={proc.returncode}) for {workload.name}"
        )
    print(f"  -> {workload.name}: {elapsed:.1f}s", flush=True)
    return workdir / "ncclFlowModel_EndToEnd.csv"


def parse_endtoend(csv_path: Path, sizes: list[int]) -> list[dict]:
    """Parse ncclFlowModel_EndToEnd.csv. Layout:

        line 1: summary header
        line 2: summary row (one)
        line 3: per-layer column header
        line 4..N: one per-layer row in workload order, then a SUM row,
                  then a "total exposed comm,..." trailer.
    """
    rows = csv_path.read_text().splitlines()
    if len(rows) < 4:
        raise RuntimeError(f"unexpected CSV layout: {csv_path}")
    layer_rows = [r for r in rows[3:] if r and not r.startswith(("SUM,", "total"))]
    if len(layer_rows) != len(sizes):
        raise RuntimeError(
            f"layer count mismatch in {csv_path}: got {len(layer_rows)}, expected {len(sizes)}"
        )
    out: list[dict] = []
    for size, line in zip(sizes, layer_rows):
        f = [c.strip() for c in line.split(",")]
        # cols: layer_name, run_name, fwd_compute, wg_compute, ig_compute,
        #       fwd_exposed, wg_exposed, ig_exposed,
        #       fwd_total_comm, fwd_algbw, fwd_busbw, ...
        out.append({
            "size_bytes": size,
            "sim_time_us":    float(f[8]),
            "sim_algbw_GiBs": float(f[9]),
            "sim_busbw_GiBs": float(f[10]),
        })
    return out


# --- NCCL ground truth --------------------------------------------------

def load_nccl_measured() -> dict[tuple[str, int], dict]:
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
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--send-lat", type=int, default=3,
                    help="AS_SEND_LAT (us). Default 3, per Tutorial's NVLS example.")
    ap.add_argument("--no-nvls", action="store_true",
                    help="Disable AS_NVLS_ENABLE. Default: enabled.")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--out", type=Path,
                    default=REPLAY_DIR / "output" / "sweep_latest")
    args = ap.parse_args()

    if not SIMAI_BIN.exists():
        print(f"SimAI_simulator binary not found at {SIMAI_BIN}", file=sys.stderr)
        return 1
    if not TOPO_FILE.exists():
        print(f"Topology file not found at {TOPO_FILE}", file=sys.stderr)
        return 1

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    workloads_dir = REPLAY_DIR / "inputs" / "workloads"
    workloads_dir.mkdir(parents=True, exist_ok=True)
    runs_root = out_dir / "ns3_runs"
    runs_root.mkdir(exist_ok=True)

    nccl = load_nccl_measured()
    summary_rows: list[dict] = []

    print(f"NS3 sweep: 3 collectives x {len(SIZES_BYTES)} sizes, "
          f"NVLS={'on' if not args.no_nvls else 'off'}, "
          f"AS_SEND_LAT={args.send_lat} us, threads={args.threads}",
          flush=True)

    for nccl_coll, simai_coll in COLLECTIVES.items():
        wl = workloads_dir / f"{nccl_coll}.txt"
        write_workload(wl, simai_coll, SIZES_BYTES, args.gpus)

        run_dir = runs_root / nccl_coll
        # Clear any stale output from a previous run.
        for stale in run_dir.glob("ncclFlowModel_*"):
            stale.unlink()

        print(f"Running {nccl_coll} ({len(SIZES_BYTES)} layers in one workload)...",
              flush=True)
        csv_path = run_ns3(
            workload=wl,
            threads=args.threads,
            send_lat=args.send_lat,
            nvls_enable=not args.no_nvls,
            workdir=run_dir,
        )
        sim_rows = parse_endtoend(csv_path, SIZES_BYTES)

        for r in sim_rows:
            size = r["size_bytes"]
            m = nccl.get((nccl_coll, size), {})
            sim_busbw_GBs = r["sim_busbw_GiBs"] * (1024 ** 3) / 1e9
            sim_algbw_GBs = r["sim_algbw_GiBs"] * (1024 ** 3) / 1e9
            row = {
                "collective":      nccl_coll,
                "size_bytes":      size,
                "nccl_time_us":    m.get("nccl_time_us"),
                "nccl_algbw_GBs":  m.get("nccl_algbw_GBs"),
                "nccl_busbw_GBs":  m.get("nccl_busbw_GBs"),
                "sim_time_us":     round(r["sim_time_us"], 3),
                "sim_algbw_GBs":   round(sim_algbw_GBs, 4),
                "sim_busbw_GBs":   round(sim_busbw_GBs, 4),
            }
            if m and m.get("nccl_time_us"):
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

    print(f"\nWrote {summary_csv}\n")
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
