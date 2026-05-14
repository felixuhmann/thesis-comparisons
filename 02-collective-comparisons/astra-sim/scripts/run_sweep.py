#!/usr/bin/env python3
"""Run the full NCCL-style size sweep through AstraSim's analytical backend.

Mirrors the size sweep emitted by `nccl-tests` with arguments
`-b 1K -e 8G -f 2` (24 sizes from 1 KiB to 8 GiB, doubling each step), for
the three collectives all_reduce, all_gather, reduce_scatter.

For each (collective, size) pair this script:
  1. Generates a 4-rank Chakra ET workload with the correct `comm_size`
     attribute (see CONVENTION note below).
  2. Invokes AstraSim's analytical congestion-aware binary.
  3. Captures the per-rank "Wall time" the simulator prints.

When all sims are done, it joins the per-(collective, size) sim times with
the corresponding rows from ../inputs/nccl-tests.out and writes
output/sweep_<UTC-TS>/summary.csv plus a pretty-printed summary.txt.

CONVENTION NOTE (NCCL vs AstraSim `comm_size`):
  AstraSim's all_gather interprets `comm_size` as the per-rank input chunk,
  not the total receive buffer. NCCL's `size` column for all_gather, on the
  other hand, is the total receive buffer. To replay an NCCL all_gather row
  faithfully we therefore feed AstraSim `comm_size = size / n`. For
  all_reduce and reduce_scatter the conventions agree (both use the total
  buffer), so we feed `comm_size = size`.
"""

from __future__ import annotations
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter")
NCCL_TEST_NAME = {
    "all_reduce":     "all_reduce_perf",
    "all_gather":     "all_gather_perf",
    "reduce_scatter": "reduce_scatter_perf",
}
# Bus-bandwidth NCCL-formula factors (algbw -> busbw), for n=4 GPUs.
BUSBW_FACTOR = {
    "all_reduce":     1.5,   # 2*(n-1)/n
    "all_gather":     0.75,  # (n-1)/n
    "reduce_scatter": 0.75,  # (n-1)/n
}

# Every NCCL test in nccl-tests.out -> human collective name.
# AstraSim only models the first 3 entries (see COLLECTIVES); we still emit
# the measured rows for the rest into all_data.csv for completeness.
NCCL_ALL_TESTS = (
    ("all_reduce_perf",     "all_reduce"),
    ("all_gather_perf",     "all_gather"),
    ("reduce_scatter_perf", "reduce_scatter"),
    ("broadcast_perf",      "broadcast"),
    ("reduce_perf",         "reduce"),
    ("alltoall_perf",       "all_to_all"),
    ("sendrecv_perf",       "sendrecv"),
)


def sweep_sizes() -> list[int]:
    """NCCL's `-b 1K -e 8G -f 2`: 1 KiB up to 8 GiB doubling each step."""
    return [1024 * (1 << k) for k in range(24)]


# -------- AstraSim workload (.et) generation -----------------------------

def write_workload(
    astra_sim_dir: Path,
    out_dir: Path,
    collective: str,
    npus: int,
    nccl_size_bytes: int,
) -> Path:
    """Write `<collective>.<rank>.et` files for `npus` ranks into `out_dir`.

    The on-disk byte layout is identical to what the upstream
    astra-sim/examples/workload/microbenchmarks/generator_scripts emits.
    """
    if str(astra_sim_dir) not in sys.path:
        sys.path.insert(0, str(astra_sim_dir))

    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (
        GlobalMetadata,
        COMM_COLL_NODE,
        ALL_REDUCE,
        ALL_GATHER,
        REDUCE_SCATTER,
    )
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (
        AttributeProto as ChakraAttr,
    )
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode
    from extern.graph_frontend.chakra.src.third_party.utils.protolib import (
        encodeMessage as encode_message,
    )

    enum_for = {
        "all_reduce":     ALL_REDUCE,
        "all_gather":     ALL_GATHER,
        "reduce_scatter": REDUCE_SCATTER,
    }
    # See module docstring for why all_gather is divided by n.
    comm_size_bytes = nccl_size_bytes // npus if collective == "all_gather" else nccl_size_bytes

    out_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(npus):
        with open(out_dir / f"{collective}.{rank}.et", "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = ChakraNode()
            node.id = 0
            node.name = (
                f"{collective}_n{npus}_ncclsize{nccl_size_bytes}_commsize{comm_size_bytes}"
            )
            node.type = COMM_COLL_NODE
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            node.attr.append(ChakraAttr(name="comm_type", int64_val=enum_for[collective]))
            node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size_bytes))
            encode_message(et, node)

    return out_dir


# -------- AstraSim invocation --------------------------------------------

_WALL_TIME_RE = re.compile(r"Wall time:\s*(\d+)")


@dataclass
class SimResult:
    sim_ns: Optional[int]
    returncode: int
    log_path: Path


def run_astra_sim(
    binary: Path,
    workload_prefix: Path,
    system_cfg: Path,
    network_cfg: Path,
    remote_mem_cfg: Path,
    run_dir: Path,
) -> SimResult:
    """Invoke AstraSim once. Side-channel `log/` dir goes inside run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "simulator.log"
    cmd = [
        str(binary),
        f"--workload-configuration={workload_prefix}",
        f"--system-configuration={system_cfg}",
        f"--network-configuration={network_cfg}",
        f"--remote-memory-configuration={remote_mem_cfg}",
    ]
    with open(log_path, "w") as log:
        log.write("# AstraSim analytical congestion-aware\n")
        log.write(f"# cmd: {' '.join(cmd)}\n")
        log.write("# ---\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT)

    sim_ns: Optional[int] = None
    if proc.returncode == 0:
        wall_times = []
        for line in log_path.read_text().splitlines():
            m = _WALL_TIME_RE.search(line)
            if m:
                wall_times.append(int(m.group(1)))
        if wall_times:
            sim_ns = max(wall_times)
    return SimResult(sim_ns=sim_ns, returncode=proc.returncode, log_path=log_path)


# -------- NCCL measurement parsing ----------------------------------------

def parse_nccl_measurements(nccl_log: Path) -> dict[tuple[str, int], float]:
    """Return {(collective, size_bytes): out-of-place time_us} from nccl-tests output.

    Lightweight wrapper around `parse_nccl_full` that only keeps the
    out-of-place time, for the 3 collectives we actually simulate. Used by
    the sweep itself.
    """
    full = parse_nccl_full(nccl_log)
    out: dict[tuple[str, int], float] = {}
    by_test = dict(NCCL_ALL_TESTS)
    for (test_name, size_bytes), row in full.items():
        coll = by_test.get(test_name)
        if coll in NCCL_TEST_NAME and row.get("time_us_oop") is not None:
            out[(coll, size_bytes)] = row["time_us_oop"]
    return out


def parse_nccl_full(nccl_log: Path) -> dict[tuple[str, int], dict]:
    """Return {(nccl_test_name, size_bytes): row_dict} with every column.

    row_dict keys: count, dtype, redop, root,
                   time_us_oop, algbw_GBs_oop, busbw_GBs_oop, wrong_oop,
                   time_us_ip,  algbw_GBs_ip,  busbw_GBs_ip,  wrong_ip
    """
    out: dict[tuple[str, int], dict] = {}
    section_start = re.compile(r"===== (\w+_perf) @ ")
    section_end = re.compile(r"^=====")
    cur_test: Optional[str] = None

    def _maybe_int(s: str) -> Optional[int]:
        try:
            return int(s)
        except ValueError:
            return None

    def _maybe_float(s: str) -> Optional[float]:
        try:
            return float(s)
        except ValueError:
            return None

    with nccl_log.open() as f:
        for line in f:
            m = section_start.search(line)
            if m:
                cur_test = m.group(1)
                continue
            if cur_test and section_end.match(line):
                cur_test = None
                continue
            if cur_test is None:
                continue
            parts = line.split()
            if not parts or not parts[0].isdigit():
                continue
            # cols: size count type redop root
            #       time(us) algbw busbw #wrong       (out-of-place)
            #       time(us) algbw busbw #wrong       (in-place)
            # alltoall/sendrecv print "N/A" instead of 0 for #wrong.
            if len(parts) < 13:
                continue
            size_bytes = int(parts[0])
            out[(cur_test, size_bytes)] = {
                "count": _maybe_int(parts[1]),
                "dtype": parts[2],
                "redop": parts[3],
                "root":  _maybe_int(parts[4]),
                "time_us_oop":   _maybe_float(parts[5]),
                "algbw_GBs_oop": _maybe_float(parts[6]),
                "busbw_GBs_oop": _maybe_float(parts[7]),
                "wrong_oop":     parts[8],
                "time_us_ip":    _maybe_float(parts[9]),
                "algbw_GBs_ip":  _maybe_float(parts[10]),
                "busbw_GBs_ip":  _maybe_float(parts[11]),
                "wrong_ip":      parts[12],
            }
    return out


# -------- Bandwidth helpers ----------------------------------------------

def algbw_GBs(size_bytes: int, time_us: float) -> float:
    """NCCL algbw: total payload size / time. Returns GB/s (decimal 1e9)."""
    if time_us <= 0:
        return float("nan")
    return size_bytes / time_us / 1e3  # bytes / us == MB/us == GB/s


def busbw_GBs(collective: str, size_bytes: int, time_us: float) -> float:
    return algbw_GBs(size_bytes, time_us) * BUSBW_FACTOR[collective]


# -------- Sweep driver ----------------------------------------------------

@dataclass
class Row:
    collective: str
    size_bytes: int
    measured_us: Optional[float]
    sim_us: Optional[float]

    @property
    def err_pct(self) -> Optional[float]:
        if self.measured_us is None or self.sim_us is None or self.measured_us <= 0:
            return None
        return (self.sim_us / self.measured_us - 1.0) * 100.0


def run_sweep(args: argparse.Namespace) -> Path:
    inputs_dir = args.inputs_dir
    astra_sim_dir = args.astra_sim_dir
    binary = (
        astra_sim_dir
        / "build" / "astra_analytical" / "build" / "bin"
        / "AstraSim_Analytical_Congestion_Aware"
    )
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit(
            f"AstraSim binary missing/not executable at:\n  {binary}\n"
            f"Build it with: {astra_sim_dir}/build/astra_analytical/build.sh"
        )

    system_cfg     = inputs_dir / "system.json"
    network_cfg    = inputs_dir / "network.yml"
    remote_mem_cfg = inputs_dir / "remote_memory.json"
    nccl_log       = inputs_dir / "nccl-tests.out"
    for f in (system_cfg, network_cfg, remote_mem_cfg, nccl_log):
        if not f.is_file():
            raise SystemExit(f"Missing required input file: {f}")

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    sweep_dir = args.output_dir / f"sweep_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=False)

    # Snapshot the inputs that drove this sweep, so the run is reproducible
    # even if inputs/ changes later.
    snapshot = sweep_dir / "inputs_snapshot"
    snapshot.mkdir()
    for f in (system_cfg, network_cfg, remote_mem_cfg, nccl_log):
        (snapshot / f.name).write_bytes(f.read_bytes())

    measurements = parse_nccl_measurements(nccl_log)
    rows: list[Row] = []
    sizes = args.sizes if args.sizes else sweep_sizes()

    total = len(args.collectives) * len(sizes)
    done = 0
    print(
        f"[sweep] {ts}: running {total} sims "
        f"({len(args.collectives)} collectives x {len(sizes)} sizes)"
    )

    for coll in args.collectives:
        for size_bytes in sizes:
            done += 1
            wl_dir = sweep_dir / "workload" / coll / str(size_bytes)
            write_workload(
                astra_sim_dir=astra_sim_dir,
                out_dir=wl_dir,
                collective=coll,
                npus=args.npus,
                nccl_size_bytes=size_bytes,
            )
            run_dir = sweep_dir / coll / str(size_bytes)
            res = run_astra_sim(
                binary=binary,
                workload_prefix=wl_dir / coll,
                system_cfg=system_cfg,
                network_cfg=network_cfg,
                remote_mem_cfg=remote_mem_cfg,
                run_dir=run_dir,
            )
            sim_us = res.sim_ns / 1000.0 if res.sim_ns is not None else None
            measured_us = measurements.get((coll, size_bytes))
            rows.append(Row(coll, size_bytes, measured_us, sim_us))

            tag = "ok" if res.returncode == 0 and sim_us is not None else "FAIL"
            print(
                f"  [{done:>3}/{total}] {coll:14s} size={size_bytes:>11d} B  "
                f"sim={sim_us if sim_us is None else f'{sim_us:>10.2f} us':>14}  "
                f"meas={measured_us if measured_us is None else f'{measured_us:>10.2f} us':>14}  "
                f"[{tag}]"
            )

    write_summary(sweep_dir, rows)
    write_all_data_csv(
        sweep_dir=sweep_dir,
        nccl_log=nccl_log,
        sim_rows=rows,
    )
    latest = args.output_dir / "sweep_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(sweep_dir.name)
    print(f"[sweep] done -> {sweep_dir}")
    print(f"[sweep] latest -> {latest}")
    return sweep_dir


def write_summary(sweep_dir: Path, rows: Iterable[Row]) -> None:
    rows = list(rows)
    csv_path = sweep_dir / "summary.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "collective",
            "size_bytes",
            "measured_us",
            "sim_us",
            "error_pct",
            "measured_algbw_GBs",
            "sim_algbw_GBs",
            "measured_busbw_GBs",
            "sim_busbw_GBs",
        ])
        for r in rows:
            m_alg = algbw_GBs(r.size_bytes, r.measured_us) if r.measured_us else None
            s_alg = algbw_GBs(r.size_bytes, r.sim_us) if r.sim_us else None
            m_bus = busbw_GBs(r.collective, r.size_bytes, r.measured_us) if r.measured_us else None
            s_bus = busbw_GBs(r.collective, r.size_bytes, r.sim_us) if r.sim_us else None
            w.writerow([
                r.collective,
                r.size_bytes,
                f"{r.measured_us:.2f}" if r.measured_us is not None else "",
                f"{r.sim_us:.2f}"      if r.sim_us      is not None else "",
                f"{r.err_pct:+.1f}"    if r.err_pct     is not None else "",
                f"{m_alg:.3f}" if m_alg is not None else "",
                f"{s_alg:.3f}" if s_alg is not None else "",
                f"{m_bus:.3f}" if m_bus is not None else "",
                f"{s_bus:.3f}" if s_bus is not None else "",
            ])

    txt_path = sweep_dir / "summary.txt"
    with txt_path.open("w") as fh:
        fh.write(format_summary_text(rows))
    print(format_summary_text(rows))
    print(f"[sweep] wrote {csv_path}")
    print(f"[sweep] wrote {txt_path}")


def write_all_data_csv(
    sweep_dir: Path,
    nccl_log: Path,
    sim_rows: list[Row],
) -> None:
    """Emit a wide spreadsheet-friendly CSV with every NCCL row and every
    sim row, joined on (collective, size_bytes).

    For collectives we did not simulate (broadcast, reduce, all_to_all,
    sendrecv) the sim_* columns are blank.
    """
    sim_by_key = {(r.collective, r.size_bytes): r for r in sim_rows}
    nccl_full = parse_nccl_full(nccl_log)

    csv_path = sweep_dir / "all_data.csv"
    cols = [
        "collective",
        "size_bytes",
        "size_KiB",
        "count",
        "dtype",
        "redop",
        "root",
        # measured (out-of-place)
        "nccl_time_us_oop",
        "nccl_algbw_GBs_oop",
        "nccl_busbw_GBs_oop",
        "nccl_wrong_oop",
        # measured (in-place)
        "nccl_time_us_ip",
        "nccl_algbw_GBs_ip",
        "nccl_busbw_GBs_ip",
        "nccl_wrong_ip",
        # simulated (only set for the 3 collectives AstraSim modelled)
        "sim_time_us",
        "sim_algbw_GBs",
        "sim_busbw_GBs",
        # joined deltas (sim - measured)
        "sim_minus_nccl_oop_us",
        "sim_minus_nccl_ip_us",
        "err_pct_vs_oop",
        "err_pct_vs_ip",
    ]

    def _fmt(v, kind: str = "default") -> str:
        if v is None or v == "":
            return ""
        if isinstance(v, float):
            if kind == "err":
                return f"{v:+.2f}"
            if kind == "us":
                return f"{v:.2f}"
            if kind == "gbs":
                return f"{v:.3f}"
            return f"{v:.4f}"
        return str(v)

    n_rows = 0
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for nccl_test, collective in NCCL_ALL_TESTS:
            keys = sorted(k for k in nccl_full if k[0] == nccl_test)
            for (_, size_bytes) in keys:
                m = nccl_full[(nccl_test, size_bytes)]
                sim = sim_by_key.get((collective, size_bytes))
                sim_us       = sim.sim_us if sim else None
                meas_oop_us  = m["time_us_oop"]
                meas_ip_us   = m["time_us_ip"]
                sim_algbw    = algbw_GBs(size_bytes, sim_us) if sim_us else None
                sim_busbw    = (
                    busbw_GBs(collective, size_bytes, sim_us)
                    if (sim_us is not None and collective in BUSBW_FACTOR)
                    else None
                )
                delta_oop = (sim_us - meas_oop_us) if (sim_us and meas_oop_us) else None
                delta_ip  = (sim_us - meas_ip_us)  if (sim_us and meas_ip_us)  else None
                err_oop = (delta_oop / meas_oop_us * 100.0) if delta_oop is not None and meas_oop_us else None
                err_ip  = (delta_ip  / meas_ip_us  * 100.0) if delta_ip  is not None and meas_ip_us  else None

                w.writerow([
                    collective,
                    size_bytes,
                    size_bytes // 1024,
                    _fmt(m["count"]),
                    _fmt(m["dtype"]),
                    _fmt(m["redop"]),
                    _fmt(m["root"]),
                    _fmt(meas_oop_us, "us"),
                    _fmt(m["algbw_GBs_oop"], "gbs"),
                    _fmt(m["busbw_GBs_oop"], "gbs"),
                    _fmt(m["wrong_oop"]),
                    _fmt(meas_ip_us, "us"),
                    _fmt(m["algbw_GBs_ip"], "gbs"),
                    _fmt(m["busbw_GBs_ip"], "gbs"),
                    _fmt(m["wrong_ip"]),
                    _fmt(sim_us, "us"),
                    _fmt(sim_algbw, "gbs"),
                    _fmt(sim_busbw, "gbs"),
                    _fmt(delta_oop, "us"),
                    _fmt(delta_ip,  "us"),
                    _fmt(err_oop, "err"),
                    _fmt(err_ip,  "err"),
                ])
                n_rows += 1
    print(f"[sweep] wrote {csv_path} ({n_rows} rows, {len(cols)} columns)")


def format_summary_text(rows: list[Row]) -> str:
    out: list[str] = []
    by_coll: dict[str, list[Row]] = {}
    for r in rows:
        by_coll.setdefault(r.collective, []).append(r)

    for coll, coll_rows in by_coll.items():
        out.append("")
        out.append(f"=== {coll} ===")
        out.append(
            f"{'size(B)':>11s}  {'meas(us)':>10s}  {'sim(us)':>10s}  "
            f"{'err':>7s}  {'meas_algbw':>11s}  {'sim_algbw':>10s}  "
            f"{'meas_busbw':>11s}  {'sim_busbw':>10s}"
        )
        out.append("-" * 100)
        for r in sorted(coll_rows, key=lambda x: x.size_bytes):
            m_alg = algbw_GBs(r.size_bytes, r.measured_us) if r.measured_us else None
            s_alg = algbw_GBs(r.size_bytes, r.sim_us) if r.sim_us else None
            m_bus = busbw_GBs(r.collective, r.size_bytes, r.measured_us) if r.measured_us else None
            s_bus = busbw_GBs(r.collective, r.size_bytes, r.sim_us) if r.sim_us else None
            out.append(
                f"{r.size_bytes:>11d}  "
                f"{(f'{r.measured_us:10.2f}' if r.measured_us else '         ?'):>10s}  "
                f"{(f'{r.sim_us:10.2f}'      if r.sim_us      else '         ?'):>10s}  "
                f"{(f'{r.err_pct:+6.1f}%'    if r.err_pct is not None else '      ?'):>7s}  "
                f"{(f'{m_alg:11.3f}' if m_alg is not None else '          ?'):>11s}  "
                f"{(f'{s_alg:10.3f}' if s_alg is not None else '         ?'):>10s}  "
                f"{(f'{m_bus:11.3f}' if m_bus is not None else '          ?'):>11s}  "
                f"{(f'{s_bus:10.3f}' if s_bus is not None else '         ?'):>10s}"
            )
    return "\n".join(out)


# -------- CLI -------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    results_dir = here.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--astra-sim-dir",
        type=Path,
        default=Path(os.environ.get("ASTRA_SIM_DIR", results_dir / ".." / ".." / "astra-sim")).resolve(),
        help="Root of the astra-sim source tree (default: ../../astra-sim).",
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=(results_dir / "inputs"),
        help="Directory containing network.yml, system.json, remote_memory.json, nccl-tests.out.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(results_dir / "output"),
        help="Where to write sweep_<timestamp>/ subdirectories.",
    )
    parser.add_argument(
        "--collectives",
        nargs="+",
        choices=COLLECTIVES,
        default=list(COLLECTIVES),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None,
        help="Override the default 1 KiB..8 GiB doubling sweep with explicit sizes (bytes).",
    )
    parser.add_argument("--npus", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    run_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
