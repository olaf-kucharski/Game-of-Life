#!/usr/bin/env python3
"""
Benchmark runner for Game of Life implementations (CPU/OpenMP, GPU/CUDA, MPI).
Runs selected grid sizes and saves results to CSV. Designed to be simple, robust,
and to work even if one of the executables is missing (records failure in CSV).
"""
import argparse
import csv
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

ROOT = Path(__file__).parent
CPU_BIN = ROOT / "gol-cpu"
GPU_BIN = ROOT / "gol-gpu"
MPI_BIN = ROOT / "gol-mpi"

RE_CPU_TOTAL = re.compile(r"Laczny czas:\s*([0-9.]+)\s*ms")
RE_CPU_AVG = re.compile(r"Sredni czas na krok:\s*([0-9.]+)\s*ms")
RE_GPU_COMP = re.compile(r"Czas obliczen:\s*([0-9.]+)\s*ms")
RE_GPU_AVG = re.compile(r"Sredni czas na krok:\s*([0-9.]+)\s*ms")
RE_MPI_MAX = re.compile(r"Maksymalny czas.*?:\s*([0-9.]+)\s*ms", re.DOTALL)
RE_MPI_AVG = re.compile(r"Średni czas na krok:\s*([0-9.]+)\s*ms")


def parse_sizes(arg: str) -> List[Tuple[int, int]]:
    sizes = []
    for part in arg.split(','):
        part = part.strip()
        if not part:
            continue
        if 'x' not in part:
            raise ValueError(f"Invalid size '{part}', expected HxW")
        h_str, w_str = part.split('x', 1)
        sizes.append((int(h_str), int(w_str)))
    return sizes


def parse_int_list(arg: str) -> List[int]:
    return [int(x.strip()) for x in arg.split(',') if x.strip()]


def run_cmd(cmd: List[str], env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    return proc.returncode, proc.stdout, proc.stderr


def ensure_built() -> None:
    print("[build] Running make ...")
    rc, out, err = run_cmd(["make"])
    if out:
        print(out.strip())
    if err:
        print(err.strip())
    if rc != 0:
        raise SystemExit(f"make failed with code {rc}")


def parse_time(regex: re.Pattern, text: str) -> Optional[float]:
    m = regex.search(text)
    return float(m.group(1)) if m else None


def run_cpu(h: int, w: int, steps: int, threads: int) -> Dict[str, Any]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    rc, out, err = run_cmd([str(CPU_BIN), str(h), str(w), str(steps)], env=env)
    total_ms = parse_time(RE_CPU_TOTAL, out)
    avg_ms = parse_time(RE_CPU_AVG, out)
    return {
        "method": "cpu",
        "h": h,
        "w": w,
        "steps": steps,
        "workers": threads,
        "metric_ms": total_ms,
        "avg_ms_per_step": avg_ms,
        "rc": rc,
        "stdout": out,
        "stderr": err,
    }


def run_gpu(h: int, w: int, steps: int) -> Dict[str, Any]:
    rc, out, err = run_cmd([str(GPU_BIN), str(h), str(w), str(steps)])
    comp_ms = parse_time(RE_GPU_COMP, out)
    avg_ms = parse_time(RE_GPU_AVG, out)
    return {
        "method": "gpu",
        "h": h,
        "w": w,
        "steps": steps,
        "workers": "gpu",
        "metric_ms": comp_ms,
        "avg_ms_per_step": avg_ms,
        "rc": rc,
        "stdout": out,
        "stderr": err,
    }


def run_mpi(h: int, w: int, steps: int, procs: int) -> Dict[str, Any]:
    rc, out, err = run_cmd([
        "mpirun", "-np", str(procs), str(MPI_BIN),
        str(h), str(w), str(steps), str(procs)
    ])
    max_ms = parse_time(RE_MPI_MAX, out)
    avg_ms = parse_time(RE_MPI_AVG, out)
    return {
        "method": "mpi",
        "h": h,
        "w": w,
        "steps": steps,
        "workers": procs,
        "metric_ms": max_ms,
        "avg_ms_per_step": avg_ms,
        "rc": rc,
        "stdout": out,
        "stderr": err,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write two tables to CSV: 1) averages, 2) all runs with repetition number."""
    fieldnames_avg = ["method", "h", "w", "steps", "workers", "avg_metric_ms", "avg_ms_per_step", "ok"]
    fieldnames_all = ["run", "method", "h", "w", "steps", "workers", "metric_ms", "avg_ms_per_step", "ok", "notes"]
    
    # Group by (method, h, w, steps, workers) and compute averages
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = (r["method"], r["h"], r["w"], r["steps"], r["workers"])
        groups[key].append(r)
    
    avg_rows = []
    for (method, h, w, steps, workers), runs in groups.items():
        ok_count = sum(1 for r in runs if r["rc"] == 0)
        metrics = [r.get("metric_ms") for r in runs if r.get("metric_ms") is not None]
        avgs = [r.get("avg_ms_per_step") for r in runs if r.get("avg_ms_per_step") is not None]
        
        avg_metric = sum(metrics) / len(metrics) if metrics else None
        avg_avg = sum(avgs) / len(avgs) if avgs else None
        
        avg_rows.append({
            "method": method,
            "h": h,
            "w": w,
            "steps": steps,
            "workers": workers,
            "avg_metric_ms": avg_metric,
            "avg_ms_per_step": avg_avg,
            "ok": ok_count == len(runs),
        })
    
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_avg)
        writer.writeheader()
        for r in avg_rows:
            writer.writerow({
                "method": r["method"],
                "h": r["h"],
                "w": r["w"],
                "steps": r["steps"],
                "workers": r["workers"],
                "avg_metric_ms": f"{r.get('avg_metric_ms'):.3f}" if r.get("avg_metric_ms") is not None else "",
                "avg_ms_per_step": f"{r.get('avg_ms_per_step'):.4f}" if r.get("avg_ms_per_step") is not None else "",
                "ok": r["ok"],
            })
        
        # Blank line separator
        f.write("\n\n")
        
        # Write all runs
        f.write("# Detailed results (all repetitions)\n")
        writer2 = csv.DictWriter(f, fieldnames=fieldnames_all)
        writer2.writeheader()
        for i, r in enumerate(rows, 1):
            writer2.writerow({
                "run": i,
                "method": r["method"],
                "h": r["h"],
                "w": r["w"],
                "steps": r["steps"],
                "workers": r["workers"],
                "metric_ms": f"{r.get('metric_ms'):.3f}" if r.get("metric_ms") is not None else "",
                "avg_ms_per_step": f"{r.get('avg_ms_per_step'):.4f}" if r.get("avg_ms_per_step") is not None else "",
                "ok": r["rc"] == 0,
                "notes": ("" if r["rc"] == 0 else f"rc={r['rc']}") + ("; missing metric" if r.get("metric_ms") is None else ""),
            })


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print("\n=== Benchmark summary ===")
    for r in rows:
        ok = "OK" if r["rc"] == 0 else "FAIL"
        metric = f"{r.get('metric_ms'):.3f} ms" if r.get("metric_ms") is not None else "n/a"
        avg = f"{r.get('avg_ms_per_step'):.4f} ms/step" if r.get("avg_ms_per_step") is not None else "n/a"
        print(f"[{ok}] {r['method']:3s} h={r['h']:<4} w={r['w']:<4} steps={r['steps']:<4} workers={r['workers']:<5} metric={metric:>12} avg={avg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU/GPU/MPI Game of Life")
    parser.add_argument("--sizes", default="200x200,400x400,800x800", help="Comma-separated sizes HxW (e.g., 200x200,400x400)")
    parser.add_argument("--steps", type=int, default=50, help="Number of steps per run")
    parser.add_argument("--cpu-threads", default="1,6", help="Comma-separated OpenMP threads to test")
    parser.add_argument("--mpi-procs", default="2,6", help="Comma-separated MPI process counts")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU runs")
    parser.add_argument("--skip-build", action="store_true", help="Do not run make before benchmarks")
    parser.add_argument("--out", default="bench-results.csv", help="Output CSV file path")
    parser.add_argument("--repeat", type=int, default=5, help="Repeat each configuration this many times")
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    cpu_threads = parse_int_list(args.cpu_threads)
    mpi_procs = parse_int_list(args.mpi_procs)

    if not args.skip_build:
        ensure_built()

    rows: List[Dict[str, Any]] = []

    for _ in range(args.repeat):
        for h, w in sizes:
            for t in cpu_threads:
                rows.append(run_cpu(h, w, args.steps, t))
            if not args.skip_gpu:
                rows.append(run_gpu(h, w, args.steps))
            for p in mpi_procs:
                rows.append(run_mpi(h, w, args.steps, p))

    out_path = ROOT / args.out
    write_csv(out_path, rows)
    print_summary(rows)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
