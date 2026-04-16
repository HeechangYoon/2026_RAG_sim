import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic JSSP instances in unified CSV format.")
    parser.add_argument("--output-dir", type=Path, default=Path("DT") / "data" / "generated")
    parser.add_argument("--num-instances", type=int, default=10)
    parser.add_argument("--name-prefix", type=str, default="gen_jssp")
    parser.add_argument("--jobs-min", type=int, default=10)
    parser.add_argument("--jobs-max", type=int, default=20)
    parser.add_argument("--machines-min", type=int, default=5)
    parser.add_argument("--machines-max", type=int, default=10)
    parser.add_argument("--proc-time-min", type=int, default=1)
    parser.add_argument("--proc-time-max", type=int, default=99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--processing-time-distribution",
        choices=["auto", "uniform", "normal", "exponential"],
        default="auto",
        help="Sampling distribution for processing times. 'auto' keeps the current size-based heuristic.",
    )
    parser.add_argument(
        "--bottleneck-prob",
        type=float,
        default=0.3,
        help="Probability that an instance gets machine-specific slowdown factors.",
    )
    parser.add_argument("--bottleneck-factor-min", type=float, default=1.5)
    parser.add_argument("--bottleneck-factor-max", type=float, default=3.0)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional CSV manifest. Defaults to <output-dir>/manifest.csv.",
    )
    parser.add_argument(
        "--append-timestamp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append a batch timestamp to generated instance names to avoid collisions.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_instances <= 0:
        raise SystemExit("--num-instances must be positive.")
    if args.jobs_min <= 0 or args.machines_min <= 0:
        raise SystemExit("Job and machine lower bounds must be positive.")
    if args.jobs_min > args.jobs_max:
        raise SystemExit("--jobs-min must be <= --jobs-max.")
    if args.machines_min > args.machines_max:
        raise SystemExit("--machines-min must be <= --machines-max.")
    if args.proc_time_min <= 0 or args.proc_time_min > args.proc_time_max:
        raise SystemExit("Invalid processing-time range.")
    if not (0.0 <= args.bottleneck_prob <= 1.0):
        raise SystemExit("--bottleneck-prob must be between 0 and 1.")
    if args.bottleneck_factor_min < 1.0 or args.bottleneck_factor_min > args.bottleneck_factor_max:
        raise SystemExit("Invalid bottleneck factor range.")


def sample_machine_multipliers(rng: random.Random, n_machines: int, args: argparse.Namespace) -> tuple[list[float], list[int]]:
    multipliers = [1.0 for _ in range(n_machines)]
    bottleneck_ids: list[int] = []
    if rng.random() >= args.bottleneck_prob:
        return multipliers, bottleneck_ids

    max_bottlenecks = max(1, n_machines // 3)
    n_bottlenecks = rng.randint(1, max_bottlenecks)
    chosen = rng.sample(list(range(n_machines)), k=n_bottlenecks)
    for idx in chosen:
        factor = rng.uniform(args.bottleneck_factor_min, args.bottleneck_factor_max)
        multipliers[idx] = factor
        bottleneck_ids.append(idx + 1)
    return multipliers, sorted(bottleneck_ids)


def sample_processing_time_matrix(
    np_rng: np.random.Generator,
    n_jobs: int,
    n_machines: int,
    proc_min: int,
    proc_max: int,
    distribution: str = "auto",
) -> tuple[np.ndarray, str]:
    selected = distribution.lower().strip()
    if selected == "auto" and n_jobs <= 3:
        dist_name = "normal"
        mean, sigma = 20.0, 5.0
        times = np_rng.normal(mean, sigma, size=(n_jobs, n_machines))
    elif selected == "auto" and n_jobs <= 8:
        dist_name = "uniform"
        low, high = 30.0, 70.0
        times = np_rng.uniform(low, high, size=(n_jobs, n_machines))
    elif selected == "uniform":
        dist_name = "uniform"
        times = np_rng.uniform(proc_min, proc_max, size=(n_jobs, n_machines))
    elif selected == "normal":
        dist_name = "normal"
        mean = (proc_min + proc_max) / 2.0
        sigma = max((proc_max - proc_min) / 6.0, 1.0)
        times = np_rng.normal(mean, sigma, size=(n_jobs, n_machines))
    elif selected == "exponential":
        dist_name = "exponential"
        scale = max((proc_min + proc_max) / 2.0, 1.0)
        times = np_rng.exponential(scale, size=(n_jobs, n_machines))
    else:
        dist_name = "exponential"
        scale = 50.0
        times = np_rng.exponential(scale, size=(n_jobs, n_machines))

    times = np.clip(times, proc_min, proc_max)
    times = np.rint(times).astype(int)
    times = np.maximum(times, 1)
    return times, dist_name


def build_instance_rows(
    rng: random.Random,
    np_rng: np.random.Generator,
    n_jobs: int,
    n_machines: int,
    proc_min: int,
    proc_max: int,
    machine_multipliers: list[float],
    distribution: str,
) -> tuple[list[dict], str]:
    rows: list[dict] = []
    machine_ids = list(range(1, n_machines + 1))
    proc_matrix, dist_name = sample_processing_time_matrix(
        np_rng,
        n_jobs,
        n_machines,
        proc_min,
        proc_max,
        distribution=distribution,
    )

    for job_idx in range(1, n_jobs + 1):
        route = machine_ids[:]
        rng.shuffle(route)
        for op_idx, machine_num in enumerate(route, start=1):
            base_time = int(proc_matrix[job_idx - 1, op_idx - 1])
            proc_time = int(round(base_time * machine_multipliers[machine_num - 1]))
            proc_time = min(proc_max, max(proc_min, proc_time))
            rows.append(
                {
                    "arrival_time": 0.0,
                    "job": f"J{job_idx}",
                    "operation": f"O{op_idx}",
                    "process": f"P{op_idx}",
                    "machine": f"M{machine_num}",
                    "capacity": 1,
                    "processing_time": float(proc_time),
                }
            )
    return rows, dist_name


def write_unified_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_path or (output_dir / "manifest.csv")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S") if args.append_timestamp else "static"

    manifest_rows: list[dict] = []

    for idx in range(1, args.num_instances + 1):
        n_jobs = rng.randint(args.jobs_min, args.jobs_max)
        n_machines = rng.randint(args.machines_min, args.machines_max)
        machine_multipliers, bottleneck_ids = sample_machine_multipliers(rng, n_machines, args)
        rows, dist_name = build_instance_rows(
            rng=rng,
            np_rng=np_rng,
            n_jobs=n_jobs,
            n_machines=n_machines,
            proc_min=args.proc_time_min,
            proc_max=args.proc_time_max,
            machine_multipliers=machine_multipliers,
            distribution=args.processing_time_distribution,
        )

        if args.append_timestamp:
            instance_name = f"{args.name_prefix}_{batch_id}_{idx:04d}"
        else:
            instance_name = f"{args.name_prefix}_{idx:04d}"
        output_path = output_dir / f"problem_{instance_name}.csv"
        write_unified_csv(output_path, rows)

        manifest_rows.append(
            {
                "batch_id": batch_id,
                "generated_at": generated_at,
                "instance": instance_name,
                "file": str(output_path),
                "n_jobs": n_jobs,
                "n_machines": n_machines,
                "processing_time_distribution": dist_name,
                "proc_time_min": args.proc_time_min,
                "proc_time_max": args.proc_time_max,
                "bottleneck_machine_ids": "|".join(f"M{x}" for x in bottleneck_ids),
                "seed": args.seed,
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Generated {len(manifest_rows)} synthetic JSSP instances in: {output_dir}")
    print(f"Batch id: {batch_id}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
