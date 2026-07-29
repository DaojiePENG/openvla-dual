#!/usr/bin/env python3
"""Continuously summarize a multi-seed frame-delay sweep into CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import tempfile
import time
from datetime import datetime
from pathlib import Path


EPISODES_RE = re.compile(r"Total episodes:\s+(\d+)")
SUCCESSES_RE = re.compile(r"Total successes:\s+(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def atomic_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_jobs(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["job_id"]: row for row in csv.DictReader(handle)}


def final_counts(path: Path) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    text = path.read_text(errors="replace")
    if "Final results:" not in text:
        return None, None
    episodes = EPISODES_RE.search(text)
    successes = SUCCESSES_RE.search(text)
    if not episodes or not successes:
        return None, None
    return int(episodes.group(1)), int(successes.group(1))


def fmt(value: float | None, digits: int) -> str:
    return "" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    args.run_root = args.run_root.resolve()
    manifest = json.loads((args.run_root / "manifest.json").read_text())
    experiments = manifest["experiments"]
    seeds = manifest["seeds"]
    delays = manifest["delays"]
    expected_episodes = manifest["expected_episodes_per_job"]
    expected_rows = len(experiments) * len(seeds) * len(delays)
    previous_signature = None

    per_seed_fields = [
        "updated_at",
        "experiment",
        "suite",
        "checkpoint_step",
        "checkpoint",
        "seed",
        "delay_steps",
        "status",
        "completed_trials",
        "completed_successes",
        "success_rate",
        "success_rate_pct",
        "log_file",
        "error_file",
    ]
    summary_fields = [
        "updated_at",
        "experiment",
        "suite",
        "checkpoint_step",
        "checkpoint",
        "delay_steps",
        "completed_seeds",
        "total_seeds",
        "mean_success_rate",
        "mean_success_rate_pct",
        "std_sample_success_rate",
        "std_sample_success_rate_pct",
        "std_population_success_rate",
        "std_population_success_rate_pct",
        "min_success_rate_pct",
        "max_success_rate_pct",
    ]

    while True:
        updated_at = datetime.now().astimezone().isoformat(timespec="seconds")
        job_state = load_jobs(args.run_root / "jobs.csv")
        per_seed_rows: list[dict[str, object]] = []
        rates: dict[tuple[str, int], list[float]] = {}

        for experiment in experiments:
            for seed in seeds:
                for delay in delays:
                    label = f"baseline_d0_seed{seed}" if delay == 0 else f"frame_delay_d{delay}_seed{seed}"
                    directory = args.run_root / experiment["name"] / f"seed_{seed}"
                    log_path = directory / f"{label}.log"
                    error_path = directory / f"{label}.err"
                    job_id = f"{experiment['name']}_s{seed}_d{delay}"
                    episodes, successes = final_counts(log_path)
                    scheduler_status = job_state.get(job_id, {}).get("status", "queued")
                    if episodes == expected_episodes and successes is not None:
                        status = "completed"
                    elif episodes is not None:
                        status = "invalid"
                    elif scheduler_status in {"failed", "stopped"}:
                        status = scheduler_status
                    elif scheduler_status == "running":
                        status = "running"
                    else:
                        status = "queued"
                    rate = successes / episodes if status == "completed" and episodes else None
                    if rate is not None:
                        rates.setdefault((experiment["name"], delay), []).append(rate)
                    per_seed_rows.append(
                        {
                            "updated_at": updated_at,
                            "experiment": experiment["name"],
                            "suite": experiment["suite"],
                            "checkpoint_step": experiment["checkpoint_step"],
                            "checkpoint": Path(experiment["checkpoint"]).name,
                            "seed": seed,
                            "delay_steps": delay,
                            "status": status,
                            "completed_trials": "" if episodes is None else episodes,
                            "completed_successes": "" if successes is None else successes,
                            "success_rate": fmt(rate, 6),
                            "success_rate_pct": fmt(None if rate is None else rate * 100, 2),
                            "log_file": str(log_path),
                            "error_file": str(error_path),
                        }
                    )

        summary_rows: list[dict[str, object]] = []
        for experiment in experiments:
            for delay in delays:
                values = rates.get((experiment["name"], delay), [])
                mean = statistics.mean(values) if values else None
                sample_std = statistics.stdev(values) if len(values) >= 2 else None
                population_std = statistics.pstdev(values) if values else None
                summary_rows.append(
                    {
                        "updated_at": updated_at,
                        "experiment": experiment["name"],
                        "suite": experiment["suite"],
                        "checkpoint_step": experiment["checkpoint_step"],
                        "checkpoint": Path(experiment["checkpoint"]).name,
                        "delay_steps": delay,
                        "completed_seeds": len(values),
                        "total_seeds": len(seeds),
                        "mean_success_rate": fmt(mean, 6),
                        "mean_success_rate_pct": fmt(None if mean is None else mean * 100, 2),
                        "std_sample_success_rate": fmt(sample_std, 6),
                        "std_sample_success_rate_pct": fmt(None if sample_std is None else sample_std * 100, 2),
                        "std_population_success_rate": fmt(population_std, 6),
                        "std_population_success_rate_pct": fmt(
                            None if population_std is None else population_std * 100, 2
                        ),
                        "min_success_rate_pct": fmt(None if not values else min(values) * 100, 2),
                        "max_success_rate_pct": fmt(None if not values else max(values) * 100, 2),
                    }
                )

        signature = tuple(
            (row["experiment"], row["seed"], row["delay_steps"], row["status"], row["success_rate"])
            for row in per_seed_rows
            if row["status"] in {"completed", "invalid", "failed", "stopped"}
        )
        should_write = previous_signature is None or signature != previous_signature or args.once
        if should_write:
            atomic_csv(args.run_root / "per_seed_results_live.csv", per_seed_rows, per_seed_fields)
            atomic_csv(args.run_root / "summary_mean_std_live.csv", summary_rows, summary_fields)
            completed = sum(row["status"] == "completed" for row in per_seed_rows)
            print(f"[{updated_at}] wrote summaries ({completed}/{expected_rows} jobs complete)", flush=True)
        previous_signature = signature

        completed = sum(row["status"] == "completed" for row in per_seed_rows)
        terminal = sum(row["status"] in {"completed", "invalid", "failed", "stopped"} for row in per_seed_rows)
        if completed == expected_rows:
            atomic_csv(args.run_root / "per_seed_results_final.csv", per_seed_rows, per_seed_fields)
            atomic_csv(args.run_root / "summary_mean_std_final.csv", summary_rows, summary_fields)
            return
        if args.once or terminal == expected_rows:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
