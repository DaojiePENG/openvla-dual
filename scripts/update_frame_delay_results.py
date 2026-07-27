#!/usr/bin/env python3
"""Continuously summarize a frame-delay evaluation run into a CSV file."""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path


FATAL_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemoryError",
    "NCCL error",
    "NCCL Error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--delays", default="0,5,10,15,20,25,30,40")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total-tasks", type=int, default=10)
    parser.add_argument("--trials-per-task", type=int, default=50)
    parser.add_argument("--output-name", default="results_live.csv")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument(
        "--update-mode",
        choices=("completion", "interval"),
        default="completion",
        help="Write only when a delay reaches a terminal state, or on every polling interval.",
    )
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def evaluator_commands() -> list[str]:
    commands: list[str] = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if raw:
            commands.append(raw.replace(b"\0", b" ").decode(errors="replace"))
    return commands


def checkpoint_step(checkpoint: Path) -> str:
    match = re.search(r"--(\d+)_chkpt$", checkpoint.name)
    return match.group(1) if match else ""


def delay_label(delay: int, seed: int) -> str:
    if delay == 0:
        return f"baseline_d0_seed{seed}"
    return f"frame_delay_d{delay}_seed{seed}"


def summarize_delay(
    *,
    run_dir: Path,
    suite: str,
    checkpoint: Path,
    delay: int,
    seed: int,
    total_tasks: int,
    trials_per_task: int,
    commands: list[str],
    updated_at: str,
) -> dict[str, str | int]:
    label = delay_label(delay, seed)
    log_path = run_dir / f"{label}.log"
    error_path = run_dir / f"{label}.err"
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    error_text = error_path.read_text(errors="replace") if error_path.exists() else ""

    completed_tasks = log_text.count("Current task success rate:")
    successes = len(re.findall(r">>\s+Success:\s+True", log_text))
    failures = len(re.findall(r">>\s+Success:\s+False", log_text))
    completed_trials = successes + failures
    final_results = "Final results:" in log_text
    fatal_error = any(pattern in error_text for pattern in FATAL_PATTERNS)
    running = any(checkpoint.name in command and label in command for command in commands)

    if final_results:
        status = "completed"
    elif fatal_error:
        status = "failed"
    elif running:
        status = "running"
    elif log_text:
        status = "stopped"
    else:
        status = "queued"

    rate = successes / completed_trials if completed_trials else None
    final_rate = rate if final_results else None
    mtime = ""
    if log_path.exists():
        mtime = datetime.fromtimestamp(log_path.stat().st_mtime).astimezone().isoformat(timespec="seconds")

    return {
        "updated_at": updated_at,
        "suite": suite,
        "checkpoint_step": checkpoint_step(checkpoint),
        "checkpoint": checkpoint.name,
        "delay_steps": delay,
        "status": status,
        "completed_tasks": completed_tasks,
        "total_tasks": total_tasks,
        "completed_trials": completed_trials,
        "total_trials": total_tasks * trials_per_task,
        "completed_successes": successes,
        "interim_success_rate": "" if rate is None else f"{rate:.4f}",
        "interim_success_rate_pct": "" if rate is None else f"{rate * 100:.1f}",
        "overall_success_rate": "" if final_rate is None else f"{final_rate:.4f}",
        "overall_success_rate_pct": "" if final_rate is None else f"{final_rate * 100:.1f}",
        "log_updated_at": mtime,
        "log_file": str(log_path.resolve()),
        "error_file": str(error_path.resolve()),
    }


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def main() -> None:
    args = parse_args()
    delays = [int(item) for item in args.delays.split(",")]
    output_path = args.run_dir / args.output_name
    previous_signature: tuple[tuple[str | int, ...], ...] | None = None

    while True:
        updated_at = datetime.now().astimezone().isoformat(timespec="seconds")
        commands = evaluator_commands()
        rows = [
            summarize_delay(
                run_dir=args.run_dir,
                suite=args.suite,
                checkpoint=args.checkpoint,
                delay=delay,
                seed=args.seed,
                total_tasks=args.total_tasks,
                trials_per_task=args.trials_per_task,
                commands=commands,
                updated_at=updated_at,
            )
            for delay in delays
        ]
        # In completion mode, queued/running transitions are intentionally
        # equivalent.  They are ordinary scheduling changes and should not
        # rewrite the CSV; only terminal outcomes trigger an update.
        signature = tuple(
            (
                row["delay_steps"],
                row["status"] if row["status"] in {"completed", "failed", "stopped"} else "pending",
                row["overall_success_rate"],
            )
            for row in rows
        )
        should_write = args.once or args.update_mode == "interval" or signature != previous_signature
        if should_write:
            write_csv(output_path, rows)
        completed = sum(row["status"] == "completed" for row in rows)
        if should_write:
            print(f"[{updated_at}] wrote {output_path} ({completed}/{len(rows)} delays complete)", flush=True)
        previous_signature = signature
        if args.once or completed == len(rows):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
