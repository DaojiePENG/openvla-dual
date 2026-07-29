#!/usr/bin/env python3
"""Globally schedule multi-checkpoint, multi-seed LIBERO frame-delay evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


FINAL_EPISODES_RE = re.compile(r"Total episodes:\s+(\d+)")
FINAL_SUCCESSES_RE = re.compile(r"Total successes:\s+(\d+)")


@dataclass(frozen=True)
class Experiment:
    name: str
    suite: str
    checkpoint: Path
    step: int


@dataclass(frozen=True)
class Job:
    experiment: Experiment
    seed: int
    delay: int
    job_id: str
    label: str
    log_path: Path
    error_path: Path


@dataclass
class RunningJob:
    job: Job
    gpu: int
    process: subprocess.Popen
    stdout_handle: object
    stderr_handle: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        action="append",
        nargs=3,
        metavar=("NAME", "SUITE", "CHECKPOINT"),
        required=True,
        help="Repeat for each checkpoint: a short name, LIBERO suite, and checkpoint path.",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tasks-per-gpu", type=int, default=4)
    parser.add_argument("--seeds", default="7,17,27,37")
    parser.add_argument("--delays", default="0,5,10,15,20,25,30,40")
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--stagger-seconds", type=int, default=30)
    parser.add_argument("--poll-seconds", type=int, default=2)
    parser.add_argument("--wait-log-seconds", type=int, default=300)
    parser.add_argument("--min-gpu-free-mib", type=int, default=18000)
    parser.add_argument("--min-ram-available-kib", type=int, default=314572800)
    parser.add_argument("--min-disk-free-kib", type=int, default=10737418240)
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=Path("experiments/robot/libero/run_libero_eval.py"),
    )
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.search(r"--(\d+)_chkpt$", path.name)
    if not match:
        raise ValueError(f"cannot parse checkpoint step from {path}")
    return int(match.group(1))


def validate_checkpoint(path: Path) -> None:
    action_heads = list(path.glob("action_head--*_checkpoint.pt"))
    proprio = list(path.glob("proprio_projector--*_checkpoint.pt"))
    shards = list(path.glob("model-*-of-*.safetensors"))
    required = [path / "model.safetensors.index.json", path / "dataset_statistics.json"]
    if not path.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {path}")
    if len(action_heads) != 1 or len(proprio) != 1 or len(shards) != 4:
        raise ValueError(f"checkpoint is not a complete merged VisionActionHead checkpoint: {path}")
    if any(not item.is_file() or item.stat().st_size == 0 for item in required + shards):
        raise ValueError(f"checkpoint contains a missing or empty required file: {path}")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def mem_available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return 0


def disk_available_kib(path: Path) -> int:
    return shutil.disk_usage(path).free // 1024


def gpu_free_mib(gpu: int) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--id",
            str(gpu),
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(output.splitlines()[0].strip())


def final_counts(log_path: Path) -> tuple[int | None, int | None]:
    if not log_path.exists():
        return None, None
    text = log_path.read_text(errors="replace")
    if "Final results:" not in text:
        return None, None
    episodes_match = FINAL_EPISODES_RE.search(text)
    successes_match = FINAL_SUCCESSES_RE.search(text)
    if not episodes_match or not successes_match:
        return None, None
    return int(episodes_match.group(1)), int(successes_match.group(1))


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    args.run_root = args.run_root.resolve()
    args.eval_script = args.eval_script.resolve()
    gpus = [int(item) for item in args.gpus.split(",")]
    seeds = [int(item) for item in args.seeds.split(",")]
    delays = [int(item) for item in args.delays.split(",")]

    if not gpus or args.tasks_per_gpu < 1:
        raise ValueError("at least one GPU and one task per GPU are required")
    if len(set(gpus)) != len(gpus) or len(set(seeds)) != len(seeds) or len(set(delays)) != len(delays):
        raise ValueError("GPU, seed, and delay lists must not contain duplicates")
    if any(delay < 0 for delay in delays):
        raise ValueError("delays must be non-negative")
    if args.run_root.exists():
        unexpected_entries = [
            item for item in args.run_root.iterdir() if item.name != "sweep_launcher.log"
        ]
        if unexpected_entries:
            raise ValueError(f"run root already exists and is not empty: {args.run_root}")
    args.run_root.mkdir(parents=True, exist_ok=True)

    experiments: list[Experiment] = []
    for name, suite, checkpoint_raw in args.experiment:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"unsafe experiment name: {name}")
        checkpoint = Path(checkpoint_raw).resolve()
        validate_checkpoint(checkpoint)
        experiments.append(Experiment(name, suite, checkpoint, checkpoint_step(checkpoint)))

    jobs: list[Job] = []
    # Interleave suites and seeds at every delay so early GPU slots cover all checkpoints.
    for delay in delays:
        for seed in seeds:
            for experiment in experiments:
                label = f"baseline_d0_seed{seed}" if delay == 0 else f"frame_delay_d{delay}_seed{seed}"
                directory = args.run_root / experiment.name / f"seed_{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                job_id = f"{experiment.name}_s{seed}_d{delay}"
                jobs.append(
                    Job(
                        experiment=experiment,
                        seed=seed,
                        delay=delay,
                        job_id=job_id,
                        label=label,
                        log_path=directory / f"{label}.log",
                        error_path=directory / f"{label}.err",
                    )
                )

    manifest = {
        "created_at": now(),
        "run_root": str(args.run_root),
        "experiments": [
            {
                "name": item.name,
                "suite": item.suite,
                "checkpoint": str(item.checkpoint),
                "checkpoint_step": item.step,
            }
            for item in experiments
        ],
        "seeds": seeds,
        "delays": delays,
        "num_trials_per_task": args.num_trials,
        "total_tasks_per_suite": 10,
        "expected_episodes_per_job": args.num_trials * 10,
        "gpus": gpus,
        "tasks_per_gpu": args.tasks_per_gpu,
        "save_rollouts": False,
        "total_jobs": len(jobs),
    }
    atomic_json(args.run_root / "manifest.json", manifest)

    fieldnames = [
        "job_id",
        "experiment",
        "suite",
        "checkpoint_step",
        "checkpoint",
        "seed",
        "delay_steps",
        "status",
        "gpu",
        "pid",
        "started_at",
        "ended_at",
        "exit_code",
        "completed_trials",
        "completed_successes",
        "success_rate",
        "log_file",
        "error_file",
    ]
    state: dict[str, dict[str, object]] = {}
    for job in jobs:
        state[job.job_id] = {
            "job_id": job.job_id,
            "experiment": job.experiment.name,
            "suite": job.experiment.suite,
            "checkpoint_step": job.experiment.step,
            "checkpoint": job.experiment.checkpoint.name,
            "seed": job.seed,
            "delay_steps": job.delay,
            "status": "queued",
            "gpu": "",
            "pid": "",
            "started_at": "",
            "ended_at": "",
            "exit_code": "",
            "completed_trials": "",
            "completed_successes": "",
            "success_rate": "",
            "log_file": str(job.log_path),
            "error_file": str(job.error_path),
        }

    jobs_csv = args.run_root / "jobs.csv"

    def write_state() -> None:
        atomic_csv(jobs_csv, [state[job.job_id] for job in jobs], fieldnames)

    write_state()

    common_args = [
        "--use_l1_regression",
        "True",
        "--use_diffusion",
        "False",
        "--use_film",
        "False",
        "--num_images_in_input",
        "2",
        "--use_proprio",
        "True",
        "--lora_rank",
        "16",
        "--center_crop",
        "True",
        "--num_trials_per_task",
        str(args.num_trials),
        "--num_open_loop_steps",
        "8",
        "--use_vision_action_head",
        "True",
        "--action_head_vision_encoder",
        "siglip-base",
        "--freeze_action_head_vision",
        "True",
        "--action_head_num_views",
        "2",
        "--save_rollouts",
        "false",
    ]

    env_base = os.environ.copy()
    env_base.update(
        {
            "HF_HOME": "/home/sheng/workspace/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )

    queue = list(jobs)
    running: dict[int, RunningJob] = {}
    slots = {gpu: 0 for gpu in gpus}
    next_gpu_index = 0
    last_launch = 0.0
    last_wait_log = 0.0
    stopping = False

    def stop_all(signum, _frame) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        print(f"[{now()}] received signal {signum}; terminating {len(running)} evaluator(s)", flush=True)
        for item in running.values():
            item.process.terminate()

    signal.signal(signal.SIGTERM, stop_all)
    signal.signal(signal.SIGINT, stop_all)

    def reap() -> None:
        finished: list[int] = []
        for pid, item in running.items():
            code = item.process.poll()
            if code is None:
                continue
            item.stdout_handle.close()
            item.stderr_handle.close()
            slots[item.gpu] -= 1
            episodes, successes = final_counts(item.job.log_path)
            valid = code == 0 and episodes == args.num_trials * 10 and successes is not None
            row = state[item.job.job_id]
            row["status"] = "completed" if valid else ("stopped" if stopping else "failed")
            row["ended_at"] = now()
            row["exit_code"] = code
            row["completed_trials"] = "" if episodes is None else episodes
            row["completed_successes"] = "" if successes is None else successes
            row["success_rate"] = "" if episodes in (None, 0) or successes is None else f"{successes / episodes:.6f}"
            finished.append(pid)
            print(
                f"[{now()}] {row['status'].upper()} {item.job.job_id} gpu={item.gpu} "
                f"episodes={episodes} successes={successes} exit={code}",
                flush=True,
            )
        for pid in finished:
            del running[pid]
        if finished:
            write_state()

    def global_resources_ok() -> bool:
        return (
            mem_available_kib() >= args.min_ram_available_kib
            and disk_available_kib(repo_root) >= args.min_disk_free_kib
        )

    def select_gpu() -> int | None:
        nonlocal next_gpu_index
        if not global_resources_ok():
            return None
        for offset in range(len(gpus)):
            index = (next_gpu_index + offset) % len(gpus)
            gpu = gpus[index]
            if slots[gpu] >= args.tasks_per_gpu:
                continue
            try:
                enough_memory = gpu_free_mib(gpu) >= args.min_gpu_free_mib
            except (OSError, ValueError, subprocess.SubprocessError):
                enough_memory = False
            if not enough_memory:
                continue
            next_gpu_index = (index + 1) % len(gpus)
            return gpu
        return None

    def launch(job: Job, gpu: int) -> None:
        if job.log_path.exists() or job.error_path.exists():
            raise RuntimeError(f"refusing to overwrite logs for {job.job_id}")
        delay_args = (
            ["--use_frame_delay_eval", "false"]
            if job.delay == 0
            else ["--use_frame_delay_eval", "true", "--max_delay_steps_eval", str(job.delay)]
        )
        command = [
            sys.executable,
            str(args.eval_script),
            "--pretrained_checkpoint",
            str(job.experiment.checkpoint),
            *common_args,
            "--task_suite_name",
            job.experiment.suite,
            "--seed",
            str(job.seed),
            *delay_args,
            "--run_id_note",
            f"multiseed_{job.experiment.name}_{job.label}",
        ]
        stdout_handle = job.log_path.open("w")
        stderr_handle = job.error_path.open("w")
        environment = env_base.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        slots[gpu] += 1
        running[process.pid] = RunningJob(job, gpu, process, stdout_handle, stderr_handle)
        row = state[job.job_id]
        row.update(
            {
                "status": "running",
                "gpu": gpu,
                "pid": process.pid,
                "started_at": now(),
            }
        )
        write_state()
        print(
            f"[{now()}] LAUNCH {job.job_id} gpu={gpu} pid={process.pid} "
            f"running={len(running)} queued={len(queue)}",
            flush=True,
        )

    print(
        f"[{now()}] starting sweep: jobs={len(jobs)} gpus={gpus} slots/gpu={args.tasks_per_gpu} "
        f"seeds={seeds} delays={delays}",
        flush=True,
    )

    while queue or running:
        reap()
        if stopping:
            if not running:
                break
            time.sleep(args.poll_seconds)
            continue

        current = time.monotonic()
        if queue and current - last_launch >= args.stagger_seconds:
            gpu = select_gpu()
            if gpu is not None:
                job = queue.pop(0)
                launch(job, gpu)
                last_launch = current
            elif current - last_wait_log >= args.wait_log_seconds:
                print(
                    f"[{now()}] waiting: running={len(running)} queued={len(queue)} "
                    f"ram_available_kib={mem_available_kib()} disk_available_kib={disk_available_kib(repo_root)}",
                    flush=True,
                )
                last_wait_log = current
        time.sleep(args.poll_seconds)

    reap()
    completed = sum(row["status"] == "completed" for row in state.values())
    failed = sum(row["status"] == "failed" for row in state.values())
    print(f"[{now()}] sweep finished: completed={completed}/{len(jobs)} failed={failed}", flush=True)
    return 0 if completed == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
