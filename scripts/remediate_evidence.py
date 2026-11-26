"""Run the fresh remediation evidence set with recorded commands and RAM.

All evidence goes under one new output root; nothing is overwritten, deleted
or merged. No full queue and no 500,000-step training is started here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl_a2.config import read_json, write_json  # noqa: E402
from rl_a2.provenance import source_digest  # noqa: E402


def _memory() -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    return {
        "total_mib": round(vm.total / 1024**2, 1),
        "available_mib": round(vm.available / 1024**2, 1),
        "free_mib": round(vm.free / 1024**2, 1),
        "used_mib": round(vm.used / 1024**2, 1),
        "percent_used": vm.percent,
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="artifacts/remediation/evidence",
        help="Fresh evidence root; an existing non-empty root is refused",
    )
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"Evidence root is not empty; choose a fresh root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    smoke_root = root / "smoke_reruns"
    preflight_path = root / "preflight" / "preflight.json"

    def command(step: str, argv: list[str]) -> dict[str, Any]:
        started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        started_epoch = time.time()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(argv, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
        finished = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        log = logs / f"{step}.log"
        log.write_text(
            f"COMMAND: {subprocess.list2cmdline(argv)}\n"
            f"STARTED: {started}\nFINISHED: {finished}\n"
            f"RETURNCODE: {result.returncode}\n\n"
            f"--- stdout ---\n{result.stdout}\n\n--- stderr ---\n{result.stderr}\n",
            encoding="utf-8",
        )
        record = {
            "step": step,
            "command": argv,
            "command_text": subprocess.list2cmdline(argv),
            "cwd": str(PROJECT_ROOT),
            "started_at": started,
            "finished_at": finished,
            "started_epoch": started_epoch,
            "wall_seconds": round(time.time() - started_epoch, 3),
            "returncode": result.returncode,
            "log": str(log),
        }
        print(f"[{step}] exit={result.returncode} {record['wall_seconds']}s", flush=True)
        return record

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    started_epoch = time.time()
    commands: list[dict[str, Any]] = []
    commands.append(
        command("preflight", [python, "scripts/preflight.py", "--output-root", str(root / "preflight")])
    )
    commands.append(
        command(
            "wrappers",
            [python, "scripts/verify_wrappers.py", "--output", str(root / "verification" / "wrapper_tests.json")],
        )
    )
    ablation_reports: dict[str, str] = {}
    for seed in (0, 1, 2):
        report = root / "ablation_check" / f"ablation_diff_full_seed{seed}.json"
        commands.append(
            command(
                f"ablations_seed{seed}",
                [
                    python,
                    "scripts/verify_ablations.py",
                    "--profile",
                    "full",
                    "--seed",
                    str(seed),
                    "--output-root",
                    str(root / "ablation_check"),
                    "--report",
                    str(report),
                ],
            )
        )
        ablation_reports[str(seed)] = str(report)

    smoke_configs = [
        ("dqn_baseline", "smoke_dqn"),
        ("ppo_baseline", "smoke_ppo"),
    ]
    for condition, step in smoke_configs:
        commands.append(
            command(
                step,
                [
                    python,
                    "-m",
                    "rl_a2.train",
                    "--condition",
                    condition,
                    "--profile",
                    "smoke",
                    "--seed",
                    "0",
                    "--output-root",
                    str(smoke_root),
                ],
            )
        )
    for algo, step in (("dqn", "reload_dqn"), ("ppo", "reload_ppo")):
        run_dir = smoke_root / f"{algo}_baseline_seed0_smoke"
        commands.append(
            command(
                step,
                [
                    python,
                    "-m",
                    "rl_a2.evaluate",
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint",
                    "final",
                    "--episodes",
                    "2",
                    "--output-dir",
                    str(root / "reloads" / f"{algo}_final"),
                    "--no-video",
                ],
            )
        )
    commands.append(
        command(
            "summarize_smoke",
            [
                python,
                "scripts/summarize_smoke.py",
                "--output-root",
                str(smoke_root),
                "--preflight",
                str(preflight_path),
            ],
        )
    )

    check_results: dict[str, bool] = {}
    check_results["all_commands_succeeded"] = all(row["returncode"] == 0 for row in commands)
    preflight = read_json(preflight_path)
    replay = preflight["replay_estimate_full_dqn"]
    check_results["preflight_ok"] = bool(preflight["ok"])
    check_results["replay_formula_corrected"] = (
        replay["bytes_per_transition"] == 56468
        and replay["total_bytes"] == 282340000
        and "done_float32" in replay["formula"]
        and "timeout_float32" in replay["formula"]
    )
    wrappers = read_json(root / "verification" / "wrapper_tests.json")
    check_results["wrappers_preserved"] = bool(wrappers["ok"])

    ablation_details: dict[str, Any] = {}
    for seed, report_path in ablation_reports.items():
        report = read_json(Path(report_path))
        check_results[f"ablations_seed{seed}_ok"] = bool(report["ok"])
        details = {}
        for name, entry in report["conditions"].items():
            diffs = [
                diff
                for diff in entry["semantic_diffs"]
                if diff["path"] not in {"condition", "metadata"}
            ]
            details[name] = {
                "expected": entry["expected"],
                "semantic_diff_paths": [diff["path"] for diff in diffs],
                "ok": entry["ok"],
            }
        expected_single = {
            "dqn_target_1000": "model.target_update_interval",
            "dqn_target_10000": "model.target_update_interval",
            "dqn_eps_010": "model.exploration_fraction",
            "dqn_eps_040": "model.exploration_fraction",
            "ppo_clip_010": "model.clip_range",
            "ppo_clip_040": "model.clip_range",
            "ppo_lambda_090": "model.gae_lambda",
            "ppo_lambda_099": "model.gae_lambda",
        }
        check_results[f"ablations_seed{seed}_eight_one_factor_diffs"] = all(
            details[name]["semantic_diff_paths"] == [path]
            for name, path in expected_single.items()
        ) and all(
            details[name]["semantic_diff_paths"] == []
            for name in ("dqn_baseline", "ppo_baseline")
        )
        ablation_details[seed] = details

    smoke_details: dict[str, Any] = {}
    for algo in ("dqn", "ppo"):
        run_dir = smoke_root / f"{algo}_baseline_seed0_smoke"
        manifest = read_json(run_dir / "run_manifest.json")
        summary = read_json(run_dir / "metrics" / "training_summary.json")
        checkpoint_rows = _csv_rows(run_dir / "metrics" / "checkpoint_manifest.csv")
        periodic = [row for row in checkpoint_rows if row["checkpoint"] != "final"]
        final_rows = [row for row in checkpoint_rows if row["checkpoint"] == "final"]
        update_rows = _csv_rows(run_dir / "metrics" / "updates.csv")
        check_results[f"{algo}_completed"] = manifest["status"] == "completed"
        check_results[f"{algo}_artifacts_ok"] = all(manifest["artifact_checks"].values())
        check_results[f"{algo}_real_optimizer_updates"] = (
            manifest["total_optimizer_updates"] > 0 and len(update_rows) > 0
        )
        check_results[f"{algo}_checkpoint_counts_consistent"] = (
            summary["periodic_checkpoints"] == len(periodic)
            and summary["checkpoint_events"] == len(checkpoint_rows)
            and summary["final_checkpoint_count"] == len(final_rows)
        )
        check_results[f"{algo}_memory_and_throughput_recorded"] = bool(
            manifest["memory"]["process_peak_sampled_rss_mib"]
            and manifest["timing"]["agent_steps_per_sec_training_only"]
        )
        check_results[f"{algo}_source_digest_recorded"] = bool(
            manifest["source_digest"]["digest"] and manifest["started_epoch"] and manifest["finished_epoch"]
        )
        smoke_details[algo] = {
            "run_dir": str(run_dir),
            "actual_timesteps": manifest["actual_timesteps"],
            "total_optimizer_updates": manifest["total_optimizer_updates"],
            "periodic_checkpoints": len(periodic),
            "final_checkpoints": len(final_rows),
            "peak_sampled_rss_mib": manifest["memory"]["process_peak_sampled_rss_mib"],
            "available_mib_at_start": manifest["memory"]["host"]["available_mib_at_start"],
            "training_only_steps_per_s": manifest["timing"]["agent_steps_per_sec_training_only"],
            "source_digest": manifest["source_digest"]["digest"],
            "started_at": manifest["started_at"],
            "finished_at": manifest["finished_at"],
            "clip": manifest["final_clips"][0] if manifest["final_clips"] else None,
        }

    reload_details: dict[str, Any] = {}
    for algo in ("dqn", "ppo"):
        manifest = read_json(smoke_root / f"{algo}_baseline_seed0_smoke" / "run_manifest.json")
        reload_manifest = read_json(root / "reloads" / f"{algo}_final" / "eval_manifest.json")
        expected = manifest["final_eval"]
        got = reload_manifest["result"]
        episode_rows = _csv_rows(root / "reloads" / f"{algo}_final" / "metrics" / "eval_episodes.csv")
        check_results[f"{algo}_reload_reproduces_final_eval"] = (
            len(episode_rows) == int(expected["episodes"])
            and abs(float(got["mean_return_raw"]) - float(expected["mean_return_raw"])) < 1e-9
            and abs(float(got["std_return_raw"]) - float(expected["std_return_raw"])) < 1e-9
        )
        reload_details[algo] = {
            "mean_return_raw": got["mean_return_raw"],
            "episodes": len(episode_rows),
            "output_dir": str(root / "reloads" / f"{algo}_final"),
        }

    dqn = smoke_details.get("dqn", {})
    process_footprint = None
    if dqn:
        replay_mib = read_json(
            smoke_root / "dqn_baseline_seed0_smoke" / "run_manifest.json"
        )["memory"]["replay_estimate"]["total_mib"]
        process_footprint = {
            "dqn_smoke_peak_rss_mib": dqn["peak_sampled_rss_mib"],
            "dqn_smoke_replay_mib": replay_mib,
            "non_replay_learn_eval_footprint_mib": round(
                dqn["peak_sampled_rss_mib"] - replay_mib, 1
            ),
            "assumption": (
                "Non-replay footprint is smoke peak RSS minus smoke replay array bytes; "
                "it is used with 25% slack to gate full DQN launches."
            ),
        }
    summary = read_json(smoke_root / "smoke_summary.json") if (smoke_root / "smoke_summary.json").is_file() else {}
    check_results["summary_replay_and_eta_wording"] = bool(
        summary.get("eta_reconstruction")
        and summary.get("checkpoint_count_definition")
        and summary.get("full_wall_time_extrapolation", {})
        .get("by_algorithm", {})
        .get("dqn", {})
        .get("representative_full_profile_pilot_measured")
        is False
    )

    report = {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "started_epoch": started_epoch,
        "wall_seconds": round(time.time() - started_epoch, 3),
        "output_root": str(root),
        "full_queue_started": False,
        "full_training_steps_executed": 0,
        "command_for_this_script": subprocess.list2cmdline([sys.executable, "-m", *sys.argv]),
        "commands": commands,
        "source_digest": source_digest(),
        "host_memory_before": _memory(),
        "host_memory_after": _memory(),
        "process_footprint": process_footprint,
        "ablation_details": ablation_details,
        "smoke_details": smoke_details,
        "reload_details": reload_details,
        "preflight_replay_estimate": replay,
        "checks": check_results,
        "ok": all(check_results.values()),
    }
    path = root / "remediation_report.json"
    write_json(path, report)
    print(json.dumps({"checks": check_results, "ok": report["ok"], "report": str(path)}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
