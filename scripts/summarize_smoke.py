"""Summarize smoke evidence and produce a conservative full-run wall-time estimate.

The estimate is stated as an extrapolation, not a benchmark: it uses the
measured training-only throughput of the smoke runs and scales periodic
evaluation/checkpoint overhead by the difference in checkpoint and episode
counts. Full runs were deliberately not started.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from rl_a2.config import read_json, write_json  # noqa: E402


def _final_eval_seeds(run_dir: Path) -> list[int]:
    path = run_dir / "metrics" / "eval_episodes.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    final = frame[frame["is_final"].astype(str).str.lower() == "true"]
    return sorted({int(value) for value in final["seed"].tolist()})


def _smoke_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for manifest_path in sorted(root.glob("*/run_manifest.json")):
        manifest = read_json(manifest_path)
        run_dir = manifest_path.parent
        summary_path = run_dir / "metrics" / "training_summary.json"
        summary = read_json(summary_path) if summary_path.is_file() else {}
        checkpoint_manifest = run_dir / "metrics" / "checkpoint_manifest.csv"
        checkpoints = (
            pd.read_csv(checkpoint_manifest) if checkpoint_manifest.is_file() else pd.DataFrame()
        )
        periodic = (
            int((checkpoints["checkpoint"] != "final").sum()) if not checkpoints.empty else 0
        )
        final_evals = (
            pd.read_csv(run_dir / "metrics" / "eval_summary.csv")
            if (run_dir / "metrics" / "eval_summary.csv").is_file()
            else pd.DataFrame()
        )
        final_row = final_evals[final_evals["is_final"].astype(str).str.lower() == "true"]
        runs.append(
            {
                "run_id": manifest["run_id"],
                "condition": manifest["condition"],
                "algorithm": manifest["algorithm"],
                "status": manifest["status"],
                "run_dir": str(run_dir),
                "actual_timesteps": manifest["actual_timesteps"],
                "requested_timesteps": manifest["requested_timesteps"],
                "rollout_overshoot": manifest["rollout_overshoot"],
                "total_optimizer_updates": manifest["total_optimizer_updates"],
                "timing": manifest["timing"],
                "memory": manifest["memory"],
                "versions": manifest["versions"],
                "env_description": manifest["env_description"],
                "final_eval": manifest["final_eval"],
                "final_clips": manifest["final_clips"],
                "artifact_checks": manifest["artifact_checks"],
                "training_summary": summary,
                "periodic_checkpoints": periodic,
                "checkpoint_counts": {
                    "periodic_from_manifest_csv": periodic,
                    "periodic_from_training_summary": int(
                        summary.get("periodic_checkpoints", -1)
                    ),
                    "final_from_manifest_csv": (
                        int((checkpoints["checkpoint"] == "final").sum())
                        if not checkpoints.empty
                        else 0
                    ),
                    "total_from_manifest_csv": int(len(checkpoints)),
                    "total_from_training_summary": int(summary.get("checkpoint_events", -1)),
                    "consistent": (
                        periodic == int(summary.get("periodic_checkpoints", -1))
                        and int(len(checkpoints)) == int(summary.get("checkpoint_events", -1))
                    ),
                    "definition": (
                        "periodic excludes checkpoint=='final'; total counts all "
                        "checkpoint_manifest.csv rows including the final model."
                    ),
                },
                "final_eval_rows": int(len(final_row)),
                "eval_seeds": _final_eval_seeds(run_dir),
                "command": manifest["command"],
            }
        )
    return runs


def _estimate_full_seconds(
    run: dict[str, Any],
    *,
    full_timesteps: int,
    full_checkpoints: int,
    full_eval_episodes: int,
) -> dict[str, Any]:
    timing = run["timing"]
    rate = timing.get("agent_steps_per_sec_training_only") or 0.0
    if rate <= 0:
        return {"error": "no measured throughput"}
    smoke_eval_episodes = int(run["final_eval"].get("episodes", 1))
    periodic = max(1, int(run["periodic_checkpoints"]))
    periodic_eval = float(timing.get("periodic_eval_seconds_in_learn", 0.0))
    periodic_save = float(timing.get("periodic_checkpoint_save_seconds", 0.0))
    final_eval = float(timing.get("final_eval_seconds", 0.0))
    eval_per_checkpoint = periodic_eval / periodic
    eval_scale = full_eval_episodes / max(1, smoke_eval_episodes)
    save_per_checkpoint = periodic_save / periodic
    training_s = full_timesteps / rate
    eval_s = eval_per_checkpoint * eval_scale * full_checkpoints + final_eval * eval_scale
    save_s = save_per_checkpoint * full_checkpoints
    total_s = training_s + eval_s + save_s
    return {
        "measured_smoke_training_only_steps_per_s": rate,
        "measured_smoke_learn_including_periodic_steps_per_s": timing.get(
            "agent_steps_per_sec_learn_including_periodic_eval"
        ),
        "assumed_full_timesteps": full_timesteps,
        "assumed_full_periodic_checkpoints": full_checkpoints,
        "assumed_full_eval_episodes_per_checkpoint": full_eval_episodes,
        "assumed_smoke_eval_episodes": smoke_eval_episodes,
        "training_seconds": round(training_s, 1),
        "eval_seconds": round(eval_s, 1),
        "checkpoint_save_seconds": round(save_s, 1),
        "one_condition_seconds": round(total_s, 1),
        "one_condition_hours": round(total_s / 3600, 2),
        "assumptions": (
            "RECONSTRUCTED PLANNING EXTRAPOLATION, not a measured full-run estimate: "
            "linear throughput from smoke, same single-env CPU device, evaluation episode "
            "lengths similar to smoke, 10 periodic checkpoints, 5 eval episodes per "
            "checkpoint (plus final), checkpoint overhead scaled from smoke, no host "
            "contention. Full PPO uses 10 epochs per rollout versus 4 in smoke (2.5x "
            "optimizer work per agent step) at the same batch size. Full DQN uses a "
            "5,000-entry replay versus 2,000 in smoke and a 20,000-step warm-up versus "
            "200. Learned policies can change evaluation length. A short representative "
            "full-profile pilot after warm-up is required before relying on this ETA."
        ),
        "ppo_epoch_ratio_full_over_smoke": 2.5,
        "dqn_replay_capacity_ratio_full_over_smoke": 2.5,
        "representative_full_profile_pilot_measured": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="artifacts/smoke")
    parser.add_argument("--full-timesteps", type=int, default=500000)
    parser.add_argument("--full-checkpoints", type=int, default=10)
    parser.add_argument("--full-eval-episodes", type=int, default=5)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument(
        "--preflight",
        default=str(PROJECT_ROOT / "artifacts" / "preflight" / "preflight.json"),
        help="Preflight JSON with the corrected replay estimate for this source revision",
    )
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    runs = _smoke_runs(root)
    if not runs:
        raise SystemExit(f"No smoke run manifests found under {root}")
    by_algo: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_algo.setdefault(run["algorithm"], []).append(run)

    estimates = {}
    for algo, algo_runs in by_algo.items():
        estimates[algo] = _estimate_full_seconds(
            algo_runs[0],
            full_timesteps=args.full_timesteps,
            full_checkpoints=args.full_checkpoints,
            full_eval_episodes=args.full_eval_episodes,
        )
    # Ten conditions: five per algorithm. If only one algorithm was smoked,
    # do not invent a missing rate; state the missing piece explicitly.
    condition_counts = {"dqn": 5, "ppo": 5}
    total_one_seed = 0.0
    missing = []
    for algo, count in condition_counts.items():
        estimate = estimates.get(algo, {})
        if "one_condition_seconds" not in estimate:
            missing.append(algo)
            continue
        total_one_seed += float(estimate["one_condition_seconds"]) * count
    total_three_seeds = total_one_seed * args.seed_count
    preflight_path = Path(args.preflight).resolve()
    preflight = read_json(preflight_path) if preflight_path.is_file() else None
    wrapper_tests_path = PROJECT_ROOT / "artifacts" / "verification" / "wrapper_tests.json"
    wrapper_tests = read_json(wrapper_tests_path) if wrapper_tests_path.is_file() else None

    summary = {
        "schema_version": 1,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "smoke_output_root": str(root),
        "full_queue_started": False,
        "note": "Smoke schedules are deliberately distinct from full 500,000-step schedules.",
        "runs": runs,
        "throughput_scope": (
            "agent_steps_per_sec_training_only = actual timesteps / "
            "(model.learn wall time minus periodic eval and checkpoint-save time); "
            "agent_steps_per_sec_learn_including_periodic_eval includes periodic eval."
        ),
        "checkpoint_count_definition": (
            "periodic_checkpoints counts checkpoint_manifest.csv rows whose "
            "checkpoint is not 'final'; checkpoint_events/total counts all rows "
            "including the final reload. Both are recomputed in this summary and "
            "cross-checked against training_summary.json."
        ),
        "eta_reconstruction": {
            "status": "planning_only_no_measured_full_profile_pilot",
            "source": "smoke training-only throughput extrapolated linearly to 500,000 steps",
            "ppo": "full n_epochs=10 vs smoke 4 at batch_size=64: 2.5x optimizer work per agent step",
            "dqn": "full replay=5,000 vs smoke 2,000; warm-up=20,000 vs 200 agent steps",
            "evaluation": "5 full-game episodes per checkpoint in full vs 2 in smoke; length can grow with skill",
            "checkpoints": "10 periodic + 1 final per condition per seed",
            "required_before_launch": (
                "short representative full-profile pilot after warm-up, with periodic "
                "save/eval and final overhead timed separately, plus verified free RAM"
            ),
        },
        "full_wall_time_extrapolation": {
            "by_algorithm": estimates,
            "ten_conditions_one_seed_seconds": round(total_one_seed, 1),
            "ten_conditions_one_seed_hours": round(total_one_seed / 3600, 2),
            "ten_conditions_three_seeds_seconds": round(total_three_seeds, 1),
            "ten_conditions_three_seeds_hours": round(total_three_seeds / 3600, 2),
            "missing_algorithm_rates": missing,
            "assumption": (
                "Reconstructed planning estimate: five DQN and five PPO conditions; "
                "matched seed sets; 500,000 agent timesteps, 10 periodic checkpoints, "
                "5 full-game eval episodes per checkpoint plus a final evaluation. "
                "PPO full epochs=10 vs smoke 4 and DQN capacity/warm-up differ, so "
                "this is not a measured full-run result and must not be used as an ETA "
                "without a representative full-profile pilot."
            ),
        },
        "host_memory_at_summary": {
            "total_mib": None,
            "available_mib": None,
        },
        "replay_estimate_full_dqn": (
            preflight.get("replay_estimate_full_dqn") if preflight else None
        ),
        "preflight": str(preflight_path) if preflight else None,
        "wrapper_tests": str(wrapper_tests_path) if wrapper_tests else None,
        "wrapper_tests_ok": wrapper_tests.get("ok") if wrapper_tests else None,
        "ablation_diff_reports": [
            str(path)
            for path in sorted((PROJECT_ROOT / "artifacts" / "ablation_check").glob("*.json"))
        ],
        "evidence_paths": {
            "smoke_runs": str(root),
            "smoke_summary_json": str(root / "smoke_summary.json"),
            "smoke_summary_md": str(root / "SMOKE_SUMMARY.md"),
            "preflight": str(preflight_path),
            "wrapper_tests": str(wrapper_tests_path),
            "ablation_check": str(PROJECT_ROOT / "artifacts" / "ablation_check"),
            "review_reloads": str(PROJECT_ROOT / "artifacts" / "review"),
        },
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        summary["host_memory_at_summary"] = {
            "total_mib": round(vm.total / 1024**2, 1),
            "available_mib": round(vm.available / 1024**2, 1),
        }
    except Exception:
        pass

    write_json(root / "smoke_summary.json", summary)
    lines = [
        "# Smoke evidence summary",
        "",
        "Full 500,000-step runs were **not** started. These are short end-to-end smoke runs.",
        "",
        "| run | algo | steps | updates | train-only steps/s | peak RSS MiB | final eval mean | clip |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in runs:
        timing = run["timing"]
        lines.append(
            "| {run_id} | {algorithm} | {actual_timesteps} | {total_optimizer_updates} | "
            "{rate} | {rss} | {score} | {clip} |".format(
                run_id=run["run_id"],
                algorithm=run["algorithm"],
                actual_timesteps=run["actual_timesteps"],
                total_optimizer_updates=run["total_optimizer_updates"],
                rate=timing.get("agent_steps_per_sec_training_only"),
                rss=run["memory"].get("process_peak_sampled_rss_mib"),
                score=run["final_eval"].get("mean_return_raw"),
                clip=(run["final_clips"][0] if run["final_clips"] else ""),
            )
        )
    lines += [
        "",
        "## Timing scope",
        "",
        summary["throughput_scope"],
        "",
        "## Checkpoint counts",
        "",
        summary["checkpoint_count_definition"],
        "",
    ]
    for run in runs:
        counts = run["checkpoint_counts"]
        lines.append(
            f"- {run['run_id']}: periodic={counts['periodic_from_manifest_csv']}, "
            f"final={counts['final_from_manifest_csv']}, "
            f"total={counts['total_from_manifest_csv']}, "
            f"consistent={counts['consistent']}."
        )
    lines += [
        "",
        "## Reconstructed full-profile ETA (planning only, not measured)",
        "",
        f"- {summary['eta_reconstruction']['ppo']}",
        f"- {summary['eta_reconstruction']['dqn']}",
        f"- {summary['eta_reconstruction']['evaluation']}",
        f"- Required before launch: {summary['eta_reconstruction']['required_before_launch']}",
        "",
    ]
    for algo, estimate in estimates.items():
        lines.append(
            f"- {algo.upper()}: one condition ~{estimate.get('one_condition_hours')} h "
            f"({estimate.get('training_seconds')} s training + {estimate.get('eval_seconds')} s eval "
            f"+ {estimate.get('checkpoint_save_seconds')} s checkpoint saves)."
        )
    lines += [
        f"- Ten conditions, one seed: ~{summary['full_wall_time_extrapolation']['ten_conditions_one_seed_hours']} h.",
        f"- Ten conditions, three matched seeds: ~{summary['full_wall_time_extrapolation']['ten_conditions_three_seeds_hours']} h.",
        f"- Missing algorithm rates: {missing or 'none'}.",
        f"- Assumption: {summary['full_wall_time_extrapolation']['assumption']}",
        "",
        "## Evidence",
        "",
        f"- Preflight: `{summary['preflight']}`",
        f"- Wrapper tests: `{summary['wrapper_tests']}` (ok={summary['wrapper_tests_ok']})",
        f"- Ablation diffs: `{summary['evidence_paths']['ablation_check']}`",
        f"- Review reloads: `{summary['evidence_paths']['review_reloads']}`",
    ]
    (root / "SMOKE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
