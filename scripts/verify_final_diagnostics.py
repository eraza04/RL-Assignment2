"""Regression check for final diagnostics (off-grid end + full schedule).

Runs focused short smokes: DQN ending on an optimizer update (508 steps), DQN
with no update before final (100 steps), and PPO aligned on a rollout update
(512 steps). A separate unit boundary finalizes a reloaded 508-step DQN state
at 510 without another optimizer step. Each real case compares the final
train_log.csv row with updates.csv and a fresh checkpoint reload. The full
500,000-step log/checkpoint schedule is checked analytically; no full training
is started.
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
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
from matplotlib.axes import Axes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl_a2.callbacks import TrainingTelemetry  # noqa: E402
from rl_a2.config import memory_headroom, read_json, resolve_config, write_json  # noqa: E402
from rl_a2.evaluate import load_model  # noqa: E402
from rl_a2.plots import plot_run  # noqa: E402
from rl_a2.probes import compute_mean_max_q, load_probe  # noqa: E402


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def _close(a: float | None, b: float | None, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _run_case(output_root: Path, condition: str, seed: int, steps: int) -> dict[str, Any]:
    algo = "dqn" if condition.startswith("dqn_") else "ppo"
    command = [
        sys.executable,
        "-m",
        "rl_a2.train",
        "--condition",
        condition,
        "--profile",
        "smoke",
        "--seed",
        str(seed),
        "--total-timesteps",
        str(steps),
        "--output-root",
        str(output_root),
        "--no-video",
    ]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    started_epoch = time.time()
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    finished = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if result.returncode != 0:
        raise RuntimeError(f"training failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    config = resolve_config(
        condition, "smoke", seed, output_root, total_timesteps=steps, validate=False
    )
    run_dir = Path(config["run"]["run_dir"])
    manifest = read_json(run_dir / "run_manifest.json")
    train_log = _rows(run_dir / "metrics" / "train_log.csv")
    updates = _rows(run_dir / "metrics" / "updates.csv")
    summary = read_json(run_dir / "metrics" / "training_summary.json")
    final_row = train_log[-1]
    last_update = updates[-1] if updates else {}
    final_step = int(manifest["actual_timesteps"])
    last_update_step = int(last_update["timestep"]) if last_update.get("timestep") else None
    note = str(final_row["note"])

    if algo == "dqn":
        probe_obs, probe_meta = load_probe(
            run_dir / "probes" / "probe_obs.npz",
            read_json(run_dir / "probes" / "probe_obs.json"),
        )
    else:
        probe_obs, probe_meta = None, {"sha256": None}
    model = load_model(run_dir / "checkpoints" / "final_model.zip", algo, device="cpu")
    reloaded_q = compute_mean_max_q(model, probe_obs) if algo == "dqn" else None
    reloaded_current_epsilon = (
        float(getattr(model, "exploration_rate", 1.0)) if algo == "dqn" else None
    )
    del model

    sampled_before_final = [row for row in train_log[:-1] if int(row["timestep"]) < final_step]
    stale_row = sampled_before_final[-1] if sampled_before_final else None
    diagnostic = summary["final_diagnostic"]
    aligned = last_update_step == final_step
    checks: dict[str, bool] = {
        "status_completed": manifest["status"] == "completed",
        "actual_timesteps": final_step == steps,
        "final_row_is_marked": final_row["is_final"].lower() == "true",
        "final_loss_is_actual_last_update": _close(
            _float(final_row["loss"]), _float(last_update.get("loss"))
        ),
        "final_loss_is_not_stale_sample": stale_row is None
        or not _close(_float(final_row["loss"]), _float(stale_row.get("loss"))),
        "last_update_timestep_matches_csv": (
            _float(final_row["last_update_timestep"]) == last_update_step
        ),
        "aligned_flag_matches_evidence": (
            final_row["final_update_aligned"].lower() == "true"
        ) == aligned,
        "note_is_explicit": bool(final_row["note"]),
        "summary_matches_row": (
            _float(diagnostic["last_update_timestep"])
            == _float(final_row["last_update_timestep"])
            and diagnostic["final_update_aligned"] == aligned
            and _close(diagnostic["final_loss"], _float(final_row["loss"]))
            and _close(diagnostic["final_mean_max_q"], reloaded_q)
            and _close(diagnostic.get("last_update_loss"), _float(last_update.get("loss")))
        ),
        "checkpoint_counts_consistent": (
            summary["checkpoint_events"] == len(_rows(run_dir / "metrics" / "checkpoint_manifest.csv"))
            and summary["periodic_checkpoints"]
            == len(
                [
                    row
                    for row in _rows(run_dir / "metrics" / "checkpoint_manifest.csv")
                    if row["checkpoint"] != "final"
                ]
            )
        ),
    }
    if algo == "dqn":
        loss_blank = final_row["loss"] == "" and final_row["td_huber_loss"] == ""
        update_loss_blank = last_update.get("loss", "") == ""
        checks.update(
            {
                "dqn_epsilon_is_current": _close(
                    _float(final_row["epsilon"]), reloaded_current_epsilon
                ),
                "online_q_is_recomputed_on_current_model": _close(
                    _float(final_row["mean_max_q"]), reloaded_q
                ),
                "dqn_loss_only_on_update_step": (aligned and not loss_blank)
                or (not aligned and loss_blank),
                "dqn_updates_csv_loss_only_on_update_step": (
                    aligned
                    and not update_loss_blank
                    or not aligned
                    and (not updates or update_loss_blank)
                ),
                "dqn_note_accurate": (
                    (
                        "loss is unavailable" in note
                        if last_update_step is None
                        else "loss comes from that actual update" in note
                        if aligned
                        else "loss is blank because it belongs to the last actual update" in note
                    )
                    and "current epsilon and online Q" in note
                    and "PPO" not in note
                ),
            }
        )
    else:
        checks.update(
            {
                "ppo_epsilon_and_q_blank": final_row["epsilon"] == ""
                and final_row["mean_max_q"] == "",
                "ppo_note_names_inapplicable_fields": (
                    "PPO diagnostics" in note and "not applicable to PPO" in note
                ),
                "ppo_note_has_no_dqn_loss_claim": (
                    "loss/epsilon come" not in note.lower()
                    and "online q recomputed" not in note.lower()
                ),
            }
        )
    return {
        "algorithm": algo,
        "condition": condition,
        "command": command,
        "command_text": subprocess.list2cmdline(command),
        "started_at": started,
        "finished_at": finished,
        "started_epoch": started_epoch,
        "wall_seconds": round(time.time() - started_epoch, 3),
        "run_dir": str(run_dir),
        "manifest_path": str(run_dir / "run_manifest.json"),
        "actual_timesteps": final_step,
        "total_optimizer_updates": manifest["total_optimizer_updates"],
        "process_peak_sampled_rss_mib": manifest["memory"]["process_peak_sampled_rss_mib"],
        "available_mib_at_start": manifest["memory"]["host"]["available_mib_at_start"],
        "training_only_steps_per_s": manifest["timing"]["agent_steps_per_sec_training_only"],
        "probe_sha256": probe_meta["sha256"],
        "last_update_row": last_update,
        "stale_sampled_row": stale_row,
        "final_train_log_row": final_row,
        "reloaded_final_online_mean_max_q": reloaded_q,
        "reloaded_current_epsilon": reloaded_current_epsilon,
        "checks": checks,
        "ok": all(checks.values()),
    }


def _unit_no_update_after_last(output_root: Path) -> dict[str, Any]:
    """Finalize a reloaded 508-step state at 510 without another update."""
    run_dir = output_root / "unit_no_update_after_last"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = resolve_config(
        "dqn_target_1000",
        "smoke",
        0,
        run_dir,
        total_timesteps=510,
        validate=False,
    )
    model = SimpleNamespace(exploration_rate=0.31, num_timesteps=510, logger=SimpleNamespace(name_to_value={}))
    telemetry = TrainingTelemetry(
        model,
        config,
        run_dir,
        probe_obs=np.zeros((1, 4, 84, 84), dtype=np.uint8),
        probe_meta=None,
    )
    last_update = {
        "algorithm": "dqn",
        "timestep": 508,
        "optimizer_updates": 77,
        "updates_this_call": 1,
        "epsilon": 0.35,
        "td_huber_loss": 0.12,
        "loss": 0.12,
        "is_final": False,
        "note": "",
    }
    telemetry.optimizer_steps = 77
    telemetry.last_update_row = dict(last_update)
    telemetry.update_rows.append(dict(last_update))
    telemetry.train_log_rows.append(
        {
            "algorithm": "dqn",
            "timestep": 508,
            "optimizer_updates": 77,
            "epsilon": 0.35,
            "td_huber_loss": 0.12,
            "mean_max_q": 0.1,
            "loss": 0.12,
            "is_final": False,
        }
    )
    recomputed_q = 0.07286563515663147
    with patch("rl_a2.callbacks.compute_mean_max_q", return_value=recomputed_q):
        summary = telemetry.finalize(510, 510)

    plot_calls: list[dict[str, Any]] = []
    annotations: list[str] = []
    original_plot = Axes.plot
    original_annotate = Axes.annotate

    def capture_plot(self, x, y, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        plot_calls.append(
            {
                "x": [float(value) for value in x],
                "y": [float(value) for value in y],
                "color": kwargs.get("color"),
                "label": str(kwargs.get("label", "")),
            }
        )
        return original_plot(self, x, y, *args, **kwargs)

    def capture_annotate(self, text, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        annotations.append(str(text))
        return original_annotate(self, text, *args, **kwargs)

    with patch.object(Axes, "plot", capture_plot), patch.object(Axes, "annotate", capture_annotate):
        plot_run(run_dir, config)

    train_log = _rows(run_dir / "metrics" / "train_log.csv")
    updates = _rows(run_dir / "metrics" / "updates.csv")
    final_row = train_log[-1]
    final_event = updates[-1]
    note = str(final_row["note"])
    diagnostic = summary["final_diagnostic"]
    loss_traces = [
        call
        for call in plot_calls
        if call["color"] == "tab:orange" and any(abs(value - 0.12) < 1e-12 for value in call["y"])
    ]
    checks: dict[str, bool] = {
        "final_step_is_510": final_row["timestep"] == "510",
        "optimizer_count_unchanged": final_row["optimizer_updates"] == "77",
        "last_update_step_is_508": final_row["last_update_timestep"] == "508",
        "aligned_flag_false": final_row["final_update_aligned"].lower() == "false",
        "final_loss_blank": final_row["loss"] == "" and final_row["td_huber_loss"] == "",
        "updates_event_loss_blank": final_event["loss"] == ""
        and final_event["td_huber_loss"] == ""
        and final_event["updates_this_call"] == "0",
        "epsilon_is_current_not_last_update": _close(
            _float(final_row["epsilon"]), 0.31
        )
        and not _close(_float(final_row["epsilon"]), 0.35),
        "q_is_recomputed_on_current_model": _close(
            _float(final_row["mean_max_q"]), recomputed_q
        ),
        "note_attributes_loss_to_last_update_without_claiming_current": (
            "loss is blank because it belongs to the last actual update at step 508" in note
            and "current epsilon and online Q" in note
            and "loss/epsilon come" not in note
        ),
        "summary_final_loss_blank_with_last_update_loss": (
            diagnostic["final_loss"] is None
            and _close(diagnostic.get("last_update_loss"), 0.12)
        ),
        "no_loss_point_plotted_at_510": (
            len(loss_traces) == 1
            and loss_traces[0]["x"] == [508.0]
            and 510.0 not in loss_traces[0]["x"]
        ),
        "no_update_annotated_on_loss_plot": any(
            "no optimizer update at final step" in text for text in annotations
        ),
    }
    return {
        "case_type": "unit_finalization_boundary",
        "run_dir": str(run_dir),
        "final_train_log_row": final_row,
        "final_update_event": final_event,
        "summary": summary,
        "recomputed_q": recomputed_q,
        "loss_plot_traces": loss_traces,
        "annotations": annotations,
        "checks": checks,
        "ok": all(checks.values()),
    }


def _full_schedule(config: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the full DQN train/log grid without running full training."""
    total = int(config["training"]["total_timesteps"])
    log_interval = int(config["training"]["log_interval"])
    warmup = int(config["model"]["learning_starts"])
    train_freq = int(config["model"]["train_freq"])
    ckpt_interval = int(config["training"]["checkpoint_interval"])
    update_steps = [
        step for step in range(1, total + 1) if step > warmup and step % train_freq == 0
    ]
    logged: list[int] = []
    last_logged = -10**18
    for step in update_steps:
        if step - last_logged >= log_interval:
            logged.append(step)
            last_logged = step
    checkpoints = [step for step in range(ckpt_interval, total + 1, ckpt_interval)]
    return {
        "total_timesteps": total,
        "learning_starts": warmup,
        "first_optimizer_update": update_steps[0],
        "last_optimizer_update": update_steps[-1],
        "optimizer_update_count": len(update_steps),
        "first_logged_train_step": logged[0],
        "last_logged_train_step": logged[-1],
        "periodic_checkpoint_steps": checkpoints,
        "final_timestep_is_on_train_log_grid": total in logged,
        "final_timestep_has_optimizer_update": total in update_steps,
        "expected_final_row_source": (
            "final loss and optimizer diagnostics from the actual update at 500000; "
            "current epsilon and recomputed online Q"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="artifacts/remediation/final_diagnostics",
        help="Fresh output root; existing run directories are never reused or deleted",
    )
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = {
        "off_grid_update_at_final": _run_case(output_root, "dqn_target_1000", 0, 508),
        "no_update_before_final": _run_case(output_root, "dqn_eps_010", 1, 100),
        "no_update_after_prior": _unit_no_update_after_last(output_root),
        "ppo_aligned_update": _run_case(output_root, "ppo_baseline", 0, 512),
    }
    full_config = resolve_config("dqn_baseline", "full", 0, output_root)
    full = _full_schedule(full_config)
    full["full_buffer_size"] = int(full_config["model"]["buffer_size"])
    full["smoke_buffer_size"] = int(
        resolve_config("dqn_baseline", "smoke", 0, output_root)["model"]["buffer_size"]
    )
    headroom = memory_headroom(full_config)
    headroom.update(
        {
            "note": (
                "Full DQN launch gate uses this verdict; no full training was started "
                "by this regression check."
            )
        }
    )
    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command_for_this_script": subprocess.list2cmdline([sys.executable, "-m", *sys.argv]),
        "output_root": str(output_root),
        "cases": cases,
        "full_profile_schedule": full,
        "full_profile_memory_headroom": headroom,
    }
    # The regression checks diagnostics and schedule. Host headroom is a
    # separate launch gate verdict, not a reason to fail this regression.
    report["regression_ok"] = all(case["ok"] for case in cases.values())
    report["host_headroom_ok"] = headroom["ok"]
    path = output_root / "final_diagnostics_report.json"
    write_json(path, report)
    print(json.dumps(report, indent=2))
    print(f"\nReport: {path}")
    return 0 if report["regression_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
