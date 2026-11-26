"""Training telemetry: optimizer counts, sampled losses, episodes, checkpoints.

SB3 does not expose a per-optimizer-step callback. We wrap ``model.train`` to
observe the actual number of optimizer steps and the logger's mean sampled
loss after that call. For DQN (train_freq=4, gradient_steps=1) each call is one
Huber-loss update. For PPO each call is one rollout update with many minibatch
optimizer steps; the optimizer step counter is incremented by wrapping
``optimizer.step``, so the logged count is the actual number, not SB3's
per-epoch ``_n_updates`` counter. Timesteps are the environment counters after
the update, which is the correct sample-efficiency alignment.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .config import write_json
from .evaluate import evaluate_checkpoint
from .probes import compute_mean_max_q

UPDATE_FIELDS = [
    "algorithm",
    "timestep",
    "optimizer_updates",
    "updates_this_call",
    "epsilon",
    "td_huber_loss",
    "loss",
    "approx_kl",
    "value_loss",
    "entropy_loss",
    "policy_entropy",
    "clip_fraction",
    "is_final",
    "note",
]

TRAIN_LOG_FIELDS = [
    "algorithm",
    "timestep",
    "optimizer_updates",
    "last_update_timestep",
    "final_update_aligned",
    "epsilon",
    "td_huber_loss",
    "mean_max_q",
    "loss",
    "approx_kl",
    "value_loss",
    "entropy_loss",
    "policy_entropy",
    "clip_fraction",
    "is_final",
    "note",
]

EPISODE_FIELDS = [
    "algorithm",
    "timestep",
    "episode_index",
    "return_raw",
    "return_training_reward",
    "length",
    "terminated",
    "truncated",
    "ale_score",
]

CHECKPOINT_FIELDS = [
    "checkpoint",
    "timestep",
    "optimizer_updates",
    "model_path",
    "eval_episodes",
    "eval_mean_return_raw",
    "eval_std_return_raw",
    "eval_mean_max_q",
    "probe_sha256",
    "reload_verified",
]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, float) and not np.isfinite(value):
                    value = ""
                clean[field] = "" if value is None else value
            writer.writerow(clean)


class TrainingTelemetry:
    def __init__(
        self,
        model,
        config: dict[str, Any],
        run_dir: Path,
        probe_obs: np.ndarray | None,
        probe_meta: dict[str, Any] | None,
    ):
        self.model = model
        self.config = config
        self.run_dir = run_dir
        self.probe_obs = probe_obs
        self.probe_meta = probe_meta
        self.algo = config["algo"]
        self.optimizer_steps = 0
        self.last_update_row: dict[str, Any] | None = None
        self.update_rows: list[dict[str, Any]] = []
        self.train_log_rows: list[dict[str, Any]] = []
        self.episode_rows: list[dict[str, Any]] = []
        self.checkpoint_rows: list[dict[str, Any]] = []
        self.episode_index = 0
        self.last_train_log_timestep = -10**18
        self.next_checkpoint = int(config["training"]["checkpoint_interval"])
        self.eval_seconds = 0.0
        self.checkpoint_seconds = 0.0
        self._orig_train: Callable | None = None
        self._orig_optimizer_step: Callable | None = None
        self._attached = False

    # ------------------------------------------------------------------ setup
    def attach(self) -> None:
        if self._attached:
            return
        optimizer = self.model.policy.optimizer
        original_step = optimizer.step

        def counted_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            self.optimizer_steps += 1
            return result

        optimizer.step = counted_step
        self._orig_optimizer_step = original_step

        original_train = self.model.train

        def wrapped_train(*args, **kwargs):
            before = self.optimizer_steps
            result = original_train(*args, **kwargs)
            if self.optimizer_steps > before:
                self._after_train(self.optimizer_steps - before)
            return result

        self.model.train = wrapped_train
        self._orig_train = original_train
        self._attached = True

    def detach(self) -> None:
        if not self._attached:
            return
        self.model.train = self._orig_train
        self.model.policy.optimizer.step = self._orig_optimizer_step
        self._attached = False

    # --------------------------------------------------------------- metrics
    def _value(self, key: str) -> float | None:
        value = self.model.logger.name_to_value.get(key)
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def _after_train(self, updates_this_call: int) -> None:
        timestep = int(self.model.num_timesteps)
        loss = self._value("train/loss")
        approx_kl = self._value("train/approx_kl")
        value_loss = self._value("train/value_loss")
        entropy_loss = self._value("train/entropy_loss")
        clip_fraction = self._value("train/clip_fraction")
        policy_entropy = -entropy_loss if entropy_loss is not None else None
        epsilon = self._value("train/epsilon")
        if epsilon is None and hasattr(self.model, "exploration_rate"):
            epsilon = float(self.model.exploration_rate)
        td_loss = loss if self.algo == "dqn" else None

        update_row = {
            "algorithm": self.algo,
            "timestep": timestep,
            "optimizer_updates": int(self.optimizer_steps),
            "updates_this_call": int(updates_this_call),
            "epsilon": epsilon,
            "td_huber_loss": td_loss,
            "loss": loss,
            "approx_kl": approx_kl,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "policy_entropy": policy_entropy,
            "clip_fraction": clip_fraction,
            "is_final": False,
            "note": "",
        }
        self.last_update_row = update_row
        stride = int(self.config["training"]["update_log_stride"])
        # Always keep PPO rollout updates; sample DQN updates in full runs.
        if self.algo == "ppo" or self.optimizer_steps % stride == 0:
            self.update_rows.append(dict(update_row))

        log_interval = int(self.config["training"]["log_interval"])
        if self.algo == "ppo" or timestep - self.last_train_log_timestep >= log_interval:
            mean_q = None
            if self.algo == "dqn" and self.probe_obs is not None:
                mean_q = compute_mean_max_q(self.model, self.probe_obs)
            log_row = {
                "algorithm": self.algo,
                "timestep": timestep,
                "optimizer_updates": int(self.optimizer_steps),
                "epsilon": epsilon,
                "td_huber_loss": td_loss,
                "mean_max_q": mean_q,
                "loss": loss,
                "approx_kl": approx_kl,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
                "policy_entropy": policy_entropy,
                "clip_fraction": clip_fraction,
                "is_final": False,
            }
            self.train_log_rows.append(log_row)
            self.last_train_log_timestep = timestep

        self._maybe_checkpoint(timestep)

    def _maybe_checkpoint(self, timestep: int) -> None:
        if timestep < self.next_checkpoint:
            return
        path = self.run_dir / "checkpoints" / f"ckpt_{timestep:07d}.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        self.model.save(str(path))
        saved = time.perf_counter()
        eval_start = time.perf_counter()
        result = evaluate_checkpoint(
            path,
            checkpoint_name=f"ckpt_{timestep:07d}",
            checkpoint_step=timestep,
            is_final=False,
            config=self.config,
            run_dir=self.run_dir,
            probe_obs=self.probe_obs,
            probe_meta=self.probe_meta,
            record_video=False,
        )
        eval_end = time.perf_counter()
        self.checkpoint_seconds += saved - start
        self.eval_seconds += eval_end - eval_start
        summary = result["summary"]
        self.checkpoint_rows.append(
            {
                "checkpoint": f"ckpt_{timestep:07d}",
                "timestep": timestep,
                "optimizer_updates": int(self.optimizer_steps),
                "model_path": str(path),
                "eval_episodes": summary["episodes"],
                "eval_mean_return_raw": summary["mean_return_raw"],
                "eval_std_return_raw": summary["std_return_raw"],
                "eval_mean_max_q": summary["mean_max_q"],
                "probe_sha256": summary["probe_sha256"],
                "reload_verified": True,
            }
        )
        while self.next_checkpoint <= timestep:
            self.next_checkpoint += int(self.config["training"]["checkpoint_interval"])

    def add_episode(self, info: dict[str, Any], timestep: int) -> None:
        raw_episode = info.get("raw_episode")
        monitor_episode = info.get("episode")
        if not raw_episode and not monitor_episode:
            return
        raw_episode = raw_episode or {}
        monitor_episode = monitor_episode or {}
        self.episode_rows.append(
            {
                "algorithm": self.algo,
                "timestep": int(timestep),
                "episode_index": int(self.episode_index),
                "return_raw": raw_episode.get("return", monitor_episode.get("r")),
                "return_training_reward": monitor_episode.get("r"),
                "length": raw_episode.get("length", monitor_episode.get("l")),
                "terminated": raw_episode.get("terminated", ""),
                "truncated": raw_episode.get("truncated", ""),
                "ale_score": raw_episode.get("ale_score", ""),
            }
        )
        self.episode_index += 1

    # --------------------------------------------------------------- finish
    def finalize(self, actual_timesteps: int, requested_timesteps: int) -> dict[str, Any]:
        """Write a final row that never misattributes a measured loss.

        The train curve is sampled on an interval, so the final timestep can
        fall off that grid. DQN loss is kept only on timesteps with a real
        optimizer update; an off-update final row leaves it blank and reports
        current epsilon plus freshly recomputed online Q. PPO diagnostics can
        describe the last rollout update, and each note states the applicable
        algorithm without referring to inapplicable epsilon/Q.
        """
        final_timestep = int(actual_timesteps)
        last = self.last_update_row
        last_update_timestep = int(last["timestep"]) if last else None
        aligned = last_update_timestep == final_timestep
        sampled_at_final = any(row["timestep"] == final_timestep for row in self.train_log_rows)
        current_epsilon = None
        if self.algo == "dqn":
            current_epsilon = getattr(self.model, "exploration_rate", None)
            if current_epsilon is not None:
                current_epsilon = float(current_epsilon)

        if last is not None:
            final_update = dict(last)
            final_update["is_final"] = True
            if aligned:
                if self.update_rows and self.update_rows[-1]["timestep"] == final_timestep:
                    self.update_rows[-1] = final_update
                else:
                    self.update_rows.append(final_update)
            else:
                final_event = dict(last)
                final_event.update(
                    {
                        "timestep": final_timestep,
                        "optimizer_updates": int(self.optimizer_steps),
                        "updates_this_call": 0,
                        "is_final": True,
                        "note": "end-of-training event; no optimizer update at this step",
                    }
                )
                if self.algo == "dqn":
                    # The copied loss belongs to the earlier measured update.
                    final_event["td_huber_loss"] = None
                    final_event["loss"] = None
                    if current_epsilon is not None:
                        final_event["epsilon"] = current_epsilon
                self.update_rows.append(final_event)

        source = dict(last) if last else {}
        mean_q = None
        if self.algo == "dqn" and self.probe_obs is not None:
            mean_q = compute_mean_max_q(self.model, self.probe_obs)
        epsilon = source.get("epsilon")
        if self.algo == "dqn":
            if current_epsilon is not None:
                epsilon = current_epsilon
            # A DQN loss belongs to the timestep of its measured optimizer
            # update. An off-update final row must not inherit it.
            final_loss = source.get("loss") if aligned else None
            final_td_loss = source.get("td_huber_loss") if aligned else None
        else:
            final_loss = source.get("loss")
            final_td_loss = None

        if self.algo == "dqn":
            if aligned:
                note = (
                    f"optimizer update occurred at final timestep {final_timestep}; "
                    "loss comes from that actual update; current epsilon and online Q "
                    "recomputed at finalize"
                )
            elif last is None:
                note = (
                    f"no optimizer update occurred before final timestep {final_timestep}; "
                    "loss is unavailable; current epsilon and online Q recomputed at finalize"
                )
            else:
                note = (
                    f"no optimizer update at final timestep {final_timestep}; "
                    f"loss is blank because it belongs to the last actual update at step "
                    f"{last_update_timestep}; current epsilon and online Q recomputed at finalize"
                )
        else:
            if aligned:
                note = (
                    f"optimizer update occurred at final timestep {final_timestep}; "
                    "PPO diagnostics come from that actual rollout update; epsilon and "
                    "online Q are not applicable to PPO"
                )
            elif last is None:
                note = (
                    f"no optimizer update occurred before final timestep {final_timestep}; "
                    "PPO diagnostics are unavailable; epsilon and online Q are not applicable to PPO"
                )
            else:
                note = (
                    f"no optimizer update at final timestep {final_timestep}; "
                    f"PPO diagnostics come from the last actual rollout update at step "
                    f"{last_update_timestep}; epsilon and online Q are not applicable to PPO"
                )
        if aligned and not sampled_at_final:
            note += "; final timestep fell off the train-log sampling grid"
        final_row = {
            "algorithm": self.algo,
            "timestep": final_timestep,
            "optimizer_updates": int(self.optimizer_steps),
            "last_update_timestep": last_update_timestep,
            "final_update_aligned": aligned,
            "epsilon": epsilon,
            "td_huber_loss": final_td_loss,
            "mean_max_q": mean_q,
            "loss": final_loss,
            "approx_kl": source.get("approx_kl"),
            "value_loss": source.get("value_loss"),
            "entropy_loss": source.get("entropy_loss"),
            "policy_entropy": source.get("policy_entropy"),
            "clip_fraction": source.get("clip_fraction"),
            "is_final": True,
            "note": note,
        }
        self.train_log_rows = [row for row in self.train_log_rows if row["timestep"] != final_timestep]
        self.train_log_rows.append(final_row)

        _write_csv(self.run_dir / "metrics" / "updates.csv", UPDATE_FIELDS, self.update_rows)
        _write_csv(self.run_dir / "metrics" / "train_log.csv", TRAIN_LOG_FIELDS, self.train_log_rows)
        _write_csv(self.run_dir / "metrics" / "train_episodes.csv", EPISODE_FIELDS, self.episode_rows)
        _write_csv(
            self.run_dir / "metrics" / "checkpoint_manifest.csv",
            CHECKPOINT_FIELDS,
            self.checkpoint_rows,
        )
        summary = {
            "algorithm": self.algo,
            "requested_timesteps": int(requested_timesteps),
            "actual_timesteps": int(actual_timesteps),
            "rollout_overshoot": int(actual_timesteps) - int(requested_timesteps),
            "total_optimizer_updates": int(self.optimizer_steps),
            "last_optimizer_update_timestep": (
                int(self.last_update_row["timestep"]) if self.last_update_row else None
            ),
            "episodes_logged": len(self.episode_rows),
            "checkpoint_events": len(self.checkpoint_rows),
            "periodic_checkpoints": len(
                [row for row in self.checkpoint_rows if row.get("checkpoint") != "final"]
            ),
            "periodic_checkpoint_steps": [
                int(row["timestep"])
                for row in self.checkpoint_rows
                if row.get("checkpoint") != "final"
            ],
            "final_checkpoint_count": len(
                [row for row in self.checkpoint_rows if row.get("checkpoint") == "final"]
            ),
            "checkpoint_count_source": "checkpoint_manifest.csv rows; periodic excludes checkpoint=='final'",
            "update_rows_logged": len(self.update_rows),
            "update_log_stride": int(self.config["training"]["update_log_stride"]),
            "loss_source": (
                "model.logger['train/loss'] after a real wrapped optimizer update; DQN final "
                "loss is blank unless an update occurred at the final timestep"
            ),
            "optimizer_count_source": "wrapped optimizer.step() call counter",
            "timestep_alignment": "after the update call, model.num_timesteps is the number of env transitions consumed",
            "final_diagnostic": {
                "source": (
                    "optimizer-update diagnostics from the aligned final update (DQN) or last "
                    "rollout update (PPO); current DQN epsilon and recomputed online Q"
                ),
                "last_update_timestep": last_update_timestep,
                "final_update_aligned": aligned,
                "final_train_log_sampled": sampled_at_final,
                "final_loss": final_loss,
                "last_update_loss": source.get("loss"),
                "final_epsilon": epsilon,
                "final_mean_max_q": mean_q,
                "note": note,
            },
        }
        write_json(self.run_dir / "metrics" / "training_summary.json", summary)
        return summary


class EpisodeLoggerCallback(BaseCallback):
    """Feed episode summaries from Monitor/RawEpisodeStats into telemetry."""

    def __init__(self, telemetry: TrainingTelemetry, verbose: int = 0):
        super().__init__(verbose)
        self.telemetry = telemetry

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        timestep = int(self.num_timesteps)
        for info in infos:
            if isinstance(info, dict):
                self.telemetry.add_episode(info, timestep)
        return True


class CheckpointFinalizerCallback(BaseCallback):
    """Detach telemetry at the end; train.py saves/evaluates the final model."""

    def __init__(self, telemetry: TrainingTelemetry, verbose: int = 0):
        super().__init__(verbose)
        self.telemetry = telemetry

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        # Keep telemetry detach explicit at the end of learn; train.py saves
        # the final model and performs the final reload evaluation.
        self.telemetry.detach()
