"""CSE 482 Assignment 2: config-driven DQN/PPO training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .callbacks import (
    CheckpointFinalizerCallback,
    EpisodeLoggerCallback,
    TrainingTelemetry,
)
from .config import (
    PROJECT_ROOT,
    memory_headroom,
    replay_estimate,
    resolve_config,
    write_json,
)
from .envs import describe_env, make_vec_env
from .evaluate import evaluate_checkpoint
from .probes import ensure_probe, load_probe, probe_fingerprint
from .plots import plot_run
from .provenance import source_digest


def _versions() -> dict[str, Any]:
    import cv2
    import gymnasium
    import imageio
    import stable_baselines3
    import torch

    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "imageio": imageio.__version__,
    }
    try:
        import ale_py

        versions["ale_py"] = ale_py.__version__
    except Exception as exc:  # pragma: no cover
        versions["ale_py_error"] = repr(exc)
    try:
        import Box2D

        versions["Box2D"] = getattr(Box2D, "__version__", "imported")
    except Exception as exc:  # pragma: no cover
        versions["Box2D_error"] = repr(exc)
    try:
        import imageio_ffmpeg

        versions["imageio_ffmpeg"] = imageio_ffmpeg.get_ffmpeg_version()
    except Exception as exc:  # pragma: no cover
        versions["imageio_ffmpeg_error"] = repr(exc)
    return versions


def _host_memory() -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    return {
        "total_bytes": int(vm.total),
        "total_mib": round(vm.total / 1024**2, 1),
        "total_gib": round(vm.total / 1024**3, 2),
        "available_bytes_at_start": int(vm.available),
        "available_mib_at_start": round(vm.available / 1024**2, 1),
    }


class MemorySampler(threading.Thread):
    def __init__(self, interval_s: float = 0.25):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.peak_rss = 0
        self.samples = 0

    def run(self) -> None:
        import psutil

        process = psutil.Process(os.getpid())
        while not self.stop_event.wait(self.interval_s):
            try:
                rss = int(process.memory_info().rss)
            except Exception:
                break
            self.peak_rss = max(self.peak_rss, rss)
            self.samples += 1

    def stop(self) -> int:
        self.stop_event.set()
        self.join(timeout=5)
        return self.peak_rss


def _peak_wset_mib() -> float | None:
    try:
        import psutil

        info = psutil.Process(os.getpid()).memory_info()
        peak = getattr(info, "peak_wset", None)
        return round(peak / 1024**2, 1) if peak else None
    except Exception:
        return None


def _setup_logging(run_dir: Path, verbose: int) -> logging.Logger:
    logger = logging.getLogger("rl_a2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _copy_probe_to_run(run_dir: Path, source: Path, meta: dict[str, Any]) -> Path:
    probe_dir = run_dir / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    destination = probe_dir / "probe_obs.npz"
    shutil.copy2(source, destination)
    meta_path = probe_dir / "probe_obs.json"
    write_json(meta_path, meta)
    # Load from the run-local copy, not the shared source, and verify the hash.
    obs, verified = load_probe(destination, meta)
    if verified["sha256"] != probe_fingerprint(obs):
        raise RuntimeError("Run-local probe hash verification failed")
    return destination


def build_model(config: dict[str, Any], vec_env, seed: int, logger: logging.Logger):
    from stable_baselines3 import DQN, PPO

    model_cfg = dict(config["model"])
    policy = model_cfg.pop("policy")
    policy_kwargs = model_cfg.pop("policy_kwargs", None)
    device = model_cfg.pop("device")
    algo = config["algo"]
    if algo == "dqn":
        model = DQN(
            policy,
            vec_env,
            policy_kwargs=policy_kwargs,
            device=device,
            seed=seed,
            verbose=int(config["training"]["verbose"]),
            tensorboard_log=None,
            **model_cfg,
        )
    else:
        model = PPO(
            policy,
            vec_env,
            policy_kwargs=policy_kwargs,
            device=device,
            seed=seed,
            verbose=int(config["training"]["verbose"]),
            tensorboard_log=None,
            **model_cfg,
        )
    logger.info(
        "model=%s policy=%s obs=%s actions=%s device=%s",
        type(model).__name__,
        type(model.policy).__name__,
        model.observation_space,
        model.action_space,
        model.device,
    )
    if algo == "dqn":
        extractor = model.q_net.features_extractor
        logger.info(
            "features_extractor=%s features_dim=%s",
            type(extractor).__name__,
            getattr(extractor, "features_dim", None),
        )
    return model


def _artifact_checks(
    run_dir: Path,
    config: dict[str, Any],
    final_model: Path,
    final_eval: dict[str, Any],
    expect_video: bool,
) -> dict[str, bool]:
    checks = {
        "config_resolved": (run_dir / "config_resolved.json").is_file(),
        "final_model": final_model.is_file() and final_model.stat().st_size > 0,
        "updates_csv": (run_dir / "metrics" / "updates.csv").is_file(),
        "train_log_csv": (run_dir / "metrics" / "train_log.csv").is_file(),
        "train_episodes_csv": (run_dir / "metrics" / "train_episodes.csv").is_file(),
        "eval_episodes_csv": (run_dir / "metrics" / "eval_episodes.csv").is_file(),
        "eval_summary_csv": (run_dir / "metrics" / "eval_summary.csv").is_file(),
        "checkpoint_manifest": (run_dir / "metrics" / "checkpoint_manifest.csv").is_file(),
        "training_summary": (run_dir / "metrics" / "training_summary.json").is_file(),
        "plots_dir": (run_dir / "plots").is_dir(),
        "final_eval_rows": len(final_eval.get("episodes", []))
        == int(config["evaluation"]["episodes"]),
    }
    if config["algo"] == "dqn":
        checks["probe_npz"] = (run_dir / "probes" / "probe_obs.npz").is_file()
        checks["probe_meta"] = (run_dir / "probes" / "probe_obs.json").is_file()
    if expect_video:
        video_paths = [
            row.get("video_path")
            for row in final_eval.get("episodes", [])
            if row.get("video_path")
        ]
        checks["final_clip"] = bool(video_paths) and all(
            Path(path).is_file() and Path(path).stat().st_size > 1024 for path in video_paths
        )
    return checks


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    config = resolve_config(
        condition=args.condition,
        profile=args.profile,
        seed=args.seed,
        output_root=args.output_root,
        total_timesteps=args.total_timesteps,
        device=args.device,
    )
    run_dir = Path(config["run"]["run_dir"]).resolve()
    if run_dir == PROJECT_ROOT or PROJECT_ROOT.is_relative_to(run_dir):
        raise SystemExit(f"Refusing to use project root or an ancestor as run dir: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(
            f"Run directory already exists and is non-empty: {run_dir}\n"
            "Choose a fresh --output-root. Evidence is never deleted or merged."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    started_epoch = time.time()
    run_start_perf = time.perf_counter()
    logger = _setup_logging(run_dir, int(config["training"]["verbose"]))
    logger.info("run_id=%s run_dir=%s", config["run"]["run_id"], run_dir)
    logger.info("condition=%s profile=%s seed=%s", args.condition, args.profile, args.seed)

    write_json(run_dir / "config_resolved.json", config)

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    host_memory = _host_memory()
    headroom = memory_headroom(config, host_memory["available_bytes_at_start"])
    versions = _versions()
    replay = replay_estimate(config)
    env_description = describe_env(config)
    logger.info("host memory: %s", host_memory)
    logger.info("env: %s", env_description)
    logger.info("replay estimate: %s", replay)
    logger.info("memory headroom: %s", headroom)
    if config["profile"] == "full" and config["algo"] == "dqn" and not headroom["ok"]:
        raise SystemExit(
            "Refusing full DQN launch: insufficient verified free RAM. "
            f"selected replay={headroom['replay_mib']} MiB, projected process peak="
            f"{headroom['projected_peak_mib']} MiB, required free="
            f"{headroom['required_free_mib']} MiB, available="
            f"{headroom['available_mib']} MiB (deficit {headroom['deficit_mib']} MiB)."
        )

    # Fixed probe set: shared per output root, copied into the run and hashed.
    probe_obs = None
    probe_meta = None
    probe_source = None
    if config["algo"] == "dqn":
        shared_dir = Path(config["run"]["output_root"]) / "_probes"
        probe_obs, probe_meta, probe_source = ensure_probe(config, shared_dir)
        _copy_probe_to_run(run_dir, probe_source, probe_meta)
        logger.info(
            "probe: source=%s sha256=%s shape=%s",
            probe_source,
            probe_meta["sha256"],
            probe_meta["shape"],
        )

    vec_env = make_vec_env(config, seed=args.seed)
    model = build_model(config, vec_env, seed=args.seed, logger=logger)
    telemetry = TrainingTelemetry(model, config, run_dir, probe_obs, probe_meta)
    telemetry.attach()
    callback = None
    from stable_baselines3.common.callbacks import CallbackList

    callback = CallbackList(
        [
            EpisodeLoggerCallback(telemetry, verbose=int(config["training"]["verbose"])),
            CheckpointFinalizerCallback(telemetry),
        ]
    )

    requested = int(config["training"]["total_timesteps"])
    sampler = MemorySampler()
    sampler.start()
    learn_start = time.perf_counter()
    status = "completed"
    error: str | None = None
    actual = 0
    learn_seconds = 0.0
    final_model = run_dir / "checkpoints" / "final_model.zip"
    final_eval: dict[str, Any] = {"episodes": [], "summary": {}}
    final_eval_seconds = 0.0
    try:
        model.learn(
            total_timesteps=requested,
            callback=callback,
            log_interval=int(config["training"]["log_interval"]),
            progress_bar=False,
            reset_num_timesteps=True,
        )
        learn_seconds = time.perf_counter() - learn_start
        actual = int(model.num_timesteps)
        logger.info("learn finished: requested=%s actual=%s", requested, actual)

        final_model.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(final_model))
        logger.info("saved final model: %s", final_model)

        final_eval_start = time.perf_counter()
        final_eval = evaluate_checkpoint(
            final_model,
            checkpoint_name="final",
            checkpoint_step=actual,
            is_final=True,
            config=config,
            run_dir=run_dir,
            probe_obs=probe_obs,
            probe_meta=probe_meta,
            record_video=bool(config["evaluation"]["record_video"]) and not args.no_video,
        )
        final_eval_seconds = time.perf_counter() - final_eval_start
        logger.info("final reload evaluation: %s", final_eval["summary"])
        telemetry.checkpoint_rows.append(
            {
                "checkpoint": "final",
                "timestep": actual,
                "optimizer_updates": int(telemetry.optimizer_steps),
                "model_path": str(final_model),
                "eval_episodes": final_eval["summary"]["episodes"],
                "eval_mean_return_raw": final_eval["summary"]["mean_return_raw"],
                "eval_std_return_raw": final_eval["summary"]["std_return_raw"],
                "eval_mean_max_q": final_eval["summary"]["mean_max_q"],
                "probe_sha256": final_eval["summary"]["probe_sha256"],
                "reload_verified": True,
            }
        )
        telemetry.finalize(actual, requested)
        plot_run(run_dir, config)
    except Exception as exc:  # keep a failed manifest, never hide a failure
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("training failed")
    finally:
        peak_rss = sampler.stop()
        try:
            vec_env.close()
        except Exception:
            pass
        if status == "failed":
            try:
                telemetry.detach()
            except Exception:
                pass

    checks = _artifact_checks(
        run_dir,
        config,
        final_model,
        final_eval,
        expect_video=bool(config["evaluation"]["record_video"]) and not args.no_video,
    )
    training_summary_path = run_dir / "metrics" / "training_summary.json"
    if training_summary_path.is_file():
        summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    timing = {
        "scope": (
            "learn_wall_seconds covers model.learn including periodic checkpoint saves and "
            "reload evaluations; final_eval_seconds is after learn; training_only_seconds "
            "subtracts telemetry-recorded periodic eval and checkpoint-save time."
        ),
        "learn_wall_seconds": round(learn_seconds, 3),
        "final_eval_seconds": round(final_eval_seconds, 3),
        "periodic_eval_seconds_in_learn": round(telemetry.eval_seconds, 3),
        "periodic_checkpoint_save_seconds": round(telemetry.checkpoint_seconds, 3),
        "training_only_seconds": round(
            max(0.0, learn_seconds - telemetry.eval_seconds - telemetry.checkpoint_seconds), 3
        ),
        "agent_steps_per_sec_training_only": (
            round(actual / max(1e-9, learn_seconds - telemetry.eval_seconds - telemetry.checkpoint_seconds), 3)
            if actual
            else None
        ),
        "agent_steps_per_sec_learn_including_periodic_eval": (
            round(actual / learn_seconds, 3) if actual and learn_seconds else None
        ),
    }
    memory = {
        "host": host_memory,
        "process_peak_sampled_rss_mib": round(peak_rss / 1024**2, 1) if peak_rss else None,
        "process_peak_wset_mib_os": _peak_wset_mib(),
        "sampling_scope": (
            "MemorySampler samples process RSS every 0.25 s from immediately before "
            "model.learn through final reload evaluation; peak_wset is the OS process peak."
        ),
        "samples": int(sampler.samples),
        "replay_estimate": replay,
        "memory_headroom": headroom,
    }
    manifest = {
        "schema_version": 1,
        "run_id": config["run"]["run_id"],
        "condition": config["condition"],
        "profile": config["profile"],
        "algorithm": config["algo"],
        "seed": int(args.seed),
        "status": status,
        "error": error,
        "command": [sys.executable, "-m", "rl_a2.train", *sys.argv[1:]],
        "cwd": os.getcwd(),
        "project_root": str(PROJECT_ROOT),
        "started_at": started_at,
        "started_epoch": started_epoch,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "finished_epoch": time.time(),
        "run_wall_seconds": round(time.perf_counter() - run_start_perf, 3),
        "source_digest": source_digest(),
        "requested_timesteps": requested,
        "actual_timesteps": actual,
        "rollout_overshoot": actual - requested,
        "total_optimizer_updates": int(telemetry.optimizer_steps),
        "versions": versions,
        "env_description": env_description,
        "memory": memory,
        "timing": timing,
        "probe": probe_meta,
        "final_eval": final_eval["summary"],
        "final_clips": [
            row["video_path"] for row in final_eval["episodes"] if row.get("video_path")
        ],
        "training_summary": summary,
        "artifact_checks": checks,
        "paths": {
            "config": str(run_dir / "config_resolved.json"),
            "metrics": str(run_dir / "metrics"),
            "plots": str(run_dir / "plots"),
            "checkpoints": str(run_dir / "checkpoints"),
            "clips": str(run_dir / "clips"),
            "train_log": str(run_dir / "logs" / "train.log"),
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    if status == "completed" and not all(checks.values()):
        missing = [name for name, ok in checks.items() if not ok]
        logger.error("missing or invalid artifacts: %s", missing)
        raise SystemExit(f"Training completed but artifact checks failed: {missing}")
    logger.info("status=%s artifacts_ok=%s run_dir=%s", status, checks, run_dir)
    if status == "failed":
        raise SystemExit(f"Training failed: {error}\nSee {run_dir / 'run_manifest.json'}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, help="e.g. dqn_baseline, ppo_clip_010")
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default="artifacts/runs")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Deliberate override for debugging; recorded in resolved config",
    )
    parser.add_argument("--device", default=None, help="cpu or cuda; default keeps resolved config")
    parser.add_argument("--no-video", action="store_true", help="Skip the final gameplay clip")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
