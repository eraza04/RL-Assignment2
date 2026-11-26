"""Checkpoint reload and full-game evaluation.

Evaluation deliberately uses a fresh environment, no reward clipping, no
EpisodicLife wrapper, and the environment's native episode limits. The primary
score is Monitor's raw episode return; RawEpisodeStats supplies an independent
raw accumulator and (for ALE) the emulator score for cross-checking.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import read_json
from .envs import make_env
from .probes import compute_mean_max_q, load_probe

EVAL_EPISODE_FIELDS = [
    "checkpoint",
    "checkpoint_step",
    "is_final",
    "seed",
    "episode_index",
    "return_raw",
    "monitor_return",
    "length",
    "terminated",
    "truncated",
    "safety_cap_hit",
    "ale_score",
    "video_path",
    "video_frames",
    "model_path",
]

EVAL_SUMMARY_FIELDS = [
    "checkpoint",
    "checkpoint_step",
    "is_final",
    "episodes",
    "mean_return_raw",
    "std_return_raw",
    "min_return_raw",
    "max_return_raw",
    "mean_length",
    "mean_max_q",
    "probe_sha256",
    "model_path",
]


def load_model(path: Path, algo: str, device: str = "cpu"):
    from stable_baselines3 import DQN, PPO

    cls = DQN if algo == "dqn" else PPO
    model = cls.load(str(path), device=device)
    model.policy.set_training_mode(False)
    return model


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> Path:
    """Write MP4 via imageio-ffmpeg; fall back to GIF if the codec is unusable."""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(
            path,
            frames,
            fps=int(fps),
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        if path.stat().st_size < 1024:
            raise RuntimeError("video file is unexpectedly small")
        return path
    except Exception:
        gif_path = path.with_suffix(".gif")
        imageio.mimsave(gif_path, frames, fps=int(fps))
        return gif_path


def _append_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        for row in rows:
            clean = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, (np.integer,)):
                    value = int(value)
                elif isinstance(value, (np.floating,)):
                    value = float(value)
                elif isinstance(value, np.bool_):
                    value = bool(value)
                if value is None:
                    value = ""
                clean[field] = value
            writer.writerow(clean)


def evaluate_checkpoint(
    model_path: Path,
    *,
    checkpoint_name: str,
    checkpoint_step: int,
    is_final: bool,
    config: dict[str, Any],
    run_dir: Path,
    probe_obs: np.ndarray | None,
    probe_meta: dict[str, Any] | None,
    episodes_override: int | None = None,
    seed_base_override: int | None = None,
    record_video: bool | None = None,
    video_dir: Path | None = None,
    csv_dir: Path | None = None,
    safety_max_steps: int | None | object = ...,
) -> dict[str, Any]:
    """Reload a checkpoint, run full games, and persist raw per-episode rows."""
    eval_cfg = config["evaluation"]
    episodes = int(episodes_override if episodes_override is not None else eval_cfg["episodes"])
    seed_base = int(seed_base_override if seed_base_override is not None else eval_cfg["seed_base"])
    do_video = bool(eval_cfg["record_video"] if record_video is None else record_video)
    video_dir = video_dir or (run_dir / "clips")
    csv_dir = csv_dir or (run_dir / "metrics")
    if safety_max_steps is ...:
        safety_max_steps = eval_cfg.get("safety_max_steps")
    if safety_max_steps is not None:
        safety_max_steps = int(safety_max_steps)

    model = load_model(model_path, config["algo"], device=str(config["model"]["device"]))
    mean_max_q: float | None = None
    probe_hash: str | None = None
    if config["algo"] == "dqn":
        if probe_obs is None or probe_meta is None:
            raise ValueError("DQN evaluation requires the fixed probe set")
        probe_hash = probe_meta["sha256"]
        mean_max_q = compute_mean_max_q(model, probe_obs)

    episode_rows: list[dict[str, Any]] = []
    returns: list[float] = []
    lengths: list[int] = []

    for episode_index in range(episodes):
        seed = seed_base + episode_index
        record_this = do_video and episode_index == 0
        render_mode = "rgb_array" if record_this else None
        env = make_env(config, training=False, render_mode=render_mode, with_monitor=True)
        frames: list[np.ndarray] = []
        try:
            obs, reset_info = env.reset(seed=seed)
            steps = 0
            sum_reward = 0.0
            terminated = False
            truncated = False
            safety_cap_hit = False
            info: dict[str, Any] = dict(reset_info)
            while True:
                action, _ = model.predict(
                    np.asarray(obs), deterministic=bool(eval_cfg["deterministic"])
                )
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                sum_reward += float(reward)
                if record_this and len(frames) < int(eval_cfg["video_max_steps"]):
                    frame = env.render()
                    if frame is not None:
                        frames.append(np.asarray(frame))
                if terminated or truncated:
                    break
                if safety_max_steps is not None and steps >= safety_max_steps:
                    safety_cap_hit = True
                    truncated = True
                    break
        finally:
            env.close()

        raw_episode = info.get("raw_episode") or {}
        monitor_episode = info.get("episode") or {}
        raw_return = raw_episode.get("return", monitor_episode.get("r", sum_reward))
        monitor_return = monitor_episode.get("r", raw_return)
        length = raw_episode.get("length", monitor_episode.get("l", steps))
        ale_score = raw_episode.get("ale_score")
        video_path = ""
        video_frames = 0
        if record_this and frames:
            target = video_dir / f"{config['run']['run_id']}_{checkpoint_name}_ep{episode_index}.mp4"
            written = _write_video(target, frames, int(eval_cfg["video_fps"]))
            video_path = str(written)
            video_frames = len(frames)

        row = {
            "checkpoint": checkpoint_name,
            "checkpoint_step": int(checkpoint_step),
            "is_final": bool(is_final),
            "seed": seed,
            "episode_index": episode_index,
            "return_raw": float(raw_return),
            "monitor_return": float(monitor_return),
            "length": int(length),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "safety_cap_hit": bool(safety_cap_hit),
            "ale_score": ale_score,
            "video_path": video_path,
            "video_frames": int(video_frames),
            "model_path": str(model_path),
        }
        episode_rows.append(row)
        returns.append(float(raw_return))
        lengths.append(int(length))

    summary = {
        "checkpoint": checkpoint_name,
        "checkpoint_step": int(checkpoint_step),
        "is_final": bool(is_final),
        "episodes": episodes,
        "mean_return_raw": float(np.mean(returns)) if returns else None,
        "std_return_raw": float(np.std(returns)) if returns else None,
        "min_return_raw": float(np.min(returns)) if returns else None,
        "max_return_raw": float(np.max(returns)) if returns else None,
        "mean_length": float(np.mean(lengths)) if lengths else None,
        "mean_max_q": mean_max_q,
        "probe_sha256": probe_hash,
        "model_path": str(model_path),
    }
    _append_csv(csv_dir / "eval_episodes.csv", EVAL_EPISODE_FIELDS, episode_rows)
    _append_csv(csv_dir / "eval_summary.csv", EVAL_SUMMARY_FIELDS, [summary])
    return {"summary": summary, "episodes": episode_rows, "mean_max_q": mean_max_q}


def evaluate_from_config(
    run_dir: Path,
    checkpoint: Path,
    *,
    output_dir: Path,
    episodes: int | None = None,
    seed_base: int | None = None,
    record_video: bool = True,
    safety_max_steps: int | None | object = ...,
) -> dict[str, Any]:
    """Standalone reload/evaluation entry point used by ``evaluate.py``."""
    config = read_json(run_dir / "config_resolved.json")
    probe_obs = None
    probe_meta = None
    if config["algo"] == "dqn":
        probe_file = run_dir / "probes" / "probe_obs.npz"
        probe_obs, probe_meta = load_probe(probe_file, read_json(probe_file.with_suffix(".json")))
    step = 0
    name = checkpoint.stem
    if (run_dir / "metrics" / "eval_summary.csv").is_file():
        with (run_dir / "metrics" / "eval_summary.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if Path(row.get("model_path", "")).resolve() == checkpoint.resolve():
                    step = int(float(row.get("checkpoint_step") or 0))
                    name = row.get("checkpoint") or name
    if step == 0:
        match = re.search(r"ckpt_(\d+)", checkpoint.stem)
        if match:
            step = int(match.group(1))
    result = evaluate_checkpoint(
        checkpoint,
        checkpoint_name=name,
        checkpoint_step=step,
        is_final=name == "final",
        config=config,
        run_dir=output_dir,
        probe_obs=probe_obs,
        probe_meta=probe_meta,
        episodes_override=episodes,
        seed_base_override=seed_base,
        record_video=record_video,
        video_dir=output_dir / "clips",
        csv_dir=output_dir / "metrics",
        safety_max_steps=safety_max_steps,
    )
    manifest = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "command": "rl_a2.evaluate",
        "result": result["summary"],
        "episodes": result["episodes"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "eval_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reload a checkpoint and run full-game evaluation")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--checkpoint",
        default="final",
        help="'final', 'latest', a ckpt_*.zip filename, or a path",
    )
    parser.add_argument("--output-dir", default=None, help="Defaults to RUN_DIR/eval_reload")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--safety-max-steps",
        type=int,
        default=None,
        help="Optional debug cap; omitted means environment-native episode limits only",
    )
    return parser.parse_args(argv)


def resolve_checkpoint(run_dir: Path, value: str) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    if value == "final":
        return checkpoint_dir / "final_model.zip"
    if value == "latest":
        candidates = sorted(checkpoint_dir.glob("ckpt_*.zip"))
        if not candidates:
            raise FileNotFoundError(f"No periodic checkpoints in {checkpoint_dir}")
        return candidates[-1]
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    candidate = checkpoint_dir / value
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Checkpoint not found: {value} (run dir {run_dir})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    checkpoint = resolve_checkpoint(run_dir, args.checkpoint)
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else run_dir / "eval_reload"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"Evaluation output directory is non-empty: {output_dir}\n"
            "Pass a fresh --output-dir to avoid overwriting evidence."
        )
    result = evaluate_from_config(
        run_dir,
        checkpoint,
        output_dir=output_dir,
        episodes=args.episodes,
        seed_base=args.seed_base,
        record_video=not args.no_video,
        safety_max_steps=args.safety_max_steps if args.safety_max_steps is not None else ...,
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
