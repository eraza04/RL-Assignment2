"""Configuration loading, validation, and reproducible run naming.

The YAML layering is deliberately small and explicit:

    configs/base_<algo>.yaml
      -> configs/profiles/<profile>.yaml:common
      -> configs/profiles/<profile>.yaml:<algo>
      -> configs/conditions/<condition>.yaml
      -> explicit CLI overrides (seed, output root, device, timesteps)

Only one experiment factor is changed per condition. The resolved file is
written next to the run so a reviewer never has to reconstruct this merge.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
CONDITION_DIR = CONFIG_DIR / "conditions"
PROFILE_DIR = CONFIG_DIR / "profiles"

TOP_KEYS = {
    "schema_version",
    "algo",
    "profile",
    "condition",
    "env",
    "wrappers",
    "model",
    "training",
    "evaluation",
    "probe",
    "metadata",
    "run",
}

ENV_KEYS = {
    "dqn": {
        "id",
        "frameskip",
        "repeat_action_probability",
        "full_action_space",
        "render_mode",
        "max_and_skip",
        "grayscale",
        "resize",
        "frame_stack",
        "fire_on_reset_and_life_loss",
        "fire_action",
        "clip_rewards",
        "clip_mode",
        "obs_dtype",
    },
    "ppo": {
        "id",
        "render_mode",
        "normalize_observations",
        "normalize_rewards",
    },
}

MODEL_KEYS = {
    "dqn": {
        "policy",
        "policy_kwargs",
        "buffer_size",
        "learning_starts",
        "batch_size",
        "gamma",
        "learning_rate",
        "train_freq",
        "gradient_steps",
        "target_update_interval",
        "exploration_fraction",
        "exploration_initial_eps",
        "exploration_final_eps",
        "max_grad_norm",
        "optimize_memory_usage",
        "device",
    },
    "ppo": {
        "policy",
        "policy_kwargs",
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "clip_range_vf",
        "normalize_advantage",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "device",
    },
}

TRAINING_KEYS = {
    "total_timesteps",
    "log_interval",
    "checkpoint_interval",
    "update_log_stride",
    "verbose",
}

EVALUATION_KEYS = {
    "episodes",
    "seed_base",
    "deterministic",
    "record_video",
    "video_max_steps",
    "video_fps",
    "safety_max_steps",
}

PROBE_KEYS = {"seed", "count", "warmup_steps", "action_cycle", "step_stride"}
RUN_KEYS = {"run_id", "seed", "output_root", "run_dir"}


class ConfigError(ValueError):
    """A configuration is incomplete or inconsistent."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a mapping in {path}, got {type(data).__name__}")
    return data


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; lists and scalars replace the old value."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def list_conditions() -> list[str]:
    return sorted(path.stem for path in CONDITION_DIR.glob("*.yaml"))


def load_condition(condition: str) -> dict[str, Any]:
    return _load_yaml(CONDITION_DIR / f"{condition}.yaml")


def load_profile(profile: str) -> dict[str, Any]:
    return _load_yaml(PROFILE_DIR / f"{profile}.yaml")


def condition_metadata(condition: str) -> dict[str, Any]:
    return load_condition(condition).get("metadata", {})


def conditions_sorted() -> list[str]:
    names = list_conditions()
    return sorted(names, key=lambda name: (condition_metadata(name).get("order", 10**9), name))


def _apply_overrides(
    config: dict[str, Any],
    *,
    condition: str,
    profile: str,
    seed: int,
    output_root: str | Path,
    total_timesteps: int | None,
    device: str | None,
) -> None:
    config["condition"] = condition
    config["profile"] = profile
    if total_timesteps is not None:
        config["training"]["total_timesteps"] = int(total_timesteps)
    if device is not None:
        config["model"]["device"] = str(device)
    root = Path(output_root).expanduser().resolve()
    run_id = f"{condition}_seed{int(seed)}_{profile}"
    config["run"] = {
        "run_id": run_id,
        "seed": int(seed),
        "output_root": str(root),
        "run_dir": str(root / run_id),
    }


def resolve_config(
    condition: str,
    profile: str = "full",
    seed: int = 0,
    output_root: str | Path = "artifacts/runs",
    total_timesteps: int | None = None,
    device: str | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    condition_cfg = load_condition(condition)
    algo = condition_cfg.get("algo")
    if algo not in {"dqn", "ppo"}:
        raise ConfigError(f"{condition}: algo must be 'dqn' or 'ppo', got {algo!r}")
    base = _load_yaml(CONFIG_DIR / f"base_{algo}.yaml")
    profile_cfg = load_profile(profile)

    config = deep_merge(base, profile_cfg.get("common", {}))
    config = deep_merge(config, profile_cfg.get(algo, {}))
    config = deep_merge(config, condition_cfg)
    config["profile"] = profile
    config["algo"] = algo
    _apply_overrides(
        config,
        condition=condition,
        profile=profile,
        seed=seed,
        output_root=output_root,
        total_timesteps=total_timesteps,
        device=device,
    )
    if validate:
        validate_config(config)
    return config


def _check_keys(section: str, data: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"{section}: unknown keys {unknown}")


def replay_estimate(config: dict[str, Any]) -> dict[str, Any]:
    """Estimate the arrays allocated by SB3 2.7.0's default ``ReplayBuffer``.

    Capacity is fixed across all full DQN conditions. SB3 allocates both
    observations and next observations (uint8), an int64 action, and **float32**
    reward, done and timeout arrays. The two mask arrays are four bytes each,
    not one; this matches a direct inspection of a smoke-sized buffer
    (112,936,000 bytes at capacity 2,000 = 56,468 bytes/transition).
    """
    if config["algo"] != "dqn":
        return {
            "applicable": False,
            "reason": "PPO uses an on-policy RolloutBuffer, not a replay buffer",
        }
    env = config["env"]
    model = config["model"]
    channels = int(env["frame_stack"])
    width, height = (int(v) for v in env["resize"])
    obs_bytes = channels * width * height  # uint8
    action_bytes = 8  # gymnasium Discrete action dtype (int64)
    reward_bytes = 4  # float32
    done_bytes = 4  # SB3 stores done as float32
    timeout_bytes = 4  # SB3 stores timeout as float32
    # DQN's n_steps defaults to 1; discounts are not allocated unless n-step > 1.
    per_transition = 2 * obs_bytes + action_bytes + reward_bytes + done_bytes + timeout_bytes
    capacity = int(model["buffer_size"])
    total_bytes = per_transition * capacity
    return {
        "applicable": True,
        "capacity": capacity,
        "obs_shape": [channels, width, height],
        "obs_dtype": "uint8",
        "next_obs_included": True,
        "bytes_per_transition": per_transition,
        "formula": (
            "2*obs_bytes(uint8) + action_int64 + reward_float32 + "
            "done_float32 + timeout_float32 (no n-step discounts for n_steps=1)"
        ),
        "total_bytes": total_bytes,
        "total_mib": round(total_bytes / (1024**2), 2),
        "total_gib": round(total_bytes / (1024**3), 3),
        "note": (
            "SB3 2.7.0 ReplayBuffer array bytes only; Python/NumPy object "
            "overhead, allocator slack and the rest of the process are excluded."
        ),
    }


# Conservative non-replay process footprint. The full 6,000-step baseline smoke
# with video and three reload evaluations peaked at 1,504.6 MiB RSS with a
# 107.7 MiB replay, i.e. 1,396.9 MiB of non-replay footprint; round up to
# 1,400.0 MiB. With the full 269.26 MiB replay and 25% slack, a launch requires
# about 2,086.6 MiB *currently available* RAM. A historical high-water sample
# is never substituted for a fresh reading at launch time.
MEASURED_PROCESS_BASELINE_MIB_DQN = 1400.0
MEASURED_PROCESS_BASELINE_MIB_PPO = 500.0
HEADROOM_FRACTION = 0.25


def memory_headroom(
    config: dict[str, Any], available_bytes: int | None = None
) -> dict[str, Any]:
    """Compare replay + measured process footprint against *available* RAM.

    The old check compared the replay estimate with total RAM, which cannot
    establish launch headroom. This computes a conservative resident estimate
    and the free RAM required to leave ``HEADROOM_FRACTION`` slack. Callers
    must supply a current reading for full-profile launch gates; a marginal
    pass is not evidence of sustained headroom under queue conditions.
    """
    estimate = replay_estimate(config)
    if available_bytes is None:
        try:
            import psutil

            available_bytes = int(psutil.virtual_memory().available)
        except Exception:  # pragma: no cover - unusual hosts only
            available_bytes = 0
    replay_bytes = int(estimate["total_bytes"]) if estimate["applicable"] else 0
    baseline_mib = (
        MEASURED_PROCESS_BASELINE_MIB_DQN
        if config["algo"] == "dqn"
        else MEASURED_PROCESS_BASELINE_MIB_PPO
    )
    baseline_bytes = int(baseline_mib * 1024**2)
    projected_peak_bytes = replay_bytes + baseline_bytes
    required_bytes = int(projected_peak_bytes * (1.0 + HEADROOM_FRACTION))
    available_mib = round(available_bytes / 1024**2, 1) if available_bytes else 0.0
    deficit_bytes = max(0, required_bytes - int(available_bytes))
    ok = bool(available_bytes) and deficit_bytes == 0
    return {
        "applicable": True,
        "available_bytes": int(available_bytes),
        "available_mib": available_mib,
        "replay_bytes": replay_bytes,
        "replay_mib": round(replay_bytes / 1024**2, 2),
        "measured_non_replay_process_mib": baseline_mib,
        "projected_peak_mib": round(projected_peak_bytes / 1024**2, 1),
        "headroom_fraction": HEADROOM_FRACTION,
        "required_free_mib": round(required_bytes / 1024**2, 1),
        "deficit_mib": round(deficit_bytes / 1024**2, 1),
        "ok": ok,
        "verdict": (
            "sufficient free RAM for the selected replay capacity"
            if ok
            else "insufficient verified free RAM; do not launch this profile on this host"
        ),
        "source": (
            "ReplayBuffer array bytes from replay_estimate(); non-replay footprint "
            "measured as smoke peak RSS minus smoke replay bytes; 25% slack required."
        ),
    }


def validate_config(config: dict[str, Any]) -> None:
    _check_keys("config", config, TOP_KEYS)
    algo = config["algo"]
    if algo not in {"dqn", "ppo"}:
        raise ConfigError(f"Unsupported algo: {algo}")
    if config.get("profile") not in {"full", "smoke"}:
        raise ConfigError("profile must be 'full' or 'smoke'")

    for section, allowed in (
        ("env", ENV_KEYS[algo]),
        ("model", MODEL_KEYS[algo]),
        ("training", TRAINING_KEYS),
        ("evaluation", EVALUATION_KEYS),
        ("run", RUN_KEYS),
    ):
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"{section} must be a mapping")
        _check_keys(section, config[section], allowed)

    if algo == "dqn":
        if not isinstance(config.get("probe"), dict):
            raise ConfigError("DQN requires a probe section")
        _check_keys("probe", config["probe"], PROBE_KEYS)

    training = config["training"]
    if int(training["total_timesteps"]) <= 0:
        raise ConfigError("training.total_timesteps must be positive")
    for key in ("log_interval", "checkpoint_interval", "update_log_stride"):
        if int(training[key]) <= 0:
            raise ConfigError(f"training.{key} must be positive")

    evaluation = config["evaluation"]
    if int(evaluation["episodes"]) < 1:
        raise ConfigError("evaluation.episodes must be >= 1")
    if int(evaluation["video_fps"]) <= 0 or int(evaluation["video_max_steps"]) <= 0:
        raise ConfigError("evaluation video settings must be positive")
    cap = evaluation.get("safety_max_steps")
    if cap is not None and int(cap) <= 0:
        raise ConfigError("evaluation.safety_max_steps must be null or positive")

    if algo == "dqn":
        env = config["env"]
        if env["id"] != "ALE/Breakout-v5":
            raise ConfigError("DQN condition must use ALE/Breakout-v5")
        if int(env["frameskip"]) != 1:
            raise ConfigError("ALE frameskip must be 1; external skip handles repeat")
        if int(env["max_and_skip"]) != 4:
            raise ConfigError("Exactly one max-and-skip layer of 4 is required")
        if int(env["frame_stack"]) != 4:
            raise ConfigError("Frame stack must be 4")
        if list(env["resize"]) != [84, 84]:
            raise ConfigError("Observation must be resized to 84x84")
        if not bool(env["grayscale"]):
            raise ConfigError("DQN observations must be grayscale")
        if str(env["obs_dtype"]) != "uint8":
            raise ConfigError("Stacked frames must remain uint8")
        if not bool(env["fire_on_reset_and_life_loss"]):
            raise ConfigError("Auto-FIRE at reset and nonterminal life loss is required")
        model = config["model"]
        if model["policy"] != "CnnPolicy":
            raise ConfigError("DQN must use CnnPolicy")
        if int(model["buffer_size"]) <= 0:
            raise ConfigError("buffer_size must be positive")
        if int(model["target_update_interval"]) <= 0:
            raise ConfigError("target_update_interval must be positive")
        if not 0.0 < float(model["exploration_fraction"]) <= 1.0:
            raise ConfigError("exploration_fraction must be in (0, 1]")
        probe = config["probe"]
        if int(probe["count"]) < 1 or int(probe["warmup_steps"]) < 0:
            raise ConfigError("probe count/warmup invalid")
        if not probe["action_cycle"]:
            raise ConfigError("probe.action_cycle must be non-empty")
    else:
        env = config["env"]
        if env["id"] != "BipedalWalker-v3":
            raise ConfigError("PPO condition must use BipedalWalker-v3")
        model = config["model"]
        if model["policy"] != "MlpPolicy":
            raise ConfigError("PPO must use MlpPolicy")
        if int(model["n_steps"]) <= 0 or int(model["batch_size"]) <= 0:
            raise ConfigError("PPO n_steps/batch_size must be positive")
        if int(model["batch_size"]) > int(model["n_steps"]):
            raise ConfigError("PPO batch_size must not exceed n_steps for this setup")

    # The replay arrays alone must fit in currently available RAM; total RAM is
    # not evidence of headroom. Full-profile launch also calls memory_headroom()
    # with the measured process baseline and explicit slack (see train.py).
    estimate = replay_estimate(config)
    if estimate.get("applicable"):
        available_bytes = 0
        try:
            import psutil

            available_bytes = int(psutil.virtual_memory().available)
        except Exception:  # pragma: no cover - only used on unusual hosts
            pass
        if available_bytes and estimate["total_bytes"] > available_bytes:
            raise ConfigError(
                "Replay allocation estimate exceeds currently available RAM: "
                f"{estimate['total_mib']} MiB needed, {available_bytes / 1024**2:.0f} MiB available"
            )


def json_ready(config: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe deep copy and reject non-finite numeric values."""
    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): walk(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [walk(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigError(f"Non-finite value in config: {value}")
        return value

    return walk(config)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ablation_diff(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    ignored_paths: tuple[tuple[str, ...], ...] = (
        ("run",),
        ("metadata",),
        ("condition",),
    ),
) -> list[dict[str, Any]]:
    """Return leaf differences between two resolved configs.

    Run identity, output paths, and metadata are intentionally ignored: those
    are bookkeeping differences, not experiment factors.
    """
    diffs: list[dict[str, Any]] = []

    def walk(left: Any, right: Any, path: tuple[str, ...]) -> None:
        if any(path[: len(ignored)] == ignored for ignored in ignored_paths):
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                walk(left.get(key), right.get(key), path + (key,))
            return
        if left != right:
            diffs.append(
                {
                    "path": ".".join(path),
                    "baseline": right,
                    "candidate": left,
                }
            )

    walk(candidate, baseline, ())
    return diffs
