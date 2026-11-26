"""Deterministic DQN probe set for mean max online Q.

The probe set is generated from a fixed seed and a fixed, documented action
cycle through the exact evaluation preprocessing chain. It is independent of
the training seed, model weights, and runtime. The SHA-256 is over the canonical
uint8 observation bytes plus shape/dtype, so a reviewer can regenerate or
compare it without relying on file-container metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .envs import make_env


def probe_fingerprint(obs: np.ndarray) -> str:
    arr = np.ascontiguousarray(obs, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(str(tuple(arr.shape)).encode("ascii"))
    digest.update(b"|")
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _probe_filename(config: dict[str, Any]) -> str:
    probe = config["probe"]
    cycle = "-".join(str(int(a)) for a in probe["action_cycle"])
    return (
        f"probe_breakout_seed{int(probe['seed'])}_n{int(probe['count'])}"
        f"_warm{int(probe['warmup_steps'])}_stride{int(probe['step_stride'])}"
        f"_actions{cycle}.npz"
    )


def generate_probe(config: dict[str, Any], path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    probe = config["probe"]
    count = int(probe["count"])
    warmup = int(probe["warmup_steps"])
    stride = int(probe["step_stride"])
    cycle = [int(a) for a in probe["action_cycle"]]
    seed = int(probe["seed"])

    env = make_env(config, training=False, with_monitor=False)
    observations: list[np.ndarray] = []
    actions_taken: list[int] = []
    resets = 0
    try:
        obs, _ = env.reset(seed=seed)
        step = 0
        # Generous bound; the action cycle is not expected to end an episode
        # before 32 observations, but a reset is handled deterministically.
        max_steps = warmup + count * stride + 5000
        while len(observations) < count and step < max_steps:
            if step >= warmup and (step - warmup) % stride == 0:
                observations.append(np.asarray(obs, dtype=np.uint8).copy())
            action = cycle[step % len(cycle)]
            actions_taken.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            step += 1
            if (terminated or truncated) and len(observations) < count:
                resets += 1
                obs, _ = env.reset(seed=seed + 10_000 + resets)
                step = 0
    finally:
        env.close()

    if len(observations) != count:
        raise RuntimeError(
            f"Could only collect {len(observations)}/{count} probe observations; "
            "increase the step bound or adjust the action cycle"
        )
    obs_array = np.stack(observations, axis=0).astype(np.uint8, copy=False)
    meta: dict[str, Any] = {
        "schema_version": 1,
        "env_id": config["env"]["id"],
        "seed": seed,
        "count": count,
        "warmup_steps": warmup,
        "step_stride": stride,
        "action_cycle": cycle,
        "action_cycle_rule": "action = action_cycle[step % len(action_cycle)]",
        "reset_rule": "after an episode ends early, reset(seed + 10000 + reset_count) and restart step counter",
        "generation_resets": resets,
        "shape": list(obs_array.shape),
        "dtype": str(obs_array.dtype),
        "sha256": probe_fingerprint(obs_array),
        "wrapper_chain": config["wrappers"]["evaluation_chain"],
        "actions_taken_prefix": actions_taken[: min(len(actions_taken), 200)],
        "note": "Fixed probes are used for every checkpoint and ablation of this algorithm.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obs=obs_array)
    with (path.with_suffix(".json")).open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    return obs_array, meta


def load_probe(path: Path, meta: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        obs = np.asarray(data["obs"], dtype=np.uint8)
    meta_path = path.with_suffix(".json")
    if meta is not None:
        expected = meta
    elif meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as handle:
            expected = json.load(handle)
    else:
        raise FileNotFoundError(meta_path)
    actual_hash = probe_fingerprint(obs)
    if expected.get("sha256") != actual_hash:
        raise ValueError(
            f"Probe hash mismatch for {path}: metadata {expected.get('sha256')} != {actual_hash}"
        )
    if list(obs.shape) != list(expected.get("shape", [])):
        raise ValueError(f"Probe shape mismatch: {obs.shape} != {expected.get('shape')}")
    return obs, expected


def ensure_probe(config: dict[str, Any], shared_dir: Path) -> tuple[np.ndarray, dict[str, Any], Path]:
    """Load the shared probe or generate it once, then verify its hash."""
    path = shared_dir / _probe_filename(config)
    if path.is_file() and path.with_suffix(".json").is_file():
        obs, meta = load_probe(path)
        expected = config["probe"]
        if (
            int(meta["seed"]) != int(expected["seed"])
            or int(meta["count"]) != int(expected["count"])
            or [int(a) for a in meta["action_cycle"]] != [int(a) for a in expected["action_cycle"]]
        ):
            raise ValueError(f"Existing probe metadata does not match config: {path}")
        return obs, meta, path
    obs, meta = generate_probe(config, path)
    return obs, meta, path


def compute_mean_max_q(model, probe_obs: np.ndarray) -> float:
    """Mean over probes of max_a Q_online(s, a), using the online Q network."""
    import torch

    if not hasattr(model, "q_net"):
        raise AttributeError("mean max Q is defined for DQN models only")
    obs_tensor, _ = model.policy.obs_to_tensor(np.asarray(probe_obs, dtype=np.uint8))
    with torch.no_grad():
        q_values = model.q_net(obs_tensor)
        return float(q_values.max(dim=1).values.mean().item())
