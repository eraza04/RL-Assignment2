"""Environment factories shared by training, evaluation, and probe generation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3.common.monitor import Monitor

try:  # ale_py registers the ALE/ namespace when imported.
    import ale_py

    gym.register_envs(ale_py)
except Exception:  # pragma: no cover - import error is reported by preflight
    ale_py = None

from .wrappers import (
    ClipRewardSign,
    FireOnResetAndLifeLoss,
    MaxAndSkipExternal,
    RawEpisodeStats,
    StackedFramesToArray,
    WarpFrame,
)


def _breakout(config: dict[str, Any], *, training: bool, render_mode: str | None, with_monitor: bool):
    env_cfg = config["env"]
    mode = render_mode if render_mode is not None else env_cfg.get("render_mode")
    env = gym.make(
        env_cfg["id"],
        frameskip=int(env_cfg["frameskip"]),
        repeat_action_probability=float(env_cfg["repeat_action_probability"]),
        full_action_space=bool(env_cfg["full_action_space"]),
        render_mode=mode,
    )
    env = MaxAndSkipExternal(env, skip=int(env_cfg["max_and_skip"]))
    env = FireOnResetAndLifeLoss(
        env,
        fire_action=int(env_cfg["fire_action"]),
        auto_fire_on_reset=bool(env_cfg["fire_on_reset_and_life_loss"]),
        auto_fire_after_life_loss=bool(env_cfg["fire_on_reset_and_life_loss"]),
    )
    env = WarpFrame(env, width=84, height=84)
    env = FrameStackObservation(env, stack_size=int(env_cfg["frame_stack"]))
    env = StackedFramesToArray(env)
    if training and bool(env_cfg["clip_rewards"]):
        env = ClipRewardSign(env, mode=str(env_cfg["clip_mode"]))
    env = RawEpisodeStats(env, use_raw_reward=True)
    if with_monitor:
        env = Monitor(env, filename=None, allow_early_resets=True)
    return env


def _walker(config: dict[str, Any], *, training: bool, render_mode: str | None, with_monitor: bool):
    env_cfg = config["env"]
    mode = render_mode if render_mode is not None else env_cfg.get("render_mode")
    env = gym.make(env_cfg["id"], render_mode=mode)
    # No observation/reward normalization: PPO sees the native 24 values and
    # native reward scale. RawEpisodeStats adds a non-destructive summary.
    env = RawEpisodeStats(env, use_raw_reward=False)
    if with_monitor:
        env = Monitor(env, filename=None, allow_early_resets=True)
    return env


def make_env(
    config: dict[str, Any],
    *,
    training: bool,
    render_mode: str | None = None,
    with_monitor: bool = True,
):
    """Create a single (non-vectorized) environment."""
    algo = config["algo"]
    if algo == "dqn":
        return _breakout(
            config, training=training, render_mode=render_mode, with_monitor=with_monitor
        )
    if algo == "ppo":
        return _walker(
            config, training=training, render_mode=render_mode, with_monitor=with_monitor
        )
    raise ValueError(f"Unsupported algo: {algo}")


def make_vec_env(config: dict[str, Any], seed: int):
    """Single-environment DummyVecEnv for SB3's DQN and PPO implementations."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv(
        [
            lambda: make_env(
                config,
                training=True,
                render_mode=None,
                with_monitor=True,
            )
        ]
    )
    venv.seed(int(seed))
    return venv


def action_meanings(env: gym.Env) -> list[str] | None:
    try:
        return list(env.unwrapped.get_action_meanings())
    except Exception:
        try:
            return list(env.unwrapped.unwrapped.get_action_meanings())
        except Exception:
            return None


def rom_name(env: gym.Env) -> str | None:
    # ale-py 0.12's Python ALEInterface no longer exposes getROMName(); the
    # Gymnasium ALE env keeps the game id in _game.
    for attribute in ("_game", "game"):
        value = getattr(env.unwrapped, attribute, None)
        if value:
            return str(value)
    return None


def describe_env(config: dict[str, Any]) -> dict[str, Any]:
    """Run one throwaway env and report the exact spaces a reviewer should see."""
    env = make_env(config, training=False, with_monitor=False)
    try:
        obs, info = env.reset(seed=0)
        obs_arr = np.asarray(obs)
        description = {
            "id": config["env"]["id"],
            "observation_space": str(env.observation_space),
            "observation_shape": list(obs_arr.shape),
            "observation_dtype": str(obs_arr.dtype),
            "action_space": str(env.action_space),
            "action_meanings": action_meanings(env),
            "native_max_episode_steps": env.spec.max_episode_steps if env.spec else None,
            "rom_name": rom_name(env),
            "reset_info_keys": sorted(info.keys()),
        }
        if config["algo"] == "dqn":
            description["configured_internal_frameskip"] = int(config["env"]["frameskip"])
            description["external_skip"] = int(config["env"]["max_and_skip"])
            description["effective_agent_frame_skip"] = int(config["env"]["max_and_skip"])
            description["frame_stack"] = int(config["env"]["frame_stack"])
            description["probe_mean"] = float(obs_arr.mean())
        return description
    finally:
        env.close()
