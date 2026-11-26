"""Atari/Breakout wrapper chain used by DQN.

Order (bottom to top):

    ALE/Breakout-v5 (frameskip=1)
      MaxAndSkipExternal(4)          one external action-repeat/max-pool layer
      FireOnResetAndLifeLoss(1)      auto-FIRE lives above that layer
      WarpFrame84x84                 RGB -> grayscale, uint8
      FrameStack(4)                  channel-first (4, 84, 84)
      StackedFramesToArray           LazyFrames -> contiguous uint8 array
      ClipRewardSign                 training only; raw reward kept in info
      RawEpisodeStats                raw return independent of clipping
      Monitor                         SB3/gym episode statistics

Why auto-FIRE is above the skip layer
--------------------------------------
If FIRE were injected below MaxAndSkipExternal, one of the four repeated ALE
frames would be replaced by FIRE. That silently changes the action-repeat
contract and makes the transition's four-frame budget conditional. Here an
auto-FIRE decision is an ordinary external action: MaxAndSkipExternal executes
FIRE four times, advances four ALE frames, and returns one stacked
observation. Therefore every agent timestep is exactly four ALE frames (except
a terminal step that ends early), and the FIRE override is visible in info as
``requested_action`` vs ``executed_action``.

Reset FIRE is executed inside ``reset`` (standard FireReset behavior). That
step advances four ALE frames before the first stacked observation but is not
an agent timestep and is not counted in ``num_timesteps``. Breakout returns
zero reward for that launch; if an emulator ever returned nonzero reward it
would be carried into the first external step's reward so the raw score stays
complete.

After a nonterminal life loss, the next policy action is overridden with FIRE.
This guarantees launch even for a greedy policy that never chooses FIRE. The
action is not injected as an extra hidden timestep: the next external step is
the FIRE step, so frame accounting and the 4-frame stack remain consistent.
"""

from __future__ import annotations

from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

FIRE = 1


class MaxAndSkipExternal(gym.Wrapper):
    """Repeat each action for ``skip`` ALE frames and max-pool the last two.

    The emulator must run with internal frameskip 1. This is the only
    action-repeat layer in the chain.
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        if int(skip) != 4:
            raise ValueError("This project requires exactly one external skip layer of 4")
        self.skip = int(skip)
        if env.observation_space.shape is None:
            raise ValueError("MaxAndSkipExternal needs an image observation")

    def reset(self, **kwargs: Any):
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        last_obs = None
        prev_obs = None
        frames = 0
        # Return max(frame_3, frame_4) for a normal four-frame action, the
        # standard Atari max-and-skip convention. If an episode ends early,
        # max-pool whatever two final frames were actually observed.
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            prev_obs = last_obs
            last_obs = obs
            frames += 1
            total_reward += float(reward)
            if terminated or truncated:
                break
        pooled = np.maximum(prev_obs, last_obs) if prev_obs is not None else last_obs
        info = dict(info)
        info["skip_reward"] = float(total_reward)
        info["frames_advanced"] = int(frames)
        # Preserve the ALE lives counter (and any other underlying fields)
        # from the final internal frame; FireOnResetAndLifeLoss reads it.
        return pooled, float(total_reward), bool(terminated), bool(truncated), info


class FireOnResetAndLifeLoss(gym.Wrapper):
    """Launch the ball on reset and after every nonterminal life loss.

    The wrapper observes ALE's ``lives`` counter. It does not terminate the
    episode on a life loss (no EpisodicLife wrapper); it only schedules one
    FIRE override for the next external action. The policy's requested action
    is still recorded in info for auditing.
    """

    def __init__(
        self,
        env: gym.Env,
        fire_action: int = FIRE,
        auto_fire_on_reset: bool = True,
        auto_fire_after_life_loss: bool = True,
    ):
        super().__init__(env)
        self.fire_action = int(fire_action)
        self.auto_fire_on_reset = bool(auto_fire_on_reset)
        self.auto_fire_after_life_loss = bool(auto_fire_after_life_loss)
        self._prev_lives: int | None = None
        self._fire_next = False
        self._pending_reset_reward = 0.0
        self.reset_fires = 0
        self.life_loss_fires = 0
        self.auto_fire_overrides = 0
        self.life_loss_events = 0

    @property
    def fire_stats(self) -> dict[str, int]:
        return {
            "reset_fires": int(self.reset_fires),
            "life_loss_events": int(self.life_loss_events),
            "life_loss_fires": int(self.life_loss_fires),
            "auto_fire_overrides": int(self.auto_fire_overrides),
            "pending_fire": int(self._fire_next),
        }

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        self._prev_lives = info.get("lives")
        self._fire_next = False
        self._pending_reset_reward = 0.0
        info["auto_fire_reset"] = False
        if self.auto_fire_on_reset:
            obs, reward, terminated, truncated, fire_info = self.env.step(self.fire_action)
            info.update(fire_info)
            self.reset_fires += 1
            self._pending_reset_reward = float(reward)
            info["auto_fire_reset"] = True
            info["auto_fire"] = True
            info["requested_action"] = None
            info["executed_action"] = self.fire_action
            info["reset_fire_reward"] = float(reward)
            if "lives" in info:
                self._prev_lives = info["lives"]
            if terminated or truncated:
                # Breakout does not do this in practice; make the condition
                # explicit rather than silently swallowing it.
                info["reset_fire_ended_episode"] = True
        return obs, info

    def step(self, action: Any):
        requested = int(np.asarray(action).item())
        executed = requested
        auto = False
        if self._fire_next:
            executed = self.fire_action
            auto = True
            self._fire_next = False
            self.auto_fire_overrides += 1
            self.life_loss_fires += 1

        obs, reward, terminated, truncated, info = self.env.step(executed)
        info = dict(info)
        reward = float(reward) + self._pending_reset_reward
        if self._pending_reset_reward:
            info["reset_fire_reward_carried"] = self._pending_reset_reward
            self._pending_reset_reward = 0.0

        info["requested_action"] = requested
        info["executed_action"] = executed
        info["auto_fire"] = auto

        lives = info.get("lives")
        if lives is not None:
            if (
                self.auto_fire_after_life_loss
                and not terminated
                and self._prev_lives is not None
                and int(lives) < int(self._prev_lives)
            ):
                self.life_loss_events += 1
                self._fire_next = True
            self._prev_lives = int(lives)
        return obs, reward, bool(terminated), bool(truncated), info


class WarpFrame(gym.ObservationWrapper):
    """Convert RGB ALE frames to 84x84 grayscale uint8."""

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        if not bool(env.observation_space.shape) or len(env.observation_space.shape) != 3:
            raise ValueError("WarpFrame expects a (H, W, C) image observation")
        self.width = int(width)
        self.height = int(height)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width), dtype=np.uint8
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(np.asarray(observation), cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return np.asarray(resized, dtype=np.uint8)


class StackedFramesToArray(gym.ObservationWrapper):
    """Convert gymnasium's LazyFrames into a contiguous uint8 array."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        shape = env.observation_space.shape
        if shape is None:
            raise ValueError("StackedFramesToArray needs a shaped observation")
        self.observation_space = spaces.Box(low=0, high=255, shape=tuple(shape), dtype=np.uint8)

    def observation(self, observation: Any) -> np.ndarray:
        return np.asarray(observation, dtype=np.uint8)


class ClipRewardSign(gym.Wrapper):
    """Clip reward to sign and preserve the unmodified reward in info."""

    def __init__(self, env: gym.Env, mode: str = "sign"):
        super().__init__(env)
        if mode not in {"sign", "none"}:
            raise ValueError("clip_mode must be 'sign' or 'none'")
        self.mode = mode

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw = float(reward)
        info = dict(info)
        info["raw_reward"] = raw
        clipped = float(np.sign(raw)) if self.mode == "sign" else raw
        info["clipped_reward"] = clipped
        return obs, clipped, terminated, truncated, info


class RawEpisodeStats(gym.Wrapper):
    """Accumulate a raw-return episode summary without changing boundaries.

    The summary is emitted as ``info["raw_episode"]``. It deliberately does not
    use the key ``episode`` so SB3's Monitor can remain the authoritative
    learning-statistics source for the reward stream it sees.
    """

    def __init__(self, env: gym.Env, use_raw_reward: bool = True):
        super().__init__(env)
        self.use_raw_reward = bool(use_raw_reward)
        self._return = 0.0
        self._length = 0

    def reset(self, **kwargs: Any):
        self._return = 0.0
        self._length = 0
        return self.env.reset(**kwargs)

    def _ale_score(self) -> int | None:
        try:
            return int(self.env.unwrapped.ale.getScore())
        except Exception:
            return None

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        raw = float(info.get("raw_reward", reward)) if self.use_raw_reward else float(reward)
        self._return += raw
        self._length += 1
        info["raw_step_reward"] = raw
        if terminated or truncated:
            info["raw_episode"] = {
                "return": float(self._return),
                "length": int(self._length),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "ale_score": self._ale_score(),
            }
        return obs, reward, terminated, truncated, info


__all__ = [
    "ClipRewardSign",
    "FireOnResetAndLifeLoss",
    "MaxAndSkipExternal",
    "RawEpisodeStats",
    "StackedFramesToArray",
    "WarpFrame",
    "FIRE",
]
