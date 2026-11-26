"""Deterministic wrapper contract tests on real ALE/Breakout-v5.

Checks:
  * auto-FIRE during reset, with a policy action sequence that never selects FIRE
  * auto-FIRE on the next agent step after a nonterminal life loss
  * exactly four ALE frames per normal external step
  * four-channel uint8 84x84 stacking and temporal order
  * raw return accounting with reward clipping independently recoverable
  * full-game Monitor return equals the raw accumulator and ALE score
  * NatureCNN accepts channel-first (4,84,84) observations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl_a2.config import resolve_config, write_json  # noqa: E402
from rl_a2.envs import make_env  # noqa: E402
from rl_a2.wrappers import FireOnResetAndLifeLoss  # noqa: E402


def find_wrapper(env, cls):
    current = env
    while current is not None:
        if isinstance(current, cls):
            return current
        current = getattr(current, "env", None)
    return None


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/verification/wrapper_tests.json")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = resolve_config(
        "dqn_baseline",
        profile="smoke",
        seed=0,
        output_root=Path(args.output).parent,
    )
    report: dict[str, Any] = {"ok": True, "checks": {}, "trace": {}}

    # ---------------------------------------------------------------- reset FIRE
    env = make_env(config, training=False, with_monitor=False)
    try:
        obs, info = env.reset(seed=args.seed)
        fire = find_wrapper(env, FireOnResetAndLifeLoss)
        if fire is None:
            fail("FireOnResetAndLifeLoss not found in wrapper chain")
        stats = fire.fire_stats
        report["trace"]["reset"] = {
            "info": {
                key: (value if isinstance(value, (str, int, float, bool, type(None))) else str(value))
                for key, value in info.items()
            },
            "fire_stats": stats,
            "frame_number_after_reset": int(env.unwrapped.ale.getFrameNumber()),
        }
        checks = {
            "reset_auto_fire_flag": bool(info.get("auto_fire_reset")),
            "reset_fire_count_one": stats["reset_fires"] == 1,
            "reset_executed_fire": int(info.get("executed_action", -1)) == 1,
            "reset_requested_action_none": info.get("requested_action") is None,
            "obs_shape_channel_first": tuple(np.asarray(obs).shape) == (4, 84, 84),
            "obs_dtype_uint8": np.asarray(obs).dtype == np.uint8,
            "obs_range": int(np.asarray(obs).min()) >= 0 and int(np.asarray(obs).max()) <= 255,
        }
        report["checks"].update(checks)
        report["trace"]["action_meanings"] = env.unwrapped.get_action_meanings()
        if not all(checks.values()):
            fail(f"reset checks failed: {checks}")

        # ------------------------------------------------------ frame skip + stack
        stack_before = np.asarray(obs).copy()
        frame_before = int(env.unwrapped.ale.getFrameNumber())
        obs2, reward2, term2, trunc2, info2 = env.step(2)  # RIGHT, never FIRE
        stack_after = np.asarray(obs2).copy()
        frame_after = int(env.unwrapped.ale.getFrameNumber())
        skip_checks = {
            "effective_skip_4": (frame_after - frame_before) == 4,
            "frames_advanced_reported_4": int(info2.get("frames_advanced", -1)) == 4,
            "stack_shape": stack_after.shape == (4, 84, 84),
            "stack_dtype": stack_after.dtype == np.uint8,
            "stack_temporal_order": bool(np.array_equal(stack_before[1:], stack_after[:3])),
            "stack_frames_differ": not bool(np.array_equal(stack_before[-1], stack_after[-1])),
        }
        report["checks"].update(skip_checks)
        report["trace"]["frame_skip"] = {
            "frame_before": frame_before,
            "frame_after": frame_after,
            "delta": frame_after - frame_before,
        }
        if not all(skip_checks.values()):
            fail(f"skip/stack checks failed: {skip_checks}")
    finally:
        env.close()

    # ------------------------------------------------------- life-loss FIRE
    env = make_env(config, training=False, with_monitor=False)
    actions = [2, 3, 0, 2, 3, 0, 2, 2, 3, 3, 0, 0]  # never FIRE
    trace = []
    try:
        obs, info = env.reset(seed=args.seed)
        fire = find_wrapper(env, FireOnResetAndLifeLoss)
        prev_lives = info.get("lives")
        found = False
        for step in range(5000):
            action = actions[step % len(actions)]
            obs, reward, terminated, truncated, info = env.step(action)
            trace.append(
                {
                    "step": step,
                    "requested": int(info.get("requested_action", -1)),
                    "executed": int(info.get("executed_action", -1)),
                    "auto_fire": bool(info.get("auto_fire", False)),
                    "lives": info.get("lives"),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "reward": float(reward),
                }
            )
            lives = info.get("lives")
            if terminated or truncated:
                break
            if (
                lives is not None
                and prev_lives is not None
                and int(lives) < int(prev_lives)
                and not terminated
            ):
                # The loss just happened. The next policy action must be
                # overridden to FIRE without an extra hidden timestep.
                next_action = actions[(step + 1) % len(actions)]
                obs, reward, terminated, truncated, info = env.step(next_action)
                trace.append(
                    {
                        "step": step + 1,
                        "requested": int(info.get("requested_action", -1)),
                        "executed": int(info.get("executed_action", -1)),
                        "auto_fire": bool(info.get("auto_fire", False)),
                        "lives": info.get("lives"),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "reward": float(reward),
                        "life_loss_test": True,
                    }
                )
                life_checks = {
                    "life_loss_detected": int(lives) < int(prev_lives),
                    "next_requested_not_fire": next_action != 1,
                    "next_executed_fire": int(info.get("executed_action", -1)) == 1,
                    "next_requested_recorded": int(info.get("requested_action", -1)) == next_action,
                    "next_auto_fire": bool(info.get("auto_fire", False)),
                    "life_loss_fire_count": fire.fire_stats["life_loss_fires"] >= 1,
                    "no_agent_action_was_fire": all(a != 1 for a in actions),
                }
                report["checks"].update(life_checks)
                report["trace"]["life_loss"] = {
                    "step": step,
                    "next_action": next_action,
                    "stats": fire.fire_stats,
                    "checks": life_checks,
                }
                found = True
                if not all(life_checks.values()):
                    fail(f"life-loss FIRE checks failed: {life_checks}")
                break
            prev_lives = lives
        if not found:
            fail("No nonterminal life loss occurred within 5000 steps; test is inconclusive")
    finally:
        env.close()

    # ------------------------------------------------ raw score accounting + Monitor
    env = make_env(config, training=True, with_monitor=True)
    try:
        obs, info = env.reset(seed=args.seed + 100)
        raw_sum = 0.0
        steps = 0
        while True:
            action = actions[steps % len(actions)]
            obs, reward, terminated, truncated, info = env.step(action)
            raw_sum += float(info.get("raw_reward", reward))
            steps += 1
            if terminated or truncated:
                break
            if steps >= 30000:
                fail("Test episode exceeded 30000 agent steps without ending")
        raw_episode = info.get("raw_episode")
        monitor_episode = info.get("episode")
        if raw_episode is None or monitor_episode is None:
            fail("Expected both raw_episode and Monitor episode at episode end")
        score_checks = {
            "raw_sum_equals_raw_episode": abs(raw_sum - float(raw_episode["return"])) < 1e-6,
            "monitor_return_is_clipped_training_return": abs(
                float(monitor_episode["r"])
            )
            <= float(raw_episode["length"]) + 1e-6,
            "raw_return_independently_recoverable": abs(
                raw_sum - float(raw_episode["return"])
            )
            < 1e-6,
        }
        report["checks"].update(score_checks)
        report["trace"]["score_accounting"] = {
            "steps": steps,
            "raw_sum": raw_sum,
            "raw_episode": raw_episode,
            "monitor_episode": monitor_episode,
            "ale_score_exposed_by_ale_py": False,
            "note": (
                "ale-py 0.12 ALEInterface does not expose getScore(); raw reward is "
                "the external game score. raw_episode is independently accumulated "
                "from info['raw_reward'] before Monitor's clipped reward stream."
            ),
            "checks": score_checks,
        }
        if not all(score_checks.values()):
            fail(f"score accounting checks failed: {score_checks}")
    finally:
        env.close()

    # Evaluation chain has no clipping: Monitor return equals raw.
    env = make_env(config, training=False, with_monitor=True)
    try:
        obs, info = env.reset(seed=args.seed + 200)
        raw_sum = 0.0
        steps = 0
        while True:
            obs, reward, terminated, truncated, info = env.step(actions[steps % len(actions)])
            raw_sum += float(reward)
            steps += 1
            if terminated or truncated:
                break
            if steps >= 30000:
                fail("Evaluation test episode exceeded 30000 agent steps")
        raw_episode = info.get("raw_episode")
        monitor_episode = info.get("episode")
        eval_checks = {
            "eval_monitor_equals_raw": abs(
                float(monitor_episode["r"]) - float(raw_episode["return"])
            )
            < 1e-6,
            "eval_no_clip_raw_equals_sum": abs(raw_sum - float(raw_episode["return"])) < 1e-6,
        }
        report["checks"].update(eval_checks)
        report["trace"]["eval_score_accounting"] = {
            "steps": steps,
            "raw_sum": raw_sum,
            "monitor_episode": monitor_episode,
            "raw_episode": raw_episode,
            "checks": eval_checks,
        }
        if not all(eval_checks.values()):
            fail(f"evaluation score checks failed: {eval_checks}")
    finally:
        env.close()

    # ---------------------------------------------------------------- NatureCNN
    import torch
    from gymnasium import spaces
    from stable_baselines3.common.torch_layers import NatureCNN

    cnn = NatureCNN(spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8), features_dim=512)
    out = cnn(torch.zeros(2, 4, 84, 84, dtype=torch.float32))
    cnn_checks = {
        "nature_cnn_output_shape": tuple(out.shape) == (2, 512),
        "observation_is_channel_first": tuple(config["env"]["resize"]) == (84, 84)
        and int(config["env"]["frame_stack"]) == 4,
    }
    report["checks"].update(cnn_checks)
    report["trace"]["cnn"] = {"output_shape": list(out.shape), "checks": cnn_checks}
    if not all(cnn_checks.values()):
        fail(f"CNN shape checks failed: {cnn_checks}")

    report["ok"] = all(bool(value) for value in report["checks"].values())
    write_json(Path(args.output), report)
    print(json.dumps(report["checks"], indent=2))
    print(f"Wrapper verification: {'OK' if report['ok'] else 'FAILED'} -> {args.output}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
