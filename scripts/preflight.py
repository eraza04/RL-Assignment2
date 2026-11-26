"""Environment preflight evidence for the implementation.

Run this before training. It actually resets and steps both native environments,
records versions and host memory, and writes a human-readable plus JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from rl_a2.config import replay_estimate, resolve_config, write_json  # noqa: E402
from rl_a2.envs import describe_env, make_env  # noqa: E402


def versions() -> dict[str, Any]:
    import cv2
    import gymnasium
    import imageio
    import stable_baselines3
    import torch

    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count_logical": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "imageio": imageio.__version__,
    }
    try:
        import ale_py

        out["ale_py"] = ale_py.__version__
    except Exception as exc:
        out["ale_py_error"] = repr(exc)
    try:
        import Box2D

        out["Box2D"] = getattr(Box2D, "__version__", "imported")
    except Exception as exc:
        out["Box2D_error"] = repr(exc)
    try:
        import imageio_ffmpeg

        out["imageio_ffmpeg"] = imageio_ffmpeg.get_ffmpeg_version()
    except Exception as exc:
        out["imageio_ffmpeg_error"] = repr(exc)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="artifacts/preflight")
    args = parser.parse_args()
    out_dir = Path(args.output_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import psutil

    vm = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    dqn_config = resolve_config(
        "dqn_baseline", profile="full", seed=0, output_root=PROJECT_ROOT / "artifacts" / "_preflight"
    )
    ppo_config = resolve_config(
        "ppo_baseline", profile="full", seed=0, output_root=PROJECT_ROOT / "artifacts" / "_preflight"
    )

    # Real ALE reset/step, not just a registration lookup.
    dqn_env = make_env(dqn_config, training=False, with_monitor=False)
    try:
        obs, info = dqn_env.reset(seed=123)
        frame_before = int(dqn_env.unwrapped.ale.getFrameNumber())
        obs2, reward, terminated, truncated, step_info = dqn_env.step(2)
        frame_after = int(dqn_env.unwrapped.ale.getFrameNumber())
        import ale_py.roms

        rom_path = Path(ale_py.roms.get_rom_path("breakout"))
        ale_check = {
            "ok": (frame_after - frame_before) == 4 and rom_path.is_file(),
            "reset_info": {
                "lives": info.get("lives"),
                "frame_number": info.get("frame_number"),
            },
            "step": {
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "requested_action": step_info.get("requested_action"),
                "executed_action": step_info.get("executed_action"),
                "frames_advanced": step_info.get("frames_advanced"),
                "lives": step_info.get("lives"),
                "frame_delta": frame_after - frame_before,
            },
            "observation_shape": list(np.asarray(obs2).shape),
            "observation_dtype": str(np.asarray(obs2).dtype),
            "action_meanings": dqn_env.unwrapped.get_action_meanings(),
            "rom_game_id": getattr(dqn_env.unwrapped, "_game", None),
            "rom_path": str(rom_path),
            "rom_exists": bool(rom_path.is_file()),
            "configured_internal_frameskip": int(dqn_config["env"]["frameskip"]),
        }
    except Exception as exc:
        ale_check = {"ok": False, "error": repr(exc)}
    finally:
        dqn_env.close()

    # Real native Box2D reset/step, including the continuous action space.
    ppo_env = make_env(ppo_config, training=False, with_monitor=False)
    try:
        obs, info = ppo_env.reset(seed=123)
        action = ppo_env.action_space.sample()
        obs2, reward, terminated, truncated, step_info = ppo_env.step(action)
        box2d_check = {
            "ok": True,
            "observation_shape": list(np.asarray(obs2).shape),
            "observation_dtype": str(np.asarray(obs2).dtype),
            "action_shape": list(np.asarray(action).shape),
            "action": [float(x) for x in np.asarray(action).ravel()],
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "native_max_episode_steps": ppo_env.spec.max_episode_steps,
        }
    except Exception as exc:
        box2d_check = {"ok": False, "error": repr(exc)}
    finally:
        ppo_env.close()

    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "project_root": str(PROJECT_ROOT),
        "versions": versions(),
        "host_memory": {
            "total_bytes": int(vm.total),
            "total_mib": round(vm.total / 1024**2, 1),
            "total_gib": round(vm.total / 1024**3, 2),
            "available_bytes": int(vm.available),
            "available_mib": round(vm.available / 1024**2, 1),
            "process_rss_at_preflight_mib": round(mem.rss / 1024**2, 1),
            "process_peak_wset_mib": (
                round(getattr(mem, "peak_wset", 0) / 1024**2, 1)
                if getattr(mem, "peak_wset", None)
                else None
            ),
        },
        "env_checks": {
            "dqn_env_description": describe_env(dqn_config),
            "dqn_ale_reset_step": ale_check,
            "ppo_env_description": describe_env(ppo_config),
            "ppo_box2d_reset_step": box2d_check,
        },
        "replay_estimate_full_dqn": replay_estimate(dqn_config),
        "paths": {
            "wrapper_tests": "artifacts/verification/wrapper_tests.json",
            "ablation_diff": "artifacts/ablation_check/",
            "smoke": "artifacts/smoke/",
        },
    }
    ok = bool(ale_check.get("ok")) and bool(box2d_check.get("ok"))
    report["ok"] = ok
    write_json(out_dir / "preflight.json", report)
    lines = [
        "CSE 482 A2 preflight",
        f"ok: {ok}",
        f"python: {report['versions']['python']} ({report['versions']['python_executable']})",
        f"torch: {report['versions']['torch']} cuda={report['versions']['cuda_available']}",
        f"gymnasium: {report['versions']['gymnasium']} sb3: {report['versions']['stable_baselines3']}",
        f"ale-py: {report['versions'].get('ale_py')} Box2D: {report['versions'].get('Box2D')}",
        f"host RAM: {report['host_memory']['total_mib']} MiB total, "
        f"{report['host_memory']['available_mib']} MiB available",
        f"ALE: {ale_check}",
        f"Box2D: {box2d_check}",
        f"replay estimate: {report['replay_estimate_full_dqn']}",
    ]
    (out_dir / "preflight.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
