"""Run the ten assignment conditions serially with matched seeds.

Examples (PowerShell, from the project root):

    .\\.venv\\Scripts\\python.exe scripts\\run_all.py --profile smoke --seeds 0 --output-root artifacts\\smoke
    .\\.venv\\Scripts\\python.exe scripts\\run_all.py --profile full  --seeds 0 --output-root artifacts\\full_seed0

The full profile is 500,000 agent timesteps per condition per seed and is not
started by the implementation smoke tests.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl_a2.config import conditions_sorted  # noqa: E402
from rl_a2.provenance import source_digest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["full", "smoke"], default="smoke")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output-root", default="artifacts/smoke")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    conditions = args.conditions or conditions_sorted()
    known = set(conditions_sorted())
    unknown = sorted(set(conditions) - known)
    if unknown:
        raise SystemExit(f"Unknown conditions: {unknown}")
    commands: list[list[str]] = []
    for condition in conditions:
        for seed in args.seeds:
            command = [
                args.python,
                "-m",
                "rl_a2.train",
                "--condition",
                condition,
                "--profile",
                args.profile,
                "--seed",
                str(seed),
                "--output-root",
                str(output_root),
            ]
            if args.no_video:
                command.append("--no-video")
            commands.append(command)

    manifest = {
        "schema_version": 1,
        "status": "planned" if args.dry_run else "running",
        "profile": args.profile,
        "seeds": args.seeds,
        "conditions": conditions,
        "output_root": str(output_root),
        "python": args.python,
        "commands": commands,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "started_epoch": time.time(),
        "source_digest": source_digest(),
        "note": (
            "Seeds are matched across every condition; each run uses the same "
            "evaluation seeds. A fresh --output-root is required: run evidence "
            "is never overwritten, deleted or merged."
        ),
    }
    (output_root / "run_all_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        for command in commands:
            print(subprocess.list2cmdline(command))
        return 0

    completed: list[dict] = []
    try:
        for index, command in enumerate(commands, start=1):
            print(f"\n[{index}/{len(commands)}] {subprocess.list2cmdline(command)}", flush=True)
            started = time.perf_counter()
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            completed.append(
                {
                    "command": command,
                    "status": "completed",
                    "wall_seconds": round(time.perf_counter() - started, 3),
                }
            )
            manifest["completed"] = completed
            (output_root / "run_all_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
    except subprocess.CalledProcessError as exc:
        manifest["status"] = "failed"
        manifest["failed_command"] = exc.cmd
        manifest["returncode"] = exc.returncode
        manifest["completed"] = completed
        manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (output_root / "run_all_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        raise
    manifest["status"] = "completed"
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["completed"] = completed
    (output_root / "run_all_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nCompleted {len(completed)} serial runs under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
