"""Verify each ablation differs from its baseline only in the intended factor.

The resolved configs contain metadata and run paths that necessarily differ.
Those are reported separately and ignored by the semantic diff. Every other
leaf (seed, timesteps, buffer capacity, evaluation seeds, wrappers, etc.) must
match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl_a2.config import (  # noqa: E402
    ablation_diff,
    conditions_sorted,
    resolve_config,
    write_json,
)

EXPECTED = {
    "dqn_baseline": None,
    "dqn_target_1000": ("model", "target_update_interval", 1000),
    "dqn_target_10000": ("model", "target_update_interval", 10000),
    "dqn_eps_010": ("model", "exploration_fraction", 0.10),
    "dqn_eps_040": ("model", "exploration_fraction", 0.40),
    "ppo_baseline": None,
    "ppo_clip_010": ("model", "clip_range", 0.10),
    "ppo_clip_040": ("model", "clip_range", 0.40),
    "ppo_lambda_090": ("model", "gae_lambda", 0.90),
    "ppo_lambda_099": ("model", "gae_lambda", 0.99),
}

BASELINES = {"dqn": "dqn_baseline", "ppo": "ppo_baseline"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default="artifacts/ablation_check")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    resolved = {
        condition: resolve_config(
            condition,
            profile=args.profile,
            seed=args.seed,
            output_root=root,
        )
        for condition in conditions_sorted()
    }
    report = {
        "schema_version": 1,
        "profile": args.profile,
        "seed": args.seed,
        "expected_factor_changes": {
            name: ("baseline" if expected is None else ".".join(map(str, expected)))
            for name, expected in EXPECTED.items()
        },
        "ignored_metadata_only_paths": ["run.*", "metadata.*", "condition"],
        "explanation": (
            "run/output paths, condition names, and metadata labels/order/comparison tags "
            "are bookkeeping only. All experiment settings (seed, timesteps, eval seeds, "
            "wrappers, buffer capacity, and every non-ablated hyperparameter) are compared."
        ),
        "conditions": {},
        "ok": True,
    }
    for condition, config in resolved.items():
        baseline_name = BASELINES[config["algo"]]
        baseline = resolved[baseline_name]
        diffs = ablation_diff(config, baseline)
        expected = EXPECTED[condition]
        semantic_diffs = [diff for diff in diffs if diff["path"] != "condition"]
        if expected is None:
            expected_ok = len(semantic_diffs) == 0
        else:
            expected_path, expected_value = f"{expected[0]}.{expected[1]}", expected[2]
            expected_ok = (
                len(semantic_diffs) == 1
                and semantic_diffs[0]["path"] == expected_path
                and semantic_diffs[0]["candidate"] == expected_value
            )
        shared = {
            "seed": config["run"]["seed"] == baseline["run"]["seed"] == args.seed,
            "total_timesteps": (
                config["training"]["total_timesteps"]
                == baseline["training"]["total_timesteps"]
            ),
            "evaluation": config["evaluation"] == baseline["evaluation"],
            "profile": config["profile"] == baseline["profile"],
        }
        ok = bool(expected_ok and all(shared.values()))
        report["conditions"][condition] = {
            "baseline": baseline_name,
            "expected": ("baseline" if expected is None else list(expected)),
            "semantic_diffs": semantic_diffs,
            "ignored_diffs": [diff for diff in diffs if diff["path"] == "condition"],
            "shared": shared,
            "ok": ok,
        }
        report["ok"] = report["ok"] and ok

    report_path = (
        Path(args.report).resolve()
        if args.report
        else root / f"ablation_diff_{args.profile}_seed{args.seed}.json"
    )
    write_json(report_path, report)
    print(json.dumps(report, indent=2))
    print(f"\nReport: {report_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
