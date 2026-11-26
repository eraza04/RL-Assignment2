"""Regression fixture for comparison plots: 30 matched runs, 3 lines per panel.

Builds 10 conditions x 3 seeds with deterministic, distinctive evaluation
values, including a duplicate periodic/final row at the same step, then runs
the real plotter. Failed, unverified, duplicate, incompatible, absent-manifest
and unmatched-seed candidates are included to prove they cannot change a panel
and are audited explicitly. Separate fixtures fix the across-seed SD at exactly
3.0 and check that one seed produces no spread band or exported SD. Three more
fixtures cover four incompatible noncanonical duplicate aliases of one baseline
seed, tied baseline signatures, and disjoint checkpoint steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib.axes import Axes  # noqa: E402

from rl_a2.config import conditions_sorted, read_json, resolve_config, write_json  # noqa: E402
from rl_a2.plots import COMPARISON_GROUPS, plot_comparisons  # noqa: E402


def _label_value(call: dict[str, Any]) -> float:
    match = re.search(r"=([0-9.eE+-]+)", call["label"])
    if not match:
        raise AssertionError(f"no factor value in label: {call['label']}")
    return float(match.group(1))


def _write_fixture_run(
    fixture: Path,
    condition: str,
    seed: int,
    index: int,
    *,
    dir_name: str | None = None,
    status: str = "completed",
    total_timesteps: int | None = None,
    add_stale_group_tag: bool = False,
    write_manifest: bool = True,
    eval_values: dict[int, float] | None = None,
    manifest_overrides: dict[str, Any] | None = None,
    config_edits: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    run_dir = fixture / (dir_name or f"{condition}_seed{seed}_full")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = resolve_config(
        condition,
        "full",
        seed,
        fixture,
        total_timesteps=total_timesteps,
        validate=False,
    )
    if add_stale_group_tag:
        # Reproduce the old misleading metadata: an epsilon variant claiming
        # to be the target baseline too.
        config["metadata"]["comparisons"].append(
            {"group": "dqn_target", "value": 5000, "baseline": True}
        )
    if config_edits is not None:
        config_edits(config)
    (run_dir / "config_resolved.json").write_text(json.dumps(config), encoding="utf-8")
    metrics = run_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    if eval_values is None:
        base = index * 10 + seed
        eval_text = (
            "checkpoint_step,is_final,episodes,mean_return_raw,std_return_raw\n"
            f"50000,false,5,{base},1.0\n"
            f"500000,false,5,-999,1.0\n"
            f"500000,true,5,{base + 1},1.0\n"
        )
    else:
        lines = ["checkpoint_step,is_final,episodes,mean_return_raw,std_return_raw"]
        final_step = max(eval_values)
        for step, value in sorted(eval_values.items()):
            is_final = "true" if step == final_step else "false"
            lines.append(f"{step},{is_final},5,{value},1.0")
        eval_text = "\n".join(lines) + "\n"
    (metrics / "eval_summary.csv").write_text(eval_text, encoding="utf-8")
    if write_manifest:
        manifest = {
            "schema_version": 1,
            "run_id": config["run"]["run_id"],
            "condition": condition,
            "profile": "full",
            "algorithm": config["algo"],
            "seed": seed,
            "status": status,
            "command": [sys.executable, "-m", "rl_a2.train", "--condition", condition],
            "artifact_checks": {"eval_summary_csv": True, "final_model": True},
        }
        manifest.update(manifest_overrides or {})
        write_json(run_dir / "run_manifest.json", manifest)
    return run_dir


def _write_sd_fixture(output_root: Path, name: str, seeds: list[int]) -> Path:
    """Three canonical DQN target runs whose seed values are exactly 3 apart."""
    fixture = output_root / name
    fixture.mkdir(parents=True, exist_ok=True)
    conditions = ["dqn_baseline", "dqn_target_1000", "dqn_target_10000"]
    indices = {condition: index for index, condition in enumerate(conditions_sorted())}
    for condition in conditions:
        for seed in seeds:
            _write_fixture_run(
                fixture,
                condition,
                seed,
                indices[condition],
                eval_values={50000: 10.0 + 3.0 * seed, 500000: 20.0 + 3.0 * seed},
            )
    return fixture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="artifacts/remediation/comparison_plots",
        help="Fresh output root; existing evidence is never overwritten",
    )
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    fixture = output_root / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    conditions = conditions_sorted()
    indices = {name: index for index, name in enumerate(conditions)}
    matched_runs = [
        _write_fixture_run(fixture, condition, seed, indices[condition])
        for condition in conditions
        for seed in (0, 1, 2)
    ]
    # Intruders: none of these may contribute a line to any panel.
    _write_fixture_run(
        fixture,
        "dqn_target_1000",
        0,
        99,
        dir_name="failed_intruder_dqn_target_1000_seed0_full",
        status="failed",
    )
    _write_fixture_run(
        fixture,
        "ppo_baseline",
        3,
        99,
        dir_name="incompatible_intruder_ppo_baseline_seed3_full",
        total_timesteps=999,
    )
    _write_fixture_run(
        fixture,
        "dqn_eps_010",
        0,
        indices["dqn_eps_010"],
        dir_name="stale_metadata_intruder_dqn_eps_010_seed0_full",
        add_stale_group_tag=True,
    )
    # Interrupted-run shape from the re-review: evaluation rows exist but no
    # completed manifest, so this candidate must never enter a panel.
    _write_fixture_run(
        fixture,
        "dqn_baseline",
        0,
        999,
        dir_name="absent_manifest_intruder_dqn_baseline_seed0_full",
        write_manifest=False,
    )
    # A manifest that belongs to another run is also refused.
    _write_fixture_run(
        fixture,
        "dqn_baseline",
        0,
        999,
        dir_name="wrong_identity_intruder_dqn_baseline_seed0_full",
        manifest_overrides={"run_id": "not_this_candidate"},
    )
    # Verified and admitted, but no matching variant seeds: must be audited as
    # an unmatched seed rather than silently counted as a contributor.
    _write_fixture_run(fixture, "dqn_baseline", 3, indices["dqn_baseline"])

    calls: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    original_plot = Axes.plot
    original_fill = Axes.fill_between

    def _floats(value: Any) -> list[float]:
        try:
            return [float(item) for item in value]
        except TypeError:
            return [float(value)]

    def capture_plot(self, x, y, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        label = str(kwargs.get("label", ""))
        if "mean±SD" in label or "mean only" in label:
            calls.append({"label": label, "x": _floats(x), "y": _floats(y)})
        return original_plot(self, x, y, *args, **kwargs)

    def capture_fill(self, x, y1, y2=0, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN004
        fills.append(
            {
                "x": _floats(x),
                "lower": _floats(y1),
                "upper": _floats(y2),
                "color": kwargs.get("color"),
            }
        )
        return original_fill(self, x, y1, y2, *args, **kwargs)

    def plot_with_capture(root: Path) -> list[Path]:
        with patch.object(Axes, "plot", capture_plot), patch.object(Axes, "fill_between", capture_fill):
            return plot_comparisons(root)

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    started_epoch = time.time()
    figures = plot_with_capture(fixture)
    wall_seconds = time.time() - started_epoch
    comparisons_dir = fixture / "comparisons"
    audit = read_json(comparisons_dir / "comparison_manifest.json")

    expected_values = {
        group: [
            spec["baseline"][1],
            *[value for _, value in spec["variants"]],
        ]
        for group, spec in COMPARISON_GROUPS.items()
    }
    expected_indices = {
        "dqn_target": [0, 1, 2],
        "dqn_exploration": [0, 3, 4],
        "ppo_clip": [5, 6, 7],
        "ppo_lambda": [5, 8, 9],
    }
    def group_calls_for(group: str) -> list[dict[str, Any]]:
        expected = expected_values[group]
        factor_name = COMPARISON_GROUPS[group]["factor"].split(".")[-1]
        selected = [
            call
            for call in calls
            if factor_name in call["label"]
            and any(abs(_label_value(call) - float(value)) < 1e-12 for value in expected)
        ]
        selected.sort(
            key=lambda call: min(
                range(len(expected)),
                key=lambda index: abs(_label_value(call) - float(expected[index])),
            )
        )
        return selected

    checks: dict[str, bool] = {
        "fixture_has_30_matched_runs": len(matched_runs) == 30,
        "four_panels_written": len(figures) == 4,
        "exactly_three_lines_per_panel": all(
            len(group_calls_for(group)) == 3 for group in COMPARISON_GROUPS
        ),
    }
    line_details: dict[str, Any] = {}
    for group in COMPARISON_GROUPS:
        group_calls = group_calls_for(group)
        expected_y = [
            [indices[conditions[index]] * 10 + 1, indices[conditions[index]] * 10 + 2]
            for index in expected_indices[group]
        ]
        y_ok = len(group_calls) == 3 and all(
            call["x"] == [50000.0, 500000.0] and call["y"] == expected_y[i]
            for i, call in enumerate(group_calls)
        )
        labels_ok = len(group_calls) == 3 and all(
            "n=3 seeds" in call["label"] and "mean±SD" in call["label"]
            for call in group_calls
        )
        checks[f"{group}_three_correct_aggregated_lines"] = y_ok
        checks[f"{group}_labels_cover_uncertainty_and_seeds"] = labels_ok
        checks[f"{group}_audit_line_count"] = audit["groups"][group].get("line_count") == 3
        checks[f"{group}_matched_seeds"] = audit["groups"][group].get("matched_seeds") == [0, 1, 2]
        checks[f"{group}_matched_steps"] = audit["groups"][group].get("matched_steps") == [
            50000,
            500000,
        ]
        excluded = audit["groups"][group].get("excluded_runs", [])
        # Intruders are only audited in panels whose algorithm they claim.
        checks[f"{group}_failed_run_excluded"] = group != "dqn_target" or any(
            "failed" in str(row.get("reason")) for row in excluded
        )
        checks[f"{group}_incompatible_run_excluded"] = (
            COMPARISON_GROUPS[group]["algo"] != "ppo"
            or any(
                "incompatible" in str(row.get("reason")) or "999" in str(row.get("run"))
                for row in excluded
            )
        )
        line_details[group] = {
            "expected_y": expected_y,
            "calls": group_calls,
            "included_runs": len(audit["groups"][group].get("included_runs", [])),
            "excluded_runs": excluded,
        }
    checks["duplicate_final_step_deduplicated"] = all(
        audit["groups"][group]["included_runs"]
        and all(
            row["dedup"]["raw_rows"] == 3 and row["dedup"]["deduped_steps"] == 2
            for row in audit["groups"][group]["included_runs"]
            if "dedup" in row
        )
        for group in COMPARISON_GROUPS
    )

    # Finding 1: unverified or ambiguous candidates must be audited, never
    # admitted; unmatched verified seeds must not be counted as contributors.
    target_audit = audit["groups"]["dqn_target"]
    exploration_audit = audit["groups"]["dqn_exploration"]
    absent_rows = [
        row
        for row in target_audit["excluded_runs"]
        if "absent_manifest_intruder" in str(row.get("run"))
    ]
    wrong_identity_rows = [
        row
        for row in target_audit["excluded_runs"]
        if "wrong_identity_intruder" in str(row.get("run"))
    ]
    admitted_paths = [str(row.get("run")) for row in target_audit["admitted_candidates"]]
    contributor_paths = [str(row.get("run")) for row in target_audit["contributors"]]
    checks["absent_manifest_intruder_excluded"] = len(absent_rows) == 1 and (
        "manifest missing" in str(absent_rows[0].get("reason"))
    )
    checks["absent_manifest_intruder_not_admitted"] = not any(
        "absent_manifest_intruder" in path for path in admitted_paths
    )
    checks["wrong_manifest_identity_excluded"] = len(wrong_identity_rows) == 1 and (
        "identity mismatch" in str(wrong_identity_rows[0].get("reason"))
    )
    checks["duplicate_prefers_canonical_run_id_directory"] = any(
        "stale_metadata_intruder" in str(row.get("run"))
        and "duplicate candidate not selected" in str(row.get("reason"))
        and "canonical run_id" in str(row.get("reason"))
        for row in exploration_audit["excluded_runs"]
    )
    for group in ("dqn_target", "dqn_exploration"):
        group_audit = audit["groups"][group]
        admitted_baseline_seeds = {
            int(row["seed"])
            for row in group_audit["admitted_candidates"]
            if row["condition"] == "dqn_baseline"
        }
        contributor_baseline_seeds = {
            int(row["seed"])
            for row in group_audit["contributors"]
            if row["condition"] == "dqn_baseline"
        }
        unmatched_rows = [
            row
            for row in group_audit["excluded_runs"]
            if int(row.get("seed", -1)) == 3 and "unmatched seed" in str(row.get("reason"))
        ]
        checks[f"{group}_unmatched_seed_audited"] = (
            3 in admitted_baseline_seeds
            and 3 not in contributor_baseline_seeds
            and len(unmatched_rows) == 1
        )
    checks["admitted_candidates_and_contributors_separate"] = all(
        len(group_audit["admitted_candidates"]) >= len(group_audit["contributors"])
        and group_audit["actual_contributor_count"] == len(group_audit["contributors"]) == 9
        and group_audit["included_runs"] == group_audit["contributors"]
        for group_audit in audit["groups"].values()
    )

    # Finding 2: exact sample SD (3.0) versus population SD (2.44949), and the
    # undefined one-seed case (no band, blank CSV).
    def _sd_rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    main_calls = list(calls)
    sd_values = (10.0, 13.0, 16.0)
    sd_mean = 13.0
    sample_sd = math.sqrt(sum((value - sd_mean) ** 2 for value in sd_values) / (len(sd_values) - 1))
    population_sd = math.sqrt(sum((value - sd_mean) ** 2 for value in sd_values) / len(sd_values))

    calls.clear()
    fills.clear()
    sd_sample_fixture = _write_sd_fixture(output_root, "sd_sample_fixture", [0, 1, 2])
    sd_sample_figures = plot_with_capture(sd_sample_fixture)
    sd_sample_calls = list(calls)
    sd_sample_fills = list(fills)
    sd_sample_rows = _sd_rows(sd_sample_fixture / "comparisons" / "dqn_target_aggregated.csv")
    sd_sample_audit = read_json(sd_sample_fixture / "comparisons" / "comparison_manifest.json")
    sample_band_widths = [
        [round(upper - lower, 9) for lower, upper in zip(fill["lower"], fill["upper"])]
        for fill in sd_sample_fills
    ]
    checks["sample_sd_is_three_not_population_2_44949"] = (
        abs(sample_sd - 3.0) < 1e-12
        and abs(population_sd - 2.449489742783178) < 1e-12
        and abs(sample_sd - population_sd) > 0.5
    )
    checks["sample_sd_fill_band_is_three"] = (
        len(sd_sample_fills) == 3
        and all(fill["x"] == [50000.0, 500000.0] for fill in sd_sample_fills)
        and all(widths == [6.0, 6.0] for widths in sample_band_widths)
    )
    checks["sample_sd_csv_is_three"] = (
        len(sd_sample_rows) == 6
        and all(
            abs(float(row["std_return_raw_across_seeds"]) - 3.0) < 1e-12
            and int(row["n_seeds"]) == 3
            for row in sd_sample_rows
        )
    )
    checks["sample_sd_audit_method"] = (
        sd_sample_audit["groups"]["dqn_target"]["standard_deviation"] == "sample (ddof=1)"
    )

    calls.clear()
    fills.clear()
    sd_single_fixture = _write_sd_fixture(output_root, "sd_single_seed_fixture", [0])
    sd_single_figures = plot_with_capture(sd_single_fixture)
    sd_single_calls = list(calls)
    sd_single_fills = list(fills)
    sd_single_rows = _sd_rows(sd_single_fixture / "comparisons" / "dqn_target_aggregated.csv")
    sd_single_audit = read_json(sd_single_fixture / "comparisons" / "comparison_manifest.json")
    checks["single_seed_has_no_spread_band"] = not sd_single_fills
    checks["single_seed_csv_sd_blank"] = (
        len(sd_single_rows) == 6
        and all(
            row["std_return_raw_across_seeds"] == "" and int(row["n_seeds"]) == 1
            for row in sd_single_rows
        )
    )
    checks["single_seed_label_omits_sd"] = (
        len(sd_single_calls) == 3
        and all("mean only, n=1 seed" in call["label"] for call in sd_single_calls)
        and all("mean±SD" not in call["label"] for call in sd_single_calls)
    )
    checks["single_seed_audit_method"] = "undefined for n=1" in str(
        sd_single_audit["groups"]["dqn_target"]["standard_deviation"]
    )

    # Follow-up finding 1: four noncanonical copies of one baseline seed that
    # differ only in learning rate must not outvote the nine canonical runs.
    calls.clear()
    fills.clear()
    vote_fixture = _write_sd_fixture(output_root, "duplicate_signature_vote_fixture", [0, 1, 2])
    for i in range(4):
        _write_fixture_run(
            vote_fixture,
            "dqn_baseline",
            0,
            999,
            dir_name=f"a_incompatible_duplicate_{i}_dqn_baseline_seed0_full",
            config_edits=lambda config: config["model"].update(learning_rate=0.0002),
            eval_values={50000: 9000.0, 500000: 9020.0},
        )
    vote_figures = plot_with_capture(vote_fixture)
    vote_calls = list(calls)
    vote_fills = list(fills)
    vote_audit = read_json(vote_fixture / "comparisons" / "comparison_manifest.json")
    vote_group = vote_audit["groups"]["dqn_target"]
    vote_rows = _sd_rows(vote_fixture / "comparisons" / "dqn_target_aggregated.csv")
    vote_reasons = [str(row.get("reason")) for row in vote_group["excluded_runs"]]
    canonical_dirs = {
        f"{condition}_seed{seed}_full"
        for condition in ("dqn_baseline", "dqn_target_1000", "dqn_target_10000")
        for seed in (0, 1, 2)
    }
    checks["duplicate_vote_keeps_nine_canonical_lines"] = (
        len(vote_figures) == 1
        and vote_figures[0].name == "compare_dqn_target.png"
        and vote_group["line_count"] == 3
        and vote_group["actual_contributor_count"] == 9
        and vote_group["contributors"] == vote_group["included_runs"]
        and {str(row["run_dir_name"]) for row in vote_group["contributors"]} == canonical_dirs
    )
    checks["duplicate_vote_excludes_aliases_before_voting"] = (
        sum(
            "duplicate candidate not selected" in reason and "canonical run_id" in reason
            for reason in vote_reasons
        )
        == 4
        and not any("incompatible resolved config" in reason for reason in vote_reasons)
        and not any(
            "incompatible_duplicate" in str(row.get("run"))
            for row in vote_group["admitted_candidates"]
        )
    )
    checks["duplicate_vote_panel_values_unchanged"] = (
        len(vote_calls) == 3
        and len(vote_fills) == 3
        and all(
            abs(call["y"][0] - 13.0) < 1e-12 and abs(call["y"][1] - 23.0) < 1e-12
            for call in vote_calls
        )
        and len(vote_rows) == 6
        and all(abs(float(row["std_return_raw_across_seeds"]) - 3.0) < 1e-12 for row in vote_rows)
    )

    # A tie among baseline signatures has no scientific reference; refuse it
    # explicitly rather than picking a winner by directory order.
    calls.clear()
    fills.clear()
    tie_fixture = output_root / "tied_reference_fixture"
    tie_fixture.mkdir(parents=True, exist_ok=True)
    for seed, rate in zip((0, 1, 2), (0.0001, 0.0002, 0.0003)):
        _write_fixture_run(
            tie_fixture,
            "dqn_baseline",
            seed,
            indices["dqn_baseline"],
            eval_values={50000: 10.0 + 3.0 * seed, 500000: 20.0 + 3.0 * seed},
            config_edits=lambda config, rate=rate: config["model"].update(learning_rate=rate),
        )
    for condition in ("dqn_target_1000", "dqn_target_10000"):
        for seed in (0, 1, 2):
            _write_fixture_run(
                tie_fixture,
                condition,
                seed,
                indices[condition],
                eval_values={50000: 10.0 + 3.0 * seed, 500000: 20.0 + 3.0 * seed},
            )
    tie_figures = plot_with_capture(tie_fixture)
    tie_audit = read_json(tie_fixture / "comparisons" / "comparison_manifest.json")
    tie_group = tie_audit["groups"]["dqn_target"]
    checks["tied_reference_refused_explicitly"] = (
        not tie_figures
        and not calls
        and "tied compatible-config reference" in str(tie_group.get("reason"))
        and tie_group["line_count"] == 0
        and tie_group["contributors"] == []
        and tie_group["actual_contributor_count"] == 0
        and sum(
            "tied compatible-config reference" in str(row.get("reason"))
            for row in tie_group["excluded_runs"]
        )
        == 9
    )

    # Follow-up finding 2: matched identities but disjoint checkpoint steps must
    # leave contributors and counts empty/zero with an explanatory reason.
    calls.clear()
    fills.clear()
    disjoint_fixture = output_root / "disjoint_steps_fixture"
    disjoint_fixture.mkdir(parents=True, exist_ok=True)
    for index, condition in enumerate(("dqn_baseline", "dqn_target_1000", "dqn_target_10000")):
        for seed in (0, 1, 2):
            _write_fixture_run(
                disjoint_fixture,
                condition,
                seed,
                indices[condition],
                eval_values={
                    50000 + index * 100: 10.0 + 3.0 * seed,
                    500000 + index * 100: 20.0 + 3.0 * seed,
                },
            )
    disjoint_figures = plot_with_capture(disjoint_fixture)
    disjoint_audit = read_json(disjoint_fixture / "comparisons" / "comparison_manifest.json")
    disjoint_group = disjoint_audit["groups"]["dqn_target"]
    checks["disjoint_steps_zero_contributors_and_lines"] = (
        not disjoint_figures
        and not calls
        and not fills
        and disjoint_group["line_count"] == 0
        and disjoint_group["contributors"] == []
        and disjoint_group["included_runs"] == []
        and disjoint_group["actual_contributor_count"] == 0
        and disjoint_group["matched_steps"] == []
        and disjoint_group["reason"] == "no steps matched across all condition/seed curves"
        and not (disjoint_fixture / "comparisons" / "dqn_target_aggregated.csv").exists()
        and not (disjoint_fixture / "plots" / "compare_dqn_target.png").exists()
    )

    report = {
        "schema_version": 1,
        "command_for_this_script": subprocess.list2cmdline([sys.executable, "-m", *sys.argv]),
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "started_epoch": started_epoch,
        "wall_seconds": round(wall_seconds, 3),
        "fixture_runs": len(matched_runs) + 6 + 3 + 1 + 13 + 9 + 9,
        "matched_fixture_runs": len(matched_runs),
        "plots": [str(path) for path in figures],
        "comparison_manifest": str(comparisons_dir / "comparison_manifest.json"),
        "line_details": line_details,
        "all_ax_plot_calls": main_calls,
        "admission_audit": {
            "dqn_target": target_audit,
            "dqn_exploration": exploration_audit,
        },
        "sd_regression": {
            "sample_values": list(sd_values),
            "sample_sd": sample_sd,
            "population_sd": population_sd,
            "sample_fixture": str(sd_sample_fixture),
            "sample_plots": [str(path) for path in sd_sample_figures],
            "sample_calls": sd_sample_calls,
            "sample_fill_band_widths": sample_band_widths,
            "sample_csv_rows": sd_sample_rows,
            "single_seed_fixture": str(sd_single_fixture),
            "single_seed_plots": [str(path) for path in sd_single_figures],
            "single_seed_calls": sd_single_calls,
            "single_seed_fill_calls": sd_single_fills,
            "single_seed_csv_rows": sd_single_rows,
        },
        "duplicate_vote_regression": {
            "fixture": str(vote_fixture),
            "plots": [str(path) for path in vote_figures],
            "audit": vote_group,
            "plot_calls": vote_calls,
            "fill_calls": vote_fills,
            "csv_rows": vote_rows,
        },
        "tied_reference_regression": {
            "fixture": str(tie_fixture),
            "plots": [str(path) for path in tie_figures],
            "audit": tie_group,
        },
        "disjoint_steps_regression": {
            "fixture": str(disjoint_fixture),
            "plots": [str(path) for path in disjoint_figures],
            "audit": disjoint_group,
        },
        "checks": checks,
        "ok": all(checks.values()),
    }
    path = output_root / "comparison_plot_report.json"
    write_json(path, report)
    print(json.dumps({key: report[key] for key in ("checks", "ok", "plots")}, indent=2))
    print(f"\nReport: {path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
