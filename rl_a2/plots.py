"""Matplotlib diagnostics and cross-condition comparison plots."""

from __future__ import annotations

import copy
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import read_json, write_json


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _empty_plot(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(title)
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    _save(fig, path)


def _rolling(values: pd.Series, window: int = 20) -> pd.Series:
    if len(values) == 0:
        return values
    return values.rolling(window=min(window, max(1, len(values))), min_periods=1).mean()


def _eval_plot(ax, summary: pd.DataFrame, label: str = "reload evaluation") -> None:
    if summary.empty:
        return
    summary = summary.sort_values("checkpoint_step")
    ax.errorbar(
        summary["checkpoint_step"],
        summary["mean_return_raw"],
        yerr=summary["std_return_raw"].fillna(0),
        marker="o",
        capsize=3,
        linewidth=1.5,
        label=label,
    )


def plot_run(run_dir: Path, config: dict[str, Any]) -> list[Path]:
    run_dir = Path(run_dir)
    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    metrics = run_dir / "metrics"
    train_log = _read_csv(metrics / "train_log.csv")
    episodes = _read_csv(metrics / "train_episodes.csv")
    evals = _read_csv(metrics / "eval_summary.csv")
    written: list[Path] = []

    prefix = "dqn" if config["algo"] == "dqn" else "ppo"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not episodes.empty:
        ax.plot(
            episodes["timestep"],
            episodes["return_raw"],
            color="tab:blue",
            alpha=0.2,
            linewidth=0.8,
            label="episode raw return",
        )
        ax.plot(
            episodes["timestep"],
            _rolling(episodes["return_raw"], 20),
            color="tab:blue",
            linewidth=1.8,
            label="raw return rolling mean (20)",
        )
    _eval_plot(ax, evals, "checkpoint reload eval mean")
    ax.set_xlabel("agent timesteps")
    ax.set_ylabel("raw episode return")
    ax.set_title(f"{config['run']['run_id']}: learning curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    path = plots / f"{prefix}_learning_curve.png"
    _save(fig, path)
    written.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _eval_plot(ax, evals)
    ax.set_xlabel("agent timesteps")
    ax.set_ylabel("mean full-game raw return")
    ax.set_title(f"{config['run']['run_id']}: reload evaluation")
    ax.grid(alpha=0.25)
    path = plots / f"{prefix}_eval_returns.png"
    _save(fig, path)
    written.append(path)

    final_note = ""
    final_step: int | None = None
    if "is_final" in train_log and "note" in train_log and not train_log.empty:
        final_rows = train_log[train_log["is_final"].astype(str).str.lower().eq("true")]
        if not final_rows.empty:
            final_note = str(final_rows.iloc[-1].get("note") or "")
            try:
                final_step = int(final_rows.iloc[-1]["timestep"])
            except (TypeError, ValueError):
                final_step = None

    if config["algo"] == "dqn":
        specs = [
            ("epsilon", "epsilon (training exploration rate)", "dqn_epsilon.png"),
            (
                "td_huber_loss",
                "Huber TD loss (optimizer-update steps only)",
                "dqn_td_huber_loss.png",
            ),
            (
                "mean_max_q",
                "mean max_a Q_online over fixed 32-probe set",
                "dqn_mean_max_q.png",
            ),
        ]
        for column, ylabel, filename in specs:
            if train_log.empty or column not in train_log or train_log[column].notna().sum() == 0:
                _empty_plot(plots / filename, ylabel)
            else:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                frame = train_log.dropna(subset=[column])
                ax.plot(frame["timestep"], frame[column], color="tab:orange", linewidth=1.5)
                if column == "td_huber_loss" and final_step is not None and "no optimizer update" in final_note:
                    ax.annotate(
                        "no optimizer update at final step",
                        xy=(final_step, 0),
                        xytext=(0, 12),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                        color="tab:red",
                    )
                ax.set_xlabel("agent timesteps")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{config['run']['run_id']}: {column}")
                ax.grid(alpha=0.25)
                _save(fig, plots / filename)
            written.append(plots / filename)
    else:
        specs = [
            (
                "approx_kl",
                "approx KL (SB3: final-epoch minibatches only)",
                "ppo_approx_kl.png",
            ),
            ("value_loss", "value loss (train/value_loss, MSE)", "ppo_value_loss.png"),
            (
                "policy_entropy",
                "policy entropy = -train/entropy_loss (nats; may be negative)",
                "ppo_entropy.png",
            ),
            ("clip_fraction", "fraction of ratios clipped", "ppo_clip_fraction.png"),
        ]
        for column, ylabel, filename in specs:
            if train_log.empty or column not in train_log or train_log[column].notna().sum() == 0:
                _empty_plot(plots / filename, ylabel)
            else:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                frame = train_log.dropna(subset=[column])
                ax.plot(frame["timestep"], frame[column], color="tab:green", linewidth=1.5)
                ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
                ax.set_xlabel("agent timesteps")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{config['run']['run_id']}: {column}")
                ax.grid(alpha=0.25)
                _save(fig, plots / filename)
            written.append(plots / filename)

    # Final scores are also useful as a compact inspection artifact.
    if not evals.empty:
        final_rows = evals[evals["is_final"].astype(str).str.lower() == "true"]
        final = final_rows.iloc[[-1]] if not final_rows.empty else evals.iloc[[-1]]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.bar(
            ["final reload eval"],
            final["mean_return_raw"].astype(float).values,
            yerr=final["std_return_raw"].astype(float).values,
        )
        ax.set_ylabel("raw return")
        ax.set_title(f"{config['run']['run_id']}: final score")
        ax.grid(alpha=0.25, axis="y")
        path = plots / f"{prefix}_final_score.png"
        _save(fig, path)
        written.append(path)
    return written


def _discover_runs(output_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for config_path in sorted(output_root.glob("*/config_resolved.json")):
        run_dir = config_path.parent
        try:
            config = read_json(config_path)
        except Exception:
            continue
        if not (run_dir / "metrics" / "eval_summary.csv").is_file():
            continue
        runs.append({"run_dir": run_dir, "config": config})
    return runs


def _verify_manifest(
    manifest_path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Load a completed manifest and require it to identify this candidate run."""
    if not manifest_path.is_file():
        return None, "manifest missing"
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:
        return None, f"manifest unreadable: {type(exc).__name__}"

    if manifest.get("status") != "completed":
        return None, f"status={manifest.get('status')!r}, not completed"
    checks = manifest.get("artifact_checks")
    if not isinstance(checks, dict) or not checks or not all(bool(flag) for flag in checks.values()):
        return None, "artifact_checks missing or not all true"

    expected = {
        "run_id": config.get("run", {}).get("run_id"),
        "condition": config.get("condition"),
        "profile": config.get("profile"),
        "algorithm": config.get("algo"),
    }
    for key, want in expected.items():
        got = manifest.get(key)
        if got != want:
            return None, f"manifest identity mismatch for {key}: {got!r} != {want!r}"
    try:
        seed_ok = int(manifest.get("seed")) == int(config.get("run", {}).get("seed"))
    except (TypeError, ValueError):
        seed_ok = False
    if not seed_ok:
        return None, (
            f"manifest identity mismatch for seed: {manifest.get('seed')!r} != "
            f"{config.get('run', {}).get('seed')!r}"
        )
    return manifest, None


def _select_duplicate_candidates(
    entries: list[dict[str, Any]], excluded: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Pick one verified run per condition/seed, preferring the canonical dir.

    A canonical directory is named exactly like the run id in the resolved
    config. If several verified candidates exist but no unique canonical one,
    all are refused as ambiguous rather than silently selecting by path order.
    """
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[(entry["condition"], entry["seed"])].append(entry)

    selected: list[dict[str, Any]] = []
    for (condition, seed), candidates in sorted(grouped.items()):
        if len(candidates) == 1:
            selected.append(candidates[0])
            continue
        canonical = [entry for entry in candidates if entry["run_dir_name"] == entry["run_id"]]
        if len(canonical) == 1:
            chosen = canonical[0]
            selected.append(chosen)
            for entry in candidates:
                if entry is not chosen:
                    excluded.append(
                        {
                            "run": entry["run_dir"],
                            "condition": condition,
                            "seed": seed,
                            "reason": (
                                "duplicate candidate not selected: canonical run_id "
                                f"directory {chosen['run_dir_name']} preferred"
                            ),
                        }
                    )
        else:
            for entry in candidates:
                excluded.append(
                    {
                        "run": entry["run_dir"],
                        "condition": condition,
                        "seed": seed,
                        "reason": (
                            f"ambiguous verified duplicates: {len(candidates)} candidates "
                            "and no unique canonical run_id directory; none used"
                        ),
                    }
                )
    return selected


# Canonical membership for each panel. Metadata comparison tags are advisory:
# they previously caused orthogonal variants to be overplotted as baseline.
COMPARISON_GROUPS: dict[str, dict[str, Any]] = {
    "dqn_target": {
        "algo": "dqn",
        "factor": "model.target_update_interval",
        "baseline": ("dqn_baseline", 5000),
        "variants": (("dqn_target_1000", 1000), ("dqn_target_10000", 10000)),
    },
    "dqn_exploration": {
        "algo": "dqn",
        "factor": "model.exploration_fraction",
        "baseline": ("dqn_baseline", 0.25),
        "variants": (("dqn_eps_010", 0.10), ("dqn_eps_040", 0.40)),
    },
    "ppo_clip": {
        "algo": "ppo",
        "factor": "model.clip_range",
        "baseline": ("ppo_baseline", 0.20),
        "variants": (("ppo_clip_010", 0.10), ("ppo_clip_040", 0.40)),
    },
    "ppo_lambda": {
        "algo": "ppo",
        "factor": "model.gae_lambda",
        "baseline": ("ppo_baseline", 0.95),
        "variants": (("ppo_lambda_090", 0.90), ("ppo_lambda_099", 0.99)),
    },
}


def _nested(config: dict[str, Any], path: str) -> Any:
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _compatible_signature(config: dict[str, Any], factor: str) -> str:
    """Resolved config with run identity, metadata and the ablated leaf removed."""
    clean = copy.deepcopy(config)
    for key in ("run", "metadata", "condition"):
        clean.pop(key, None)
    parts = factor.split(".")
    node = clean
    for part in parts[:-1]:
        node = node.get(part, {}) if isinstance(node, dict) else {}
    if isinstance(node, dict) and parts[-1] in node:
        node[parts[-1]] = "<ablated>"
    return json.dumps(clean, sort_keys=True, default=str)


def _eval_curve(path: Path) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    """One evaluation record per step; final reload wins over a periodic row."""
    frame = _read_csv(path)
    empty_info = {"raw_rows": 0, "deduped_steps": 0, "final_step_rows": 0}
    if frame.empty or not {"checkpoint_step", "mean_return_raw"}.issubset(frame.columns):
        return {}, empty_info
    frame = frame.copy()
    frame["checkpoint_step"] = pd.to_numeric(frame["checkpoint_step"], errors="coerce")
    frame["mean_return_raw"] = pd.to_numeric(frame["mean_return_raw"], errors="coerce")
    frame = frame.dropna(subset=["checkpoint_step", "mean_return_raw"])
    frame = frame[frame["mean_return_raw"].apply(lambda value: np.isfinite(float(value)))]
    if frame.empty:
        return {}, empty_info
    frame["is_final"] = (
        frame["is_final"].astype(str).str.lower().eq("true") if "is_final" in frame else False
    )
    frame["episodes"] = (
        pd.to_numeric(frame["episodes"], errors="coerce").fillna(0)
        if "episodes" in frame
        else 0
    )
    frame["std_return_raw"] = (
        pd.to_numeric(frame["std_return_raw"], errors="coerce").fillna(0.0)
        if "std_return_raw" in frame
        else 0.0
    )
    frame = frame.sort_values(
        ["checkpoint_step", "is_final", "episodes"], ascending=[True, False, False]
    )
    deduped = frame.drop_duplicates("checkpoint_step", keep="first")
    curve = {
        int(row.checkpoint_step): {
            "mean": float(row.mean_return_raw),
            "std": float(row.std_return_raw),
            "is_final": bool(row.is_final),
        }
        for row in deduped.itertuples()
    }
    info = {
        "raw_rows": int(len(frame)),
        "deduped_steps": int(len(curve)),
        "final_step_rows": int(frame["is_final"].sum()),
    }
    return curve, info


def plot_comparisons(output_root: Path) -> list[Path]:
    """Plot exactly baseline + two variants per panel, aggregated over matched seeds.

    Inclusion is by canonical condition identity and factor value, not by
    advisory metadata tags. Runs must have a completed, artifact-checked
    manifest whose identity matches the resolved config. Verified duplicate
    condition/seed candidates are resolved first, preferring the canonical
    run_id directory and refusing ambiguous uncanonical copies; each identity
    then casts one compatibility vote. Tied reference signatures are refused
    explicitly. Admitted candidates and actual contributors are audited
    separately, and contributors are populated only for a rendered panel.
    """
    output_root = Path(output_root)
    runs = _discover_runs(output_root)
    plots_dir = output_root / "plots"
    comparisons_dir = output_root / "comparisons"
    plots_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    audit: dict[str, Any] = {
        "schema_version": 1,
        "output_root": str(output_root),
        "canonical_groups": COMPARISON_GROUPS,
        "discovered_runs": len(runs),
        "groups": {},
        "rule": (
            "Exactly the canonical baseline and its two variants; require a completed "
            "manifest with all artifact checks true and matching run identity; one "
            "evaluation record per run/step (final wins); mean +/- sample SD (ddof=1) "
            "across matched seeds, blank when n=1; duplicates resolved before the "
            "compatibility vote; failed, unverified, incompatible, ambiguous, "
            "tied-reference and unmatched-seed runs excluded."
        ),
    }
    for group, spec in COMPARISON_GROUPS.items():
        expected: list[tuple[str, str, float]] = [
            ("baseline", *spec["baseline"]),
            *[("variant", *variant) for variant in spec["variants"]],
        ]
        included: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for run in runs:
            config = run["config"]
            if config.get("algo") != spec["algo"]:
                continue
            condition = str(config.get("condition", ""))
            match = next(
                (
                    (role, name, value)
                    for role, name, value in expected
                    if condition == name
                    and _nested(config, spec["factor"]) is not None
                    and np.isclose(
                        float(_nested(config, spec["factor"])), float(value), rtol=1e-9, atol=1e-12
                    )
                ),
                None,
            )
            if match is None:
                continue
            role, name, value = match
            run_dir = run["run_dir"]
            seed = int(config["run"]["seed"])
            manifest, manifest_error = _verify_manifest(run_dir / "run_manifest.json", config)
            if manifest_error is not None:
                excluded.append(
                    {
                        "run": str(run_dir),
                        "condition": name,
                        "seed": seed,
                        "reason": manifest_error,
                    }
                )
                continue
            curve, dedup_info = _eval_curve(run_dir / "metrics" / "eval_summary.csv")
            if not curve:
                excluded.append(
                    {
                        "run": str(run_dir),
                        "condition": name,
                        "seed": seed,
                        "reason": "no usable eval curve",
                    }
                )
                continue
            included.append(
                {
                    "role": role,
                    "condition": name,
                    "value": float(value),
                    "seed": seed,
                    "run_id": str(config["run"]["run_id"]),
                    "run_dir_name": run_dir.name,
                    "profile": str(config.get("profile")),
                    "signature": _compatible_signature(config, spec["factor"]),
                    "metadata_tags": config.get("metadata", {}).get("comparisons", []),
                    "manifest_status": manifest.get("status"),
                    "curve": curve,
                    "dedup": dedup_info,
                    "run_dir": str(run_dir),
                }
            )

        audit["groups"][group] = {
            "admitted_candidates": [],
            "included_runs": [],
            "contributors": [],
            "excluded_runs": excluded,
            "line_count": 0,
            "actual_contributor_count": 0,
            "matched_seeds": [],
            "matched_steps": [],
        }
        group_audit = audit["groups"][group]
        conditions = [name for _, name, _ in expected]
        present = {entry["condition"] for entry in included}
        if not set(conditions).issubset(present):
            group_audit["reason"] = (
                f"incomplete canonical set: missing {sorted(set(conditions) - present)}"
            )
            continue

        # Resolve verified duplicates BEFORE choosing the compatible-config
        # reference. Otherwise several noncanonical copies of one identity can
        # win the signature vote and displace every legitimate canonical run.
        # A unique canonical run_id directory takes precedence per identity;
        # genuinely ambiguous uncanonical copies are refused with a reason.
        selected = _select_duplicate_candidates(included, excluded)
        baseline_candidates = [
            entry for entry in selected if entry["condition"] == conditions[0]
        ]
        if not baseline_candidates:
            group_audit["reason"] = (
                "no verified baseline candidate remains after duplicate resolution"
            )
            group_audit["excluded_runs"] = excluded
            continue
        # Each resolved condition/seed identity casts exactly one vote. A tie
        # has no scientific basis for a reference, so refuse it explicitly
        # instead of silently falling back to directory/insertion order.
        signature_counts = Counter(entry["signature"] for entry in baseline_candidates)
        top_votes = max(signature_counts.values())
        leaders = [sig for sig, count in signature_counts.items() if count == top_votes]
        if len(leaders) > 1:
            for entry in selected:
                excluded.append(
                    {
                        "run": entry["run_dir"],
                        "condition": entry["condition"],
                        "seed": entry["seed"],
                        "reason": (
                            f"tied compatible-config reference: {len(leaders)} signatures "
                            f"with {top_votes} baseline votes each; none used"
                        ),
                    }
                )
            group_audit["excluded_runs"] = excluded
            group_audit["reason"] = (
                f"tied compatible-config reference signatures ({len(leaders)} at "
                f"{top_votes} votes each); no panel"
            )
            continue
        signature = leaders[0]
        incompatible = [entry for entry in selected if entry["signature"] != signature]
        for entry in incompatible:
            excluded.append(
                {
                    "run": entry["run_dir"],
                    "condition": entry["condition"],
                    "seed": entry["seed"],
                    "reason": "incompatible resolved config",
                }
            )
        admitted = [entry for entry in selected if entry["signature"] == signature]
        # The audit keeps every verified run compatible with the reference,
        # including noncanonical copies, while the vote and contributors use
        # the deduplicated `admitted` list above.
        group_audit["admitted_candidates"] = [
            {key: value for key, value in entry.items() if key != "curve"}
            for entry in included
            if entry["signature"] == signature
        ]
        group_audit["excluded_runs"] = excluded

        by_condition: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in conditions}
        for entry in admitted:
            by_condition[entry["condition"]][entry["seed"]] = entry
        matched_seeds = sorted(set.intersection(*[set(per_seed) for per_seed in by_condition.values()]))
        for entry in admitted:
            if entry["seed"] not in matched_seeds:
                excluded.append(
                    {
                        "run": entry["run_dir"],
                        "condition": entry["condition"],
                        "seed": entry["seed"],
                        "reason": (
                            f"unmatched seed {entry['seed']}: not present in all "
                            f"{len(conditions)} canonical conditions"
                        ),
                    }
                )
        group_audit["excluded_runs"] = excluded
        if not matched_seeds:
            group_audit["reason"] = "no seeds matched across all three conditions"
            continue
        contributors = [entry for entry in admitted if entry["seed"] in matched_seeds]
        by_condition = {name: {} for name in conditions}
        for entry in contributors:
            by_condition[entry["condition"]][entry["seed"]] = entry
        matched_steps = sorted(
            set.intersection(
                *[
                    set(by_condition[name][seed]["curve"])
                    for name in conditions
                    for seed in matched_seeds
                ]
            )
        )
        if not matched_steps:
            group_audit["reason"] = "no steps matched across all condition/seed curves"
            continue
        # Contributors and counts only describe a panel that actually rendered;
        # they stay empty/zero with the reason above when no steps matched.
        group_audit["contributors"] = [
            {key: value for key, value in entry.items() if key != "curve"}
            for entry in contributors
        ]
        group_audit["included_runs"] = group_audit["contributors"]
        group_audit["actual_contributor_count"] = len(contributors)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        colors = ["tab:blue", "tab:orange", "tab:green"]
        agg_rows: list[dict[str, Any]] = []
        n_seeds = len(matched_seeds)
        sample_sd = n_seeds >= 2
        for index, (role, name, value) in enumerate(expected):
            values = np.array(
                [
                    [by_condition[name][seed]["curve"][step]["mean"] for seed in matched_seeds]
                    for step in matched_steps
                ],
                dtype=float,
            )
            means = values.mean(axis=1)
            # Across-seed uncertainty is descriptive sample SD. It is undefined
            # for a single seed, so no spread band is drawn and the CSV is blank.
            stds = values.std(axis=1, ddof=1) if sample_sd else np.full(len(matched_steps), np.nan)
            factor_name = spec["factor"].split(".")[-1]
            uncertainty = (
                f"mean±SD (sample), n={n_seeds} seeds"
                if sample_sd
                else f"mean only, n={n_seeds} seed"
            )
            label = f"{role} {factor_name}={value:g} ({uncertainty})"
            ax.plot(matched_steps, means, marker="o", linewidth=1.8, color=colors[index], label=label)
            if sample_sd:
                ax.fill_between(
                    matched_steps,
                    means - stds,
                    means + stds,
                    color=colors[index],
                    alpha=0.15,
                )
            for step, mean, std in zip(matched_steps, means, stds):
                agg_rows.append(
                    {
                        "group": group,
                        "condition": name,
                        "role": role,
                        "factor": spec["factor"],
                        "value": value,
                        "step": step,
                        "mean_return_raw": float(mean),
                        "std_return_raw_across_seeds": (float(std) if sample_sd else None),
                        "n_seeds": n_seeds,
                        "seeds": ",".join(str(seed) for seed in matched_seeds),
                        "n_steps": len(matched_steps),
                    }
                )
        ax.set_xlabel("agent timesteps")
        ax.set_ylabel(
            "full-game raw return: mean ± sample SD across matched seeds"
            if sample_sd
            else "full-game raw return: mean across one seed (no SD)"
        )
        ax.set_title(f"{group}: baseline vs two variants ({len(matched_seeds)} matched seeds)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        path = plots_dir / f"compare_{group}.png"
        _save(fig, path)
        written.append(path)
        pd.DataFrame(agg_rows).to_csv(
            comparisons_dir / f"{group}_aggregated.csv", index=False
        )
        group_audit.update(
            {
                "line_count": len(expected),
                "matched_seeds": matched_seeds,
                "matched_steps": matched_steps,
                "standard_deviation": (
                    "sample (ddof=1)" if sample_sd else "undefined for n=1; band omitted, CSV blank"
                ),
                "plot": str(path),
                "aggregated_csv": str(comparisons_dir / f"{group}_aggregated.csv"),
            }
        )
    write_json(comparisons_dir / "comparison_manifest.json", audit)
    return written
