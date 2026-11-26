# CSE 482 Reinforcement Learning — Assignment 2

Small, readable Stable-Baselines3 implementation of the two assigned experiments:

| Task | Environment | Agent | Comparison |
| --- | --- | --- | --- |
| Visual discrete control | `ALE/Breakout-v5` | DQN + `CnnPolicy` (SB3 `NatureCNN`) | target update interval, epsilon decay fraction |
| Continuous control | `BipedalWalker-v3` | PPO + `MlpPolicy` | clip range, GAE lambda |

This worktree contains the implementation, resolved configs, checkpoint-reload
evaluation, diagnostic plots, deterministic wrapper tests, and fresh short
verification evidence. **No 500,000-step training run and no full experiment
queue was started.** Full and smoke schedules are deliberately different, and
the current host did not demonstrate enough free RAM for a full DQN launch.

## Contents

```
configs/
  base_dqn.yaml, base_ppo.yaml       shared hyperparameters and wrapper chains
  profiles/full.yaml, smoke.yaml     schedule overlay (500k vs short smoke)
  conditions/*.yaml                  ten one-factor conditions
rl_a2/
  config.py        YAML layering, validation, replay formula, RAM headroom gate
  wrappers.py      MaxAndSkip, auto-FIRE, warp, clipping, raw episode stats
  envs.py          DQN/PPO env factories and environment descriptions
  probes.py        deterministic fixed probe set + mean max Q
  callbacks.py     optimizer counts, sampled losses, episodes, final diagnostics
  evaluate.py      checkpoint reload + full-game evaluation + clips
  train.py         train/evaluate/plot orchestration, manifest, source provenance
  plots.py         per-run diagnostics and canonical comparison plots
  provenance.py    reproducible SHA-256 source digest and Git state
scripts/
  run_all.py                serial ten-condition runner
  preflight.py              native ALE/Box2D, versions, replay estimate
  verify_wrappers.py        deterministic wrapper contract tests
  verify_ablations.py       resolved-config one-factor diff check
  summarize_smoke.py        smoke evidence + explicitly reconstructed ETA
  plot_comparisons.py       comparison plots from completed runs
  verify_final_diagnostics.py  off-grid/no-update DQN final-row regression
  verify_comparison_plots.py   canonical comparison fixture regression
  remediate_evidence.py     fresh isolated evidence driver
```

## Environment used here

The host had only the WindowsApps Python alias and no WSL/`nvidia-smi`. A
workspace-local standalone `uv` and `uv`-managed CPython 3.11 were installed
under `.tools/`; all packages and caches are workspace-local. The final
interpreter is:

```text
C:\Users\ezanr\orca\workspaces\rl-a2\rl-a2-implementation\.venv\Scripts\python.exe
```

Versions recorded in the fresh manifests: Python 3.11.18, Stable-Baselines3
2.7.0, Gymnasium 1.2.1, PyTorch 2.15.0+cpu, NumPy 2.4.6, ale-py 0.12.1,
Box2D 2.3.10. PyTorch is CPU-only (`torch_cuda_available=false`). Native ALE
ROMs and the native Box2D wheel were actually reset and stepped; see
`artifacts/remediation/evidence_final2/preflight/preflight.json`.

### Recreate the environment (PowerShell, from the project root)

```powershell
New-Item -ItemType Directory -Force .tools | Out-Null
curl.exe -sSL -o .tools\uv.zip https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip
Expand-Archive .tools\uv.zip .tools -Force
$env:UV_PYTHON_INSTALL_DIR="$PWD\.tools\python"
$env:UV_CACHE_DIR="$PWD\.tools\uv-cache"
.\.tools\uv.exe python install 3.11
.\.tools\uv.exe venv --python 3.11 .venv
.\.tools\uv.exe pip install --python .\.venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
.\.tools\uv.exe pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

`Box2D==2.3.10` supplies the native `Box2D` module with a cp311 Windows wheel;
`gymnasium[box2d]` would try to build `box2d-py` with SWIG, so this project
installs the wheel directly. `pygame` is installed because Gymnasium imports it
when the Box2D environments are registered. No system packages were installed
during remediation.

## Quick verification (safe, short)

Use fresh output roots when reproducing; training and standalone evaluation
refuse non-empty run/output directories.

```powershell
# Native dependencies, versions, RAM, ROM path, reset/step evidence
.\.venv\Scripts\python.exe scripts\preflight.py --output-root artifacts\verification\preflight

# Deterministic wrapper tests on real Breakout (auto-FIRE, skip, stacking, raw score)
.\.venv\Scripts\python.exe scripts\verify_wrappers.py --output artifacts\verification\wrapper_tests.json

# Exactly one semantic factor per ablation; metadata-only differences ignored
.\.venv\Scripts\python.exe scripts\verify_ablations.py --profile full --seed 0 --output-root artifacts\verification\ablation_check --report artifacts\verification\ablation_check\ablation_diff_full_seed0.json
.\.venv\Scripts\python.exe scripts\verify_ablations.py --profile smoke --seed 0 --output-root artifacts\verification\ablation_check --report artifacts\verification\ablation_check\ablation_diff_smoke_seed0.json
```

These write only inside `artifacts/` and are read-only with respect to
implementation files.

## Smoke runs that were actually executed

The final fresh evidence root is `artifacts/remediation/evidence_final2`. The
recorded commands were:

```powershell
.\.venv\Scripts\python.exe -m rl_a2.train --condition dqn_baseline --profile smoke --seed 0 --output-root artifacts\remediation\evidence_final2\smoke_reruns
.\.venv\Scripts\python.exe -m rl_a2.train --condition ppo_baseline --profile smoke --seed 0 --output-root artifacts\remediation\evidence_final2\smoke_reruns
.\.venv\Scripts\python.exe -m rl_a2.evaluate --run-dir artifacts\remediation\evidence_final2\smoke_reruns\dqn_baseline_seed0_smoke --checkpoint final --episodes 2 --output-dir artifacts\remediation\evidence_final2\reloads\dqn_final --no-video
.\.venv\Scripts\python.exe -m rl_a2.evaluate --run-dir artifacts\remediation\evidence_final2\smoke_reruns\ppo_baseline_seed0_smoke --checkpoint final --episodes 2 --output-dir artifacts\remediation\evidence_final2\reloads\ppo_final --no-video
.\.venv\Scripts\python.exe scripts\summarize_smoke.py --output-root artifacts\remediation\evidence_final2\smoke_reruns --preflight artifacts\remediation\evidence_final2\preflight\preflight.json
```

Fresh measured outcomes:

| Outcome | DQN smoke | PPO smoke |
| --- | ---: | ---: |
| Requested / actual agent steps | 6,000 / 6,000 | 4,000 / 4,096 (rollout overshoot) |
| Optimizer updates | 1,450 | 256 |
| Periodic / final checkpoints | 2 / 1 | 2 / 1 |
| Sampled process peak RSS | 1,504.7 MiB | 495.7 MiB |
| Available RAM at run start | 1,806.2 MiB | 1,733.4 MiB |
| Training-only agent steps/s | 105.129 | 1,133.897 |
| Final reload mean raw return | 0.0 | -84.1275181483943 |

Both runs completed real optimizer updates, saved and reloaded checkpoints,
reproduced their final evaluation, wrote all diagnostics, and produced MP4
gameplay clips. Smoke scores are implementation evidence only, not scientific
results. The DQN final score of 0 means the 6,000-step run never cleared a
brick.

## Full experiment commands (not run here)

The ten conditions run **serially**. There is no `--overwrite` option: a
non-empty run directory is refused, and existing evidence is never deleted or
merged. Use a fresh output root for every seed set. `--seeds 0 1 2` gives three
matched seeds across all ten conditions.

```powershell
# One matched seed: 10 conditions x 500,000 agent steps
.\.venv\Scripts\python.exe scripts\run_all.py --profile full --seeds 0 --output-root artifacts\full_seed0

# Three matched seeds: 30 runs, still serial
.\.venv\Scripts\python.exe scripts\run_all.py --profile full --seeds 0 1 2 --output-root artifacts\full_seeds012

# One condition / one seed directly
.\.venv\Scripts\python.exe -m rl_a2.train --condition dqn_target_1000 --profile full --seed 0 --output-root artifacts\full_seed0
```

Full profile: 500,000 requested agent timesteps per condition per seed;
checkpoints every 50,000 requested steps (PPO's first rollout boundary at or
after each boundary, e.g. 51,200); 5 full-game evaluation episodes per
periodic/final checkpoint. All other settings are shared. A full DQN launch is
gated by currently available RAM (see below) and will refuse to start when the
verified headroom is insufficient. **These queues were not started here.**

## Conditions and one-factor map

| Condition | Factor | Baseline | Comparison groups |
| --- | --- | --- | --- |
| `dqn_baseline` | target=5000, eps_fraction=0.25 | — | target + exploration |
| `dqn_target_1000` | `model.target_update_interval=1000` | 5000 | target |
| `dqn_target_10000` | `model.target_update_interval=10000` | 5000 | target |
| `dqn_eps_010` | `model.exploration_fraction=0.10` | 0.25 | exploration |
| `dqn_eps_040` | `model.exploration_fraction=0.40` | 0.25 | exploration |
| `ppo_baseline` | clip=0.20, lambda=0.95 | — | clip + lambda |
| `ppo_clip_010` | `model.clip_range=0.10` | 0.20 | clip |
| `ppo_clip_040` | `model.clip_range=0.40` | 0.20 | clip |
| `ppo_lambda_090` | `model.gae_lambda=0.90` | 0.95 | lambda |
| `ppo_lambda_099` | `model.gae_lambda=0.99` | 0.95 | lambda |

`scripts/verify_ablations.py` resolves all configs and checks that each
condition differs from its baseline in exactly the intended leaf. Run identity,
output paths, condition names, and metadata labels are metadata-only and
excluded from the semantic diff. Seeds, total timesteps, evaluation settings,
wrapper chains, replay capacity, and every non-ablated hyperparameter must
match. Cross-group comparison tags were removed from all eight variant YAMLs;
comparison membership is now canonical code, not advisory metadata.

## Breakout wrapper chain and temporal accounting

```
ALE/Breakout-v5(frameskip=1, repeat_action_probability=0.0, full_action_space=False)
  MaxAndSkipExternal(skip=4)          # the only external action-repeat/max-pool layer
  FireOnResetAndLifeLoss(fire_action=1)
  WarpFrame84x84                      # RGB -> grayscale, INTER_AREA, uint8
  FrameStackObservation(stack_size=4) # channel-first (4, 84, 84)
  StackedFramesToArray
  ClipRewardSign                      # training only
  RawEpisodeStats                     # raw return from info["raw_reward"]
  Monitor                             # SB3 learning statistics
```

* ALE runs at internal frameskip **1**. `MaxAndSkipExternal` repeats the
  selected action for four ALE frames, returns `max(frame 3, frame 4)` (the
  standard Atari max-pool convention), sums reward over the four frames, and
  advances exactly four frames on normal steps.
* Auto-FIRE is **above** the skip layer. A FIRE override is an ordinary
  external action: it executes four ALE frames and produces one stacked
  observation, so the frame budget and frame-stack timing stay consistent.
  Reset FIRE runs inside `reset` (four frames, not an agent timestep, not
  counted in `num_timesteps`). Because Breakout returns zero reward for launch,
  no score is hidden; nonzero reset reward would be carried into the first
  external step's reward.
* `FireOnResetAndLifeLoss` watches ALE's `lives` counter. On a **nonterminal**
  life loss it sets a one-step override; the next policy action is executed as
  FIRE, with `info["requested_action"]` retaining what the policy asked for and
  `info["executed_action"]=1`. Thus even a greedy policy that never outputs
  FIRE launches the next ball. No extra hidden timestep is inserted.
* Training clips reward to sign only for optimization. `info["raw_reward"]` and
  `RawEpisodeStats` keep the unmodified external score recoverable.
  `train_episodes.csv` records both the raw and training reward streams.
* There is **no EpisodicLife wrapper anywhere**. Training and evaluation
  boundaries are full games. Evaluation uses no reward clipping, so
  `Monitor`'s episode return is the raw external game score.
* The deterministic wrapper test asserts reset FIRE, next-step FIRE after a
  life loss while the scripted policy only uses NOOP/RIGHT/LEFT, a 4-frame
  step delta, channel ordering/temporal stack order, and raw-return
  recoverability. Evidence:
  `artifacts/remediation/evidence_final2/verification/wrapper_tests.json`.

Action meanings: `0=NOOP`, `1=FIRE`, `2=RIGHT`, `3=LEFT` (verified from the
environment). Observation shape `(4, 84, 84)`, dtype `uint8`, channel-first;
NatureCNN output for a batch of two was verified as `(2, 512)`.

## Evaluation protocol

* Fixed evaluation seeds: DQN `1001, 1002, ...`; PPO `2001, 2002, ...`.
  Every checkpoint and every ablation uses the same seed sequence.
* Deterministic greedy actions; fresh environment per episode; no training
  clip; no EpisodicLife; full games.
* Native limits only: `BipedalWalker-v3` has a 1,600-step `TimeLimit`;
  Breakout has no Gymnasium time limit, so ALE's own 108,000-internal-frame
  (27,000 agent-step) cap is the only potential truncation. `terminated` and
  `truncated` are recorded separately; `safety_cap_hit` is false by default
  (`evaluation.safety_max_steps: null`). No artificial evaluation cap is used.
* Primary return is Monitor's episode return, cross-checked by
  `RawEpisodeStats`; per-episode rows store both. `ale-py 0.12`'s Python
  `ALEInterface` does not expose `getScore`, so raw reward is the authoritative
  external score and no fake ALE-score field is invented.
* Gameplay clips are separate: only episode 0 of the final reload evaluation is
  recorded, trimmed to `video_max_steps` (600 full / 300 smoke), MP4 H.264 with
  an imageio-ffmpeg fallback to GIF. Short clips never shorten evaluation.

## Diagnostics and artifacts

Each run directory has:

```
config_resolved.json                 exact merged model/env/wrapper/training config
run_manifest.json                    command, versions, timings, RAM, digest, status, checks
metrics/updates.csv                  DQN: sampled optimizer updates; PPO: every rollout update
metrics/train_log.csv                epsilon, TD/Huber loss, mean max Q, PPO diagnostics
metrics/train_episodes.csv           raw and training-reward episode returns
metrics/eval_episodes.csv            raw per-episode scores, seeds, length, terminated/truncated
metrics/eval_summary.csv             per-checkpoint means/std, mean max Q, probe hash
metrics/checkpoint_manifest.csv      periodic and final checkpoint reload verification
metrics/training_summary.json        actual timesteps, update counts, final diagnostic
checkpoints/ckpt_*.zip, final_model.zip
probes/probe_obs.npz + .json         fixed probe observations and hash
clips/*.mp4|gif                      short representative final-evaluation gameplay
plots/*.png                          per-run diagnostics
logs/train.log                       stdout training log
```

### Final DQN diagnostic row

The train curve is sampled on `log_interval`, so the final timestep can fall
off the grid. The final row is **never** copied from the last sampled row:

* DQN Huber/TD loss is recorded only on timesteps with a real optimizer update
  from the wrapped `model.train`/`optimizer.step` counters. A final event with
  `updates_this_call=0` leaves both `loss` and `td_huber_loss` blank in
  `train_log.csv` and `updates.csv`; the measured loss stays on its update row.
* Current DQN epsilon and online mean max Q are retained at finalization. Q is
  recomputed on the current model from the run-local fixed probe set; target
  network Q is not used. PPO's epsilon/Q fields stay blank.
* `last_update_timestep`, `final_update_aligned`, and an explicit `note` state
  the update situation. Notes are per algorithm: DQN says the loss is blank
  when no update happened at the final step, while PPO names its diagnostics
  as rollout statistics and calls epsilon/Q inapplicable.
* A final event is also written to `updates.csv` when needed. The summary's
  `final_diagnostic` mirrors the final CSV row and records `last_update_loss`
  separately so an off-update final still preserves the measured update loss.

Analytic full-profile schedule (not a run): first optimizer update at 20,004,
last logged train step 499,004, final optimizer update and final diagnostic at
500,000, 120,000 updates, 10 periodic checkpoints plus the final model. The
final 500,000 step is off the train-log sample grid but does have an optimizer
update. Regression evidence covers that off-grid case, a 100-step warm-up with
no update, a forced 510-step final after a 508-step update (with no loss point
or updated epsilon misattributed to step 510), and an aligned PPO rollout:
`artifacts/remediation/final_diagnostics_final3/final_diagnostics_report.json`.

### Fixed mean max Q probe

For DQN, the probe set is generated once per output root with seed `20261119`,
32 observations, four warmup steps, stride 1, and action cycle `[2, 3, 0]`
(NOOP/RIGHT/LEFT; never FIRE). The exact algorithm is recorded in
`probes/probe_obs.json`; the SHA-256 is over canonical uint8 bytes + shape +
dtype. Every run copies the probe into its own directory and verifies the hash.
Evidence hash:

```text
3cae32dca407415b02b4639cad2b41b4856fcf0e2a461f3e6ef43978fa1449e6
```

The same observations are used at every checkpoint and every DQN ablation;
`mean_max_q` is the mean over probes of `max_a Q_online(s,a)` from the reloaded
checkpoint.

### Update alignment, losses, and checkpoint counts

`model.train` is wrapped. After each call, the actual optimizer-step counter is
read and `train/loss` is the mean sampled loss from that call. DQN uses SB3's
Huber loss (`smooth_l1_loss`) and one optimizer step per `train_freq=4` call.
PPO's row is per rollout; `updates_this_call` is the actual number of minibatch
optimizer steps (e.g. `32`), and the row reports the actual sampled
policy/value/entropy/KL means. PPO's actual timestep can exceed the request
(rollout overshoot), which is recorded.

`periodic_checkpoints` counts `checkpoint_manifest.csv` rows whose `checkpoint`
is not `final`; `checkpoint_events`/`total` counts all rows including the final
reload. Both counts are recomputed in `smoke_summary.json` and cross-checked
against `training_summary.json`, fixing the old historical mismatch.

PPO plots label `policy_entropy = -train/entropy_loss` in nats. SB3 logs the
negative entropy loss; the sign is reversed for the diagnostic. Continuous
Gaussian differential entropy can legitimately be negative. The plot label for
approximate KL is explicitly scoped to **SB3's final-epoch minibatches only**;
SB3's value loss and entropy are averaged across optimization epochs. Do not
read the KL trace as an all-epoch aggregate.

## Plots

Per run: `plots/{dqn,ppo}_learning_curve.png`, `*_eval_returns.png`,
`*_final_score.png`; DQN additionally `dqn_epsilon.png`,
`dqn_td_huber_loss.png`, `dqn_mean_max_q.png`; PPO additionally
`ppo_approx_kl.png`, `ppo_value_loss.png`, `ppo_entropy.png`,
`ppo_clip_fraction.png`.

Cross-condition comparison plots (once multiple full conditions exist):

```powershell
.\.venv\Scripts\python.exe scripts\plot_comparisons.py --output-root artifacts\full_seeds012
```

Outputs identify the four comparison families:
`plots/compare_dqn_target.png`, `compare_dqn_exploration.png`,
`compare_ppo_clip.png`, `compare_ppo_lambda.png`. Each panel contains exactly
the canonical baseline and its two factor variants, identified from the
resolved config's condition name and factor value. One evaluation record per
run/step is used (the final reload wins over a periodic row with the same
step), and curves are averaged across matched seeds with mean ± **sample** SD
(`ddof=1`) bands for `n>=2`. A single seed has undefined between-seed SD: no
band is drawn, the CSV SD is blank, and the label says mean only. These bands
describe between-seed spread, not significance or episode-level variability.

A candidate needs a completed manifest whose `run_id`, condition, profile,
algorithm, and seed match its resolved config and whose artifact checks are all
true; absent, unreadable, failed, or wrong-identity manifests are excluded.
Verified duplicate condition/seed candidates prefer the unique canonical
`run_id` directory, and ambiguous duplicates are refused with an explicit
reason. Incompatible configs, unmatched seeds, and unusable curves are also
excluded. The audit lists `admitted_candidates`, `contributors`,
`actual_contributor_count`, and per-run exclusion reasons separately.

Audit outputs: `comparisons/comparison_manifest.json` and
`comparisons/<group>_aggregated.csv`. Fixture evidence:
`artifacts/remediation/comparison_plots_final4/comparison_plot_report.json`
(36 fixture runs, 30 matched, 47 checks: four panels with three correct lines
each, duplicate final-step deduplication, absent/wrong-identity manifest and
unmatched-seed exclusion, canonical duplicate preference, sample SD exactly
3.0 versus population 2.44949, and single-seed blank SD with no band).

## Provenance and evidence isolation

Run and queue manifests carry a `source_digest` object:

* Algorithm: SHA-256 over each included file's sorted relative path, byte
  length, and bytes; per-file hashes and file count are recorded.
* Included: `rl_a2/`, `scripts/`, `configs/`, `requirements.txt`,
  `requirements-lock.txt`, and `.gitignore`.
* Excluded: `README.md` (narrative evidence that records digest values, so
  including it would be circular), `artifacts/`, `.venv/`, `.tools/`, caches,
  and Git internals.
* Git HEAD, dirty status, and dirty line count are recorded separately. This
  snapshot's HEAD is `c50b5d84c989a77ff076f142c48721e275bd5857` and the
  implementation is untracked/dirty; no commit was made.

Manifests also record `started_at`/`finished_at`, epoch values, run wall time,
versions, command, seed, paths, status, artifact checks, timing scopes, host
memory, replay estimate, and memory-headroom verdict.

Training refuses a non-empty run directory and standalone evaluation refuses a
non-empty output directory. There is no overwrite flag; nothing is deleted or
merged. Earlier partial and `_final` remediation directories were retained;
the current evidence uses new fresh roots only.

## Fresh remediation evidence and exact commands

All three checks were run serially into fresh directories:

```powershell
.\.venv\Scripts\python.exe scripts\verify_comparison_plots.py --output-root artifacts\remediation\comparison_plots_final4
.\.venv\Scripts\python.exe scripts\verify_final_diagnostics.py --output-root artifacts\remediation\final_diagnostics_final3
.\.venv\Scripts\python.exe scripts\remediate_evidence.py --output-root artifacts\remediation\evidence_final2
```

Reports and outcomes:

* `artifacts/remediation/comparison_plots_final4/comparison_plot_report.json` —
  `ok: true`, 47 checks.
* `artifacts/remediation/final_diagnostics_final3/final_diagnostics_report.json`
  — `regression_ok: true`; full-profile RAM headroom `ok: false` (correctly
  blocking launch).
* `artifacts/remediation/evidence_final2/remediation_report.json` — `ok: true`;
  all 25 checks true, including corrected replay formula, preserved wrappers,
  one-factor ablations at seeds 0/1/2, real optimizer updates, consistent
  checkpoint counts, recorded provenance, and reproduced final evaluations.

Final source digest from the fresh evidence:
`04e0be85b43148ff01aaf74a90af6e2b13b0c1f52e59ca06f7bc5e0eeb979f67`
over 36 files. Host available RAM was 1,997.6 MiB before and after the evidence
driver; the fresh smoke runs started with 1,806.2 MiB (DQN) and 1,733.4 MiB
(PPO). Earlier observations ranged roughly 670–2,132 MiB.

## Memory, throughput, and full-run planning evidence

Replay accounting was corrected for installed SB3 2.7.0. `done` and `timeout`
are `float32`, not one byte, so a transition is **56,468 bytes**:
`2 x obs(uint8, 4x84x84) + action(int64) + reward(float32) + done(float32) +
timeout(float32)`. `n_steps=1` allocates no discount array.

* Full DQN replay capacity is **5,000** in all five DQN conditions and seeds:
  282,340,000 bytes = **269.26 MiB** of SB3 array bytes. Smoke uses 2,000
  (107.7 MiB) only for short plumbing checks. The old 20,000 capacity would
  need about 1,077 MiB in arrays alone and was reduced.
* These are array bytes only; Python/NumPy object overhead, allocator slack,
  and the rest of the process are additional.
* The conservative non-replay process constant is **1,400.0 MiB**, rounded up
  from the fresh smoke's 1,504.7 MiB peak RSS minus 107.7 MiB replay
  (1,397.0 MiB). Projected full DQN process peak is **1,669.3 MiB**.
* With 25% slack, the launch gate requires **2,086.6 MiB currently available**
  RAM. At the final diagnostic check only 1,617.2 MiB was available (deficit
  469.4 MiB); the evidence driver saw 1,997.6 MiB, and smoke starts saw less.
  Sustained headroom under queue conditions was not demonstrated. Do not
  launch a full DQN run on this host without freeing substantially more RAM,
  rechecking the gate immediately before launch, and running a short
  representative full-profile pilot first.
* Throughput numbers in this README are training-only and exclude periodic
  evaluation and checkpoint saves; exact scopes are in each manifest and in
  `smoke_summary.json`.

### Reconstructed ETA (planning only, not a measured full-run estimate)

From fresh smoke training-only rates (DQN 105.129 and PPO 1,133.897 steps/s),
linear extrapolation to 500,000 steps gives approximately 1.32 h per DQN
condition, 0.12 h per PPO condition, 7.22 h for ten conditions with one seed,
and 21.7 h for ten conditions with three matched seeds. These are **not**
launch estimates: full PPO uses 10 epochs per rollout versus 4 in smoke (2.5x
optimizer work per agent step at the same batch size), full DQN uses 5,000
replay entries and a 20,000-step warm-up versus 2,000/200 in smoke, evaluation
length can grow with learned skill, and checkpoint/final overhead is scaled
from smoke. A short representative full-profile pilot with periodic save/eval
and final overhead timed separately is required before relying on any ETA.

## Known limitations

* Full 500k runs and three-seed replication were not executed; all scientific
  comparisons must be produced with the serial commands above after the RAM
  and pilot prerequisites are met.
* Smoke profiles change buffer capacity, learning starts, PPO rollout length
  and epochs, checkpoint cadence, and eval episode count. They are not directly
  comparable to full runs; only full-profile runs are the intended experiment.
* The DQN evaluation cap is ALE's native frame cap; `ale-py` does not expose a
  Python `getScore` in this version, so raw reward is the score definition.
* CPU-only PyTorch and low, variable host RAM make DQN slow; the full DQN
  launch gate currently fails.
* No `VecNormalize` for BipedalWalker: this keeps the setup standard/readable
  but may change PPO sample efficiency relative to normalized implementations.
* Determinism is seeded (SB3/PyTorch/Gymnasium) but not bitwise-guaranteed
  across library versions or CPU thread scheduling.
* The 4–5 page synthesis, behavioral clip analysis, and comparative scientific
  claims remain outstanding; the smoke clips are implementation checks only.
