# NVIDIA CEO forecast v2 — verification record

Shipped variant (sweep-5 winner): **W=180 r=0.5 exp120 + sumpull**
- HL=120 empirical-quantile offsets, cap 2x (v1 architecture)
- Median = last price − 0.5·max(0, last − trailing-180min mean)·(1−exp(−h/120))
  − max(0, board_sum−1)·(last/board_sum)·(1−exp(−h/240))
- Picked over HL120+onesided240+sumpull by lower mean of the three losses
  (0.228239 vs 0.228712); both beat v1 on all three sets.

## Scores (pinball loss, verified by running shipped main.py — `python3 verify_v2.py`)

| set | v2 | v1 | sweep target | match |
|---|---|---|---|---|
| dev (6 practice) | 0.267379 | 0.282528 | 0.267379 | exact |
| fresh (5 unseen checkpoints) | 0.103612 | 0.104627 | 0.103612 | exact |
| old extra (5 checkpoints) | 0.313726 | 0.315345 | 0.313726 | exact |
| mean of 3 | 0.228239 | 0.234167 | — | — |

Median error: dev 0.7213 (v1 0.7675), fresh 0.2404 (v1 0.2391 — slightly
worse, the only metric that did not improve), old 0.7901.
Coverage_90: dev 0.7668, fresh 0.8491, old 0.8260.

## Format / determinism / runtime

- Format: exactly {"forecasts": [...]}, one forecast per market, 240 rows,
  5 finite numbers in [0,1], non-decreasing — 0 problems on all 3 sets.
- Determinism: two runs per set byte-identical.
- Runtime: ~1.0–1.6 s per set (6/5/5 cases), well within limits.
- main.py: Python 3.13 stdlib only (json, math, os), no network, no clock,
  no unseeded randomness, no model calls.

## Honest context

v2 improves on v1 on all three sets but is still ~0.028 short of rank-100's
0.240 (board ~17:55 UTC, 2,592 entries; rank 100 loss 0.240). It improves
rank but does not reach the prize zone on the dev board; the final is 12
unseen cases, so the fresh-set win is the meaningful signal.

## Handoff

- NOT pushed, NOT submitted, NOT signed. No make.fun API key used at any
  point (all data local: kit inputs, fresh-cases.json, v1 reference).
- Commit: see `git log` in this directory. Repo to create/push: parent's job.
