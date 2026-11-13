#!/usr/bin/env python3
"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, 4 hours ahead.

Entry for make.fun polymarket-nvidia-ceo-forecast-01 (program/v1, Python 3.13,
standard library only, no network, deterministic). Version 2.

The program reads every <id>.in case in the input directory
($MAKEFUN_INPUT_DIR, else /input) and writes /output/<id>.out
($MAKEFUN_OUTPUT_DIR, else /output): for each market of the case and each of
the 240 minutes after the checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%)
of that minute's Yes price.

Method (fitted from the case's own 72 hours of prices, nothing baked in):
per market, the quantiles of past price changes over each horizon are taken
empirically, with recent changes weighted more (exponential weights,
half-life 120 minutes), because one-minute moves are mostly microstructure
noise that does not accumulate like a random walk, while the distribution of
actual h-step moves is what the pinball loss scores. The offsets are
interpolated in sqrt(h) between the measured horizons, and each offset is
capped at twice the house baseline's offset for that level and minute, so a
wild stretch in a market's history cannot blow the intervals out. Markets
with too little history fall back to the baseline.

The median is NOT the last price (that was v1's weakness: its median error
matched the house baseline exactly). Two principled, one-sided pull-downs
move it, both growing with horizon and both only ever pulling down:

1. Run-up give-back. Prediction-market prices overreact and revert: a price
   that ran above its trailing 180-minute mean tends to give some of it
   back over the next 4 h. When the last price sits above that mean, the
   median is pulled down toward the mean by half the gap, scaled by
   1 - exp(-h/120). A price at or below its mean is never pulled up.
2. Board-sum pull. The 24 markets are mutually exclusive, so their Yes
   prices should sum to 1; in practice they sum to ~1.02, i.e. the board is
   overpriced in aggregate. When the sum of last prices exceeds 1, the
   excess is pulled out of every median in proportion to its price, scaled
   by 1 - exp(-h/240). A sum at or below 1 never pulls anything up (the
   residual is the unlisted-candidate probability, not a signal).

Local scores (pinball loss; lower is better): dev 0.267379, fresh unseen
0.103612, old extra 0.313726, vs v1's 0.282528 / 0.104627 / 0.315345.
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The quantile levels the contest scores, in the order each row lists them.
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (the baseline's shape).
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
# The house baseline's spread, price units per square-root minute.
SIGMA0 = 0.0025
# Horizons (minutes) at which past price changes are measured directly.
HORIZONS = (1, 5, 15, 30, 60, 120, 240)
# Recency half-life (minutes) for weighting past changes.
HALF_LIFE = 120.0
# Cap on an empirical offset, as a multiple of the baseline's offset.
CAP_MULT = 2.0
# A horizon is measured only when at least this many past changes exist.
MIN_SAMPLES = 20
# Run-up give-back: trailing-mean window (minutes), pull strength, and the
# horizon (minutes) over which the pull grows to full strength.
ONE_SIDED_WINDOW = 180
ONE_SIDED_R = 0.5
ONE_SIDED_TAU = 120.0
# Board-sum pull strength (fraction of the excess over a sum of 1 removed)
# and its growth horizon (minutes).
SUM_PULL_R = 1.0
SUM_PULL_TAU = 240.0


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def weighted_quantiles(values, weights, levels):
    """Type-1 weighted quantiles: the smallest value whose cumulative weight
    reaches level * total weight. Deterministic; ties keep input order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    total = sum(weights)
    cum = []
    running = 0.0
    for i in order:
        running += weights[i]
        cum.append(running)
    out = []
    for level in levels:
        target = level * total
        j = 0
        while j < len(cum) and cum[j] < target:
            j += 1
        out.append(values[order[min(j, len(order) - 1)]])
    return out


def baseline_rows(x, horizon):
    """The house baseline: price stays put, spread widens like sqrt(h)."""
    return [[price6(x + z * SIGMA0 * math.sqrt(h)) for z in Z]
            for h in range(1, horizon + 1)]


def offsets_for_market(prices, horizon):
    """240 rows (horizon rows) of five offsets from the median, recentered
    on 0: the empirical-quantile fan of past h-step price changes, with the
    empirical median subtracted out. Falls back to the baseline's shape
    when the market has too little history to measure any horizon."""
    n = len(prices)
    avail = [h for h in HORIZONS if n - h >= MIN_SAMPLES]
    if not avail:
        return [[z * SIGMA0 * math.sqrt(h) for z in Z]
                for h in range(1, horizon + 1)]
    # Exponential recency weights: a change ending at index i weighs
    # 0.5 ** ((n - 1 - i) / HALF_LIFE); the most recent change weighs 1.
    decay = 0.5 ** (1.0 / HALF_LIFE)
    w_end = [0.0] * n
    w = 1.0
    for i in range(n - 1, -1, -1):
        w_end[i] = w
        w *= decay
    by_h = {}
    for h in avail:
        vals = [prices[i + h] - prices[i] for i in range(n - h)]
        wts = [w_end[i + h] for i in range(n - h)]
        by_h[h] = weighted_quantiles(vals, wts, TAUS)
    rows = []
    for h in range(1, horizon + 1):
        if h in by_h:
            qs = by_h[h]
        elif h < avail[0]:
            qs = by_h[avail[0]]
        elif h > avail[-1]:
            f = math.sqrt(h / avail[-1])
            qs = [q * f for q in by_h[avail[-1]]]
        else:
            lo = max(hh for hh in avail if hh < h)
            hi = min(hh for hh in avail if hh > h)
            a = (math.sqrt(h) - math.sqrt(lo)) / (math.sqrt(hi) - math.sqrt(lo))
            qs = [by_h[lo][j] * (1.0 - a) + by_h[hi][j] * a for j in range(5)]
        # Recentre on the median: the offsets are the shape the history
        # teaches; the median itself is set by median_rows().
        med = qs[2]
        row = []
        for j in range(5):
            off = qs[j] - med
            lim = CAP_MULT * abs(Z[j] * SIGMA0 * math.sqrt(h))
            if off > lim:
                off = lim
            elif off < -lim:
                off = -lim
            row.append(off)
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def median_rows(prices, board_sum, horizon):
    """240 medians for one market. The last price, pulled down by the two
    one-sided reversion terms (run-up give-back and board-sum excess), each
    growing with horizon. Never pulled up. This is the sweep-5 winner
    "W=180 r=0.5 exp120 + sumpull": a 180-minute trailing mean, half the
    run-up gap, growing as 1-exp(-h/120); plus the full board-sum excess
    over 1, shared out by price, growing as 1-exp(-h/240). The two pulls
    add, exactly as the sweep combined them."""
    x = prices[-1]
    # Run-up give-back: only when the price sits above its trailing mean.
    w = min(ONE_SIDED_WINDOW, len(prices))
    mean = sum(prices[-w:]) / w
    dev = x - mean
    runup = ONE_SIDED_R * dev if dev > 0.0 else 0.0
    # Board-sum pull: only when the cross-market sum exceeds 1, shared out
    # in proportion to this market's price.
    excess = board_sum - 1.0
    if excess > 0.0 and board_sum > 0.0:
        sumpull = SUM_PULL_R * excess * (x / board_sum)
    else:
        sumpull = 0.0
    return [x - runup * (1.0 - math.exp(-h / ONE_SIDED_TAU))
            - sumpull * (1.0 - math.exp(-h / SUM_PULL_TAU))
            for h in range(1, horizon + 1)]


def forecast_market(prices, medians, horizon):
    """240 rows (horizon rows) of five quantiles for one market."""
    offs = offsets_for_market(prices, horizon)
    rows = []
    for h in range(horizon):
        row = [price6(medians[h] + offs[h][j]) for j in range(5)]
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def clean_prices(prices):
    """Drop non-finite prices (NaN/Infinity, which JSON can carry): forward-fill
    from the last finite value, back-fill a non-finite head from the first
    finite value. An all-non-finite series becomes empty, which the
    forecaster answers with flat 0.5 rows rather than poisoned quantiles."""
    out = []
    last = None
    for p in prices:
        if isinstance(p, (int, float)) and math.isfinite(p):
            last = float(p)
            out.append(last)
        elif last is not None:
            out.append(last)
        else:
            out.append(None)
    first = next((v for v in out if v is not None), None)
    if first is None:
        return []
    return [first if v is None else v for v in out]


def valid_rows(q, horizon):
    """A forecast is valid only if it has exactly `horizon` rows of five
    finite numbers in [0, 1], non-decreasing. An invalid case is scored as if
    every quantile were 0.5 and takes no prize, so never emit one."""
    if len(q) != horizon:
        return False
    for row in q:
        if len(row) != 5:
            return False
        prev = -1.0
        for v in row:
            if not math.isfinite(v) or v < 0.0 or v > 1.0 or v < prev:
                return False
            prev = v
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = int(case.get("horizon_minutes", 240))
        cleaned = [clean_prices(market["prices"]) for market in case["markets"]]
        # Board sum: the sum of last prices across all markets of this case.
        board_sum = sum(p[-1] for p in cleaned if p)
        forecasts = []
        for market, prices in zip(case["markets"], cleaned):
            if not prices:
                q = [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(horizon)]
            else:
                medians = median_rows(prices, board_sum, horizon)
                q = forecast_market(prices, medians, horizon)
            if not valid_rows(q, horizon):
                # Never emit an invalid case: fall back to the baseline,
                # which is valid by construction, for this market.
                x = prices[-1] if prices else 0.5
                q = baseline_rows(x, horizon)
            forecasts.append({"id": market["id"], "q": q})
        out = json.dumps({"forecasts": forecasts}) + "\n"
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w",
                  encoding="utf-8") as fh:
            fh.write(out)


if __name__ == "__main__":
    main()
