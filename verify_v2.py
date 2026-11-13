#!/usr/bin/env python3
"""Verify shipped main.py (v2) reproduces the sweep-5 scores on all 3 sets.

Builds input dirs for dev (kit), fresh (fresh-cases.json), old (experiments
OLD_EXTRA), runs main.py twice per set (determinism + runtime), scores with
v1's pinball logic, validates output format. Run from the work dir:
  python3 verify_v2.py
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

WORK = os.path.dirname(os.path.abspath(__file__))
V1 = "/home/hatch/workspace/makefun/nvidia-forecast-01"
KIT = os.path.join(V1, "polymarket-nvidia-ceo-forecast-01-kit")
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]

TARGETS = {
    "dev": 0.267379,    # sweep5: W=180 r=0.5 exp120 + sumpull
    "fresh": 0.103612,
    "old": 0.313726,
}
V1_SCORES = {"dev": 0.282528, "fresh": 0.104627, "old": 0.315345}


def build_inputs():
    base = os.path.join(WORK, "verify-tmp")
    if os.path.exists(base):
        shutil.rmtree(base)
    sets = {}
    # dev: kit inputs, answers from v1 reference
    dev_in = os.path.join(base, "dev-in")
    os.makedirs(dev_in)
    for cid in ["p00", "p01", "p02", "p03", "p04", "p05"]:
        shutil.copy(os.path.join(KIT, "input", f"{cid}.in"), dev_in)
    dev_ans = json.load(open(os.path.join(V1, "reference", "answers.json")))["cases"]
    sets["dev"] = (dev_in, dev_ans, sorted(dev_ans.keys()))
    # fresh: fresh-cases.json / fresh-answers.json
    fresh = json.load(open(os.path.join(WORK, "fresh-cases.json")))["cases"]
    fresh_ans = json.load(open(os.path.join(WORK, "fresh-answers.json")))["cases"]
    fr_in = os.path.join(base, "fresh-in")
    os.makedirs(fr_in)
    for cid, case in fresh.items():
        json.dump(case, open(os.path.join(fr_in, f"{cid}.in"), "w"))
    sets["fresh"] = (fr_in, fresh_ans, sorted(fresh_ans.keys()))
    # old: reconstructed by experiments.py from p05 + v1 raw histories
    sys.path.insert(0, WORK)
    import experiments as E
    old_in = os.path.join(base, "old-in")
    os.makedirs(old_in)
    for cid, case in E.OLD_EXTRA.items():
        safe = cid.replace(":", "").replace("-", "")
        json.dump(case, open(os.path.join(old_in, f"{safe}.in"), "w"))
    old_ans = {cid.replace(":", "").replace("-", ""): ans
               for cid, ans in E.OLD_EXTRA_ANS.items()}
    sets["old"] = (old_in, old_ans, sorted(old_ans.keys()))
    return base, sets


def run_main(indir, outdir):
    env = dict(os.environ, MAKEFUN_INPUT_DIR=indir, MAKEFUN_OUTPUT_DIR=outdir)
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(WORK, "main.py")],
                       env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print("main.py FAILED:", r.stdout, r.stderr)
        sys.exit(1)
    return dt


def dir_hash(d):
    h = hashlib.sha256()
    for f in sorted(os.listdir(d)):
        h.update(open(os.path.join(d, f), "rb").read())
    return h.hexdigest()


def score_set(outdir, answers, cids):
    tot_loss = tot_n = tot_med = tot_mm = tot_cov = 0.0, 0, 0.0, 0, 0
    tot_loss = 0.0
    tot_n = 0
    tot_med = 0.0
    tot_mm = 0
    tot_cov = 0
    per = {}
    for cid in cids:
        fc = json.load(open(os.path.join(outdir, f"{cid}.out")))["forecasts"]
        forecasts = {f["id"]: f["q"] for f in fc}
        ls = nn = 0.0, 0
        ls = 0.0
        nn = 0
        for mid, ys in answers[cid].items():
            q = forecasts[mid]
            for hh in range(240):
                y = ys[hh]
                row = q[hh]
                tot_med += abs(y - row[2])
                tot_mm += 1
                if row[0] <= y <= row[4]:
                    tot_cov += 1
                for j, tau in enumerate(TAUS):
                    d = y - row[j]
                    v = 100.0 * max(tau * d, (tau - 1.0) * d)
                    tot_loss += v
                    ls += v
                    tot_n += 1
                    nn += 1
        per[cid] = ls / nn
    return tot_loss / tot_n, 100 * tot_med / tot_mm, tot_cov / tot_mm, per


def validate(outdir, cids, indir):
    """Exactly {"forecasts": [...]}, one row per market, 240 rows, 5 finite
    numbers in [0,1], non-decreasing."""
    problems = []
    for cid in cids:
        case = json.load(open(os.path.join(indir, f"{cid}.in")))
        data = json.load(open(os.path.join(outdir, f"{cid}.out")))
        if set(data.keys()) != {"forecasts"}:
            problems.append(f"{cid}: top-level keys {sorted(data.keys())}")
        fc = data["forecasts"]
        if len(fc) != len(case["markets"]):
            problems.append(f"{cid}: {len(fc)} forecasts vs "
                            f"{len(case['markets'])} markets")
        ids_in = [m["id"] for m in case["markets"]]
        ids_out = [f["id"] for f in fc]
        if ids_out != ids_in:
            problems.append(f"{cid}: forecast ids/order differ")
        for f in fc:
            q = f["q"]
            if len(q) != 240:
                problems.append(f"{cid}/{f['id']}: {len(q)} rows")
                continue
            for row in q:
                if len(row) != 5:
                    problems.append(f"{cid}/{f['id']}: row len {len(row)}")
                    break
                prev = -1.0
                for v in row:
                    if not math.isfinite(v) or v < 0.0 or v > 1.0 or v < prev:
                        problems.append(
                            f"{cid}/{f['id']}: bad value {v} in {row}")
                        break
                    prev = v
    return problems


def main():
    base, sets = build_inputs()
    print("=== v2 verification (shipped main.py) ===")
    ok = True
    for name, (indir, answers, cids) in sets.items():
        out1 = os.path.join(base, f"{name}-out1")
        out2 = os.path.join(base, f"{name}-out2")
        t1 = run_main(indir, out1)
        t2 = run_main(indir, out2)
        h1, h2 = dir_hash(out1), dir_hash(out2)
        det = "IDENTICAL" if h1 == h2 else "DIFFERENT"
        if h1 != h2:
            ok = False
        loss, med, cov, per = score_set(out1, answers, cids)
        tgt = TARGETS[name]
        diff = loss - tgt
        match = "MATCH" if abs(diff) < 5e-7 else "MISMATCH"
        if abs(diff) >= 5e-7:
            ok = False
        probs = validate(out1, cids, indir)
        print(f"{name:6s} loss={loss:.6f} target={tgt:.6f} diff={diff:+.6f} "
              f"[{match}] med_err={med:.4f} cov90={cov:.4f} "
              f"v1={V1_SCORES[name]:.6f} beats_v1={loss < V1_SCORES[name]}")
        print(f"       runtime run1={t1:.2f}s run2={t2:.2f}s "
              f"determinism={det} format_problems={len(probs)}")
        for cid, v in per.items():
            print(f"       {cid}: loss={v:.6f}")
        for p in probs[:5]:
            print("       FORMAT PROBLEM:", p)
            ok = False
    print("=== OVERALL:", "PASS" if ok else "FAIL", "===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
