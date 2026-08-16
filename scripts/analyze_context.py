#!/usr/bin/env python3
"""
T3 analysis: context length scaling.

Merges every T3 run under results/raw/, then answers two questions:

  1. How does prefill cost grow with context? A linear fit is adequate up to
     ~16k tokens; beyond that the quadratic attention term dominates, so both
     a linear and a quadratic fit are reported with their residuals.

  2. How much does a long context slow down decode? Each decoded token must
     read the full weight set plus the accumulated KV cache, so the predicted
     slowdown is (weights + KV) / weights. The measured loss is compared
     against that first-principles model.

Usage:
    python scripts/analyze_context.py
    python scripts/analyze_context.py --plot
    python scripts/analyze_context.py --runs id1,id2 --plot
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

RAW_DIR = Path("results/raw")
PROC_DIR = Path("results/processed")
PLOT_DIR = Path("results/plots")

# Hardware / model constants used by the first-principles decode model.
WEIGHTS_GB = 5.42          # BF16 weights, from vLLM startup log (5.05 GiB)
KV_BYTES_PER_TOKEN = 16_400  # derived: 7.46 GiB / 488,404 tokens
BANDWIDTH_GB_S = 672.0     # RTX 4070 Ti SUPER, 256-bit GDDR6X


def pct(values, p):
    s = sorted(values)
    return s[max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))]


def discover(scenario="T3"):
    out = []
    for d in sorted(RAW_DIR.iterdir()):
        if not d.is_dir() or not (d / "manifest.json").exists():
            continue
        try:
            man = json.loads((d / "manifest.json").read_text())
        except Exception:
            continue
        if man.get("scenario") == scenario:
            out.append((d.name, man))
    return out


def load_cells(run_ids):
    """Return {isl_actual: stats} merged across runs."""
    buckets = {}
    for rid in run_ids:
        with (RAW_DIR / rid / "requests.jsonl").open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("status") != "ok":
                    continue
                buckets.setdefault(r["isl_target"], []).append(r)

    cells = []
    for isl_target, recs in sorted(buckets.items()):
        ttfts = [r["ttft_ms"] for r in recs]
        tpots = [statistics.mean(r["itls_ms"]) for r in recs if r["itls_ms"]]
        pooled = [x for r in recs for x in r["itls_ms"]]
        tpot = statistics.mean(tpots)
        isl = statistics.median([r["prompt_tokens"] for r in recs])
        cells.append({
            "isl_target": isl_target,
            "isl_actual": isl,
            "n": len(recs),
            "ttft_p50": pct(ttfts, 50),
            "ttft_p95": pct(ttfts, 95),
            "tpot_mean": tpot,
            "speed_tok_s": 1000 / tpot,
            "itl_p99": pct(pooled, 99),
            "cv_pct": (statistics.stdev(tpots) / tpot * 100) if len(tpots) > 1 else 0.0,
            "prefill_tok_s": isl / (pct(ttfts, 50) / 1000),
        })
    return cells


def polyfit(xs, ys, degree):
    """Least squares via normal equations. Returns coefficients low->high."""
    n = degree + 1
    A = [[sum(x ** (i + j) for x in xs) for j in range(n)] for i in range(n)]
    b = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(n)]
    # Gaussian elimination with partial pivoting.
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        for r in range(col + 1, n):
            f = A[r][col] / A[col][col]
            for c in range(col, n):
                A[r][c] -= f * A[col][c]
            b[r] -= f * b[col]
    coef = [0.0] * n
    for i in reversed(range(n)):
        coef[i] = (b[i] - sum(A[i][j] * coef[j] for j in range(i + 1, n))) / A[i][i]
    return coef


def evaluate(coef, x):
    return sum(c * x ** i for i, c in enumerate(coef))


def r_squared(xs, ys, coef):
    my = statistics.mean(ys)
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - evaluate(coef, x)) ** 2 for x, y in zip(xs, ys))
    return 1 - ss_res / ss_tot if ss_tot else float("nan")


def predicted_decode_loss(isl):
    """Bytes read per decoded token: weights + KV cache. Returns percent."""
    kv_gb = isl * KV_BYTES_PER_TOKEN / 1e9
    return kv_gb / WEIGHTS_GB * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--linear-cutoff", type=int, default=16384,
                    help="fit the linear model only up to this ISL")
    ap.add_argument("--label", default="t3")
    args = ap.parse_args()

    if args.runs:
        run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
        manifests = {r: json.loads((RAW_DIR / r / "manifest.json").read_text())
                     for r in run_ids}
    else:
        found = discover()
        if not found:
            raise SystemExit("No T3 runs found under results/raw/")
        run_ids = [r for r, _ in found]
        manifests = dict(found)

    print(f"Runs merged ({len(run_ids)}):")
    for r in run_ids:
        m = manifests[r]
        print(f"  {r}  note={m.get('note')!r}")
    first = manifests[run_ids[0]]
    print(f"\nModel  : {first.get('model')}")
    print(f"Commit : {first.get('git_commit')}")

    cells = load_cells(run_ids)

    # --- Main table ---
    print("\n=== Context scaling ===")
    hdr = (f"{'ISL':>9}{'n':>4}{'TTFT p50':>12}{'prefill tok/s':>15}"
           f"{'TPOT ms':>10}{'decode tok/s':>14}{'loss':>8}{'CV%':>7}")
    print(hdr)
    print("-" * len(hdr))
    base = cells[0]["tpot_mean"]
    for c in cells:
        loss = (c["tpot_mean"] / base - 1) * 100
        ttft = (f"{c['ttft_p50']/1000:>11.2f}s" if c["ttft_p50"] >= 1000
                else f"{c['ttft_p50']:>10.1f}ms")
        print(f"{c['isl_actual']:>9.0f}{c['n']:>4}{ttft}"
              f"{c['prefill_tok_s']:>15.0f}{c['tpot_mean']:>10.2f}"
              f"{c['speed_tok_s']:>14.1f}{loss:>7.1f}%{c['cv_pct']:>7.2f}")

    xs = [c["isl_actual"] for c in cells]
    ys = [c["ttft_p50"] for c in cells]

    # --- Prefill: linear on the short range, quadratic on the full range ---
    short = [(x, y) for x, y in zip(xs, ys) if x <= args.linear_cutoff]
    if len(short) >= 3:
        sx, sy = zip(*short)
        lin = polyfit(list(sx), list(sy), 1)
        print(f"\n=== Prefill, linear fit (ISL <= {args.linear_cutoff}) ===")
        print(f"  TTFT = {lin[0]:.2f} ms + {lin[1]*1000:.3f} us/token x ISL")
        print(f"  prefill speed : {1000/lin[1]:.0f} tok/s")
        print(f"  R^2           : {r_squared(list(sx), list(sy), lin):.5f}")
        worst = max(xs)
        print(f"  extrapolated to ISL={worst:.0f}: {evaluate(lin, worst)/1000:.2f}s "
              f"vs measured {ys[xs.index(worst)]/1000:.2f}s "
              f"({ys[xs.index(worst)]/evaluate(lin, worst)-1:+.0%})")

    if len(xs) >= 4:
        quad = polyfit(xs, ys, 2)
        lin_all = polyfit(xs, ys, 1)
        print("\n=== Prefill, full range ===")
        print(f"  linear    : R^2 = {r_squared(xs, ys, lin_all):.5f}, "
              f"intercept {lin_all[0]:.1f} ms"
              + ("   <-- negative intercept: the linear model is wrong here"
                 if lin_all[0] < 0 else ""))
        print(f"  quadratic : R^2 = {r_squared(xs, ys, quad):.5f}")
        print(f"    TTFT = {quad[0]:.2f} + {quad[1]*1000:.3f} us/token x ISL "
              f"+ {quad[2]*1e6:.5f} ns/token^2 x ISL^2")
        big = max(xs)
        share = quad[2] * big ** 2 / evaluate(quad, big) * 100
        print(f"    quadratic term accounts for {share:.0f}% of TTFT at ISL={big:.0f}")

    # --- Decode: measured vs first-principles model ---
    print("\n=== Decode degradation vs bandwidth model ===")
    print(f"  weights {WEIGHTS_GB} GB, KV {KV_BYTES_PER_TOKEN/1024:.1f} KB/token, "
          f"bandwidth {BANDWIDTH_GB_S} GB/s")
    print(f"{'ISL':>9}{'KV size':>11}{'predicted':>12}{'measured':>11}{'ratio':>8}")
    print("-" * 51)
    for c in cells:
        isl = c["isl_actual"]
        pred = predicted_decode_loss(isl)
        meas = (c["tpot_mean"] / base - 1) * 100
        kv_mb = isl * KV_BYTES_PER_TOKEN / 1e6
        ratio = meas / pred if pred > 0.05 else float("nan")
        kv_s = f"{kv_mb/1000:.2f} GB" if kv_mb >= 1000 else f"{kv_mb:.0f} MB"
        r_s = f"{ratio:>7.2f}" if ratio == ratio else "      -"
        print(f"{isl:>9.0f}{kv_s:>11}{pred:>11.1f}%{meas:>10.1f}%{r_s}")
    print("\n  ratio < 1 means the model reads KV cache more efficiently than")
    print("  weights, so the pure bandwidth model is pessimistic.")

    # --- CSV ---
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PROC_DIR / f"{args.label}_context_scaling.csv"
    for c in cells:
        c["decode_loss_pct"] = (c["tpot_mean"] / base - 1) * 100
        c["predicted_loss_pct"] = predicted_decode_loss(c["isl_actual"])
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
        w.writeheader()
        w.writerows(cells)
    print(f"\nCSV  : {out_csv}")

    if not args.plot:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: uv pip install matplotlib")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: TTFT with linear extrapolation for contrast
    ax1.plot(xs, [y / 1000 for y in ys], "o-", color="#ff7f0e", lw=2,
             label="Measured TTFT p50")
    if len(short) >= 3:
        grid = [min(xs) + i * (max(xs) - min(xs)) / 100 for i in range(101)]
        ax1.plot(grid, [evaluate(lin, g) / 1000 for g in grid], "--",
                 color="gray", lw=1.4,
                 label=f"Linear model (fit on ISL<={args.linear_cutoff})")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{int(x/1024)}k" if x >= 1024 else str(int(x)) for x in xs])
    ax1.set_xlabel("Input length (tokens)")
    ax1.set_ylabel("Time to first token (s)")
    ax1.set_title("Prefill cost: quadratic term emerges past 16k", fontsize=11)
    ax1.grid(alpha=.3)
    ax1.legend(fontsize=9)

    # Panel 2: decode speed, measured vs predicted
    ax2.plot(xs, [c["speed_tok_s"] for c in cells], "s-", color="#1f77b4",
             lw=2, label="Measured decode speed")
    ax2.plot(xs, [(1000 / base) / (1 + predicted_decode_loss(x) / 100) for x in xs],
             "^--", color="gray", lw=1.4, label="Bandwidth model prediction")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{int(x/1024)}k" if x >= 1024 else str(int(x)) for x in xs])
    ax2.set_xlabel("Input length (tokens)")
    ax2.set_ylabel("Decode speed (tokens/s)")
    ax2.set_ylim(0, 115)
    ax2.set_title("Decode slowdown from KV cache reads", fontsize=11)
    ax2.grid(alpha=.3)
    ax2.legend(fontsize=9)

    model = (first.get("model") or "").split("/")[-1]
    fig.suptitle(f"T3 — Context scaling — {model} on RTX 4070 Ti SUPER "
                 f"(OSL=128, concurrency=1)", fontsize=11)
    fig.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = PLOT_DIR / f"{args.label}_context_scaling.png"
    fig.savefig(out_png, dpi=150)
    print(f"Plot : {out_png}")


if __name__ == "__main__":
    main()
